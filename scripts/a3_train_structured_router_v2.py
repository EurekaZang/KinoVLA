#!/usr/bin/env python
"""Train five structured evidence-router v2 seeds on method-development corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier

from kino_vla.eval.a3_successor_stats import accuracy_metrics, appearance_cluster
from kino_vla.eval.a7_ablation import load_yaml, repo_path, write_json
from kino_vla.eval.evidence_routed_v2 import (
    StructuredEvidencePolicyV2,
    StructuredFeatures,
    observable_features,
)


@dataclass(frozen=True)
class Row:
    meta: dict[str, Any]
    features: StructuredFeatures
    material_regime: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _t3_sub(record: dict[str, Any]) -> str:
    if record.get("a3_direction") in {"looks_safe", "reverse"}:
        return str(record["a3_direction"])
    if record["snapshot"]["operator_name"] == "O8_invisible_collider":
        return "O8"
    return "other"


def _truth(record: dict[str, Any]) -> str:
    if record["taxonomy_cell"] == "T5" or record.get("a3_direction") == "reverse":
        return "nominal"
    return str(record["ground_truth"]["category"])


def _load_rows(cfg: dict[str, Any]) -> tuple[list[Row], dict[str, Any], dict[str, str]]:
    checkpoint = torch.load(
        repo_path(cfg["sources"]["material_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    material_models = checkpoint["material_models"]
    material_classes = tuple(checkpoint["material_classes"])
    rows = []
    evidence = {}
    for source_name, value in cfg["sources"].items():
        if source_name == "material_checkpoint":
            continue
        directory = repo_path(value)
        samples_path = directory / "samples.jsonl"
        frames_path = directory / "frames.npz"
        records = [
            json.loads(line) for line in samples_path.read_text().splitlines() if line
        ]
        frames = np.load(frames_path)
        for record in records:
            sid = str(record["sample_id"])
            truth = _truth(record)
            cell = str(record["taxonomy_cell"])
            meta = {
                "sid": sid,
                "cell": cell,
                "t3_sub": _t3_sub(record),
                "truth": truth,
                "appearance_id": str(record["appearance_id"]),
                "appearance_split": str(record["appearance_split"]),
                "source": source_name,
            }
            rows.append(
                Row(
                    meta=meta,
                    features=observable_features(
                        np.asarray(frames[f"{sid}__rgb"][-1], dtype=np.float32),
                        np.asarray(frames[f"{sid}__proprio"], dtype=np.float32),
                        material_models,
                        material_classes,
                    ),
                    material_regime=cell == "T2"
                    or (cell == "T1" and truth == "compliant_terrain"),
                )
            )
        evidence[source_name] = {
            "directory": str(directory),
            "n": len(records),
            "samples_sha256": _sha256(samples_path),
            "frames_sha256": _sha256(frames_path),
        }
    material = {
        "material_models": material_models,
        "material_classes": list(material_classes),
        "categories": list(checkpoint["categories"]),
    }
    hashes = {str(repo_path(cfg["sources"]["material_checkpoint"])): _sha256(
        repo_path(cfg["sources"]["material_checkpoint"])
    )}
    source_hashes = {
        name: item["samples_sha256"] for name, item in evidence.items()
    }
    return rows, material, {**hashes, **source_hashes}


def _cluster_split(rows: list[Row], *, seed: int, fraction: float) -> tuple[list[Row], list[Row]]:
    by_stratum: dict[str, dict[str, list[Row]]] = {}
    for row in rows:
        case = row.meta["t3_sub"] if row.meta["cell"] == "T3" else row.meta["truth"]
        stratum = f"{row.meta['cell']}|{case}"
        cluster = appearance_cluster(row.meta)
        by_stratum.setdefault(stratum, {}).setdefault(cluster, []).append(row)
    dev_clusters = set()
    for stratum, clusters in sorted(by_stratum.items()):
        ordered = sorted(
            clusters,
            key=lambda cluster: hashlib.sha256(f"{seed}|{stratum}|{cluster}".encode()).digest(),
        )
        count = max(1, int(round(float(fraction) * len(ordered))))
        if count >= len(ordered):
            count = len(ordered) - 1
        if count <= 0:
            raise ValueError(f"stratum {stratum} needs at least two appearance clusters")
        dev_clusters.update(ordered[:count])
    train = [row for row in rows if appearance_cluster(row.meta) not in dev_clusters]
    dev = [row for row in rows if appearance_cluster(row.meta) in dev_clusters]
    return train, dev


def _estimator(cfg: dict[str, Any], seed: int) -> ExtraTreesClassifier:
    model = cfg["model"]
    return ExtraTreesClassifier(
        n_estimators=int(model["n_estimators"]),
        max_features=str(model["max_features"]),
        class_weight=str(model["class_weight"]),
        min_samples_leaf=int(model["min_samples_leaf"]),
        random_state=int(seed),
        n_jobs=-1,
    )


def _fit_policy(
    rows: list[Row],
    cfg: dict[str, Any],
    material: dict[str, Any],
    seed: int,
    threshold: float,
) -> StructuredEvidencePolicyV2:
    joint = np.stack([row.features.joint for row in rows])
    proprio = np.stack([row.features.proprio for row in rows])
    intervention = _estimator(cfg, seed).fit(
        joint, [row.meta["truth"] != "nominal" for row in rows]
    )
    regime = _estimator(cfg, seed + 1000).fit(
        proprio, [row.material_regime for row in rows]
    )
    anomalous = [row for row in rows if row.meta["truth"] != "nominal"]
    category = _estimator(cfg, seed + 2000).fit(
        np.stack([row.features.joint for row in anomalous]),
        [row.meta["truth"] for row in anomalous],
    )
    return StructuredEvidencePolicyV2(
        intervention_model=intervention,
        regime_model=regime,
        category_model=category,
        material_models=material["material_models"],
        material_classes=material["material_classes"],
        categories=material["categories"],
        continue_threshold=threshold,
        material_pair_logit_scale=float(cfg["model"]["material_pair_logit_scale"]),
    )


def _score(policy: StructuredEvidencePolicyV2, rows: list[Row]) -> list[dict[str, Any]]:
    predictions, p_intervene, p_material = policy.predict_features(
        [row.features for row in rows]
    )
    return [
        {
            **row.meta,
            "attribution": prediction,
            "attr_ok": prediction == row.meta["truth"],
            "p_intervene": float(intervention),
            "p_material_regime": float(material_probability),
        }
        for row, prediction, intervention, material_probability in zip(
            rows, predictions, p_intervene, p_material, strict=True
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train A3 structured router v2")
    parser.add_argument("--config", default="configs/eval/a3_successor_v2.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    rows, material, source_hashes = _load_rows(cfg)
    output = repo_path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in [int(value) for value in cfg["training"]["seeds"]]:
        train, dev = _cluster_split(
            rows,
            seed=seed,
            fraction=float(cfg["protocol"]["cluster_development_fraction"]),
        )
        selection_model = _fit_policy(train, cfg, material, seed, threshold=0.5)
        best = None
        best_scored = []
        for threshold in [float(value) for value in cfg["selection"]["continue_thresholds"]]:
            selection_model.continue_threshold = threshold
            scored = _score(selection_model, dev)
            metrics = accuracy_metrics(scored)
            key = (metrics["worst_cell"], metrics["macro"])
            if best is None or key > best[0]:
                best = (key, threshold, metrics)
                best_scored = scored
        assert best is not None
        final_model = _fit_policy(rows, cfg, material, seed, threshold=best[1])
        seed_dir = output / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        with (seed_dir / "router_v2.pkl").open("wb") as handle:
            pickle.dump(final_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        write_json(seed_dir / "per_item_cluster_dev.json", best_scored)
        in_sample = _score(final_model, rows)
        write_json(seed_dir / "per_item_method_development_fit.json", in_sample)
        result = {
            "seed": seed,
            "n_train_cluster_split": len(train),
            "n_dev_cluster_split": len(dev),
            "n_final_fit": len(rows),
            "continue_threshold": best[1],
            "cluster_dev_metrics": best[2],
            "method_development_fit_metrics_not_confirmatory": accuracy_metrics(in_sample),
        }
        write_json(seed_dir / "metrics.json", result)
        results.append(result)
    summary = {
        "schema_version": 1,
        "experiment": cfg["experiment"],
        "status": "method_development_ready_for_new_confirmation",
        "revision_reason": cfg["revision_reason"],
        "n_rows": len(rows),
        "n_appearance_clusters": len({appearance_cluster(row.meta) for row in rows}),
        "source_sha256": source_hashes,
        "failed_confirmation_preserved": (
            "outputs/eval/a3_successor/confirmatory_eval/evaluation_manifest.json"
        ),
        "deployment_inputs": [
            "proprio temporal distribution summary",
            "train-fitted material prototype distances",
            "observable colour chroma",
        ],
        "forbidden_deployment_inputs": cfg["protocol"]["forbidden_deployment_inputs"],
        "results": results,
    }
    write_json(output / "training_summary.json", summary)
    print(json.dumps({"seeds": [item["seed"] for item in results], "results": results}, indent=2))


if __name__ == "__main__":
    main()
