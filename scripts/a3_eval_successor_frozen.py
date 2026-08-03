#!/usr/bin/env python
"""Evaluate frozen A3 successor checkpoints on a never-inspected confirmatory corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.eval.evidence_routed import EXPERT_NAMES, EvidenceRoutedNetwork, predict_category
from kino_vla.eval.material_support import (
    dominant_chromatic_rgb,
    matched_material_pair_logits,
    material_evidence_features,
)
from kino_vla.eval.proprio_baseline import ProprioBaselinePolicy
from kino_vla.map.clip_appearance import ClipAppearanceEncoder
from kino_vla.utils.config import load_config
from kino_vla.vla.dataset_build import _snapshot_from_record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_method(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    changed = {
        path: {"expected": expected, "actual": _sha256(Path(path))}
        for path, expected in manifest["file_sha256"].items()
        if not Path(path).exists() or _sha256(Path(path)) != expected
    }
    if changed:
        raise ValueError(f"frozen method changed before confirmatory evaluation: {changed}")
    return manifest


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


def _row(record: dict[str, Any], attribution: str | None, **extra: Any) -> dict[str, Any]:
    truth = _truth(record)
    return {
        "sid": str(record["sample_id"]),
        "cell": str(record["taxonomy_cell"]),
        "t3_sub": _t3_sub(record),
        "truth": truth,
        "appearance_id": str(record["appearance_id"]),
        "appearance_split": str(record["appearance_split"]),
        "attribution": attribution,
        "attr_ok": attribution == truth,
        **extra,
    }


def _load_corpora(
    directories: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    evidence = {}
    for directory in directories:
        source_records = [
            json.loads(line)
            for line in (directory / "samples.jsonl").read_text().splitlines()
            if line
        ]
        archive = np.load(directory / "frames.npz")
        records.extend(source_records)
        arrays.update({key: np.asarray(archive[key]) for key in archive.files})
        evidence[str(directory)] = {
            "samples_sha256": _sha256(directory / "samples.jsonl"),
            "frames_sha256": _sha256(directory / "frames.npz"),
            "n": len(source_records),
        }
    return records, arrays, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen A3 successor confirmatory evaluation")
    parser.add_argument(
        "--corpora",
        default=(
            "outputs/eval/a3_successor/confirmatory_main,"
            "outputs/eval/a3_successor/confirmatory_t3"
        ),
    )
    parser.add_argument(
        "--conflict-predictions",
        default=(
            "outputs/eval/a3_successor/confirmatory_conflict_main.json,"
            "outputs/eval/a3_successor/confirmatory_conflict_t3.json"
        ),
    )
    parser.add_argument(
        "--manifest", default="outputs/eval/a3_successor/frozen_method_manifest.json"
    )
    parser.add_argument("--out", default="outputs/eval/a3_successor/confirmatory_eval")
    parser.add_argument("--b1-model", default="outputs/eval/e2/b1_attributor/monitor")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = _verify_method(manifest_path)
    directories = [Path(value) for value in args.corpora.split(",") if value]
    records, arrays, corpus_evidence = _load_corpora(directories)
    conflict_rows = []
    conflict_evidence = {}
    for value in args.conflict_predictions.split(","):
        path = Path(value)
        conflict_rows.extend(json.loads(path.read_text()))
        conflict_evidence[str(path)] = _sha256(path)
    conflict = {str(row["sid"]): str(row.get("attribution") or "") for row in conflict_rows}
    if set(conflict) != {str(record["sample_id"]) for record in records}:
        raise ValueError("frozen conflict predictions do not exactly cover confirmatory records")

    rgbs = [
        np.asarray(arrays[f"{record['sample_id']}__rgb"][-1], dtype=np.float32)
        for record in records
    ]
    encoder = ClipAppearanceEncoder()
    clip_chunks = [
        encoder.embed_batch(rgbs[start : start + 64]).astype(np.float32)
        for start in range(0, len(rgbs), 64)
    ]
    clip = np.concatenate(clip_chunks, axis=0)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_results = {}
    for seed in manifest["training_seeds"]:
        checkpoint = Path(f"outputs/eval/a3_successor/seed{seed}/router.pt")
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        categories = list(blob["categories"])
        material_rgb = np.stack([dominant_chromatic_rgb(rgb) for rgb in rgbs])
        material_features = np.stack(
            [
                material_evidence_features(
                    value, blob["material_models"], classes=tuple(blob["material_classes"])
                )
                for value in material_rgb
            ]
        )
        visual = np.concatenate([clip, material_features], axis=1).astype(np.float32)
        material_logits = np.stack(
            [
                matched_material_pair_logits(
                    value,
                    blob["material_models"],
                    categories,
                    scale=float(blob["material_pair_logit_scale"]),
                )
                for value in material_rgb
            ]
        )
        proprio = np.stack(
            [arrays[f"{record['sample_id']}__proprio"] for record in records]
        ).astype(np.float32)
        proprio = (proprio - np.asarray(blob["proprio_mean"])) / np.asarray(
            blob["proprio_std"]
        )
        conflict_logits = np.zeros((len(records), len(categories)), dtype=np.float32)
        for index, record in enumerate(records):
            prediction = conflict[str(record["sample_id"])]
            if prediction in categories:
                conflict_logits[index, categories.index(prediction)] = float(
                    blob["conflict_logit_scale"]
                )
        model = EvidenceRoutedNetwork.build(
            visual_dim=int(blob["visual_dim"]),
            proprio_dim=int(blob["proprio_dim"]),
            n_categories=len(categories),
            hidden=int(blob["hidden"]),
            material_feature_dim=int(blob["material_feature_dim"]),
        ).to(device)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        with torch.no_grad():
            outputs = model(
                torch.from_numpy(visual).to(device),
                torch.from_numpy(proprio).to(device),
                torch.from_numpy(material_logits).to(device),
                torch.from_numpy(conflict_logits).to(device),
            )
        predictions, p_intervene, router = predict_category(
            outputs, categories, continue_threshold=float(blob["continue_threshold"])
        )
        scored = [
            _row(
                record,
                prediction,
                p_intervene=float(probability),
                router_weights={
                    name: float(weight) for name, weight in zip(EXPERT_NAMES, weights, strict=True)
                },
            )
            for record, prediction, probability, weights in zip(
                records, predictions, p_intervene, router, strict=True
            )
        ]
        (output / f"per_item_seed{seed}.json").write_text(json.dumps(scored, indent=2) + "\n")
        seed_results[int(seed)] = scored

    conflict_scored = [
        _row(record, conflict[str(record["sample_id"])]) for record in records
    ]
    (output / "per_item_B5_conflict_bi.json").write_text(
        json.dumps(conflict_scored, indent=2) + "\n"
    )
    prompt_cfg = load_config("data/hindsight.yaml")
    b1 = ProprioBaselinePolicy.from_deployed(
        prompt_cfg,
        FailureTaxonomy(prompt_cfg),
        model_path=args.b1_model,
        device="cpu",
    )
    b1_scored = []
    for record in records:
        sid = str(record["sample_id"])
        snapshot = _snapshot_from_record(
            record,
            {kind: arrays[f"{sid}__{kind}"] for kind in ("rgb", "depth", "proprio")},
        )
        decision = b1.decide(snapshot)
        attribution = decision.attribution if decision.ok else None
        b1_scored.append(_row(record, attribution))
    (output / "per_item_B1.json").write_text(json.dumps(b1_scored, indent=2) + "\n")
    meta = {
        "schema_version": 1,
        "status": "untouched_confirmatory_evaluation",
        "method_sha256": manifest["method_sha256"],
        "method_unchanged": True,
        "n": len(records),
        "n_training_seeds": len(seed_results),
        "device": str(device),
        "corpus_evidence": corpus_evidence,
        "conflict_prediction_sha256": conflict_evidence,
        "manifest_sha256": _sha256(manifest_path),
        "no_fit_or_threshold_selection_on_confirmatory": True,
    }
    (output / "evaluation_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"n": len(records), "seeds": sorted(seed_results)}, indent=2))


if __name__ == "__main__":
    main()
