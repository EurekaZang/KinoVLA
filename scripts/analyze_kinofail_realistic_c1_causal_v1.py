#!/usr/bin/env python3
"""Analyze the frozen realistic C1 causal matched battery."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _weights(case_ids: np.ndarray) -> np.ndarray:
    values, counts = np.unique(case_ids.astype(str), return_counts=True)
    lookup = dict(zip(values.tolist(), counts.tolist(), strict=True))
    return np.asarray([1.0 / lookup[value] for value in case_ids], dtype=np.float64)


def _model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=2026072411,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _scene_auc(
    truth: np.ndarray, probability: np.ndarray, scenes: np.ndarray
) -> dict[str, float]:
    return {
        scene: float(roc_auc_score(truth[scenes == scene], probability[scenes == scene]))
        for scene in sorted(set(scenes.astype(str)))
    }


def _bootstrap_scene_mean(
    scene_values: dict[str, float], *, draws: int, seed: int
) -> list[float]:
    values = np.asarray(list(scene_values.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(int(draws), len(values)), replace=True).mean(
        axis=1
    )
    return [
        float(np.percentile(sampled, 2.5)),
        float(np.percentile(sampled, 97.5)),
    ]


def _texture_consistency(
    rows: list[dict[str, Any]], probability: np.ndarray
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(str(row["case_id"]), str(row["target_operator"]))].append(index)
    values = []
    for (case_id, cause), indices in sorted(groups.items()):
        if len(indices) != 3:
            raise RuntimeError("C1 texture consistency group is not three views")
        scores = probability[indices]
        predictions = scores >= 0.5
        values.append(
            {
                "case_id": case_id,
                "target_operator": cause,
                "hard_agreement": bool(np.all(predictions == predictions[0])),
                "probability_range": float(scores.max() - scores.min()),
            }
        )
    return {
        "group_count": len(values),
        "hard_consistency": float(np.mean([row["hard_agreement"] for row in values])),
        "mean_probability_range": float(
            np.mean([row["probability_range"] for row in values])
        ),
        "maximum_probability_range": float(
            np.max([row["probability_range"] for row in values])
        ),
        "groups": values,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    feature_dir = args.features.resolve()
    output = args.output.resolve()
    protocol = _json(protocol_path)
    feature_manifest_path = feature_dir / "feature_manifest.json"
    feature_manifest = _json(feature_manifest_path)
    records_path = feature_dir / "records.jsonl"
    features_path = feature_dir / "features.npz"
    if feature_manifest.get("passed") is not True:
        raise RuntimeError("C1 feature manifest is not complete")
    if (
        _sha(records_path) != feature_manifest["output_sha256"]["records"]
        or _sha(features_path) != feature_manifest["output_sha256"]["features"]
    ):
        raise RuntimeError("C1 feature outputs no longer match their manifest")
    if feature_manifest["protocol_id"] != protocol["protocol_id"]:
        raise RuntimeError("C1 features and analysis protocol do not match")

    rows = _jsonl(records_path)
    archive = np.load(features_path, allow_pickle=False)
    if archive["sample_ids"].astype(str).tolist() != [
        str(row["sample_id"]) for row in rows
    ]:
        raise RuntimeError("C1 feature rows are misaligned")
    visual = np.asarray(archive["visual"], dtype=np.float32)
    proprio = np.asarray(archive["proprio"], dtype=np.float32)
    truth = np.asarray(
        [row["target_operator"] == "O4_tether" for row in rows], dtype=np.int64
    )
    splits = np.asarray([str(row["split"]) for row in rows])
    scenes = np.asarray([str(row["scene_cluster"]) for row in rows])
    domains = np.asarray([str(row["domain"]) for row in rows])
    case_ids = np.asarray([str(row["case_id"]) for row in rows])
    train = splits == "train"
    val = splits == "val"
    test = splits == "test"
    if not all(mask.any() for mask in (train, val, test)):
        raise RuntimeError("C1 train/val/test split is incomplete")

    selection = []
    candidates = [
        float(value) for value in protocol["frozen_analysis"]["candidate_C"]
    ]
    for c_value in candidates:
        candidate = _model(c_value)
        candidate.fit(
            visual[train],
            truth[train],
            classifier__sample_weight=_weights(case_ids[train]),
        )
        probability = candidate.predict_proba(visual[val])[:, 1]
        per_scene = _scene_auc(truth[val], probability, scenes[val])
        selection.append(
            {
                "C": c_value,
                "validation_macro_within_scene_auc": float(
                    np.mean(list(per_scene.values()))
                ),
                "validation_per_scene_auc": per_scene,
            }
        )
    selected = sorted(
        selection,
        key=lambda row: (
            -row["validation_macro_within_scene_auc"],
            row["C"],
        ),
    )[0]
    selected_c = float(selected["C"])
    fit = train | val
    visual_model = _model(selected_c)
    visual_model.fit(
        visual[fit],
        truth[fit],
        classifier__sample_weight=_weights(case_ids[fit]),
    )
    proprio_model = _model(selected_c)
    proprio_model.fit(
        proprio[fit],
        truth[fit],
        classifier__sample_weight=_weights(case_ids[fit]),
    )
    visual_probability = visual_model.predict_proba(visual[test])[:, 1]
    proprio_probability = proprio_model.predict_proba(proprio[test])[:, 1]
    test_rows = [row for row, keep in zip(rows, test, strict=True) if keep]
    test_truth = truth[test]
    test_scenes = scenes[test]
    per_scene = _scene_auc(test_truth, visual_probability, test_scenes)
    macro_auc = float(np.mean(list(per_scene.values())))
    bootstrap = protocol["frozen_analysis"]["uncertainty"]
    macro_ci = _bootstrap_scene_mean(
        per_scene,
        draws=int(bootstrap["draws"]),
        seed=int(bootstrap["seed"]),
    )

    paired_probability_deltas = []
    index_by_pair: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(test_rows):
        index_by_pair[
            (str(row["case_id"]), str(row["appearance_view_id"]))
        ][str(row["target_operator"])] = index
    for causes in index_by_pair.values():
        if set(causes) != {"O2_compliance", "O4_tether"}:
            raise RuntimeError("C1 test pair is incomplete")
        paired_probability_deltas.append(
            abs(
                float(proprio_probability[causes["O2_compliance"]])
                - float(proprio_probability[causes["O4_tether"]])
            )
        )
    proprio_hash_identity = float(
        np.mean(
            [
                len(
                    {
                        row["shared_proprio_sha256"]
                        for row in test_rows
                        if row["case_id"] == case_id
                    }
                )
                == 1
                for case_id in sorted(set(row["case_id"] for row in test_rows))
            ]
        )
    )
    texture = _texture_consistency(test_rows, visual_probability)
    acceptance = protocol["acceptance"]
    checks = {
        "all_case_manifests_pass": bool(
            feature_manifest["checks"]["all_case_manifests_pass"]
        ),
        "minimum_test_scene_clusters": (
            len(set(test_scenes)) >= int(acceptance["minimum_test_scene_clusters"])
        ),
        "minimum_test_domains": (
            len(set(domains[test])) >= int(acceptance["minimum_test_domains"])
        ),
        "proprio_byte_identity_rate": (
            proprio_hash_identity == float(acceptance["proprio_byte_identity_rate"])
        ),
        "maximum_paired_proprio_probability_delta": (
            max(paired_probability_deltas)
            <= float(acceptance["maximum_paired_proprio_probability_delta"])
        ),
        "proprio_roc_auc_target": (
            float(roc_auc_score(test_truth, proprio_probability))
            == float(acceptance["proprio_roc_auc_target"])
        ),
        "minimum_visual_macro_within_scene_auc": (
            macro_auc >= float(acceptance["minimum_visual_macro_within_scene_auc"])
        ),
        "minimum_visual_per_scene_auc": (
            min(per_scene.values())
            >= float(acceptance["minimum_visual_per_scene_auc"])
        ),
        "minimum_visual_scene_bootstrap_ci95_lower": (
            macro_ci[0]
            >= float(acceptance["minimum_visual_scene_bootstrap_ci95_lower"])
        ),
        "minimum_texture_swap_hard_consistency": (
            texture["hard_consistency"]
            >= float(acceptance["minimum_texture_swap_hard_consistency"])
        ),
    }

    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"
    predictions = []
    for index, row in enumerate(test_rows):
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "case_id": row["case_id"],
                "scene_cluster": row["scene_cluster"],
                "domain": row["domain"],
                "target_operator": row["target_operator"],
                "appearance_view_id": row["appearance_view_id"],
                "truth_o4": bool(test_truth[index]),
                "visual_probability_o4": float(visual_probability[index]),
                "proprio_probability_o4": float(proprio_probability[index]),
            }
        )
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    checkpoint_path = output / "visual_model.pkl"
    with checkpoint_path.open("wb") as handle:
        pickle.dump(visual_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    proprio_checkpoint_path = output / "proprio_model.pkl"
    with proprio_checkpoint_path.open("wb") as handle:
        pickle.dump(proprio_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "schema_version": "kinofail.realistic-c1-causal-report.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha(protocol_path),
        "passed": all(checks.values()),
        "checks": checks,
        "selection": {
            "test_outcomes_used": False,
            "candidates": selection,
            "selected_C": selected_c,
            "selected_from": "validation split only",
            "refit": "train plus validation before one-shot test",
        },
        "test_counts": {
            "samples": int(test.sum()),
            "shared_prefix_cases": len(set(case_ids[test])),
            "scene_clusters": len(set(test_scenes)),
            "domains": len(set(domains[test])),
            "causes": 2,
            "appearance_views_per_cause_case": 3,
        },
        "visual": {
            "primary_macro_within_scene_roc_auc": macro_auc,
            "scene_cluster_bootstrap_ci95": macro_ci,
            "per_scene_roc_auc": per_scene,
            "pooled_roc_auc_secondary": float(
                roc_auc_score(test_truth, visual_probability)
            ),
            "texture_swap": texture,
        },
        "proprio": {
            "roc_auc": float(roc_auc_score(test_truth, proprio_probability)),
            "byte_identity_rate": proprio_hash_identity,
            "maximum_paired_probability_delta": max(paired_probability_deltas),
            "paired_comparisons": len(paired_probability_deltas),
        },
        "claim_boundary": protocol["claim_boundary"],
        "artifacts": {
            "feature_manifest_sha256": _sha(feature_manifest_path),
            "predictions": str(predictions_path.relative_to(ROOT)),
            "predictions_sha256": _sha(predictions_path),
            "visual_checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "visual_checkpoint_sha256": _sha(checkpoint_path),
            "proprio_checkpoint": str(proprio_checkpoint_path.relative_to(ROOT)),
            "proprio_checkpoint_sha256": _sha(proprio_checkpoint_path),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": checks,
                "visual": {
                    "macro_auc": macro_auc,
                    "ci95": macro_ci,
                    "per_scene": per_scene,
                    "texture_hard_consistency": texture["hard_consistency"],
                },
                "proprio": report["proprio"],
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
