#!/usr/bin/env python3
"""One-shot analysis of the untouched realistic C1 confirmation extension."""

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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _load(
    feature_dir: Path, geometry_dir: Path
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    feature_manifest_path = feature_dir / "feature_manifest.json"
    feature_manifest = _json(feature_manifest_path)
    records_path = feature_dir / "records.jsonl"
    features_path = feature_dir / "features.npz"
    geometry_manifest_path = geometry_dir / "geometry_manifest.json"
    geometry_manifest = _json(geometry_manifest_path)
    geometry_path = geometry_dir / "geometry.npz"
    if feature_manifest.get("passed") is not True:
        raise RuntimeError("C1 confirmation refuses incomplete CLIP features")
    if geometry_manifest.get("passed") is not True:
        raise RuntimeError("C1 confirmation refuses incomplete geometry features")
    if (
        _sha(records_path) != feature_manifest["output_sha256"]["records"]
        or _sha(features_path) != feature_manifest["output_sha256"]["features"]
        or _sha(geometry_path) != geometry_manifest["output_sha256"]
    ):
        raise RuntimeError("C1 confirmation feature provenance mismatch")
    rows = _jsonl(records_path)
    archive = np.load(features_path, allow_pickle=False)
    geometry_archive = np.load(geometry_path, allow_pickle=False)
    sample_ids = np.asarray([str(row["sample_id"]) for row in rows])
    if (
        archive["sample_ids"].astype(str).tolist() != sample_ids.tolist()
        or geometry_archive["sample_ids"].astype(str).tolist()
        != sample_ids.tolist()
    ):
        raise RuntimeError("C1 confirmation feature streams are misaligned")
    return (
        rows,
        np.asarray(archive["visual"], dtype=np.float32),
        np.asarray(archive["proprio"], dtype=np.float32),
        np.asarray(geometry_archive["geometry"], dtype=np.float32),
        {
            "feature_manifest_path": feature_manifest_path,
            "feature_manifest": feature_manifest,
            "geometry_manifest_path": geometry_manifest_path,
            "geometry_manifest": geometry_manifest,
        },
    )


def _model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=2026072414,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _scene_auc(
    truth: np.ndarray, probability: np.ndarray, scenes: np.ndarray
) -> dict[str, float]:
    return {
        scene: float(
            roc_auc_score(
                truth[scenes == scene], probability[scenes == scene]
            )
        )
        for scene in sorted(set(scenes.astype(str)))
    }


def _scene_bootstrap(
    values: dict[str, float], *, draws: int, seed: int
) -> list[float]:
    array = np.asarray(list(values.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        array, size=(int(draws), len(array)), replace=True
    ).mean(axis=1)
    return [
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    ]


def _texture(
    rows: list[dict[str, Any]], probability: np.ndarray
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(str(row["case_id"]), str(row["target_operator"]))].append(index)
    detail = []
    for (case_id, operator), indices in sorted(groups.items()):
        if len(indices) != 3:
            raise RuntimeError("C1 confirmation texture group is incomplete")
        values = probability[indices]
        prediction = values >= 0.5
        detail.append(
            {
                "case_id": case_id,
                "target_operator": operator,
                "hard_agreement": bool(
                    np.all(prediction == prediction[0])
                ),
                "probability_range": float(values.max() - values.min()),
            }
        )
    return {
        "groups": len(detail),
        "hard_consistency": float(
            np.mean([row["hard_agreement"] for row in detail])
        ),
        "mean_probability_range": float(
            np.mean([row["probability_range"] for row in detail])
        ),
        "maximum_probability_range": float(
            np.max([row["probability_range"] for row in detail])
        ),
        "detail": detail,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--development-features", type=Path, required=True)
    parser.add_argument("--development-geometry", type=Path, required=True)
    parser.add_argument("--confirmation-features", type=Path, required=True)
    parser.add_argument("--confirmation-geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = _json(protocol_path)
    if protocol.get("analyzer_sha256") != _sha(Path(__file__).resolve()):
        raise RuntimeError("C1 confirmation analyzer hash mismatch")

    dev = _load(
        args.development_features.resolve(),
        args.development_geometry.resolve(),
    )
    confirm = _load(
        args.confirmation_features.resolve(),
        args.confirmation_geometry.resolve(),
    )
    dev_rows, dev_visual, dev_proprio, dev_geometry, dev_meta = dev
    rows, visual, proprio, geometry, meta = confirm
    frozen = protocol["frozen_analysis"]
    for key, actual in (
        (
            "development_feature_manifest_sha256",
            _sha(dev_meta["feature_manifest_path"]),
        ),
        (
            "development_geometry_manifest_sha256",
            _sha(dev_meta["geometry_manifest_path"]),
        ),
    ):
        if protocol.get(key) != actual:
            raise RuntimeError(f"C1 confirmation {key} mismatch")
    if {str(row["split"]) for row in rows} != {"test"}:
        raise RuntimeError("C1 confirmation extension must be test-only")
    if (
        meta["feature_manifest"]["protocol_id"]
        != protocol["protocol_id"]
    ):
        raise RuntimeError("C1 confirmation protocol ID mismatch")

    dev_truth = np.asarray(
        [row["target_operator"] == "O4_tether" for row in dev_rows],
        dtype=np.int64,
    )
    truth = np.asarray(
        [row["target_operator"] == "O4_tether" for row in rows],
        dtype=np.int64,
    )
    train_visual = np.concatenate([dev_visual, dev_geometry], axis=1)
    test_visual = np.concatenate([visual, geometry], axis=1)
    visual_model = _model(float(frozen["visual_C"]))
    visual_model.fit(train_visual, dev_truth)
    proprio_model = _model(float(frozen["proprio_C"]))
    proprio_model.fit(dev_proprio, dev_truth)
    visual_probability = visual_model.predict_proba(test_visual)[:, 1]
    proprio_probability = proprio_model.predict_proba(proprio)[:, 1]
    prediction = visual_probability >= float(
        frozen["hard_prediction_threshold"]
    )
    scenes = np.asarray([str(row["scene_cluster"]) for row in rows])
    domains = np.asarray([str(row["domain"]) for row in rows])
    per_scene = _scene_auc(truth, visual_probability, scenes)
    macro_auc = float(np.mean(list(per_scene.values())))
    ci = _scene_bootstrap(
        per_scene,
        draws=int(frozen["uncertainty"]["draws"]),
        seed=int(frozen["uncertainty"]["seed"]),
    )
    texture = _texture(rows, visual_probability)

    by_pair: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(rows):
        by_pair[
            (str(row["case_id"]), str(row["appearance_view_id"]))
        ][str(row["target_operator"])] = index
    deltas = []
    for pair in by_pair.values():
        if set(pair) != {"O2_compliance", "O4_tether"}:
            raise RuntimeError("C1 confirmation paired case is incomplete")
        deltas.append(
            abs(
                float(proprio_probability[pair["O2_compliance"]])
                - float(proprio_probability[pair["O4_tether"]])
            )
        )
    identity = float(
        np.mean(
            [
                len(
                    {
                        row["shared_proprio_sha256"]
                        for row in rows
                        if row["case_id"] == case_id
                    }
                )
                == 1
                for case_id in sorted(set(row["case_id"] for row in rows))
            ]
        )
    )
    acceptance = protocol["acceptance"]["c1_confirmation"]
    checks = {
        "all_case_manifests_pass": bool(
            meta["feature_manifest"]["checks"]["all_case_manifests_pass"]
        ),
        "fresh_case_ids": set(row["case_id"] for row in rows).isdisjoint(
            {row["case_id"] for row in dev_rows}
        ),
        "minimum_scene_clusters": len(set(scenes))
        >= int(acceptance["minimum_scene_clusters"]),
        "minimum_domains": len(set(domains))
        >= int(acceptance["minimum_domains"]),
        "minimum_macro_scene_auc": macro_auc
        >= float(acceptance["minimum_macro_scene_auc"]),
        "minimum_per_scene_auc": min(per_scene.values())
        >= float(acceptance["minimum_per_scene_auc"]),
        "minimum_scene_ci95_lower": ci[0]
        >= float(acceptance["minimum_scene_ci95_lower"]),
        "minimum_balanced_accuracy": float(np.mean(prediction == truth))
        >= float(acceptance["minimum_balanced_accuracy"]),
        "minimum_texture_hard_consistency": texture["hard_consistency"]
        >= float(acceptance["minimum_texture_hard_consistency"]),
        "proprio_byte_identity_rate": identity == 1.0,
        "proprio_auc_is_chance": float(
            roc_auc_score(truth, proprio_probability)
        )
        == 0.5,
        "maximum_paired_proprio_probability_delta": max(deltas) <= 1e-12,
    }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    predictions_path = output / "predictions.jsonl"
    predictions_path.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": row["sample_id"],
                    "case_id": row["case_id"],
                    "scene_cluster": row["scene_cluster"],
                    "domain": row["domain"],
                    "target_operator": row["target_operator"],
                    "appearance_view_id": row["appearance_view_id"],
                    "truth_o4": bool(truth[index]),
                    "visual_probability_o4": float(
                        visual_probability[index]
                    ),
                    "visual_prediction_o4": bool(prediction[index]),
                    "proprio_probability_o4": float(
                        proprio_probability[index]
                    ),
                },
                sort_keys=True,
            )
            + "\n"
            for index, row in enumerate(rows)
        ),
        encoding="utf-8",
    )
    visual_checkpoint = output / "visual_model.pkl"
    with visual_checkpoint.open("wb") as handle:
        pickle.dump(visual_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    proprio_checkpoint = output / "proprio_model.pkl"
    with proprio_checkpoint.open("wb") as handle:
        pickle.dump(proprio_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "schema_version": "kinofail.realistic-c1-confirmation-report.v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha(protocol_path),
        "passed": all(checks.values()),
        "checks": checks,
        "selection": {
            "confirmation_outcomes_used": False,
            "development_source": (
                "all 27 v1 cases; v1 formal outcomes became development only "
                "after the frozen v1 consistency failure"
            ),
            "model": frozen["visual_model"],
            "visual_C": frozen["visual_C"],
            "hard_prediction_threshold": frozen[
                "hard_prediction_threshold"
            ],
        },
        "counts": {
            "samples": len(rows),
            "cases": len(set(row["case_id"] for row in rows)),
            "scene_clusters": len(set(scenes)),
            "domains": len(set(domains)),
            "appearance_views_per_cause_case": 3,
        },
        "visual": {
            "primary_macro_within_scene_roc_auc": macro_auc,
            "scene_cluster_bootstrap_ci95": ci,
            "per_scene_roc_auc": per_scene,
            "pooled_roc_auc_secondary": float(
                roc_auc_score(truth, visual_probability)
            ),
            "balanced_accuracy": float(np.mean(prediction == truth)),
            "texture_swap": texture,
        },
        "proprio": {
            "roc_auc": float(roc_auc_score(truth, proprio_probability)),
            "byte_identity_rate": identity,
            "maximum_paired_probability_delta": max(deltas),
            "paired_comparisons": len(deltas),
        },
        "claim_boundary": protocol["claim_boundary"],
        "artifacts": {
            "confirmation_feature_manifest_sha256": _sha(
                meta["feature_manifest_path"]
            ),
            "confirmation_geometry_manifest_sha256": _sha(
                meta["geometry_manifest_path"]
            ),
            "predictions": str(predictions_path.relative_to(ROOT)),
            "predictions_sha256": _sha(predictions_path),
            "visual_checkpoint": str(
                visual_checkpoint.relative_to(ROOT)
            ),
            "visual_checkpoint_sha256": _sha(visual_checkpoint),
            "proprio_checkpoint": str(
                proprio_checkpoint.relative_to(ROOT)
            ),
            "proprio_checkpoint_sha256": _sha(proprio_checkpoint),
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
                    "ci95": ci,
                    "per_scene": per_scene,
                    "balanced_accuracy": report["visual"][
                        "balanced_accuracy"
                    ],
                    "texture_hard_consistency": texture[
                        "hard_consistency"
                    ],
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
