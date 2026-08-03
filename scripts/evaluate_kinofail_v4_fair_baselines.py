#!/usr/bin/env python3
"""Evaluate strong scene-disjoint baselines for KiNO v4 development.

The generic baselines are fit independently per battery, which gives them a
favorable task-specific training regime.  Conflict visual inputs include the
same frozen CLIP+DINOv2 descriptors available to KiNO's visual specialist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import recall_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.conflict_invariant_kino import (  # noqa: E402
    invariant_relative_proprio_features,
    physical_group_weights,
    visual_context_features,
)
from scripts.develop_kinofail_conflict_invariant_kino_v4 import (  # noqa: E402
    _load_conflict,
    _load_scale,
)


DINO = ROOT / "outputs/eval/kino_conflict_dinov2_v4_development/features.npz"
OURS = ROOT / "outputs/eval/kino_conflict_invariant_v4_development_dino_eta020"
DEFAULT_OUTPUT = ROOT / "outputs/eval/kino_v4_fair_baselines_development"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _estimator(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=512,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=-1,
    )


def _aligned(
    model: ExtraTreesClassifier, values: np.ndarray, classes: list[str]
) -> np.ndarray:
    raw = model.predict_proba(values)
    lookup = {str(label): index for index, label in enumerate(model.classes_)}
    output = np.zeros((len(values), len(classes)), dtype=np.float64)
    for index, label in enumerate(classes):
        if label in lookup:
            output[:, index] = raw[:, lookup[label]]
    normalizer = output.sum(axis=1, keepdims=True)
    return output / np.where(normalizer <= 0.0, 1.0, normalizer)


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    classes = sorted(set(labels.tolist()))
    recalls = recall_score(labels, predictions, labels=classes, average=None)
    return {
        "samples": len(labels),
        "accuracy": float(np.mean(labels == predictions)),
        "balanced_accuracy": float(recalls.mean()),
        "per_class_recall": {
            label: float(value)
            for label, value in zip(classes, recalls, strict=True)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dino", type=Path, default=DINO)
    parser.add_argument(
        "--dino-features",
        default="cls",
        help="Comma-separated frozen DINOv2 pools used in Conflict.",
    )
    parser.add_argument("--ours", type=Path, default=OURS)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    scale = _load_scale()
    conflict = _load_conflict()
    dino_path = args.dino
    if not dino_path.is_absolute():
        dino_path = ROOT / dino_path
    dino_feature_names = [
        value.strip()
        for value in str(args.dino_features).split(",")
        if value.strip()
    ]
    if not dino_feature_names:
        raise ValueError("at least one DINO feature pool is required")
    with np.load(dino_path, allow_pickle=False) as archive:
        if not np.array_equal(
            archive["sample_ids"].astype(str), conflict["sample_ids"]
        ):
            raise RuntimeError("Conflict DINO cache does not align with records")
        missing = [
            name
            for name in dino_feature_names
            if f"visual_{name}" not in archive.files
        ]
        if missing:
            raise KeyError(f"missing Conflict DINO pools: {missing}")
        conflict_dino = np.concatenate(
            [
                np.asarray(archive[f"visual_{name}"], dtype=np.float32)
                for name in dino_feature_names
            ],
            axis=1,
        )
    scale["visual_v4"] = visual_context_features(scale["visual"])
    conflict["visual_v4"] = np.concatenate(
        [visual_context_features(conflict["visual"]), conflict_dino], axis=1
    )
    for source in (scale, conflict):
        source["proprio_v4"] = invariant_relative_proprio_features(
            source["invariant"], source["full"]
        )

    ours_path = args.ours
    if not ours_path.is_absolute():
        ours_path = ROOT / ours_path
    ours_rows = [
        json.loads(line)
        for line in (ours_path / "scene_disjoint_predictions.jsonl").read_text().splitlines()
        if line
    ]
    ours_lookup = {
        (str(row["dataset"]), str(row["sample_id"])): str(row["prediction"])
        for row in ours_rows
    }
    prediction_rows: list[dict[str, object]] = []
    fold_reports: dict[str, object] = {}
    for fold in range(5):
        fold_report: dict[str, object] = {}
        for dataset_name, source in (("scale", scale), ("conflict", conflict)):
            train = source["folds"] != fold
            test = ~train
            classes = sorted(set(source["labels"].tolist()))
            weights = physical_group_weights(source["groups"][train])
            visual_model = _estimator(2026081300 + fold).fit(
                source["visual_v4"][train],
                source["labels"][train],
                sample_weight=weights,
            )
            proprio_model = _estimator(2026082300 + fold).fit(
                source["proprio_v4"][train],
                source["labels"][train],
                sample_weight=weights,
            )
            early_values = np.concatenate(
                [source["visual_v4"], source["proprio_v4"]], axis=1
            )
            early_model = _estimator(2026083300 + fold).fit(
                early_values[train],
                source["labels"][train],
                sample_weight=weights,
            )
            visual_probability = _aligned(
                visual_model, source["visual_v4"][test], classes
            )
            proprio_probability = _aligned(
                proprio_model, source["proprio_v4"][test], classes
            )
            early_probability = _aligned(
                early_model, early_values[test], classes
            )
            probability = {
                "vision": visual_probability,
                "proprioception": proprio_probability,
                "joint_early_fusion": early_probability,
                "late_fusion": 0.5 * (visual_probability + proprio_probability),
            }
            if dataset_name == "conflict":
                cells = source["cells"][test]
                oracle = proprio_probability.copy()
                oracle[cells == "T2_vision_decisive"] = visual_probability[
                    cells == "T2_vision_decisive"
                ]
                probability["oracle_route"] = oracle
            sample_ids = source["sample_ids"][test]
            labels = source["labels"][test]
            method_predictions = {
                method: np.asarray(classes)[values.argmax(axis=1)]
                for method, values in probability.items()
            }
            method_predictions["kino_v4"] = np.asarray(
                [ours_lookup[(dataset_name, str(sample_id))] for sample_id in sample_ids]
            )
            fold_report[dataset_name] = {
                method: _metrics(labels, predictions)
                for method, predictions in method_predictions.items()
            }
            for index, sample_id in enumerate(sample_ids):
                for method, predictions in method_predictions.items():
                    prediction_rows.append(
                        {
                            "fold": fold,
                            "dataset": dataset_name,
                            "sample_id": str(sample_id),
                            "truth": str(labels[index]),
                            "method": method,
                            "prediction": str(predictions[index]),
                            "group": str(source["groups"][test][index]),
                            "scene": str(source["scenes"][test][index]),
                            **(
                                {"cell": str(source["cells"][test][index])}
                                if dataset_name == "conflict"
                                else {}
                            ),
                        }
                    )
        fold_reports[str(fold)] = fold_report
        print(
            json.dumps(
                {
                    "fold": fold,
                    "scale": {
                        method: values["balanced_accuracy"]
                        for method, values in fold_report["scale"].items()
                    },
                    "conflict": {
                        method: values["balanced_accuracy"]
                        for method, values in fold_report["conflict"].items()
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )

    aggregate: dict[str, dict[str, object]] = defaultdict(dict)
    for dataset_name in ("scale", "conflict"):
        methods = sorted(
            {
                str(row["method"])
                for row in prediction_rows
                if row["dataset"] == dataset_name
            }
        )
        for method in methods:
            selected = [
                row
                for row in prediction_rows
                if row["dataset"] == dataset_name and row["method"] == method
            ]
            aggregate[dataset_name][method] = _metrics(
                np.asarray([row["truth"] for row in selected]),
                np.asarray([row["prediction"] for row in selected]),
            )
    cross_battery = {}
    deployable = sorted(set(aggregate["scale"]) & set(aggregate["conflict"]))
    for method in deployable:
        scale_score = float(aggregate["scale"][method]["balanced_accuracy"])
        conflict_score = float(aggregate["conflict"][method]["balanced_accuracy"])
        cross_battery[method] = {
            "scale": scale_score,
            "conflict": conflict_score,
            "macro": float(0.5 * (scale_score + conflict_score)),
            "worst": float(min(scale_score, conflict_score)),
        }

    output.mkdir(parents=True, exist_ok=False)
    predictions_path = output / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows)
    )
    report = {
        "schema_version": "kinofail.v4-fair-baselines-development.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "five_fold_scene_disjoint_development_complete",
        "confirmatory_evidence": False,
        "baseline_advantage": (
            "generic baselines are independently fit per battery; KiNO remains one "
            "shared architecture"
        ),
        "features": {
            "scale_visual": "frozen CLIP mean+final",
            "conflict_visual": (
                "frozen CLIP mean+final plus DINOv2 "
                + "+".join(dino_feature_names)
            ),
            "proprioception": "80-D invariant plus 171-D relative dynamics",
        },
        "folds": fold_reports,
        "aggregate": aggregate,
        "cross_battery": cross_battery,
        "source_sha256": {
            "conflict_dino": _sha256(dino_path),
            "kino_v4_predictions": _sha256(
                ours_path / "scene_disjoint_predictions.jsonl"
            ),
        },
        "artifacts": {
            "predictions": str(predictions_path.relative_to(ROOT)),
            "predictions_sha256": _sha256(predictions_path),
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(cross_battery, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
