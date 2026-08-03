#!/usr/bin/env python3
"""Crossed-cluster paired bootstrap for KiNO v4 development results."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.develop_kinofail_conflict_invariant_kino_v4 import (  # noqa: E402
    _load_conflict,
    _load_scale,
)


BASELINES = ROOT / "outputs/eval/kino_v4_fair_baselines_development"
DEFAULT_OUTPUT = ROOT / "outputs/eval/kino_v4_paired_statistics_development"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    return {
        "estimate": float(np.mean(finite)),
        "ci95_low": float(np.quantile(finite, 0.025)),
        "ci95_high": float(np.quantile(finite, 0.975)),
        "probability_gt_zero": float(np.mean(finite > 0.0)),
        "probability_le_minus_one_point": float(np.mean(finite <= -0.01)),
    }


def _metric(labels: np.ndarray, predictions: np.ndarray) -> float:
    classes = sorted(set(labels.tolist()))
    return float(
        np.mean(
            [
                np.mean(predictions[labels == label] == label)
                for label in classes
            ]
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baselines", type=Path, default=BASELINES)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    baselines = args.baselines
    if not baselines.is_absolute():
        baselines = ROOT / baselines
    prediction_path = baselines / "predictions.jsonl"
    rows = [
        json.loads(line) for line in prediction_path.read_text().splitlines() if line
    ]
    scale = _load_scale()
    conflict = _load_conflict()
    metadata = {
        "scale": {
            str(sample_id): str(row["cluster_material"])
            for sample_id, row in zip(scale["sample_ids"], scale["rows"], strict=True)
        },
        "conflict": {
            str(sample_id): str(row["cluster_material"])
            for sample_id, row in zip(
                conflict["sample_ids"], conflict["rows"], strict=True
            )
        },
    }
    scenes = sorted({str(row["scene"]) for row in rows})
    materials = sorted(
        {
            metadata[str(row["dataset"])][str(row["sample_id"])]
            for row in rows
        }
    )
    scene_index = {value: index for index, value in enumerate(scenes)}
    material_index = {value: index for index, value in enumerate(materials)}
    methods = sorted({str(row["method"]) for row in rows if row["method"] != "oracle_route"})
    datasets = ("scale", "conflict")
    arrays: dict[str, dict[str, object]] = {}
    point: dict[str, dict[str, float]] = {}
    for dataset in datasets:
        selected = [row for row in rows if row["dataset"] == dataset]
        sample_ids = sorted({str(row["sample_id"]) for row in selected})
        by_method = {
            method: {
                str(row["sample_id"]): str(row["prediction"])
                for row in selected
                if row["method"] == method
            }
            for method in methods
        }
        truth = {
            str(row["sample_id"]): str(row["truth"])
            for row in selected
        }
        representative_by_id: dict[str, dict[str, object]] = {}
        for row in selected:
            representative_by_id.setdefault(str(row["sample_id"]), row)
        classes = sorted(set(truth.values()))
        class_index = {value: index for index, value in enumerate(classes)}
        totals = np.zeros(
            (len(classes), len(scenes), len(materials)), dtype=np.float64
        )
        correct = np.zeros(
            (len(methods), len(classes), len(scenes), len(materials)),
            dtype=np.float64,
        )
        for sample_id in sample_ids:
            representative = representative_by_id[sample_id]
            c = class_index[truth[sample_id]]
            s = scene_index[str(representative["scene"])]
            m = material_index[metadata[dataset][sample_id]]
            totals[c, s, m] += 1.0
            for method_index, method in enumerate(methods):
                correct[method_index, c, s, m] += float(
                    by_method[method][sample_id] == truth[sample_id]
                )
        arrays[dataset] = {
            "totals": totals,
            "correct": correct,
            "classes": classes,
        }
        point[dataset] = {
            method: _metric(
                np.asarray([truth[sample_id] for sample_id in sample_ids]),
                np.asarray([by_method[method][sample_id] for sample_id in sample_ids]),
            )
            for method in methods
        }

    rng = np.random.default_rng(int(args.seed))
    draws_by_dataset = {
        dataset: np.empty((int(args.draws), len(methods)), dtype=np.float64)
        for dataset in datasets
    }
    batch_size = 250
    for start in range(0, int(args.draws), batch_size):
        size = min(batch_size, int(args.draws) - start)
        scene_weights = rng.multinomial(
            len(scenes), np.full(len(scenes), 1.0 / len(scenes)), size=size
        ).astype(np.float64)
        material_weights = rng.multinomial(
            len(materials),
            np.full(len(materials), 1.0 / len(materials)),
            size=size,
        ).astype(np.float64)
        crossed = scene_weights[:, :, None] * material_weights[:, None, :]
        for dataset in datasets:
            totals = arrays[dataset]["totals"]
            correct = arrays[dataset]["correct"]
            denominator = np.einsum("bsm,csm->bc", crossed, totals)
            numerator = np.einsum("bsm,kcsm->bkc", crossed, correct)
            recall = numerator / np.where(
                denominator[:, None, :] <= 0.0,
                np.nan,
                denominator[:, None, :],
            )
            draws_by_dataset[dataset][start : start + size] = np.nanmean(
                recall, axis=2
            )

    method_index = {value: index for index, value in enumerate(methods)}
    comparisons = {}
    pairs = (
        ("kino_v4", "joint_early_fusion"),
        ("kino_v4", "late_fusion"),
        ("kino_v4", "proprioception"),
    )
    for left, right in pairs:
        key = f"{left}_minus_{right}"
        comparisons[key] = {}
        left_index = method_index[left]
        right_index = method_index[right]
        per_dataset = {
            dataset: draws_by_dataset[dataset][:, left_index]
            - draws_by_dataset[dataset][:, right_index]
            for dataset in datasets
        }
        for dataset, values in per_dataset.items():
            summary = _summary(values)
            summary["point_difference"] = float(
                point[dataset][left] - point[dataset][right]
            )
            comparisons[key][dataset] = summary
        left_macro = 0.5 * (
            draws_by_dataset["scale"][:, left_index]
            + draws_by_dataset["conflict"][:, left_index]
        )
        right_macro = 0.5 * (
            draws_by_dataset["scale"][:, right_index]
            + draws_by_dataset["conflict"][:, right_index]
        )
        left_worst = np.minimum(
            draws_by_dataset["scale"][:, left_index],
            draws_by_dataset["conflict"][:, left_index],
        )
        right_worst = np.minimum(
            draws_by_dataset["scale"][:, right_index],
            draws_by_dataset["conflict"][:, right_index],
        )
        comparisons[key]["macro"] = _summary(left_macro - right_macro)
        comparisons[key]["worst"] = _summary(left_worst - right_worst)

    per_cell = {}
    for cell in ("T2_vision_decisive", "T3_proprio_decisive"):
        cell_rows = [row for row in rows if row.get("cell") == cell]
        per_cell[cell] = {}
        for method in methods:
            method_rows = [row for row in cell_rows if row["method"] == method]
            per_cell[cell][method] = _metric(
                np.asarray([row["truth"] for row in method_rows]),
                np.asarray([row["prediction"] for row in method_rows]),
            )

    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": "kinofail.v4-paired-statistics-development.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "crossed_scene_material_cluster_bootstrap_complete",
        "confirmatory_evidence": False,
        "draws": int(args.draws),
        "seed": int(args.seed),
        "clusters": {"scenes": len(scenes), "materials": len(materials)},
        "point_balanced_accuracy": point,
        "per_conflict_cell_balanced_accuracy": per_cell,
        "comparisons": comparisons,
        "noninferiority_margin": -0.01,
        "source_sha256": {"predictions": _sha256(prediction_path)},
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
