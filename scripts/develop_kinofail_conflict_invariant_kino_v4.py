#!/usr/bin/env python3
"""Five-fold scene-disjoint development evaluation of conflict-invariant KiNO.

F42 is treated strictly as development data.  The script reconstructs the
retained 190-D proprio summaries for the final Scale overlay, joins them with
the shared 80-D invariant descriptors, and evaluates every scene exactly once
in a domain-balanced held-out fold.  No result is described as independent
confirmation; a new frozen collection is required after architecture freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import recall_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.conflict_invariant_kino import (  # noqa: E402
    ConflictInvariantKiNO,
)


F42 = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f42"
SCALE_OVERLAY = ROOT / "outputs/eval/unified_moe_v3_scale_direct_o9_f35"
ORIGINAL_SCALE = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2/shards"
REPLENISHMENT = ROOT / "outputs/eval/unified_moe_v3_replenishment_f25/shards"
DIRECT_O9 = ROOT / "outputs/eval/unified_moe_v3_o9_direct_f34"
DEFAULT_OUTPUT = ROOT / "outputs/eval/kino_conflict_invariant_v4_development"
DINO_DETAIL = ROOT / "outputs/eval/kino_t2_dinov2_v4_development/features.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _scene_fold(scene: str) -> int:
    return int(str(scene).rsplit("_", 1)[1]) % 5


def _load_scale() -> dict[str, Any]:
    truth_rows = _jsonl(F42 / "evaluation_inputs/scale_truth.jsonl")
    truth = {
        str(row["sample_id"]): row
        for row in truth_rows
        if bool(row.get("valid", True))
    }
    feature_path = F42 / "evaluation_inputs/scale_features.npz"
    with np.load(feature_path, allow_pickle=False) as archive:
        sample_ids = archive["sample_ids"].astype(str)
        visual = np.asarray(archive["visual"], dtype=np.float32)
        invariant = np.asarray(archive["proprio"], dtype=np.float32)
    if set(sample_ids.tolist()) != set(truth):
        raise RuntimeError("Scale features and valid truth registry differ")
    rows = [truth[sample_id] for sample_id in sample_ids]

    final_records: dict[str, dict[str, Any]] = {}
    record_paths = sorted(
        SCALE_OVERLAY.glob("shards/*/scale/snapshots/snapshot_records.jsonl")
    )
    for path in record_paths:
        for row in _jsonl(path):
            sample_id = str(row["sample_id"])
            if sample_id in final_records:
                raise RuntimeError(f"duplicate final Scale sample: {sample_id}")
            final_records[sample_id] = row
    if set(final_records) != set(sample_ids.tolist()):
        raise RuntimeError("final Scale overlay records do not match F42 features")

    source_id = {
        sample_id: str(
            row.get(
                "source_replacement_sample_id",
                row.get("source_f33_sample_id", sample_id),
            )
        )
        for sample_id, row in final_records.items()
    }
    needed = set(source_id.values())
    rich: dict[str, np.ndarray] = {}
    rich_sources = [
        *sorted(ORIGINAL_SCALE.glob("*/scale/features/features.npz")),
        *sorted(REPLENISHMENT.glob("*/scale/features/features.npz")),
        DIRECT_O9 / "visual_features/features.npz",
    ]
    for path in rich_sources:
        with np.load(path, allow_pickle=False) as archive:
            ids = archive["sample_ids"].astype(str)
            values = np.asarray(archive["proprio"], dtype=np.float32)
        for index, sample_id in enumerate(ids):
            if sample_id in needed:
                if sample_id in rich and not np.allclose(
                    rich[sample_id], values[index], rtol=1.0e-5, atol=1.0e-6
                ):
                    raise RuntimeError(f"rich Scale feature drift: {sample_id}")
                rich[sample_id] = values[index].copy()
    missing = needed - set(rich)
    if missing:
        raise RuntimeError(
            f"missing {len(missing)} retained rich Scale features: "
            f"{sorted(missing)[:3]}"
        )
    full = np.stack([rich[source_id[sample_id]] for sample_id in sample_ids])
    return {
        "rows": rows,
        "sample_ids": sample_ids,
        "visual": visual,
        "full": full.astype(np.float32),
        "invariant": invariant,
        "labels": np.asarray([str(row["truth"]) for row in rows]),
        "groups": np.asarray(
            [f"scale::{row['group_id']}::{row['truth']}" for row in rows]
        ),
        "scenes": np.asarray([str(row["scene"]) for row in rows]),
        "folds": np.asarray([_scene_fold(str(row["scene"])) for row in rows]),
        "source_paths": [feature_path, *record_paths, *rich_sources],
    }


def _load_conflict() -> dict[str, Any]:
    root = F42 / "conflict"
    rows = _jsonl(root / "features/records.jsonl")
    base_path = root / "c2_base/features.npz"
    invariant_path = root / "features/features.npz"
    local_visual_path = root / "features/geometry.npz"
    with np.load(base_path, allow_pickle=False) as archive:
        sample_ids = archive["sample_ids"].astype(str)
        visual = np.asarray(archive["visual"], dtype=np.float32)
        full = np.asarray(archive["proprio"], dtype=np.float32)
    with np.load(invariant_path, allow_pickle=False) as archive:
        invariant_ids = archive["sample_ids"].astype(str)
        invariant = np.asarray(archive["proprio"], dtype=np.float32)
    with np.load(local_visual_path, allow_pickle=False) as archive:
        local_ids = archive["sample_ids"].astype(str)
        local_visual = np.asarray(archive["geometry"], dtype=np.float32)
    expected = [str(row["sample_id"]) for row in rows]
    if sample_ids.tolist() != expected or not np.array_equal(
        sample_ids, invariant_ids
    ) or not np.array_equal(
        sample_ids, local_ids
    ):
        raise RuntimeError("Conflict feature blocks are misaligned")
    cells = np.asarray([str(row["cell"]) for row in rows])
    scenes = np.asarray([str(row["scene_cluster"]) for row in rows])
    return {
        "rows": rows,
        "sample_ids": sample_ids,
        "visual": visual,
        "local_visual": local_visual,
        "full": full,
        "invariant": invariant,
        "labels": np.asarray([str(row["attribution_category"]) for row in rows]),
        "groups": np.asarray([f"conflict::{row['case_id']}" for row in rows]),
        "scenes": scenes,
        "folds": np.asarray([_scene_fold(scene) for scene in scenes]),
        "cells": cells,
        "vision_decisive": cells == "T2_vision_decisive",
        "source_paths": [
            root / "features/records.jsonl",
            base_path,
            invariant_path,
            local_visual_path,
        ],
    }


def _metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    classes: list[str] | None = None,
) -> dict[str, Any]:
    labels = np.asarray(labels).astype(str)
    predictions = np.asarray(predictions).astype(str)
    support = sorted(set(labels.tolist())) if classes is None else list(classes)
    recalls = recall_score(labels, predictions, labels=support, average=None)
    balanced_accuracy = float(recalls.mean())
    return {
        "samples": len(labels),
        "accuracy": float(np.mean(labels == predictions)),
        "balanced_accuracy": balanced_accuracy,
        "per_class_recall": {
            label: float(value)
            for label, value in zip(support, recalls, strict=True)
        },
        "worst_class_recall": float(recalls.min()),
        "worst_class": support[int(recalls.argmin())],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--route-threshold", type=float, default=0.11)
    parser.add_argument("--visual-regularization", type=float, default=0.03)
    parser.add_argument(
        "--dino-detail",
        action="store_true",
        help="Add the frozen DINOv2 CLS temporal context to the T2 specialist.",
    )
    parser.add_argument("--dino-detail-path", type=Path, default=DINO_DETAIL)
    parser.add_argument(
        "--dino-detail-features",
        default="cls",
        help=(
            "Comma-separated frozen DINOv2 pools used by the T2 specialist "
            "(for example cls,ground_mean,ground_max)."
        ),
    )
    parser.add_argument(
        "--detail-only-visual-specialist",
        action="store_true",
        help=(
            "Use only the frozen detail descriptor in the T2 specialist; "
            "the route gate remains proprioceptive."
        ),
    )
    parser.add_argument(
        "--local-visual",
        action="store_true",
        help=(
            "Use the retained ground-ROI HOG branch for Conflict. Scale's "
            "pruned cache is represented by zeros; its conservative gate "
            "normally retains the proprio default."
        ),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    scale = _load_scale()
    conflict = _load_conflict()
    if args.dino_detail:
        dino_detail_path = args.dino_detail_path
        if not dino_detail_path.is_absolute():
            dino_detail_path = ROOT / dino_detail_path
        dino_feature_names = [
            value.strip()
            for value in str(args.dino_detail_features).split(",")
            if value.strip()
        ]
        if not dino_feature_names:
            raise ValueError("at least one DINO detail feature is required")
        with np.load(dino_detail_path, allow_pickle=False) as archive:
            dino_ids = archive["sample_ids"].astype(str)
            missing = [
                name
                for name in dino_feature_names
                if f"visual_{name}" not in archive.files
            ]
            if missing:
                raise KeyError(f"missing DINO detail pools: {missing}")
            dino_values = np.concatenate(
                [
                    np.asarray(archive[f"visual_{name}"], dtype=np.float32)
                    for name in dino_feature_names
                ],
                axis=1,
            )
        t2 = conflict["vision_decisive"]
        if not np.array_equal(dino_ids, conflict["sample_ids"][t2]):
            raise RuntimeError("DINO detail cache does not align with Conflict T2")
        scale["detail_visual"] = np.zeros(
            (len(scale["sample_ids"]), dino_values.shape[1]), dtype=np.float32
        )
        conflict["detail_visual"] = np.zeros(
            (len(conflict["sample_ids"]), dino_values.shape[1]), dtype=np.float32
        )
        conflict["detail_visual"][t2] = dino_values
        conflict["source_paths"].append(dino_detail_path)
    else:
        scale["detail_visual"] = np.empty(
            (len(scale["sample_ids"]), 0), dtype=np.float32
        )
        conflict["detail_visual"] = np.empty(
            (len(conflict["sample_ids"]), 0), dtype=np.float32
        )
    if args.local_visual:
        local_dimension = int(conflict["local_visual"].shape[1])
        scale["visual"] = np.concatenate(
            [
                scale["visual"],
                np.zeros(
                    (len(scale["visual"]), local_dimension), dtype=np.float32
                ),
            ],
            axis=1,
        )
        conflict["visual"] = np.concatenate(
            [conflict["visual"], conflict["local_visual"]], axis=1
        )
    prediction_rows: list[dict[str, Any]] = []
    fold_reports: dict[str, Any] = {}
    for fold in range(5):
        scale_train = scale["folds"] != fold
        scale_test = ~scale_train
        conflict_train = conflict["folds"] != fold
        conflict_test = ~conflict_train
        train_visual = np.concatenate(
            [scale["visual"][scale_train], conflict["visual"][conflict_train]],
            axis=0,
        )
        train_full = np.concatenate(
            [scale["full"][scale_train], conflict["full"][conflict_train]],
            axis=0,
        )
        train_invariant = np.concatenate(
            [
                scale["invariant"][scale_train],
                conflict["invariant"][conflict_train],
            ],
            axis=0,
        )
        train_labels = np.concatenate(
            [scale["labels"][scale_train], conflict["labels"][conflict_train]]
        )
        train_detail = np.concatenate(
            [
                scale["detail_visual"][scale_train],
                conflict["detail_visual"][conflict_train],
            ],
            axis=0,
        )
        train_groups = np.concatenate(
            [scale["groups"][scale_train], conflict["groups"][conflict_train]]
        )
        train_vision_decisive = np.concatenate(
            [
                np.zeros(int(scale_train.sum()), dtype=bool),
                conflict["vision_decisive"][conflict_train],
            ]
        )
        model = ConflictInvariantKiNO.fit(
            train_visual,
            train_full,
            train_invariant,
            train_labels,
            train_groups,
            train_vision_decisive,
            seed=2026080300 + fold,
            route_threshold=float(args.route_threshold),
            detail_visual=train_detail,
            visual_regularization=float(args.visual_regularization),
            include_base_visual_in_specialist=(
                not args.detail_only_visual_specialist
            ),
        )
        scale_result = model.predict_with_routes(
            scale["visual"][scale_test],
            scale["full"][scale_test],
            scale["invariant"][scale_test],
            scale["detail_visual"][scale_test],
        )
        conflict_result = model.predict_with_routes(
            conflict["visual"][conflict_test],
            conflict["full"][conflict_test],
            conflict["invariant"][conflict_test],
            conflict["detail_visual"][conflict_test],
        )
        fold_report: dict[str, Any] = {
            "heldout_scenes": {
                "scale": sorted(set(scale["scenes"][scale_test].tolist())),
                "conflict": sorted(set(conflict["scenes"][conflict_test].tolist())),
            },
            "fit_audit": model.fit_audit,
        }
        for dataset, source, mask, result in (
            ("scale", scale, scale_test, scale_result),
            ("conflict", conflict, conflict_test, conflict_result),
        ):
            labels = source["labels"][mask]
            predictions = np.asarray(model.classes)[
                result.probabilities.argmax(axis=1)
            ]
            default_predictions = np.asarray(model.classes)[
                result.proprio_probabilities.argmax(axis=1)
            ]
            visual_predictions = np.asarray(("adhesion", "compliant_terrain"))[
                result.visual_t2_probabilities.argmax(axis=1)
            ]
            expected_vision = (
                np.zeros(len(labels), dtype=bool)
                if dataset == "scale"
                else source["vision_decisive"][mask]
            )
            fold_report[dataset] = {
                "kino": _metrics(labels, predictions),
                "proprio_default": _metrics(labels, default_predictions),
                "route_fidelity": float(
                    np.mean((result.routes == "vision") == expected_vision)
                ),
                "vision_route_rate": float(np.mean(result.routes == "vision")),
            }
            if dataset == "conflict":
                cells = source["cells"][mask]
                fold_report[dataset]["per_cell"] = {}
                for cell in sorted(set(cells.tolist())):
                    selected = cells == cell
                    fold_report[dataset]["per_cell"][cell] = _metrics(
                        labels[selected], predictions[selected]
                    )
            for index, sample_id in enumerate(source["sample_ids"][mask]):
                row = {
                    "fold": fold,
                    "dataset": dataset,
                    "sample_id": str(sample_id),
                    "scene": str(source["scenes"][mask][index]),
                    "truth": str(labels[index]),
                    "prediction": str(predictions[index]),
                    "proprio_default_prediction": str(default_predictions[index]),
                    "proprio_default_confidence": float(
                        result.proprio_probabilities[index].max()
                    ),
                    "visual_t2_prediction": str(visual_predictions[index]),
                    "visual_t2_confidence": float(
                        result.visual_t2_probabilities[index].max()
                    ),
                    "route": str(result.routes[index]),
                    "vision_decisive_probability": float(
                        result.vision_decisive_probability[index]
                    ),
                }
                if dataset == "conflict":
                    row["cell"] = str(source["cells"][mask][index])
                prediction_rows.append(row)
        fold_report["cross_battery"] = {
            "macro_balanced_accuracy": float(
                np.mean(
                    [
                        fold_report["scale"]["kino"]["balanced_accuracy"],
                        fold_report["conflict"]["kino"]["balanced_accuracy"],
                    ]
                )
            ),
            "worst_battery_balanced_accuracy": float(
                min(
                    fold_report["scale"]["kino"]["balanced_accuracy"],
                    fold_report["conflict"]["kino"]["balanced_accuracy"],
                )
            ),
        }
        fold_reports[str(fold)] = fold_report
        print(
            json.dumps(
                {
                    "fold": fold,
                    "scale": fold_report["scale"]["kino"]["balanced_accuracy"],
                    "conflict": fold_report["conflict"]["kino"][
                        "balanced_accuracy"
                    ],
                    "t2": fold_report["conflict"]["per_cell"][
                        "T2_vision_decisive"
                    ]["balanced_accuracy"],
                    "t3": fold_report["conflict"]["per_cell"][
                        "T3_proprio_decisive"
                    ]["balanced_accuracy"],
                    "worst": fold_report["cross_battery"][
                        "worst_battery_balanced_accuracy"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        by_dataset[str(row["dataset"])].append(row)
    aggregate: dict[str, Any] = {}
    for dataset, rows in by_dataset.items():
        labels = np.asarray([row["truth"] for row in rows])
        predictions = np.asarray([row["prediction"] for row in rows])
        default = np.asarray([row["proprio_default_prediction"] for row in rows])
        expected_vision = np.asarray(
            [
                row.get("cell") == "T2_vision_decisive"
                if dataset == "conflict"
                else False
                for row in rows
            ]
        )
        routes = np.asarray([row["route"] for row in rows])
        aggregate[dataset] = {
            "kino": _metrics(labels, predictions),
            "proprio_default": _metrics(labels, default),
            "route_fidelity": float(
                np.mean((routes == "vision") == expected_vision)
            ),
            "vision_route_rate": float(np.mean(routes == "vision")),
        }
        if dataset == "conflict":
            cells = np.asarray([row["cell"] for row in rows])
            aggregate[dataset]["per_cell"] = {
                cell: _metrics(labels[cells == cell], predictions[cells == cell])
                for cell in sorted(set(cells.tolist()))
            }
    aggregate["cross_battery"] = {
        "macro_balanced_accuracy": float(
            np.mean(
                [
                    aggregate["scale"]["kino"]["balanced_accuracy"],
                    aggregate["conflict"]["kino"]["balanced_accuracy"],
                ]
            )
        ),
        "worst_battery_balanced_accuracy": float(
            min(
                aggregate["scale"]["kino"]["balanced_accuracy"],
                aggregate["conflict"]["kino"]["balanced_accuracy"],
            )
        ),
    }
    expected_rows = len(scale["sample_ids"]) + len(conflict["sample_ids"])
    if len(prediction_rows) != expected_rows or len(
        {f"{row['dataset']}::{row['sample_id']}" for row in prediction_rows}
    ) != expected_rows:
        raise RuntimeError("scene-fold evaluation did not predict every sample exactly once")

    output.mkdir(parents=True, exist_ok=False)
    predictions_path = output / "scene_disjoint_predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows)
    )
    source_paths = [*scale["source_paths"], *conflict["source_paths"]]
    report = {
        "schema_version": "kinofail.conflict-invariant-kino-v4-development.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "five_fold_scene_disjoint_development_complete",
        "confirmatory_evidence": False,
        "architecture_selection_consumed_f42": True,
        "independent_confirmation_required": True,
        "split": (
            "five domain-balanced scene folds; scene suffix modulo five; "
            "every sample evaluated exactly once"
        ),
        "route_threshold": float(args.route_threshold),
        "visual_regularization_C": float(args.visual_regularization),
        "dino_detail_branch": bool(args.dino_detail),
        "dino_detail_path": (
            str(args.dino_detail_path)
            if args.dino_detail
            else None
        ),
        "dino_detail_features": (
            [
                value.strip()
                for value in str(args.dino_detail_features).split(",")
                if value.strip()
            ]
            if args.dino_detail
            else []
        ),
        "detail_only_visual_specialist": bool(
            args.detail_only_visual_specialist
        ),
        "scale_detail_cache_policy": (
            {
                "policy": "zero placeholder for the pruned Scale detail cache",
                "vision_route_samples": sum(
                    row["dataset"] == "scale" and row["route"] == "vision"
                    for row in prediction_rows
                ),
                "vision_route_truth_disjoint_from_t2_support": all(
                    row["truth"] not in {"adhesion", "compliant_terrain"}
                    for row in prediction_rows
                    if row["dataset"] == "scale" and row["route"] == "vision"
                ),
                "accuracy_status": (
                    "exact because every Scale vision-route truth lies outside the "
                    "T2 specialist support; probability calibration is not evaluated"
                    if all(
                        row["truth"] not in {"adhesion", "compliant_terrain"}
                        for row in prediction_rows
                        if row["dataset"] == "scale" and row["route"] == "vision"
                    )
                    else "approximate until the Scale detail cache is reconstructed"
                ),
            }
            if args.dino_detail
            else {"policy": "not used"}
        ),
        "local_visual_branch": bool(args.local_visual),
        "counts": {
            "scale_samples": len(scale["sample_ids"]),
            "conflict_samples": len(conflict["sample_ids"]),
            "scenes": len(set(scale["scenes"].tolist())),
            "folds": 5,
        },
        "folds": fold_reports,
        "aggregate": aggregate,
        "comparison_to_sealed_f42_checkpoint": {
            "sealed_kino": {
                "scale": 0.688,
                "conflict": 0.037,
                "macro": 0.362,
                "worst": 0.037,
            },
            "note": (
                "Reference values are the one-shot F42 scores of the prior frozen "
                "checkpoint. V4 values are scene-disjoint development estimates on "
                "F42 and require a fresh confirmation collection."
            ),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in source_paths
            if path.is_file()
        },
        "artifacts": {
            "predictions": str(predictions_path.relative_to(ROOT)),
            "predictions_sha256": _sha256(predictions_path),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
