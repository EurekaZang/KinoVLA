#!/usr/bin/env python3
"""Build a reviewer-facing split and per-operator ledger for scale-v8."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "vision_only",
    "proprio_only",
    "early_fusion",
    "structured_bidirectional",
)


def _sha256(path: Path) -> str:
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


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records",
        type=Path,
        default=ROOT
        / "outputs/eval/realistic_a0_a7_v6/snapshots/snapshot_records.jsonl",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v6/predictions.jsonl",
    )
    parser.add_argument(
        "--scene-registry",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_realistic_scale_scene_registry_v1.json",
    )
    parser.add_argument(
        "--extraction-audit",
        type=Path,
        default=ROOT
        / "outputs/eval/realistic_a0_a7_v6/snapshots/extraction_audit.json",
    )
    parser.add_argument(
        "--readiness",
        type=Path,
        default=ROOT
        / "outputs/eval/realistic_a0_a7_v6/icra_benchmark_readiness.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/eval/scale_v8_operator_ledger_v1/report.json",
    )
    args = parser.parse_args()
    records_path = args.records.resolve()
    predictions_path = args.predictions.resolve()
    registry_path = args.scene_registry.resolve()
    extraction_path = args.extraction_audit.resolve()
    readiness_path = args.readiness.resolve()
    output_path = args.output.resolve()

    records = _jsonl(records_path)
    registry = _json(registry_path)
    extraction = _json(extraction_path)
    readiness = _json(readiness_path)
    scene_split = {
        str(row["scene_id"]): str(row["split"]) for row in registry["scenes"]
    }

    def is_primary(row: dict[str, Any]) -> bool:
        return str(row["appearance_intervention_id"]) == "primary"

    def material_split(row: dict[str, Any]) -> str:
        return str(row["material_family"]).split("_", 1)[0]

    split_masks = {
        "scene": (
            lambda row: scene_split[str(row["scene_family"])] != "test",
            lambda row: (
                scene_split[str(row["scene_family"])] == "test"
                and is_primary(row)
            ),
        ),
        "material": (
            lambda row: material_split(row) != "test" and is_primary(row),
            lambda row: material_split(row) == "test" and is_primary(row),
        ),
        "scene_and_material": (
            lambda row: (
                scene_split[str(row["scene_family"])] != "test"
                and material_split(row) != "test"
                and is_primary(row)
            ),
            lambda row: (
                scene_split[str(row["scene_family"])] == "test"
                and material_split(row) == "test"
                and is_primary(row)
            ),
        ),
        "camera_profile": (
            lambda row: (
                str(row["camera_profile"]) != "go2_front_calib_c"
                and is_primary(row)
            ),
            lambda row: (
                str(row["camera_profile"]) == "go2_front_calib_c"
                and is_primary(row)
            ),
        ),
    }
    split_ledger: dict[str, Any] = {}
    for axis, (fit_mask, test_mask) in split_masks.items():
        fit_rows = [row for row in records if fit_mask(row)]
        test_rows = [row for row in records if test_mask(row)]
        split_ledger[axis] = {
            "fit_records": len(fit_rows),
            "fit_physical_episodes": len(
                {str(row["physical_episode_id"]) for row in fit_rows}
            ),
            "fit_counterfactual_pairs": len(
                {str(row["counterfactual_group_id"]) for row in fit_rows}
            ),
            "test_primary_records": len(test_rows),
            "test_physical_episodes": len(
                {str(row["physical_episode_id"]) for row in test_rows}
            ),
            "test_counterfactual_pairs": len(
                {str(row["counterfactual_group_id"]) for row in test_rows}
            ),
        }

    operators = sorted({str(row["target_operator"]) for row in records})
    operator_ledger: dict[str, Any] = {}
    for operator in operators:
        subset = [
            row for row in records if str(row["target_operator"]) == operator
        ]
        anomaly_causes = sorted(
            {
                str(row["attribution_category"])
                for row in subset
                if str(row["condition"]) == "anomaly"
            }
        )
        if len(anomaly_causes) != 1:
            raise RuntimeError(f"non-unique cause mapping for {operator}")
        operator_ledger[operator] = {
            "diagnostic_cause": anomaly_causes[0],
            "planned_counterfactual_pairs": int(
                readiness["planned"]["counterfactual_pairs"]
                // readiness["planned"]["operators"]
            ),
            "model_counterfactual_pairs": len(
                {str(row["counterfactual_group_id"]) for row in subset}
            ),
            "appearance_sequences": len(subset),
        }

    prediction_rows = [
        row
        for row in _jsonl(predictions_path)
        if str(row["axis"]) == "scene"
        and bool(row["headline_primary_view"])
        and str(row["method"]) in METHODS
    ]
    cells: dict[tuple[str, str, str, int], list[bool]] = defaultdict(list)
    for row in prediction_rows:
        condition = str(row["condition"])
        if condition not in {"anomaly", "nominal_counterfactual"}:
            raise RuntimeError(f"unexpected condition: {condition}")
        cells[
            (
                str(row["method"]),
                str(row["target_operator"]),
                condition,
                int(row["seed"]),
            )
        ].append(str(row["prediction"]) == str(row["truth"]))

    per_operator_results: dict[str, Any] = {}
    for operator in operators:
        per_operator_results[operator] = {}
        for method in METHODS:
            method_result: dict[str, Any] = {}
            for condition in ("anomaly", "nominal_counterfactual"):
                per_seed = {
                    str(seed): {
                        "accuracy": float(
                            np.mean(cells[(method, operator, condition, seed)])
                        ),
                        "samples": len(
                            cells[(method, operator, condition, seed)]
                        ),
                    }
                    for seed in range(5)
                }
                method_result[condition] = {
                    "accuracy_across_training_seeds": _summary(
                        [
                            float(per_seed[str(seed)]["accuracy"])
                            for seed in range(5)
                        ]
                    ),
                    "per_seed": per_seed,
                }
            per_operator_results[operator][method] = method_result

    report = {
        "schema_version": "kinofail.scale-v8-operator-ledger.v1",
        "status": "registered_reanalysis_of_frozen_records_and_predictions",
        "passed": True,
        "dataset_scope": "kinofail_realistic_scale_v8",
        "planned_corpus": readiness["planned"],
        "model_snapshot_corpus": extraction["counts"],
        "model_snapshot_exclusion": (
            "Two complete O10-moderate pairs were excluded at pair level "
            "because the frozen RGB window was temporally infeasible."
        ),
        "split_ledger": split_ledger,
        "operator_ontology_and_counts": operator_ledger,
        "scene_heldout_primary_per_operator_results": per_operator_results,
        "statistical_contract": {
            "training_seeds": [0, 1, 2, 3, 4],
            "appearance_views_count_as_independent_physics": False,
            "primary_operator_table_unit": (
                "one held-out primary-view physical episode per training seed"
            ),
        },
        "source_sha256": {
            "records": _sha256(records_path),
            "predictions": _sha256(predictions_path),
            "scene_registry": _sha256(registry_path),
            "extraction_audit": _sha256(extraction_path),
            "readiness": _sha256(readiness_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
