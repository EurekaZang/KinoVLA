#!/usr/bin/env python3
"""Preregistered expanded-corpus operator and boundary analysis for A6.

The analysis is frozen before F36 predictions are generated.  It measures
classification stability across all 11 operators, both Scale severity strata,
three appearance views, 30 scene clusters, and the 16 direct-contact O9
parameter points.  It never fits a model or changes an operating point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
F36 = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f36"
SCALE_SCHEDULE = ROOT / "outputs/kinofail_reconfirmation_v2/schedules/scale_schedule.jsonl"
F33_SCHEDULE = ROOT / "outputs/kinofail_confirmatory_o9_final_f33/schedule.jsonl"
CHECKPOINTS = tuple(f"seed{index}" for index in range(5))
METHODS = ("learned_router", "late_average")
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2_027_016_601
MIN_OPERATOR_POINT_ACCURACY = 0.85
MIN_OPERATOR_SCENE_LOWER_BOUND = 0.75
MIN_O9_PARAMETER_POINT_ACCURACY = 0.80
MAX_APPEARANCE_VIEW_GAP = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def scene_bootstrap(
    rows: list[dict[str, Any]], *, seed: int
) -> dict[str, Any]:
    by_scene: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_scene[str(row["scene"])].append(float(row["correct"]))
    scenes = sorted(by_scene)
    if not scenes:
        raise RuntimeError("empty A6 analysis cell")
    scene_means = np.asarray(
        [np.mean(by_scene[scene]) for scene in scenes], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(scenes), size=(BOOTSTRAP_DRAWS, len(scenes)))
    values = scene_means[sampled].mean(axis=1)
    return {
        "five_checkpoint_sample_mean": float(np.mean([row["correct"] for row in rows])),
        "scene_cluster_mean": float(scene_means.mean()),
        "scene_cluster_ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
        "scene_cluster_count": len(scenes),
        "sample_checkpoint_units": len(rows),
    }


def anomaly_only(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conditions = {str(row["condition"]) for row in rows}
    if conditions != {"anomaly", "nominal_counterfactual"}:
        raise RuntimeError(f"unexpected A6 conditions: {sorted(conditions)}")
    selected = [row for row in rows if row["condition"] == "anomaly"]
    if not selected:
        raise RuntimeError("A6 anomaly attribution set is empty")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--truth", type=Path, default=F36 / "blind_bundle/truth_key.jsonl"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=F36 / "blind_predictions/blind_predictions.jsonl",
    )
    parser.add_argument(
        "--out", type=Path, default=F36 / "a6_operator_boundary_report.json"
    )
    args = parser.parse_args()
    truth_path = args.truth.resolve()
    predictions_path = args.predictions.resolve()
    out = args.out.resolve()
    for path in (truth_path, predictions_path, SCALE_SCHEDULE, F33_SCHEDULE):
        if not path.is_file():
            raise FileNotFoundError(path)
    if out.exists():
        raise FileExistsError(out)

    truth = {
        str(row["sample_id"]): row
        for row in jsonl(truth_path)
        if row.get("dataset") == "scale" and bool(row.get("valid", True))
    }
    schedule_anomaly = {
        str(row["counterfactual_group_id"]): row
        for row in jsonl(SCALE_SCHEDULE)
        if row.get("condition") == "anomaly"
    }
    f33_anomaly = [
        row for row in jsonl(F33_SCHEDULE) if row.get("condition") == "anomaly"
    ]
    o9_lambda_by_source = {
        str(row["o9_semantic_recollection"]["source_counterfactual_group_id"]): float(
            row["parameter_interpolation"]["direct_o9_continuum"]
        )
        for row in f33_anomaly
    }
    lambda_values = sorted(set(o9_lambda_by_source.values()))
    if len(lambda_values) != 16:
        raise RuntimeError(f"F33 does not define 16 O9 parameter points: {len(lambda_values)}")
    lambda_index = {value: index for index, value in enumerate(lambda_values)}

    prediction_rows = [
        row
        for row in jsonl(predictions_path)
        if row.get("method") in METHODS and str(row.get("sample_id")) in truth
    ]
    expected = len(truth) * len(METHODS) * len(CHECKPOINTS)
    if len(prediction_rows) != expected:
        raise RuntimeError(
            f"A6 prediction coverage is incomplete: {len(prediction_rows)} != {expected}"
        )
    observations: list[dict[str, Any]] = []
    for prediction in prediction_rows:
        row = truth[str(prediction["sample_id"])]
        group_id = str(row["group_id"])
        schedule = schedule_anomaly[group_id]
        observations.append(
            {
                "sample_id": str(row["sample_id"]),
                "group_id": group_id,
                "scene": str(row["scene"]),
                "operator": str(row["operator"]),
                "severity": str(schedule["severity_id"]),
                "appearance_view_id": str(row.get("appearance_view_id", "")),
                "condition": str(row.get("condition", "")),
                "method": str(prediction["method"]),
                "checkpoint": str(prediction["checkpoint_id"]),
                "correct": float(str(prediction["prediction"]) == str(row["truth"])),
                "o9_lambda": o9_lambda_by_source.get(group_id),
            }
        )

    # Operator, severity, appearance, and O9 boundary claims concern failure
    # attribution. Nominal counterfactuals are intentionally excluded so easy
    # nominal recognition cannot conceal a weak anomaly/operator cell.
    attribution_rows = anomaly_only(observations)
    operators = sorted({row["operator"] for row in attribution_rows})
    severities = sorted({row["severity"] for row in attribution_rows})
    if len(operators) != 11 or severities != ["hard", "moderate"]:
        raise RuntimeError(f"unexpected A6 strata: operators={operators}, severity={severities}")

    cells: dict[str, Any] = {}
    operator_min_lower = 1.0
    operator_min_point = 1.0
    minimum_scene_count = 10**9
    for op_index, operator in enumerate(operators):
        method_cells: dict[str, Any] = {}
        for method_index, method in enumerate(METHODS):
            rows = [
                row
                for row in attribution_rows
                if row["operator"] == operator and row["method"] == method
            ]
            aggregate = scene_bootstrap(
                rows, seed=BOOTSTRAP_SEED + 100 * op_index + method_index
            )
            severity_cells = {
                severity: scene_bootstrap(
                    [row for row in rows if row["severity"] == severity],
                    seed=BOOTSTRAP_SEED
                    + 10_000
                    + 100 * op_index
                    + 10 * method_index
                    + severity_index,
                )
                for severity_index, severity in enumerate(severities)
            }
            method_cells[method] = {"overall": aggregate, "severity": severity_cells}
            if method == "learned_router":
                operator_min_lower = min(
                    operator_min_lower, float(aggregate["scene_cluster_ci95"][0])
                )
                operator_min_point = min(
                    operator_min_point, float(aggregate["five_checkpoint_sample_mean"])
                )
                minimum_scene_count = min(
                    minimum_scene_count, int(aggregate["scene_cluster_count"])
                )
        cells[operator] = method_cells

    # One common scene resample preserves all operator correlations and yields
    # a simultaneous lower bound for the weakest operator, the actual gate.
    scene_ids = sorted({row["scene"] for row in attribution_rows})
    learned_operator_scene = np.asarray(
        [
            [
                np.mean(
                    [
                        row["correct"]
                        for row in attribution_rows
                        if row["method"] == "learned_router"
                        and row["operator"] == operator
                        and row["scene"] == scene
                    ]
                )
                for scene in scene_ids
            ]
            for operator in operators
        ],
        dtype=np.float64,
    )
    if not np.isfinite(learned_operator_scene).all():
        raise RuntimeError("A6 operator-by-scene matrix is incomplete")
    joint_rng = np.random.default_rng(BOOTSTRAP_SEED + 90_000)
    joint_sampled = joint_rng.integers(
        0, len(scene_ids), size=(BOOTSTRAP_DRAWS, len(scene_ids))
    )
    joint_operator_means = learned_operator_scene[:, joint_sampled].mean(axis=2).T
    joint_worst_operator = joint_operator_means.min(axis=1)
    simultaneous_worst_operator_lower = float(
        np.quantile(joint_worst_operator, 0.025)
    )

    view_accuracy: dict[str, float] = {}
    for view in sorted({row["appearance_view_id"] for row in attribution_rows}):
        values = [
            row["correct"]
            for row in attribution_rows
            if row["method"] == "learned_router"
            and row["appearance_view_id"] == view
        ]
        view_accuracy[view] = float(np.mean(values))
    appearance_gap = max(view_accuracy.values()) - min(view_accuracy.values())
    views = sorted(view_accuracy)
    learned_view_scene = np.asarray(
        [
            [
                np.mean(
                    [
                        row["correct"]
                        for row in attribution_rows
                        if row["method"] == "learned_router"
                        and row["appearance_view_id"] == view
                        and row["scene"] == scene
                    ]
                )
                for scene in scene_ids
            ]
            for view in views
        ],
        dtype=np.float64,
    )
    if not np.isfinite(learned_view_scene).all():
        raise RuntimeError("A6 appearance-by-scene matrix is incomplete")
    joint_view_means = learned_view_scene[:, joint_sampled].mean(axis=2).T
    joint_view_gap = joint_view_means.max(axis=1) - joint_view_means.min(axis=1)
    appearance_gap_upper = float(np.quantile(joint_view_gap, 0.975))

    operator_view_accuracy: dict[str, dict[str, float]] = {}
    for operator in operators:
        operator_view_accuracy[operator] = {}
        for view in views:
            values = [
                row["correct"]
                for row in attribution_rows
                if row["method"] == "learned_router"
                and row["operator"] == operator
                and row["appearance_view_id"] == view
            ]
            operator_view_accuracy[operator][view] = float(np.mean(values))
    maximum_operator_view_point_gap = max(
        max(values.values()) - min(values.values())
        for values in operator_view_accuracy.values()
    )
    learned_operator_view_scene = np.asarray(
        [
            [
                [
                    np.mean(
                        [
                            row["correct"]
                            for row in attribution_rows
                            if row["method"] == "learned_router"
                            and row["operator"] == operator
                            and row["appearance_view_id"] == view
                            and row["scene"] == scene
                        ]
                    )
                    for scene in scene_ids
                ]
                for view in views
            ]
            for operator in operators
        ],
        dtype=np.float64,
    )
    if not np.isfinite(learned_operator_view_scene).all():
        raise RuntimeError("A6 operator-by-appearance-by-scene matrix is incomplete")
    # Shape after the transpose is draw x operator x view. Taking the view
    # range and then the worst operator inside each common scene resample gives
    # a simultaneous upper bound that cannot be hidden by averaging operators.
    joint_operator_view_means = learned_operator_view_scene[
        :, :, joint_sampled
    ].mean(axis=3).transpose(2, 0, 1)
    joint_operator_view_gaps = (
        joint_operator_view_means.max(axis=2)
        - joint_operator_view_means.min(axis=2)
    )
    joint_worst_operator_view_gap = joint_operator_view_gaps.max(axis=1)
    simultaneous_operator_view_gap_upper = float(
        np.quantile(joint_worst_operator_view_gap, 0.975)
    )

    o9_points: dict[str, Any] = {}
    for value in lambda_values:
        rows = [
            row
            for row in attribution_rows
            if row["method"] == "learned_router" and row["o9_lambda"] == value
        ]
        if not rows:
            raise RuntimeError(f"accepted F36 O9 evidence misses lambda={value}")
        o9_points[f"p{lambda_index[value]:02d}"] = {
            "direct_o9_continuum": value,
            "five_checkpoint_sample_accuracy": float(
                np.mean([row["correct"] for row in rows])
            ),
            "scene_cluster_count": len({row["scene"] for row in rows}),
            "sample_checkpoint_units": len(rows),
        }
    minimum_o9_point = min(
        row["five_checkpoint_sample_accuracy"] for row in o9_points.values()
    )

    acceptance = {
        "all_11_operators_have_30_scene_clusters": len(operators) == 11
        and minimum_scene_count >= 30,
        "both_frozen_severity_strata_present_for_every_operator": all(
            all(
                cells[operator]["learned_router"]["severity"][severity][
                    "scene_cluster_count"
                ]
                >= 30
                for severity in severities
            )
            for operator in operators
        ),
        "worst_operator_point_accuracy_at_least_0_85": operator_min_point
        >= MIN_OPERATOR_POINT_ACCURACY,
        "simultaneous_worst_operator_lower_bound_above_0_75": simultaneous_worst_operator_lower
        > MIN_OPERATOR_SCENE_LOWER_BOUND,
        "all_16_direct_o9_points_at_least_0_80": minimum_o9_point
        >= MIN_O9_PARAMETER_POINT_ACCURACY,
        "appearance_view_gap_upper_bound_at_most_0_05": appearance_gap_upper
        <= MAX_APPEARANCE_VIEW_GAP,
        "simultaneous_worst_operator_appearance_gap_upper_bound_at_most_0_05": simultaneous_operator_view_gap_upper
        <= MAX_APPEARANCE_VIEW_GAP,
    }
    acceptance["all_preregistered_boundary_checks_passed"] = all(
        acceptance.values()
    )
    report = {
        "schema_version": "kinofail.reconfirmation-a6-v7-report.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": (
            "confirmatory_passed"
            if acceptance["all_preregistered_boundary_checks_passed"]
            else "confirmatory_gate_failed"
        ),
        "passed": acceptance["all_preregistered_boundary_checks_passed"],
        "operator_count": len(operators),
        "boundary_analysis_condition": "anomaly_only",
        "nominal_counterfactuals_excluded_from_operator_and_boundary_gates": True,
        "anomaly_prediction_checkpoint_units": len(attribution_rows),
        "minimum_scene_clusters_per_operator": minimum_scene_count,
        "severity_strata": severities,
        "o9_direct_parameter_point_count": len(o9_points),
        "thresholds": {
            "minimum_operator_point_accuracy": MIN_OPERATOR_POINT_ACCURACY,
            "minimum_operator_scene_cluster_lower_bound": MIN_OPERATOR_SCENE_LOWER_BOUND,
            "minimum_o9_parameter_point_accuracy": MIN_O9_PARAMETER_POINT_ACCURACY,
            "maximum_appearance_view_gap": MAX_APPEARANCE_VIEW_GAP,
        },
        "summaries": {
            "worst_operator_point_accuracy": operator_min_point,
            "worst_operator_scene_cluster_lower_bound": operator_min_lower,
            "simultaneous_worst_operator_97_5_percent_lower_bound": simultaneous_worst_operator_lower,
            "minimum_o9_parameter_point_accuracy": minimum_o9_point,
            "appearance_view_accuracy": view_accuracy,
            "maximum_appearance_view_accuracy_gap": appearance_gap,
            "appearance_view_gap_97_5_percent_upper_bound": appearance_gap_upper,
            "operator_view_accuracy": operator_view_accuracy,
            "maximum_operator_view_point_gap": maximum_operator_view_point_gap,
            "simultaneous_worst_operator_view_gap_97_5_percent_upper_bound": simultaneous_operator_view_gap_upper,
        },
        "operator_cells": cells,
        "o9_parameter_points": o9_points,
        "acceptance": acceptance,
        "statistical_contract": {
            "bootstrap_unit": "scene cluster",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "five_checkpoints_are_not_independent_units": True,
            "no_refitting_or_threshold_selection": True,
            "simultaneous_worst_operator_inference": (
                "one common scene-cluster bootstrap; take the minimum operator "
                "accuracy inside every replicate before the lower quantile"
            ),
            "appearance_gap_inference": (
                "one common scene-cluster bootstrap; take max-minus-min view "
                "accuracy inside every replicate before the upper quantile"
            ),
            "simultaneous_operator_appearance_gap_inference": (
                "one common scene-cluster bootstrap; compute the three-view "
                "accuracy range for each operator, then take the maximum over "
                "all 11 operators inside every replicate before the upper quantile"
            ),
        },
        "source_sha256": {
            "f36_blind_predictions": sha256(predictions_path),
            "f36_truth_key": sha256(truth_path),
            "scale_schedule": sha256(SCALE_SCHEDULE),
            "f33_schedule": sha256(F33_SCHEDULE),
            "analysis_script": sha256(Path(__file__).resolve()),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
