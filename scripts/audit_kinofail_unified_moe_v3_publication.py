#!/usr/bin/env python3
"""Build a compact publication audit for the unified Kino-Fail MoE.

Random seeds are treated as training sensitivity checks, not independent
experimental units.  Counterfactual pairs, matched conflict cases, scene
instances, and material families remain explicit in the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_interval(successes: int, total: int) -> list[float]:
    alpha = 0.05
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2.0, successes, total - successes + 1))
    )
    upper = (
        1.0
        if successes == total
        else float(
            beta.ppf(
                1.0 - alpha / 2.0,
                successes + 1,
                total - successes,
            )
        )
    )
    return [lower, upper]


def _seed_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "training_seed_sensitivity_range": float(
            np.max(values) - np.min(values)
        ),
    }


def _strict_all_seed_groups(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    axis: str,
    method: str,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["dataset"] == dataset
        and row["axis"] == axis
        and row["method"] == method
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[str(row["group_id"])].append(row)
    correct = sum(
        all(row["prediction"] == row["truth"] for row in values)
        for values in grouped.values()
    )
    return {
        "groups": len(grouped),
        "strict_all_training_seeds_and_group_members_correct": int(correct),
        "rate": float(correct / len(grouped)),
        "exact_pair_or_case_level_ci95": _exact_interval(
            correct, len(grouped)
        ),
        "training_seeds_are_not_counted_as_independent_units": True,
    }


def _operator_strata(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["dataset"] == "scale"
        and row["axis"] == "scene_and_material"
        and row["method"] == "learned_router"
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[(str(row["operator"]), str(row["condition"]))].append(row)
    return {
        f"{operator}/{condition}": {
            "prediction_rows_over_five_training_seeds": len(values),
            "accuracy": float(
                np.mean(
                    [
                        row["prediction"] == row["truth"]
                        for row in values
                    ]
                )
            ),
            "physical_samples": len(
                {str(row["sample_id"]) for row in values}
            ),
        }
        for (operator, condition), values in sorted(grouped.items())
    }


def _critical_route_audit(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["dataset"] == "c2_conflict"
        and row["axis"] == "scene_and_material"
    ]
    lookup = {
        (int(row["seed"]), str(row["sample_id"]), str(row["method"])): row
        for row in selected
    }
    learned = [
        row for row in selected if row["method"] == "learned_router"
    ]
    expected = {
        "T2_vision_decisive": "vision",
        "T3_proprio_decisive": "proprio",
    }
    output: dict[str, Any] = {}
    for cell, expected_route in expected.items():
        cell_rows = [row for row in learned if row["cell"] == cell]
        critical = []
        for row in cell_rows:
            key = (int(row["seed"]), str(row["sample_id"]))
            vision = lookup[(*key, "fixed_vision")]["prediction"]
            proprio = lookup[(*key, "fixed_proprio")]["prediction"]
            if vision != proprio:
                critical.append(row)
        output[cell] = {
            "all_prediction_rows": len(cell_rows),
            "decision_critical_prediction_rows": len(critical),
            "decision_critical_fraction": float(
                len(critical) / len(cell_rows)
            ),
            "evidence_route_fidelity": float(
                np.mean(
                    [
                        row["evidence_route"] == expected_route
                        for row in critical
                    ]
                )
            ),
            "attribution_accuracy_on_decision_critical_rows": float(
                np.mean(
                    [
                        row["prediction"] == row["truth"]
                        for row in critical
                    ]
                )
            ),
            "unique_matched_cases": len(
                {str(row["group_id"]) for row in critical}
            ),
        }
    return {
        "definition": (
            "Decision-critical means that the fixed vision and fixed "
            "proprioception experts predict different causes."
        ),
        "cells": output,
    }


def _selective_risk(
    report: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method in (
        "learned_router",
        "late_average",
        "fixed_joint",
        "fixed_proprio",
        "fixed_vision",
        "random_route",
        "oracle_route",
    ):
        datasets: dict[str, list[dict[str, float]]] = {}
        for dataset, values in (
            (
                "scale",
                report["metrics"]["scale"]["scene_and_material"][method],
            ),
            ("conflict", report["metrics"]["c2_conflict"][method]),
        ):
            curves = [values[str(seed)]["selective_risk"] for seed in range(5)]
            datasets[dataset] = [
                {
                    "requested_coverage": float(
                        curves[0][index]["requested_coverage"]
                    ),
                    "mean_risk_over_training_seeds": float(
                        np.mean(
                            [
                                curve[index]["selective_risk"]
                                for curve in curves
                            ]
                        )
                    ),
                }
                for index in range(len(curves[0]))
            ]
        output[method] = datasets
    return output


def _loso_summary(report: dict[str, Any]) -> dict[str, Any] | None:
    loso = report.get("nine_scene_loso")
    if not loso:
        return None
    methods = list(next(iter(next(iter(loso.values())).values())).keys())
    output: dict[str, Any] = {}
    for method in methods:
        scene_values = {
            scene: float(
                np.mean(
                    [
                        seed_values[str(seed)][method][
                            "balanced_accuracy"
                        ]
                        for seed in range(5)
                    ]
                )
            )
            for scene, seed_values in loso.items()
        }
        values = list(scene_values.values())
        output[method] = {
            "scene_balanced_accuracy": scene_values,
            "scene_mean": float(np.mean(values)),
            "scene_sample_sd": float(np.std(values, ddof=1)),
            "worst_scene": float(np.min(values)),
            "best_scene": float(np.max(values)),
            "independent_scene_instances": len(values),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=ROOT / "outputs/eval/unified_moe_v3_development_full",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/eval/unified_moe_v3_publication_audit/report.json",
    )
    args = parser.parse_args()
    evaluation_root = args.evaluation_root.resolve()
    report_path = evaluation_root / "report.json"
    predictions_path = evaluation_root / "predictions.jsonl"
    report = _json(report_path)
    predictions = _jsonl(predictions_path)

    cross_battery = report["cross_battery_summary"]
    comparisons = {}
    learned = cross_battery["learned_router"]
    for method in (
        "late_average",
        "fixed_joint",
        "fixed_proprio",
        "fixed_vision",
        "legacy_class_rule",
        "random_route",
        "oracle_route",
    ):
        value = cross_battery[method]
        comparisons[method] = {
            "equal_battery_macro_difference": float(
                learned["equal_battery_macro_mean"]
                - value["equal_battery_macro_mean"]
            ),
            "worst_battery_difference": float(
                learned["worst_battery_mean"]
                - value["worst_battery_mean"]
            ),
        }

    primary_metrics = report["metrics"]["scale"]["scene_and_material"]
    per_material = {}
    for method in ("learned_router", "late_average", "fixed_proprio"):
        material_names = sorted(
            primary_metrics[method]["0"]["per_material"]
        )
        per_material[method] = {
            material: _seed_summary(
                [
                    primary_metrics[method][str(seed)][
                        "per_material"
                    ][material]["balanced_accuracy"]
                    for seed in range(5)
                ]
            )
            for material in material_names
        }

    output = {
        "schema_version": "kinofail.unified-moe-publication-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete_retrospective_publication_audit",
        "confirmatory": False,
        "evidence_boundary": report["evidence_boundary"],
        "counts": report["counts"],
        "single_system_contract": report["single_system_contract"],
        "cross_battery_summary": cross_battery,
        "learned_router_comparisons": comparisons,
        "same_coverage_selective_risk": _selective_risk(report),
        "strict_group_certificates": {
            "scale": _strict_all_seed_groups(
                predictions,
                dataset="scale",
                axis="scene_and_material",
                method="learned_router",
            ),
            "conflict": _strict_all_seed_groups(
                predictions,
                dataset="c2_conflict",
                axis="scene_and_material",
                method="learned_router",
            ),
        },
        "decision_critical_route_audit": _critical_route_audit(
            predictions
        ),
        "operator_condition_strata": _operator_strata(predictions),
        "per_material_balanced_accuracy": per_material,
        "nine_scene_loso": _loso_summary(report),
        "c4_action_audit": report["c4_action_audit"],
        "source_sha256": {
            "evaluation_report": _sha256(report_path),
            "predictions": _sha256(predictions_path),
            "auditor": _sha256(Path(__file__).resolve()),
        },
        "statistical_unit_note": (
            "Training seeds quantify model sensitivity. The independent "
            "experimental units are counterfactual pairs, matched conflict "
            "cases, scene instances, and material families as registered."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.out),
                "scale_strict": output["strict_group_certificates"][
                    "scale"
                ],
                "critical_routes": output[
                    "decision_critical_route_audit"
                ],
                "pairwise_recovery": output["c4_action_audit"].get(
                    "learned_router_pairwise"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
