#!/usr/bin/env python3
"""Analyze the frozen realistic A6 four-level, five-seed boundary extension."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
LEVELS = ("nominal", "mild", "moderate", "severe")


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _accepted(row: dict, manifest: dict) -> bool:
    validation = manifest.get("runtime_validation", {})
    if validation.get("passed") is True:
        return True
    issues = [str(value) for value in validation.get("issues", [])]
    certified_o4 = False
    if row["condition"] == "anomaly" and row["target_operator"] == "O4_tether":
        telemetry = manifest.get("operator_readback", {}).get("telemetry", {})
        certified_o4 = (
            int(telemetry.get("total_attachment_cycles", 0)) > 0
            and float(telemetry.get("total_applied_force_n", 0.0)) > 0.0
            and float(telemetry.get("total_tangential_work_j", 0.0)) > 0.0
        )
    return bool(issues) and all(
        issue.endswith("appearance_effect_too_small")
        or issue == "rgb_spatial_contrast_too_low"
        or (issue == "operator_local_qa_failed" and certified_o4)
        for issue in issues
    )


def _mean_ci(values: list[float], *, reps: int, seed: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.asarray([np.mean(rng.choice(array, len(array), replace=True)) for _ in range(reps)])
    return {
        "mean": float(array.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "n": len(array),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eval/kinofail_realistic_a6_analysis_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/eval/realistic_a0_a7_v4/a6_boundary.json")
    args = parser.parse_args()
    config = _json(args.config)
    collection_protocol_path = ROOT / config["collection_protocol"]
    collection_protocol = _json(collection_protocol_path)
    schedule_path = ROOT / config["schedule"]
    if collection_protocol["schedule_sha256"] != _sha(schedule_path):
        raise RuntimeError("A6 schedule/protocol hash mismatch")
    corpus_root = ROOT / config["corpus_root"]
    rows = [json.loads(line) for line in schedule_path.read_text(encoding="utf-8").splitlines() if line]
    by_pair = defaultdict(list)
    for row in rows:
        by_pair[row["counterfactual_group_id"]].append(row)
        manifest_path = corpus_root / row["required_outputs"]["episode_manifest"]
        if not manifest_path.is_file() or not _accepted(row, _json(manifest_path)):
            raise RuntimeError(f"A6 episode missing or invalid: {row['episode_id']}")

    event = config["effect_event"]
    result_rows = []
    for pair_id, pair in sorted(by_pair.items()):
        condition = {row["condition"]: row for row in pair}
        anomaly, nominal = condition["anomaly"], condition["nominal_counterfactual"]
        summary = _json(corpus_root / "pair_summaries" / f"{pair_id}.json")
        outcomes = {row["condition"]: row for row in summary["results"]}
        a, n = outcomes["anomaly"], outcomes["nominal_counterfactual"]
        route_loss = float(n["route_progress_m"]) - float(a["route_progress_m"])
        tilt_increase = float(a["max_tilt_rad"]) - float(n["max_tilt_rad"])
        excess_fall = bool(a["fallen"] and not n["fallen"])
        effect = excess_fall or route_loss >= float(event["or_route_progress_loss_m_ge"]) or tilt_increase >= float(event["or_max_tilt_increase_rad_ge"])
        scalar_cfg = config["parameter_scalar"][anomaly["target_operator"]]
        result_rows.append({
            "operator": anomaly["target_operator"],
            "level": anomaly["severity_id"],
            "level_rank": {"mild": 1, "moderate": 2, "severe": 3}[anomaly["severity_id"]],
            "boundary_seed_index": anomaly["boundary_seed_index"],
            "scene_seed": anomaly["scene_seed"],
            "pair_id": pair_id,
            "parameter_field": scalar_cfg["field"],
            "parameter_value": float(anomaly["physics_parameters"][scalar_cfg["field"]]),
            "route_progress_loss_m": route_loss,
            "max_tilt_increase_rad": tilt_increase,
            "paired_excess_fall": excess_fall,
            "effect_event": effect,
        })
    # Add one non-duplicated nominal level per operator/seed using the mild pair's paired control.
    for operator in sorted(config["parameter_scalar"]):
        for seed_index in range(5):
            source = next(row for row in result_rows if row["operator"] == operator and row["level"] == "mild" and row["boundary_seed_index"] == seed_index)
            schedule_nominal = next(
                row for row in by_pair[source["pair_id"]]
                if row["condition"] == "nominal_counterfactual"
            )
            scalar_cfg = config["parameter_scalar"][operator]
            result_rows.append({
                "operator": operator, "level": "nominal", "level_rank": 0,
                "boundary_seed_index": seed_index, "scene_seed": source["scene_seed"],
                "pair_id": source["pair_id"], "parameter_field": scalar_cfg["field"],
                "parameter_value": float(schedule_nominal["physics_parameters"][scalar_cfg["field"]]),
                "route_progress_loss_m": 0.0, "max_tilt_increase_rad": 0.0,
                "paired_excess_fall": False, "effect_event": False,
            })

    reps, base_seed = int(config["bootstrap"]["repetitions"]), int(config["bootstrap"]["seed"])
    operators = {}
    for operator_index, operator in enumerate(sorted(config["parameter_scalar"])):
        op_rows = [row for row in result_rows if row["operator"] == operator]
        levels = {}
        rates = []
        parameter_values = []
        for level_index, level in enumerate(LEVELS):
            subset = [row for row in op_rows if row["level"] == level]
            if len(subset) != 5 or len({row["scene_seed"] for row in subset}) != 5:
                raise RuntimeError(f"{operator}/{level} lacks five independent seeds")
            levels[level] = {
                "parameter_value": sorted({row["parameter_value"] for row in subset}),
                "effect_rate": _mean_ci([float(row["effect_event"]) for row in subset], reps=reps, seed=base_seed + 100 * operator_index + level_index),
                "route_progress_loss_m": _mean_ci([row["route_progress_loss_m"] for row in subset], reps=reps, seed=base_seed + 1000 + 100 * operator_index + level_index),
                "max_tilt_increase_rad": _mean_ci([row["max_tilt_increase_rad"] for row in subset], reps=reps, seed=base_seed + 2000 + 100 * operator_index + level_index),
            }
            rates.append(levels[level]["effect_rate"]["mean"])
            parameter_values.append(float(np.mean(levels[level]["parameter_value"])))
        crossing = next((index for index in range(1, 4) if rates[index - 1] < 0.5 <= rates[index]), None)
        bracket = None if crossing is None else {
            "lower_level": LEVELS[crossing - 1], "upper_level": LEVELS[crossing],
            "lower_parameter": parameter_values[crossing - 1], "upper_parameter": parameter_values[crossing],
        }
        rho = float(spearmanr(np.arange(4), rates).statistic) if len(set(rates)) > 1 else 0.0
        operators[operator] = {
            "worse_direction": config["parameter_scalar"][operator]["worse_direction"],
            "levels": levels,
            "effect_rate_spearman_vs_rank": rho,
            "identified_boundary_bracket": bracket,
            "interpolated_boundary_point_reported": False,
        }
    checks = {
        "five_operator_parameter_families": len(operators) == 5,
        "four_levels_per_family": all(len(row["levels"]) == 4 for row in operators.values()),
        "five_independent_seeds_per_level": all(all(value["effect_rate"]["n"] == 5 for value in row["levels"].values()) for row in operators.values()),
        "interval_estimates_reported": all(all(len(value["effect_rate"]["ci95"]) == 2 for value in row["levels"].values()) for row in operators.values()),
        "no_unidentified_interpolated_point": all(row["interpolated_boundary_point_reported"] is False for row in operators.values()),
    }
    certificate = _json(ROOT / "outputs/eval/realistic_a0_a7_v4/a0_certificate.json")
    result = {
        "schema_version": "kinofail.realistic-a6-boundary.v1",
        "status": "confirmatory_complete",
        "dataset_scope": "kinofail_realistic",
        "a0_evidence_bundle_sha256": certificate["a0_evidence_bundle_sha256"],
        "analysis_protocol_id": config["protocol_id"],
        "analysis_protocol_sha256": _sha(args.config),
        "collection_protocol_id": collection_protocol["protocol_id"],
        "collection_protocol_sha256": _sha(collection_protocol_path),
        "operators": operators,
        "per_seed_rows": result_rows,
        "acceptance": checks,
        "passed": all(checks.values()),
        "interpretation": "Boundary points are not interpolated. Each family reports the measured adjacent-level bracket or explicitly remains unidentified.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "identified_brackets": {key: value["identified_boundary_bracket"] for key, value in operators.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
