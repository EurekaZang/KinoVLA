#!/usr/bin/env python3
"""Freeze a model-blind operator capability-boundary development sweep.

The sweep changes only physical operator dose.  It uses one source scene and
one pair-shared nuisance realization per operator, three predeclared doses,
and the same three checkpoint branches: continue, universal halt, and the
registered cause-matched recovery.  Results are development-only and select a
recoverable physical band before the larger scene-disjoint action study is
frozen; no classifier output is read by this builder.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/kinofail_action_full_v1_pilot_p5"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_action_full_v1.py"
BASE_COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_reconfirmation_a4_v7.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


# Ordered from the least to the most demanding development dose.  These are
# physical values, not multipliers, so every tested condition is independently
# inspectable in the frozen schedule.
DOSES: dict[str, tuple[tuple[str, dict[str, Any]], ...]] = {
    "O1_mu_field": (
        ("d1", {"mu_s": 0.16, "mu_d": 0.12, "restitution": 0.0}),
        ("d2", {"mu_s": 0.10, "mu_d": 0.07, "restitution": 0.0}),
        ("d3", {"mu_s": 0.06, "mu_d": 0.04, "restitution": 0.0}),
    ),
    "O2_compliance": (
        ("d1", {"sink_depth_m": 0.04, "shear_gain": 0.70, "stiffness_n_per_m": 105.0}),
        ("d2", {"sink_depth_m": 0.06, "shear_gain": 0.55, "stiffness_n_per_m": 150.0}),
        ("d3", {"sink_depth_m": 0.08, "shear_gain": 0.44, "stiffness_n_per_m": 195.0}),
    ),
    "O3_collapse": (
        ("d1", {"damage_threshold_ns": 75.0, "drop_m": 0.025, "residual_support": 0.72}),
        ("d2", {"damage_threshold_ns": 45.0, "drop_m": 0.06, "residual_support": 0.45}),
        ("d3", {"damage_threshold_ns": 35.0, "drop_m": 0.08, "residual_support": 0.34}),
    ),
    "O4_tether": (
        ("d1", {"attachment_enabled": 1.0, "tangential_force_cap_n": 18.0, "normal_force_cap_n": 8.0, "peel_height_m": 0.025, "unload_steps_to_peel": 3.0}),
        ("d2", {"attachment_enabled": 1.0, "tangential_force_cap_n": 24.0, "normal_force_cap_n": 10.0, "peel_height_m": 0.025, "unload_steps_to_peel": 3.0}),
        ("d3", {"attachment_enabled": 1.0, "tangential_force_cap_n": 30.0, "normal_force_cap_n": 12.0, "peel_height_m": 0.025, "unload_steps_to_peel": 3.0}),
    ),
    "O5_payload": (
        ("d1", {"mass_kg": 3.0, "com_offset_x_m": 0.0, "com_offset_y_m": 0.0}),
        ("d2", {"mass_kg": 6.0, "com_offset_x_m": 0.04, "com_offset_y_m": 0.0}),
        ("d3", {"mass_kg": 8.0, "com_offset_x_m": 0.08, "com_offset_y_m": 0.0}),
    ),
    "O6_push": (
        ("d1", {"impulse_ns": 2.0, "duration_s": 0.12, "application_point_body_m": [0.0, -0.085, 0.075]}),
        ("d2", {"impulse_ns": 4.0, "duration_s": 0.12, "application_point_body_m": [0.0, -0.085, 0.075]}),
        ("d3", {"impulse_ns": 6.0, "duration_s": 0.12, "application_point_body_m": [0.0, -0.085, 0.075]}),
    ),
    "O7_visual_remap": (
        ("d1", {"mu_s": 0.16, "mu_d": 0.12, "depth_bias_m": 0.18}),
        ("d2", {"mu_s": 0.10, "mu_d": 0.07, "depth_bias_m": 0.24}),
        ("d3", {"mu_s": 0.06, "mu_d": 0.04, "depth_bias_m": 0.32}),
    ),
    "O8_invisible_collider": (
        ("d1", {"collision_enabled": 1.0, "obstacle_height_m": 0.12, "optical_transmission": 0.92}),
        ("d2", {"collision_enabled": 1.0, "obstacle_height_m": 0.20, "optical_transmission": 0.95}),
        ("d3", {"collision_enabled": 1.0, "obstacle_height_m": 0.28, "optical_transmission": 0.98}),
    ),
    "O9_high_centering": (
        ("d1", {"ridge_height_m": 0.19, "ridge_width_m": 0.24, "residual_support": 0.46}),
        ("d2", {"ridge_height_m": 0.205, "ridge_width_m": 0.26, "residual_support": 0.34}),
        ("d3", {"ridge_height_m": 0.22, "ridge_width_m": 0.28, "residual_support": 0.25}),
    ),
    "O10_effort_decay": (
        ("d1", {"effort_floor": 0.65, "decay_rate_per_s": 0.35, "onset_s": 2.0}),
        ("d2", {"effort_floor": 0.55, "decay_rate_per_s": 0.50, "onset_s": 1.75}),
        ("d3", {"effort_floor": 0.45, "decay_rate_per_s": 0.65, "onset_s": 1.50}),
    ),
    "O11_obs_bias": (
        ("d1", {"tilt_bias_rad": 0.12, "random_walk_rad_sqrt_s": 0.006, "latency_s": 0.06}),
        ("d2", {"tilt_bias_rad": 0.22, "random_walk_rad_sqrt_s": 0.012, "latency_s": 0.12}),
        ("d3", {"tilt_bias_rad": 0.32, "random_walk_rad_sqrt_s": 0.018, "latency_s": 0.18}),
    ),
}


RECOVERY_CALIBRATION_DOSES: dict[
    str, tuple[tuple[str, dict[str, Any]], ...]
] = {
    "O1_mu_field": (
        ("r1", {"mu_s": 0.10, "mu_d": 0.07, "restitution": 0.0}),
        ("r2", {"mu_s": 0.06, "mu_d": 0.04, "restitution": 0.0}),
        ("r3", {"mu_s": 0.03, "mu_d": 0.02, "restitution": 0.0}),
    ),
    "O2_compliance": (
        ("r1", {"sink_depth_m": 0.04, "shear_gain": 0.70, "stiffness_n_per_m": 105.0}),
        ("r2", {"sink_depth_m": 0.06, "shear_gain": 0.55, "stiffness_n_per_m": 150.0}),
        ("r3", {"sink_depth_m": 0.08, "shear_gain": 0.44, "stiffness_n_per_m": 195.0}),
    ),
    "O3_collapse": (
        ("r1", {"damage_threshold_ns": 75.0, "drop_m": 0.025, "residual_support": 0.72}),
        ("r2", {"damage_threshold_ns": 45.0, "drop_m": 0.06, "residual_support": 0.45}),
        ("r3", {"damage_threshold_ns": 35.0, "drop_m": 0.08, "residual_support": 0.34}),
    ),
    "O6_push": (
        ("r1", {"impulse_ns": 4.0, "duration_s": 0.12, "application_point_body_m": [0.0, -0.085, 0.075]}),
        ("r2", {"impulse_ns": 6.0, "duration_s": 0.12, "application_point_body_m": [0.0, -0.085, 0.075]}),
        ("r3", {"impulse_ns": 8.0, "duration_s": 0.12, "application_point_body_m": [0.0, -0.085, 0.075]}),
    ),
    "O7_visual_remap": (
        ("r1", {"mu_s": 0.10, "mu_d": 0.07, "depth_bias_m": 0.24}),
        ("r2", {"mu_s": 0.06, "mu_d": 0.04, "depth_bias_m": 0.32}),
        ("r3", {"mu_s": 0.03, "mu_d": 0.02, "depth_bias_m": 0.36}),
    ),
    "O8_invisible_collider": (
        ("r1", {"collision_enabled": 1.0, "obstacle_height_m": 0.12, "optical_transmission": 0.92}),
        ("r2", {"collision_enabled": 1.0, "obstacle_height_m": 0.20, "optical_transmission": 0.95}),
        ("r3", {"collision_enabled": 1.0, "obstacle_height_m": 0.28, "optical_transmission": 0.98}),
    ),
    "O9_high_centering": (
        ("r1", {"ridge_height_m": 0.19, "ridge_width_m": 0.24, "residual_support": 0.46}),
        ("r2", {"ridge_height_m": 0.205, "ridge_width_m": 0.26, "residual_support": 0.34}),
        ("r3", {"ridge_height_m": 0.22, "ridge_width_m": 0.28, "residual_support": 0.25}),
    ),
    "O10_effort_decay": (
        ("r1", {"effort_floor": 0.92, "decay_rate_per_s": 0.20, "onset_s": 0.80}),
        ("r2", {"effort_floor": 0.85, "decay_rate_per_s": 0.30, "onset_s": 0.80}),
        ("r3", {"effort_floor": 0.78, "decay_rate_per_s": 0.40, "onset_s": 0.80}),
    ),
}


RECOVERY_CONTROL_DOSES: dict[str, tuple[tuple[str, dict[str, Any]], ...]] = {
    "O1_mu_field": RECOVERY_CALIBRATION_DOSES["O1_mu_field"],
    "O2_compliance": (
        ("v1", {"sink_depth_m": 0.035, "shear_gain": 0.74, "stiffness_n_per_m": 95.0}),
        ("v2", {"sink_depth_m": 0.045, "shear_gain": 0.66, "stiffness_n_per_m": 120.0}),
        ("v3", {"sink_depth_m": 0.055, "shear_gain": 0.58, "stiffness_n_per_m": 142.0}),
    ),
    "O3_collapse": (
        ("v1", {"damage_threshold_ns": 80.0, "drop_m": 0.020, "residual_support": 0.78}),
        ("v2", {"damage_threshold_ns": 70.0, "drop_m": 0.030, "residual_support": 0.68}),
        ("v3", {"damage_threshold_ns": 55.0, "drop_m": 0.045, "residual_support": 0.56}),
    ),
    "O6_push": (
        ("v1", {"impulse_ns": 4.0, "duration_s": 0.12, "application_point_body_m": [0.0, -0.085, 0.075]}),
        ("v2", {"impulse_ns": 5.0, "duration_s": 0.12, "application_point_body_m": [0.0, -0.085, 0.075]}),
        ("v3", {"impulse_ns": 6.0, "duration_s": 0.12, "application_point_body_m": [0.0, -0.085, 0.075]}),
    ),
    "O7_visual_remap": RECOVERY_CALIBRATION_DOSES["O7_visual_remap"],
    "O9_high_centering": RECOVERY_CALIBRATION_DOSES["O9_high_centering"],
    "O10_effort_decay": (
        ("v1", {"effort_floor": 0.78, "decay_rate_per_s": 0.40, "onset_s": 0.80}),
        ("v2", {"effort_floor": 0.72, "decay_rate_per_s": 0.48, "onset_s": 0.80}),
        ("v3", {"effort_floor": 0.68, "decay_rate_per_s": 0.56, "onset_s": 0.80}),
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("coarse", "recovery-calibration", "recovery-control"),
        default="coarse",
    )
    parser.add_argument(
        "--operators",
        help="optional comma-separated subset of the selected profile",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)

    source_protocol = load(SOURCE / "protocol.json")
    source_cases = jsonl(SOURCE / "schedule.jsonl")
    by_operator = {str(row["operator"]): row for row in source_cases}
    dose_grid = {
        "coarse": DOSES,
        "recovery-calibration": RECOVERY_CALIBRATION_DOSES,
        "recovery-control": RECOVERY_CONTROL_DOSES,
    }[args.profile]
    if args.operators:
        requested = tuple(value.strip() for value in args.operators.split(","))
        unknown = set(requested) - set(dose_grid)
        if unknown:
            raise ValueError(f"operators absent from profile: {sorted(unknown)}")
        dose_grid = {operator: dose_grid[operator] for operator in requested}
    if not set(dose_grid).issubset(by_operator):
        raise RuntimeError("P5 source does not cover the requested operators")

    cases: list[dict[str, Any]] = []
    for operator, levels in dose_grid.items():
        source = by_operator[operator]
        correct = str(source["operator_recovery"])
        for dose_id, parameters in levels:
            row = copy.deepcopy(source)
            row["schema_version"] = "kinofail.action-capability-sweep-v1-schedule.v1"
            row["case_id"] = (
                f"actioncapv1__{row['scene_id']}__{operator}__{dose_id}__"
                f"{row['source_counterfactual_group_id']}"
            )
            row["severity_id"] = dose_id
            row["source_record"]["physics_parameters"] = copy.deepcopy(parameters)
            row["actions"] = ["continue", "always_safe_halt", correct]
            if args.profile == "recovery-control" and operator in {
                "O1_mu_field",
                "O7_visual_remap",
            }:
                row["operator_engagement_dwell_steps"] = 2
            if args.profile == "recovery-control" and operator == "O9_high_centering":
                row["operator_engagement_dwell_steps"] = 1
                row["strict_o9_semantic_certificate_role"] = (
                    "continue-arm outcome validation, not a delayed branch trigger"
                )
            row["pairing"] = "one exact checkpoint restored across three capability branches"
            row["development_only"] = True
            row["counts_as_publication_evidence"] = False
            row["capability_sweep"] = {
                "dose_id": dose_id,
                "parameters": copy.deepcopy(parameters),
                "ordered_low_to_high": True,
                "selection_uses_model_predictions_or_action_outcomes": False,
            }
            cases.append(row)

    output.mkdir(parents=True)
    schedule = output / "schedule.jsonl"
    schedule.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in cases),
        encoding="utf-8",
    )
    registry = output / "scene_registry.json"
    registry.write_text(
        (SOURCE / "scene_registry.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    protocol = copy.deepcopy(source_protocol)
    protocol.update(
        {
            "schema_version": "kinofail.action-capability-sweep-v1-protocol.v1",
            "protocol_id": (
                f"kinofail-action-capability-sweep-v1-{args.profile}-20260809"
            ),
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "development_capability_sweep_frozen",
            "development_only": True,
            "schedule": str(schedule),
            "schedule_sha256": sha256(schedule),
            "scene_registry": str(registry),
            "scene_registry_sha256": sha256(registry),
            "collector_sha256": sha256(COLLECTOR),
            "base_collector_sha256": sha256(BASE_COLLECTOR),
            "builder": str(Path(__file__).resolve().relative_to(ROOT)),
            "builder_sha256": sha256(Path(__file__).resolve()),
            "counts": {
                "scenes": len({row["scene_id"] for row in cases}),
                "operators": len(dose_grid),
                "physical_cases": len(cases),
                "action_arms": 3,
                "recovery_action_arms": 1,
                "physical_episodes": len(cases) * 3,
            },
            "action_arms": "continue, always_safe_halt, and the operator-registered action",
            "capability_contract": {
                "changed_factor": "operator physical dose only",
                "levels_per_operator": 3,
                "model_outputs_read": False,
                "action_outcomes_read_before_freeze": False,
                "formal_evidence": False,
                "selection_rule": (
                    "choose the highest dose with successful registered recovery and "
                    "a nontrivial continue/control deficit; confirm on new scenes before formal freeze"
                ),
            },
        }
    )
    protocol_path = output / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = {
        "schema_version": "kinofail.action-capability-sweep-v1-design-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "development_only": True,
        "selection_used_model_predictions_or_action_outcomes": False,
        "counts": protocol["counts"],
        "profile": args.profile,
        "dose_grid": dose_grid,
        "hashes": {
            "schedule": sha256(schedule),
            "scene_registry": sha256(registry),
            "protocol": sha256(protocol_path),
            "collector": sha256(COLLECTOR),
            "base_collector": sha256(BASE_COLLECTOR),
        },
    }
    (output / "design_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
