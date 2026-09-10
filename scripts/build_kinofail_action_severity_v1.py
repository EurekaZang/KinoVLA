#!/usr/bin/env python3
"""Freeze mild/hard action branches around the publication capability band."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/kinofail_action_multiscene_v1_formal_v4"
OUTPUT = ROOT / "outputs/kinofail_action_severity_v1"
COLLECTOR = "scripts/isaac_collect_kinofail_action_full_v1.py"
PROTOCOL_ID = "kinofail-action-severity-v1-post-freeze-20260814"
HORIZON_STEPS: int | None = None
SEVERITY_LEVELS = ("mild", "hard")

REGISTERED_ACTION = {
    "O1_mu_field": "recover_as_low_friction",
    "O2_compliance": "recover_as_O2_compliance",
    "O3_collapse": "recover_as_O3_collapse",
    "O4_tether": "recover_as_O4_tether",
    "O5_payload": "recover_as_payload_unload",
    "O6_push": "recover_as_O6_push",
    "O7_visual_remap": "recover_as_low_friction",
    "O8_invisible_collider": "recover_as_O8_invisible_collider",
    "O9_high_centering": "recover_as_O9_high_centering",
    "O10_effort_decay": "recover_as_O10_effort_decay",
    "O11_obs_bias": "recover_as_sensor_recalibrate",
}

SEVERITY_PARAMETERS: dict[str, dict[str, dict[str, Any]]] = {
    "mild": {
        "O1_mu_field": {"mu_s": 0.08, "mu_d": 0.055, "restitution": 0.0},
        "O2_compliance": {"sink_depth_m": 0.035, "shear_gain": 0.74, "stiffness_n_per_m": 150.0},
        "O3_collapse": {"damage_threshold_ns": 78.0, "drop_m": 0.025, "residual_support": 0.74},
        "O4_tether": {"attachment_enabled": 1.0, "tangential_force_cap_n": 24.0, "normal_force_cap_n": 10.0, "peel_height_m": 0.020, "unload_steps_to_peel": 3.0},
        "O5_payload": {"mass_kg": 6.0, "com_offset_x_m": 0.06, "com_offset_y_m": 0.0},
        "O6_push": {"impulse_ns": 2.2, "duration_s": 0.12, "application_point_body_m": [0.0, -0.085, 0.075]},
        "O7_visual_remap": {"mu_s": 0.08, "mu_d": 0.055, "depth_bias_m": 0.24},
        "O8_invisible_collider": {"collision_enabled": 1.0, "obstacle_height_m": 0.10, "optical_transmission": 0.92},
        "O9_high_centering": {"ridge_height_m": 0.185, "ridge_width_m": 0.23, "residual_support": 0.55},
        "O10_effort_decay": {"effort_floor": 0.80, "decay_rate_per_s": 0.35, "onset_s": 0.90},
        "O11_obs_bias": {"tilt_bias_rad": 0.24, "random_walk_rad_sqrt_s": 0.014, "latency_s": 0.14},
    },
    "hard": {
        "O1_mu_field": {"mu_s": 0.045, "mu_d": 0.028, "restitution": 0.0},
        "O2_compliance": {"sink_depth_m": 0.055, "shear_gain": 0.58, "stiffness_n_per_m": 95.0},
        "O3_collapse": {"damage_threshold_ns": 55.0, "drop_m": 0.042, "residual_support": 0.55},
        "O4_tether": {"attachment_enabled": 1.0, "tangential_force_cap_n": 36.0, "normal_force_cap_n": 15.0, "peel_height_m": 0.030, "unload_steps_to_peel": 4.0},
        "O5_payload": {"mass_kg": 10.5, "com_offset_x_m": 0.11, "com_offset_y_m": 0.0},
        "O6_push": {"impulse_ns": 4.2, "duration_s": 0.12, "application_point_body_m": [0.0, -0.085, 0.075]},
        "O7_visual_remap": {"mu_s": 0.045, "mu_d": 0.028, "depth_bias_m": 0.40},
        "O8_invisible_collider": {"collision_enabled": 1.0, "obstacle_height_m": 0.15, "optical_transmission": 0.92},
        "O9_high_centering": {"ridge_height_m": 0.205, "ridge_width_m": 0.26, "residual_support": 0.36},
        "O10_effort_decay": {"effort_floor": 0.60, "decay_rate_per_s": 0.62, "onset_s": 0.70},
        "O11_obs_bias": {"tilt_bias_rad": 0.40, "random_walk_rad_sqrt_s": 0.024, "latency_s": 0.24},
    },
}

O9_PRODUCTION_004 = {
    "mild": {"ridge_height_m": 0.215, "ridge_width_m": 0.27, "residual_support": 0.34},
    "hard": {"ridge_height_m": 0.235, "ridge_width_m": 0.30, "residual_support": 0.18},
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_protocol = load(SOURCE / "protocol.json")
    replicate_a = [
        row for row in jsonl(SOURCE / "schedule.jsonl") if row.get("source_rank_contract") == "replicate_a"
    ]
    if len(replicate_a) != 132:
        raise RuntimeError(f"expected 132 replicate-a cells, found {len(replicate_a)}")

    cases: list[dict[str, Any]] = []
    for source in replicate_a:
        operator = str(source["operator"])
        scene = str(source["scene_id"])
        for severity in SEVERITY_LEVELS:
            case = copy.deepcopy(source)
            case["schema_version"] = "kinofail.action-severity-v1-schedule.v1"
            case["case_id"] = (
                f"actionseverityv1__{severity}__{scene}__{operator}__"
                f"{case['source_counterfactual_group_id']}"
            )
            case["severity_id"] = severity
            parameters = copy.deepcopy(SEVERITY_PARAMETERS[severity][operator])
            if operator == "O9_high_centering" and scene == "kino4c_production_004":
                parameters = copy.deepcopy(O9_PRODUCTION_004[severity])
            case["source_record"]["physics_parameters"] = parameters
            case["capability_band_parameters"] = parameters
            case["actions"] = ["continue", "always_safe_halt", REGISTERED_ACTION[operator]]
            case["pairing"] = "one severity-specific physical checkpoint restored across three arms"
            case["severity_schedule_frozen_before_collection"] = True
            case["outcome_selection_used"] = False
            if operator in {"O4_tether", "O9_high_centering"}:
                case["source_f35_decision"]["decision_time_s"] = 0.0
                case["source_f35_decision"]["event_time_s"] = 0.0
            cases.append(case)

    scenes = sorted({str(row["scene_id"]) for row in cases})
    operators = sorted({str(row["operator"]) for row in cases})
    expected_cases = 132 * len(SEVERITY_LEVELS)
    if len(cases) != expected_cases or len(scenes) != 12 or len(operators) != 11:
        raise RuntimeError({"cases": len(cases), "scenes": len(scenes), "operators": len(operators)})

    OUTPUT.mkdir(parents=True)
    schedule = OUTPUT / "schedule.jsonl"
    schedule.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in cases), encoding="utf-8"
    )
    registry = load(SOURCE / "scene_registry.json")
    (OUTPUT / "scene_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    protocol = copy.deepcopy(source_protocol)
    protocol.update(
        {
            "schema_version": "kinofail.action-severity-v1-protocol.v1",
            "protocol_id": PROTOCOL_ID,
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "formal_post_freeze_confirmation",
            "development_only": False,
            "integrity_mode": "path_and_count",
            "schedule": str(schedule),
            "scene_registry": str(OUTPUT / "scene_registry.json"),
            "collector": COLLECTOR,
            "runner": "scripts/run_kinofail_experiment_v1.py",
            "counts": {
                "scenes": 12,
                "operators": 11,
                "severity_levels": len(SEVERITY_LEVELS),
                "physical_cases": len(cases),
                "action_arms": 3,
                "recovery_action_arms": 1,
                "unique_recovery_control_programs": 1,
                "physical_episodes": len(cases) * 3,
            },
            "severity_levels": {
                severity: SEVERITY_PARAMETERS[severity]
                for severity in SEVERITY_LEVELS
            },
            "o9_scene_parameter_overrides": O9_PRODUCTION_004,
            "operator_to_registered_action": REGISTERED_ACTION,
            "action_arms": ["continue", "always_safe_halt", "cause-matched registered recovery"],
            "analysis_plan": {
                "unit": "operator-by-scene physical checkpoint",
                "severity_order": list(SEVERITY_LEVELS),
                "primary_contrast": "cause-matched registered recovery versus continue",
                "reference_contrast": "cause-matched registered recovery versus always-safe halt",
                "primary_endpoint": "operator_recovery_success",
                "secondary_endpoints": ["fell", "terminal_cost", "safety_exposure_auc"],
                "cluster": "scene",
                "severity_interaction": "scene-clustered matched difference across all three levels",
                "unfavorable_outcomes_retained": True,
            },
            "selection_contract": {
                "source_rank": "replicate_a fixed for every scene-operator cell",
                "reads_model_predictions": False,
                "reads_action_outcomes": False,
                "result_dependent_parameter_change": False,
            },
        }
    )
    for key in list(protocol):
        if key.endswith("_sha256"):
            protocol.pop(key)
    if HORIZON_STEPS is not None:
        protocol["horizon_steps"] = HORIZON_STEPS
    (OUTPUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "schema_version": "kinofail.action-severity-v1-design-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "counts": protocol["counts"],
        "source_rank": "replicate_a",
        "outcome_selection_used": False,
        "severity_schedule_frozen_before_collection": True,
    }
    (OUTPUT / "design_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
