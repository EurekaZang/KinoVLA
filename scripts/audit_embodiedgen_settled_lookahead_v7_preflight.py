#!/usr/bin/env python3
"""Fail-closed preflight for the Kitchen31 settled-lookahead calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.realistic_route_scene_contract import (
    audit_realistic_route_scene_contract,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_embodiedgen_settled_lookahead_v7_calibration_v1.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_settled_lookahead_v7_calibration_v1/preflight_audit.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _json(config_path)
    boundary = config.get("evidence_boundary", {})
    policy = config.get("execution_policy", {})
    runtime = config.get("runtime", {})
    schema = config.get("schema_version")
    if schema in {
        "kinofail.embodiedgen-lateral-authority-v10-calibration.v1",
        "kinofail.embodiedgen-stable-speed-v11-calibration.v1",
    }:
        controller_contract_frozen = all(
            (
                abs(float(runtime.get("cross_track_gain_per_s", -1.0)) - 1.0)
                <= 1.0e-9,
                abs(float(runtime.get("lateral_velocity_damping", -1.0))) <= 1.0e-9,
                abs(float(runtime.get("heading_gain_per_s", -1.0)) - 2.0)
                <= 1.0e-9,
                abs(float(runtime.get("lateral_limit_mps", -1.0)) - 0.20)
                <= 1.0e-9,
                abs(float(runtime.get("yaw_rate_limit_radps", -1.0)) - 0.60)
                <= 1.0e-9,
            )
        )
    elif schema in {
        "kinofail.embodiedgen-route-aligned-reset-v8-calibration.v1",
        "kinofail.embodiedgen-fixed-joint-reset-v9-calibration.v1",
    }:
        controller_contract_frozen = all(
            (
                abs(float(runtime.get("cross_track_gain_per_s", -1.0)) - 1.0)
                <= 1.0e-9,
                abs(float(runtime.get("lateral_velocity_damping", -1.0))) <= 1.0e-9,
                abs(float(runtime.get("heading_gain_per_s", -1.0)) - 2.0)
                <= 1.0e-9,
                abs(float(runtime.get("lateral_limit_mps", -1.0)) - 0.16)
                <= 1.0e-9,
                abs(float(runtime.get("yaw_rate_limit_radps", -1.0)) - 0.60)
                <= 1.0e-9,
            )
        )
    else:
        controller_contract_frozen = all(
            (
                abs(float(runtime.get("alignment_gain_per_s", -1.0)) - 0.8)
                <= 1.0e-9,
                abs(float(runtime.get("alignment_lateral_limit_mps", -1.0)) - 0.12)
                <= 1.0e-9,
                abs(float(runtime.get("alignment_tolerance_m", -1.0)) - 0.035)
                <= 1.0e-9,
                int(runtime.get("alignment_dwell_steps", -1)) == 10,
                abs(float(runtime.get("lookahead_m", -1.0)) - 0.45) <= 1.0e-9,
            )
        )
    checks: dict[str, bool] = {
        "supported_schema": schema
        in {
            "kinofail.embodiedgen-settled-lookahead-v7-calibration.v1",
            "kinofail.embodiedgen-settled-lookahead-v7-calibration.v2",
            "kinofail.embodiedgen-route-aligned-reset-v8-calibration.v1",
            "kinofail.embodiedgen-fixed-joint-reset-v9-calibration.v1",
            "kinofail.embodiedgen-lateral-authority-v10-calibration.v1",
            "kinofail.embodiedgen-stable-speed-v11-calibration.v1",
        },
        "calibration_only": boundary.get("evidence_role")
        in {
            "controller_architecture_calibration_only",
            "reset_contract_calibration_only",
            "controller_authority_calibration_only",
            "stable_speed_calibration_only",
        },
        "not_registry_evidence": boundary.get("scene_registry_eligible") is False,
        "not_a0_a7_evidence": boundary.get("counts_as_a0_a7_evidence") is False,
        "a0_a7_readiness_zero": boundary.get("realistic_a0_a7_readiness") == "0/8",
        "full_realistic_replication_required": boundary.get(
            "completion_requires_new_realistic_corpus_training_inference_and_a0_a7"
        )
        is True,
        "one_attempt": policy.get("one_attempt") is True,
        "anomaly_blocked_before_nominal_pass": policy.get(
            "run_o2_anomaly_only_after_nominal_passes"
        )
        is True,
        "common_controller_or_start_contract_required": policy.get(
            "same_controller_contract_required_for_o1_o2_o3"
        )
        is True
        or policy.get("same_start_and_controller_required_for_all_operators_and_lanes")
        is True
        or policy.get("hidden_joint_reset_randomization_disabled") is True,
        "nominal_o2_only": runtime.get("operator") == "o2"
        and runtime.get("lane") == "nominal",
        "tracking_budget_frozen": abs(
            float(runtime.get("maximum_route_deviation_m", -1.0)) - 0.30
        )
        <= 1.0e-9,
        "controller_contract_frozen": controller_contract_frozen,
    }
    if schema in {
        "kinofail.embodiedgen-settled-lookahead-v7-calibration.v2",
        "kinofail.embodiedgen-route-aligned-reset-v8-calibration.v1",
    }:
        checks.update(
            {
                "start_matches_target_lane": abs(
                    float(runtime.get("start_lateral_offset_m", 1.0))
                    - float(runtime.get("target_lateral_offset_m", -1.0))
                )
                <= 1.0e-9,
                "start_state_is_operator_and_lane_independent": policy.get(
                    "start_state_may_not_depend_on_operator_or_lane"
                )
                is True,
            }
        )
        for name, spec in config.get("calibration_basis", {}).items():
            if not isinstance(spec, dict) or "path" not in spec:
                continue
            path = _resolve(spec["path"])
            checks[f"basis_{name}_hash"] = path.is_file() and _sha256(path) == spec[
                "sha256"
            ]
    elif schema == "kinofail.embodiedgen-fixed-joint-reset-v9-calibration.v1":
        reset_contract = config.get("inference_reset_contract", {})
        checks.update(
            {
                "hidden_joint_reset_randomization_disabled": policy.get(
                    "hidden_joint_reset_randomization_disabled"
                )
                is True,
                "pose_nuisance_must_be_explicit_and_paired": policy.get(
                    "future_pose_perturbations_must_be_explicit_paired_schedule_factors"
                )
                is True,
                "fixed_default_joint_positions": reset_contract.get(
                    "position_scale_range"
                )
                == [1.0, 1.0]
                and reset_contract.get("seed_may_randomize_initial_joints") is False,
                "fixed_zero_joint_velocities": reset_contract.get(
                    "velocity_scale_range"
                )
                == [0.0, 0.0],
            }
        )
        for name, spec in config.get("calibration_basis", {}).items():
            if not isinstance(spec, dict) or "path" not in spec:
                continue
            path = _resolve(spec["path"])
            checks[f"basis_{name}_hash"] = path.is_file() and _sha256(path) == spec[
                "sha256"
            ]
    elif schema == "kinofail.embodiedgen-lateral-authority-v10-calibration.v1":
        checks.update(
            {
                "authority_change_is_single_preregistered_change": config.get(
                    "calibration_basis", {}
                ).get("single_change")
                == "lateral_limit_mps_from_0.16_to_0.20",
                "authority_remains_inside_policy_training_range": abs(
                    float(runtime.get("lateral_limit_mps", -1.0))
                )
                <= 1.0,
                "no_further_gain_tuning_after_attempt": policy.get(
                    "no_further_gain_tuning_after_this_attempt"
                )
                is True,
            }
        )
        for name, spec in config.get("calibration_basis", {}).items():
            if not isinstance(spec, dict) or "path" not in spec:
                continue
            path = _resolve(spec["path"])
            checks[f"basis_{name}_hash"] = path.is_file() and _sha256(path) == spec[
                "sha256"
            ]
    elif schema == "kinofail.embodiedgen-stable-speed-v11-calibration.v1":
        checks.update(
            {
                "speed_change_is_single_preregistered_change": config.get(
                    "calibration_basis", {}
                ).get("single_change")
                == "forward_speed_mps_from_0.20_to_0.18",
                "stable_speed_is_positive_and_within_training_range": 0.0
                < float(runtime.get("forward_speed_mps", -1.0))
                <= 1.0,
                "stop_rule_frozen": policy.get(
                    "stop_parameter_search_and_retrain_path_tracking_policy_if_this_attempt_fails"
                )
                is True,
            }
        )
        for name, spec in config.get("calibration_basis", {}).items():
            if not isinstance(spec, dict) or "path" not in spec:
                continue
            path = _resolve(spec["path"])
            checks[f"basis_{name}_hash"] = path.is_file() and _sha256(path) == spec[
                "sha256"
            ]
    frozen_inputs: list[dict[str, Any]] = []
    for name, spec in config.get("source_freeze", {}).items():
        path = _resolve(spec["path"])
        valid = path.is_file() and _sha256(path) == spec["sha256"]
        checks[f"source_{name}_hash"] = valid
        frozen_inputs.append({"name": name, "path": str(path), "valid": valid})

    scene_payloads: dict[str, dict[str, Any]] = {}
    for name in (
        "route_surface_admission",
        "terrain_compiled_audit",
        "offline_usd_audit",
    ):
        spec = config["scene"][name]
        path = _resolve(spec["path"])
        valid = path.is_file() and _sha256(path) == spec["sha256"]
        checks[f"scene_{name}_hash"] = valid
        scene_payloads[name] = _json(path) if path.is_file() else {}
    episode_spec = config["scene"]["terrain_episode"]
    episode = _resolve(episode_spec["path"])
    checks["scene_episode_hash"] = episode.is_file() and _sha256(episode) == episode_spec[
        "sha256"
    ]
    checks["scene_inputs_passed"] = all(
        payload.get("passed") is True for payload in scene_payloads.values()
    )
    scene_contract = audit_realistic_route_scene_contract(
        scene_payloads["terrain_compiled_audit"]
    )
    checks["scene_route_contract_passed"] = scene_contract.get("passed") is True
    planned_output = _resolve(config["planned_output"])
    checks["planned_output_unused"] = not planned_output.exists()

    result = {
        "schema_version": "kinofail.embodiedgen-settled-lookahead-v7-preflight.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "frozen_inputs": frozen_inputs,
        "scene_contract": scene_contract,
        "planned_output": str(planned_output),
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
    }
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "passed": result["passed"]}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
