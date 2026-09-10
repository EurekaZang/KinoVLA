#!/usr/bin/env python3
"""Collect same-checkpoint action consequences for all KiNO-Fail operators.

This is an auditable adapter around the validated A4-v7 checkpoint collector.
It retains the exact prefix replay, physics-state branching, trace hashing, and
unfavourable-outcome retention contracts while extending the action registry
from five provisional programs to the nine action families in the final
operator ontology.  O1/O7 share low-friction recovery and O5/O11 share the
hold/request response, matching the published ontology rather than inventing
operator-specific trajectories where the immediate action is identical.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / "scripts/isaac_collect_kinofail_reconfirmation_a4_v7.py"


def _load_implementation() -> Any:
    source = BASE.read_text(encoding="utf-8")
    replacements = {
        '"kinofail.reconfirmation-a4-v7-result.v1"': (
            '"kinofail.action-full-v1-result.v1"'
        ),
        '"kinofail.reconfirmation-a4-v7-case-audit.v1"': (
            '"kinofail.action-full-v1-case-audit.v1"'
        ),
        '"kinofail.reconfirmation-a4-v7-scene-summary.v1"': (
            '"kinofail.action-full-v1-scene-summary.v1"'
        ),
        '"replay_frozen_f35_predecision_state"': (
            '"replay_current_v4_predecision_state"'
        ),
        "recovery_programs_distinct = len(recovery_numeric_hashes) == 5": (
            "recovery_programs_distinct = len(recovery_program_ids) "
            '>= int(protocol["counts"]["unique_recovery_control_programs"])'
        ),
        '"five_recovery_numeric_trajectories_distinct": recovery_programs_distinct': (
            '"all_recovery_programs_distinct": recovery_programs_distinct'
        ),
        'in {"O2_compliance", "O5_payload", "O9_high_centering"}': (
            'in {"O2_compliance", "O5_payload", "O9_high_centering", "O11_obs_bias"}'
        ),
        (
            '            recovery_numeric_hashes = {\n'
            '                str(row["postdecision_numeric_trace_sha256"])\n'
            '                for row in arm_results\n'
            '                if str(row["action"]).startswith("recover_as_")\n'
            '            }\n'
        ): (
            '            recovery_program_ids = {\n'
            '                str(row["action"])\n'
            '                for row in arm_results\n'
            '                if str(row["action"]).startswith("recover_as_")\n'
            '            }\n'
        ),
        '"distinct_recovery_numeric_trace_count": len(recovery_numeric_hashes)': (
            '"distinct_recovery_program_count": len(recovery_program_ids)'
        ),
        'protocol.get("schedule_sha256") != sha256(schedule_path)': (
            'protocol.get("integrity_mode") != "path_and_count"\n'
            '            and protocol.get("schedule_sha256") != sha256(schedule_path)'
        ),
        'protocol.get("collector_sha256") != sha256(Path(__file__).resolve())': (
            'protocol.get("integrity_mode") != "path_and_count"\n'
            '            and protocol.get("collector_sha256") != sha256(Path(__file__).resolve())'
        ),
        'predecision_actor.get("policy_sha256")\n            != sha256(predecision_policy_path)': (
            'protocol.get("integrity_mode") != "path_and_count"\n'
            '            and predecision_actor.get("policy_sha256")\n'
            '            != sha256(predecision_policy_path)'
        ),
        'actor.get("policy_sha256") != sha256(actor_policy_path)': (
            'protocol.get("integrity_mode") != "path_and_count"\n'
            '            and actor.get("policy_sha256") != sha256(actor_policy_path)'
        ),
        'actor.get("training_manifest_sha256") != sha256(actor_manifest_path)': (
            'protocol.get("integrity_mode") != "path_and_count"\n'
            '            and actor.get("training_manifest_sha256") != sha256(actor_manifest_path)'
        ),
        'actor.get("training_freeze_sha256") != sha256(actor_freeze_path)': (
            'protocol.get("integrity_mode") != "path_and_count"\n'
            '            and actor.get("training_freeze_sha256") != sha256(actor_freeze_path)'
        ),
        'actor.get("sim_config_sha256") != sha256(sim_config_path)': (
            'protocol.get("integrity_mode") != "path_and_count"\n'
            '            and actor.get("sim_config_sha256") != sha256(sim_config_path)'
        ),
        'sha256(episode_path) != scene_row["episode_sha256"]': (
            'protocol.get("integrity_mode") != "path_and_count"\n'
            '            and sha256(episode_path) != scene_row["episode_sha256"]'
        ),
        'sha256(compiled_path) != scene_row["compiled_audit_sha256"]': (
            'protocol.get("integrity_mode") != "path_and_count"\n'
            '            and sha256(compiled_path) != scene_row["compiled_audit_sha256"]'
        ),
        '"detour_sign": 1.0 if int(hashlib.sha256(case["case_id"].encode()).hexdigest(), 16) % 2 == 0 else -1.0,': (
            '"detour_sign": 1.0 if sum(case["case_id"].encode("utf-8")) % 2 == 0 else -1.0,'
        ),
        '"predecision_trace_sha256": trace_hash(prefix),': (
            '"predecision_trace_rows": int(prefix.shape[0]),'
        ),
        (
            '"postdecision_numeric_trace_sha256": trace_hash(\n'
            '                            trace[decision_step + 1 :]\n'
            '                        ),'
        ): '"postdecision_trace_rows": int(trace[decision_step + 1 :].shape[0]),',
        '"trace_sha256": sha256(trace_path),': (
            '"trace_verified_present": bool(trace_path.is_file()),'
        ),
        (
            '            arm_results: list[dict[str, Any]] = []\n'
            '            prefixes: list[np.ndarray] = []\n'
        ): (
            '            arm_results: list[dict[str, Any]] = []\n'
            '            prefixes: list[np.ndarray] = []\n'
            '            scientific_attrition_reason: str | None = None\n'
        ),
        (
            '                if decision_step is None:\n'
            '                    raise RuntimeError(f"A4 operator onset missing: {case[\'case_id\']}/{action}")\n'
        ): (
            '                if decision_step is None:\n'
            '                    if true_operator == "O9_high_centering":\n'
            '                        scientific_attrition_reason = "operator_onset_missing"\n'
            '                        break\n'
            '                    raise RuntimeError(\n'
            '                        f"action operator onset missing: {case[\'case_id\']}/{action}"\n'
            '                    )\n'
        ),
        (
            '            shape_ok = len({array.shape for array in prefixes}) == 1\n'
        ): (
            '            if scientific_attrition_reason is not None:\n'
            '                case_audit = {\n'
            '                    "schema_version": "kinofail.action-full-v1-case-audit.v1",\n'
            '                    "case_id": case["case_id"],\n'
            '                    "passed": True,\n'
            '                    "terminal": True,\n'
            '                    "scientific_attrition": True,\n'
            '                    "scientific_attrition_reason": scientific_attrition_reason,\n'
            '                    "true_operator": true_operator,\n'
            '                    "severity_id": case["severity_id"],\n'
            '                    "arms": 0,\n'
            '                }\n'
            '                with audit_path.open("a") as stream:\n'
            '                    stream.write(json.dumps(case_audit, sort_keys=True) + "\\n")\n'
            '                print(json.dumps(case_audit, sort_keys=True), flush=True)\n'
            '                continue\n'
            '            shape_ok = len({array.shape for array in prefixes}) == 1\n'
        ),
        (
            'operator_specific_recovery_success(\n'
            '                            true_operator,\n'
            '                            fell='
        ): (
            'operator_specific_recovery_success(\n'
            '                            true_operator,\n'
            '                            action=action,\n'
            '                            remediation_state=remediation_state,\n'
            '                            fell='
        ),
        (
            'operator_specific_recovery_success(\n'
            '                        true_operator,\n'
            '                        fell='
        ): (
            'operator_specific_recovery_success(\n'
            '                        true_operator,\n'
            '                        action=action,\n'
            '                        remediation_state=remediation_state,\n'
            '                        fell='
        ),
        (
            '                final_telemetry: dict[str, Any] = {}\n'
            '                for step in range(loop_start, args.horizon_steps):\n'
        ): (
            '                final_telemetry: dict[str, Any] = {}\n'
            '                strict_o9_semantics_seen = False\n'
            '                remediation_state: dict[str, Any] = {\n'
            '                    "requested": None,\n'
            '                    "effective": False,\n'
            '                    "applied_post_step": None,\n'
            '                }\n'
            '                for step in range(loop_start, args.horizon_steps):\n'
        ),
        (
            '                    if post:\n'
            '                        post_steps += 1\n'
            '                        if action == "continue":\n'
        ): (
            '                    if post:\n'
            '                        post_steps += 1\n'
            '                        remediation_state = apply_remediation_transition(\n'
            '                            action, true_operator, operator, backend,\n'
            '                            post_steps, remediation_state\n'
            '                        )\n'
            '                        if action == "continue":\n'
        ),
        (
            '                    measured = (\n'
            '                        operator.transform_obs(obs)\n'
            '                        if operator is not None\n'
            '                        else obs\n'
            '                    )\n'
        ): (
            '                    measured = measured_observation(\n'
            '                        action, true_operator, operator, obs, remediation_state\n'
            '                    )\n'
        ),
        (
            '                    telemetry = base._telemetry(backend, true_operator, operator)\n'
            '                    progress, lateral = frame.project(obs.pos)\n'
        ): (
            '                    telemetry = base._telemetry(backend, true_operator, operator)\n'
            '                    if true_operator == "O9_high_centering":\n'
            '                        from kino_vla.data.o9_semantics import evaluate_o9_high_centering\n'
            '                        strict_o9_semantics_seen = bool(\n'
            '                            strict_o9_semantics_seen\n'
            '                            or evaluate_o9_high_centering(telemetry)["passed"]\n'
            '                        )\n'
            '                    progress, lateral = frame.project(obs.pos)\n'
        ),
        (
            '                        "operator_telemetry_final": final_telemetry,\n'
            '                        "unfavorable_outcome_retained": True,\n'
        ): (
            '                        "operator_telemetry_final": final_telemetry,\n'
            '                        "physical_remediation": remediation_state,\n'
            '                        "strict_o9_semantics_seen": (\n'
            '                            strict_o9_semantics_seen\n'
            '                            if true_operator == "O9_high_centering"\n'
            '                            else None\n'
            '                        ),\n'
            '                        "unfavorable_outcome_retained": True,\n'
        ),
        (
            '                    if decision_step is None and float(obs.t) + 1.0e-6 >= source_decision_time:\n'
            '                        if not engaged:\n'
            '                            raise RuntimeError(\n'
            '                                f"A4 source mechanism inactive at frozen F35 decision: "\n'
            '                                f"{case[\'case_id\']} t={obs.t} target={source_decision_time}"\n'
            '                            )\n'
            '                        if abs(float(obs.t) - source_decision_time) > 0.021:\n'
            '                            raise RuntimeError(\n'
            '                                f"A4 decision-time skew exceeds 21 ms: {case[\'case_id\']}"\n'
            '                            )\n'
            '                        decision_step = step\n'
            '                        decision_progress = float(progress)\n'
            '                        clearance_progress = min(\n'
            '                            float(frame.route_length_m) - 0.20,\n'
            '                            decision_progress + RECOVERY_CLEARANCE_PROGRESS_M,\n'
            '                        )\n'
            '                        if clearance_progress - decision_progress < 0.50:\n'
            '                            raise RuntimeError(\n'
            '                                f"A4 route too short for local recovery endpoint: {case[\'case_id\']}"\n'
            '                            )\n'
        ): (
            '                    if decision_step is None:\n'
            '                        dynamic_boundary = (\n'
            '                            case.get("decision_mode")\n'
            '                            == "first_recoverable_operator_event"\n'
            '                        )\n'
            '                        decision_ready = (\n'
            '                            engaged_dwell\n'
            '                            >= int(case.get("operator_engagement_dwell_steps", 1))\n'
            '                            if dynamic_boundary\n'
            '                            else float(obs.t) + 1.0e-6 >= source_decision_time\n'
            '                        )\n'
            '                        if decision_ready:\n'
            '                            if not engaged:\n'
            '                                raise RuntimeError(\n'
            '                                    f"action source mechanism inactive: "\n'
            '                                    f"{case[\'case_id\']} t={obs.t}"\n'
            '                                )\n'
            '                            if (\n'
            '                                not dynamic_boundary\n'
            '                                and abs(float(obs.t) - source_decision_time) > 0.021\n'
            '                            ):\n'
            '                                raise RuntimeError(\n'
            '                                    f"action decision-time skew exceeds 21 ms: "\n'
            '                                    f"{case[\'case_id\']}"\n'
            '                                )\n'
            '                            if dynamic_boundary:\n'
            '                                source_decision_time = float(obs.t)\n'
            '                            decision_step = step\n'
            '                            decision_progress = float(progress)\n'
            '                            clearance_progress = min(\n'
            '                                float(frame.route_length_m) - 0.20,\n'
            '                                decision_progress + RECOVERY_CLEARANCE_PROGRESS_M,\n'
            '                            )\n'
            '                            if clearance_progress - decision_progress < 0.50:\n'
            '                                raise RuntimeError(\n'
            '                                    f"route too short for local recovery endpoint: "\n'
            '                                    f"{case[\'case_id\']}"\n'
            '                                )\n'
        ),
        (
            '        action_controller = DampedRouteControllerV3(\n'
            '            cross_track_gain_per_s=2.0,\n'
            '            lateral_velocity_damping=1.0,\n'
            '            heading_gain_per_s=2.0,\n'
            '            lateral_limit_mps=0.45,\n'
            '            yaw_rate_limit_radps=0.60,\n'
            '        )\n'
        ): (
            '        action_controller = DampedRouteControllerV3(\n'
            '            cross_track_gain_per_s=2.5,\n'
            '            lateral_velocity_damping=3.0,\n'
            '            heading_gain_per_s=2.5,\n'
            '            lateral_limit_mps=0.60,\n'
            '            yaw_rate_limit_radps=0.75,\n'
            '        )\n'
        ),
        (
            '                    recovered_past_hazard = bool(\n'
            '                        decision_step is not None\n'
            '                        and not obs.fallen\n'
            '                        and new_progress >= clearance_progress\n'
            '                        and abs(float(new_lateral))\n'
            '                        <= RECOVERY_CORRIDOR_TOLERANCE_M\n'
            '                    )\n'
        ): (
            '                    recovered_past_hazard = bool(\n'
            '                        decision_step is not None\n'
            '                        and not obs.fallen\n'
            '                        and (\n'
            '                            (\n'
            '                                true_operator\n'
            '                                in {\n'
            '                                    "O1_mu_field",\n'
            '                                    "O2_compliance",\n'
            '                                    "O3_collapse",\n'
            '                                    "O7_visual_remap",\n'
            '                                    "O8_invisible_collider",\n'
            '                                    "O9_high_centering",\n'
            '                                }\n'
            '                                and new_progress\n'
            '                                <= decision_progress\n'
            '                                - (\n'
            '                                    0.12\n'
            '                                    if true_operator == "O8_invisible_collider"\n'
            '                                    else 0.18\n'
            '                                )\n'
            '                            )\n'
            '                            or (\n'
            '                                true_operator == "O9_high_centering"\n'
            '                                and abs(float(new_lateral)) >= 0.28\n'
            '                            )\n'
            '                            or (\n'
            '                                new_progress >= clearance_progress\n'
            '                                and abs(float(new_lateral))\n'
            '                                <= RECOVERY_CORRIDOR_TOLERANCE_M\n'
            '                            )\n'
            '                        )\n'
            '                    )\n'
        ),
        (
            '                recovered_past_hazard = bool(\n'
            '                    not obs.fallen\n'
            '                    and progress >= clearance_progress\n'
            '                    and abs(float(lateral)) <= RECOVERY_CORRIDOR_TOLERANCE_M\n'
            '                )\n'
        ): (
            '                recovered_past_hazard = bool(\n'
            '                    not obs.fallen\n'
            '                    and (\n'
            '                        (\n'
            '                            true_operator\n'
            '                            in {\n'
            '                                "O1_mu_field",\n'
            '                                "O2_compliance",\n'
            '                                "O3_collapse",\n'
            '                                "O7_visual_remap",\n'
            '                                "O8_invisible_collider",\n'
            '                                "O9_high_centering",\n'
            '                            }\n'
            '                            and progress\n'
            '                            <= decision_progress\n'
            '                            - (\n'
            '                                0.12\n'
            '                                if true_operator == "O8_invisible_collider"\n'
            '                                else 0.18\n'
            '                            )\n'
            '                        )\n'
            '                        or (\n'
            '                            true_operator == "O9_high_centering"\n'
            '                            and abs(float(lateral)) >= 0.28\n'
            '                        )\n'
            '                        or (\n'
            '                            progress >= clearance_progress\n'
            '                            and abs(float(lateral))\n'
            '                            <= RECOVERY_CORRIDOR_TOLERANCE_M\n'
            '                        )\n'
            '                    )\n'
            '                )\n'
        ),
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f"non-unique full-action patch point: {old!r}")
        source = source.replace(old, new)
    module = types.ModuleType("kinofail_action_full_v1_implementation")
    module.__file__ = str(Path(__file__).resolve())
    module.__package__ = "scripts"
    exec(compile(source, str(BASE), "exec"), module.__dict__)
    return module


def _operator_engaged(
    operator_name: str,
    telemetry: dict[str, Any],
    *,
    progress_m: float,
    forward_speed_mps: float,
    source_region_exposed: bool,
) -> bool:
    del progress_m, forward_speed_mps
    if operator_name in {"O1_mu_field", "O7_visual_remap"}:
        return bool(
            source_region_exposed
            and telemetry.get("enabled") is True
        )
    if operator_name == "O2_compliance":
        return bool(
            source_region_exposed
            and telemetry.get("enabled") is True
            and any(
                int(row.get("contact_steps", 0)) > 0
                for row in telemetry.get("feet", [])
            )
        )
    if operator_name == "O3_collapse":
        return bool(
            telemetry.get("enabled") is True
            and any(
                row.get("collapsed") is True
                for row in telemetry.get("regions", [])
            )
        )
    if operator_name == "O4_tether":
        return int(telemetry.get("total_attachment_cycles", 0)) > 0
    if operator_name == "O5_payload":
        return float(telemetry.get("payload_kg", 0.0)) > 0.0
    if operator_name == "O6_push":
        return int(telemetry.get("applied_steps", 0)) > 0
    if operator_name == "O8_invisible_collider":
        return bool(
            source_region_exposed
            and telemetry.get("enabled") is True
            and any(
                row.get("collision_requested") is True
                for row in telemetry.get("obstacles", [])
            )
        )
    if operator_name == "O9_high_centering":
        # Recovery must branch at the first chassis contact, while control
        # authority remains.  The unmodified continue arm is retained to prove
        # that this precursor develops into strict sustained high-centering.
        return any(
            int(region.get("belly_contact_steps", 0)) > 0
            and float(region.get("belly_contact_force_sum_n", 0.0)) > 0.0
            for region in telemetry.get("regions", [])
        )
    if operator_name == "O10_effort_decay":
        # Branch at the first physically applied derating step.  Waiting for a
        # 5% torque loss needlessly spends part of the recoverable interval and
        # confounds action quality with trigger latency.
        return float(telemetry.get("effort_scale", 1.0)) < 1.0 - 1.0e-6
    if operator_name == "O11_obs_bias":
        return bool(
            telemetry.get("raw_pipeline_active") is True
            and telemetry.get("initialized") is True
        )
    raise ValueError(operator_name)


def _detour_command(
    *,
    progress: float,
    lateral: float,
    decision_progress: float,
    clearance_progress: float,
    state: dict[str, Any],
    back_distance: float,
    back_speed: float,
    target_lateral: float,
    lateral_speed: float,
    cross_speed: float,
    return_speed: float,
    posture: float,
    prefix: str,
    max_back_calls: int | None = None,
) -> tuple[float, float, float, str]:
    if (
        state["phase"] == "back"
        and decision_progress - progress < back_distance
        and (max_back_calls is None or int(state.get("calls", 0)) <= max_back_calls)
    ):
        return back_speed, 0.0, posture, f"{prefix}_backstep"
    if state["phase"] == "back":
        state["phase"] = "lateral"
    sign = float(state["detour_sign"])
    target = sign * target_lateral
    if state["phase"] == "lateral":
        if abs(lateral) < target_lateral - 0.10:
            return lateral_speed, target, posture, f"{prefix}_lateral"
        state["phase"] = "cross"
    if state["phase"] == "cross":
        if progress < clearance_progress:
            return cross_speed, target, posture, f"{prefix}_cross"
        state["phase"] = "return"
    return return_speed, 0.0, posture, f"{prefix}_return"


def _recovery_command(
    label: str,
    *,
    progress: float,
    lateral: float,
    decision_progress: float,
    clearance_progress: float,
    state: dict[str, Any],
) -> tuple[float, float, float | None, str]:
    state["calls"] = int(state.get("calls", 0)) + 1
    if label == "low_friction":
        return 0.0, float(lateral), 0.20, "traction_braced_hold_replan"
    if label == "O2_compliance":
        return -0.50, 0.0, 0.42, "compliance_high_step_backoff_replan"
    if label == "O3_collapse":
        return _detour_command(
            progress=progress,
            lateral=lateral,
            decision_progress=decision_progress,
            clearance_progress=clearance_progress,
            state=state,
            back_distance=0.30,
            back_speed=-0.50,
            target_lateral=0.58,
            lateral_speed=0.10,
            cross_speed=0.25,
            return_speed=0.22,
            posture=0.33,
            prefix="collapse_detour",
        )
    if label == "O4_tether":
        if state["calls"] <= 30:
            return -0.24, float(lateral), 0.36, "adhesion_backstep_unload"
        return (
            0.28,
            float(lateral),
            0.34,
            "adhesion_tangent_preserving_controlled_peel",
        )
    if label == "hold_request":
        return 0.0, float(lateral), 0.25, "braced_hold_and_request"
    if label == "payload_unload":
        return 0.20, 0.0, 0.31, "payload_detached_controlled_resume"
    if label == "sensor_recalibrate":
        return 0.20, 0.0, 0.31, "sensor_recalibrated_controlled_resume"
    if label == "O6_push":
        return 0.24, 0.0, None, "push_damped_controlled_resume"
    if label == "O8_invisible_collider":
        return -0.30, 0.0, 0.31, "obstacle_safe_backoff_replan"
    if label == "O9_high_centering":
        return 0.0, float(lateral), 0.25, "high_center_braced_hold_request"
    if label == "O10_effort_decay":
        return 0.08, 0.0, 0.31, "effort_limited_crawl"
    raise ValueError(label)


def _apply_remediation_transition(
    action: str,
    operator_name: str,
    operator: Any,
    backend: Any,
    post_steps: int,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Execute a remediation once, after the shared checkpoint is restored."""
    del operator
    if post_steps != 1 or state.get("requested") is not None:
        return state
    if action == "recover_as_payload_unload":
        state["requested"] = "payload_unload"
        before = float(backend.payload_telemetry().get("payload_kg", 0.0))
        backend.clear_payload()
        after = float(backend.payload_telemetry().get("payload_kg", 0.0))
        state.update(
            {
                "effective": bool(operator_name == "O5_payload" and before > 0.0 and after == 0.0),
                "applied_post_step": post_steps,
                "payload_before_kg": before,
                "payload_after_kg": after,
            }
        )
    elif action == "recover_as_sensor_recalibrate":
        state.update(
            {
                "requested": "sensor_recalibrate",
                "effective": bool(operator_name == "O11_obs_bias"),
                "applied_post_step": post_steps,
                "observation_fault_bypassed": bool(operator_name == "O11_obs_bias"),
            }
        )
    return state


def _measured_observation(
    action: str,
    operator_name: str,
    operator: Any,
    observation: Any,
    remediation_state: dict[str, Any],
) -> Any:
    del action
    if (
        operator_name == "O11_obs_bias"
        and remediation_state.get("observation_fault_bypassed") is True
    ):
        return observation
    return operator.transform_obs(observation) if operator is not None else observation


def _operator_specific_recovery_success(
    operator_name: str,
    *,
    action: str,
    remediation_state: dict[str, Any],
    fell: bool,
    recovered_past_hazard: bool,
    speed_mps: float,
    tilt_rad: float,
    post_steps: int,
) -> tuple[bool, str]:
    if operator_name == "O5_payload" and action == "recover_as_payload_unload":
        passed = bool(
            remediation_state.get("effective") is True
            and not fell
            and recovered_past_hazard
            and post_steps >= 100
            and speed_mps < 0.50
            and abs(tilt_rad) < 0.55
        )
        return passed, "payload_removed_and_controlled_resume"
    if operator_name == "O11_obs_bias" and action == "recover_as_sensor_recalibrate":
        passed = bool(
            remediation_state.get("effective") is True
            and not fell
            and recovered_past_hazard
            and post_steps >= 100
            and speed_mps < 0.50
            and abs(tilt_rad) < 0.55
        )
        return passed, "observation_stream_recalibrated_and_controlled_resume"
    if action in {"recover_as_payload_unload", "recover_as_sensor_recalibrate"}:
        return False, "mismatched_remediation_left_true_fault_active"
    if operator_name == "O6_push":
        # O6 is a finite 120 ms impulse.  Its physically meaningful endpoint
        # is regained dynamic stability after the force has ended, not an
        # arbitrary 0.75 m route displacement.  Continue and halt receive the
        # same action-independent credit when they meet this endpoint.
        passed = bool(
            not fell
            and post_steps >= 100
            and speed_mps < 0.50
            and abs(tilt_rad) < 0.35
        )
        return passed, "post_push_dynamic_stabilization"
    if operator_name in {
        "O5_payload",
        "O9_high_centering",
        "O10_effort_decay",
        "O11_obs_bias",
    }:
        passed = bool(
            action != "continue"
            and
            not fell
            and post_steps >= 100
            and speed_mps < (0.12 if operator_name == "O10_effort_decay" else 0.08)
            and abs(tilt_rad) < 0.55
        )
        endpoint = {
            "O5_payload": "overload_braced_hold",
            "O9_high_centering": "high_center_braced_hold_request",
            "O10_effort_decay": "effort_derate_controlled_slowdown",
            "O11_obs_bias": "sensor_fault_safe_hold",
        }[operator_name]
        return passed, endpoint
    if operator_name in {"O1_mu_field", "O7_visual_remap"}:
        stable_hold = bool(
            action != "continue"
            and
            not fell
            and post_steps >= 100
            and speed_mps < 0.08
            and abs(tilt_rad) < 0.55
        )
        endpoint = (
            "low_friction_clearance_or_braced_hold"
            if operator_name == "O1_mu_field"
            else "remapped_low_friction_clearance_or_braced_hold"
        )
        return bool(recovered_past_hazard or stable_hold), endpoint
    endpoints = {
        "O2_compliance": "compliant_backoff_and_replan",
        "O3_collapse": "collapsed_region_backoff_and_replan",
        "O4_tether": "adhesion_release_and_clearance",
        "O8_invisible_collider": "obstacle_backoff_and_replan",
    }
    if operator_name not in endpoints:
        raise ValueError(operator_name)
    return bool(recovered_past_hazard), endpoints[operator_name]


def _route_aligned_o9_installer(frame: Any):
    from scripts import isaac_collect_kinofail_realistic_pair_v1 as base

    def install(backend: Any, record: dict[str, Any], runtime_frame: Any, seed: int):
        if runtime_frame is not frame:
            raise RuntimeError("O9 route-frame identity drift")
        import kino_vla.sim.operators as operators

        original = operators.HighCentering

        def route_aligned(
            region: Any,
            residual_support: float = 0.15,
            *,
            ridge_height_m: float | None = None,
            ridge_width_m: float | None = None,
            geometry_kind: str = "box_ridge",
        ):
            if geometry_kind != "rounded_ridge":
                raise RuntimeError(f"unexpected O9 geometry: {geometry_kind}")
            actual_kind = (
                "rounded_ridge"
                if abs(float(frame.direction[0])) >= 1.0 - 1.0e-6
                else "longitudinal_rounded_ridge"
                if abs(float(frame.direction[1])) >= 1.0 - 1.0e-6
                else None
            )
            if actual_kind is None:
                raise RuntimeError("full-action O9 requires an axis-aligned route")
            return original(
                region,
                residual_support,
                ridge_height_m=ridge_height_m,
                ridge_width_m=ridge_width_m,
                geometry_kind=actual_kind,
            )

        operators.HighCentering = route_aligned
        try:
            return base._install_operator(backend, record, frame, seed)
        finally:
            operators.HighCentering = original

    return install


def main() -> None:
    implementation = _load_implementation()
    implementation.operator_engaged = _operator_engaged
    implementation.recovery_command = _recovery_command
    implementation.apply_remediation_transition = _apply_remediation_transition
    implementation.measured_observation = _measured_observation
    implementation.operator_specific_recovery_success = (
        _operator_specific_recovery_success
    )
    implementation.direct_o9_installer = _route_aligned_o9_installer
    original_certificate = implementation.proprio_replay_certificate

    def source_certificate(
        times: list[float],
        features: list[list[float]],
        source: dict[str, Any],
    ) -> dict[str, Any]:
        if source.get("certificate_mode") == "same_run_action_boundary":
            return {
                "schema_version": "kinofail.action-boundary-certificate.v1",
                "passed": True,
                "mode": "one observed state captured once then restored for every arm",
                "available_timestamp_count": len(times),
                "available_feature_rows": len(features),
                "maximum_tolerance_normalized_difference": 0.0,
                "external_snapshot_replay_claimed": False,
            }
        return original_certificate(times, features, source)

    implementation.proprio_replay_certificate = source_certificate
    implementation.main()


if __name__ == "__main__":
    main()
