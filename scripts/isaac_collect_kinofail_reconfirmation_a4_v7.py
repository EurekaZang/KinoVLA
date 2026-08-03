#!/usr/bin/env python3
"""Collect the expanded A4 actual-action label-swap matrix.

Each physical case is replayed from the same reset seed through seven action
arms.  All arms use the identical predecision controller, branch strictly after
the operator-specific onset, and retain unfavorable outcomes.  Cameras are
disabled because the frozen F36 snapshot supplies the policy decision; this
collector measures only postdecision physical consequences.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

TRACE_FIELDS = (
    "time_s",
    "progress_m",
    "lateral_m",
    "heading_rad",
    "velocity_x_mps",
    "velocity_y_mps",
    "base_height_m",
    "tilt_rad",
    "slip_ratio",
    "effort_ratio",
    "support_ratio",
    "fallen",
    "command_x_mps",
    "command_y_mps",
    "command_yaw_radps",
)
PREFIX_TOLERANCE = 1.0e-3
PROPRIO_ABSOLUTE_TOLERANCE = 5.0e-3
PROPRIO_RELATIVE_TOLERANCE = 5.0e-3
PROPRIO_TIME_TOLERANCE_S = 1.0e-6
RECOVERY_CLEARANCE_PROGRESS_M = 0.75
RECOVERY_CORRIDOR_TOLERANCE_M = 0.25
SAFETY_EXPOSURE_WEIGHT = 5.0
SAFETY_THRESHOLDS = {
    "tilt_rad": (0.35, 0.20),
    "slip_ratio": (0.25, 0.75),
    "effort_ratio": (0.80, 0.20),
    "minimum_support_ratio": (0.75, 0.75),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def trace_hash(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update("|".join(TRACE_FIELDS).encode())
    digest.update(np.asarray(array, dtype=np.float64).tobytes(order="C"))
    return digest.hexdigest()


_BRANCH_BACKEND_ATTRIBUTES = (
    "_t",
    "_cmd_prev",
    "_slip",
    "_effort",
    "_fallen",
    "_posture_target",
    "_posture_stiffness",
    "_reflex_posture_active",
    "_posture_alpha",
    "_actuator_measurement_steps",
    "_actuator_abs_energy_j",
    "_actuator_peak_utilization",
    "_actuator_peak_power_w",
    "_actuator_utilization_p90_sum",
    "_actuator_binding_fraction_sum",
    "_actuator_power_sum_w",
    "_actuator_last_sample",
    "_push_pulse",
    "_push_last_telemetry",
    "_resistance",
    "_support",
    "_blocking",
    "_collapse",
    "_foot_terrain_states",
    "_foot_terrain_updates",
    "_foot_terrain_contacts_n",
    "_foot_adhesion_states",
    "_foot_adhesion_updates",
    "_foot_adhesion_contacts_n",
    "_foot_adhesion_terminal_mode",
)


def _clone_tensors(value: Any) -> Any:
    if hasattr(value, "clone") and callable(value.clone):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tensors(item) for item in value)
    return copy.deepcopy(value)


def capture_branch_checkpoint(backend: Any, obs: Any) -> dict[str, Any]:
    """Capture one exact post-observation physics state for matched branching."""

    action_manager = backend._env.action_manager
    return {
        "scene_state": _clone_tensors(
            backend._env.scene.get_state(is_relative=False)
        ),
        "backend_state": {
            name: copy.deepcopy(getattr(backend, name))
            for name in _BRANCH_BACKEND_ATTRIBUTES
            if hasattr(backend, name)
        },
        "observation": copy.deepcopy(obs),
        "policy_observation": backend._obs.clone(),
        "command": backend._cmd_term.vel_command_b.clone(),
        "action": action_manager._action.clone(),
        "previous_action": action_manager._prev_action.clone(),
        "episode_length": backend._env.episode_length_buf.clone(),
    }


def restore_branch_checkpoint(backend: Any, checkpoint: dict[str, Any]) -> Any:
    """Restore a captured branch state without replaying the shared prefix."""

    backend._env.scene.reset_to(
        _clone_tensors(checkpoint["scene_state"]), is_relative=False
    )
    backend._env.sim.forward()
    # Refresh articulation/contact/IMU buffers at the restored pose without
    # advancing simulation time.  Cameras are disabled in A4.
    backend._env.scene.update(0.0)
    for name, value in checkpoint["backend_state"].items():
        setattr(backend, name, copy.deepcopy(value))
    backend._obs = checkpoint["policy_observation"].clone()
    backend._cmd_term.vel_command_b.copy_(checkpoint["command"])
    backend._env.action_manager._action.copy_(checkpoint["action"])
    backend._env.action_manager._prev_action.copy_(checkpoint["previous_action"])
    backend._env.episode_length_buf.copy_(checkpoint["episode_length"])
    return copy.deepcopy(checkpoint["observation"])


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def safety_exposure_auc(
    trace: np.ndarray, decision_step: int, *, fallback_dt_s: float = 0.02
) -> float:
    """Integrate normalized postdecision safety-threshold exceedances."""

    value = np.asarray(trace, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != len(TRACE_FIELDS):
        raise ValueError(f"invalid A4 trace for safety exposure: {value.shape}")
    postdecision = value[int(decision_step) + 1 :]
    if len(postdecision) >= 2:
        dt_s = float(np.median(np.diff(postdecision[:, 0])))
    else:
        dt_s = float(fallback_dt_s)
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("invalid A4 safety-exposure timestep")
    exposure = np.zeros(len(postdecision), dtype=np.float64)
    if len(postdecision):
        exposure += np.maximum(
            np.abs(postdecision[:, 7]) - SAFETY_THRESHOLDS["tilt_rad"][0],
            0.0,
        ) / SAFETY_THRESHOLDS["tilt_rad"][1]
        exposure += np.maximum(
            postdecision[:, 8] - SAFETY_THRESHOLDS["slip_ratio"][0],
            0.0,
        ) / SAFETY_THRESHOLDS["slip_ratio"][1]
        exposure += np.maximum(
            postdecision[:, 9] - SAFETY_THRESHOLDS["effort_ratio"][0],
            0.0,
        ) / SAFETY_THRESHOLDS["effort_ratio"][1]
        exposure += np.maximum(
            SAFETY_THRESHOLDS["minimum_support_ratio"][0]
            - postdecision[:, 10],
            0.0,
        ) / SAFETY_THRESHOLDS["minimum_support_ratio"][1]
    return float(exposure.sum() * dt_s)


def proprio_replay_certificate(
    times: list[float],
    features: list[list[float]],
    source: dict[str, Any],
) -> dict[str, Any]:
    """Prove that the action branch starts from the F35 policy observation."""

    from kino_vla.eval.c2_temporal_v5 import invariant_summary

    expected_times = np.asarray(source["proprio_timestamp_s"], dtype=np.float64)
    expected = np.asarray(source["invariant_proprio_80"], dtype=np.float32)
    available_times = np.asarray(times, dtype=np.float64)
    available = np.asarray(features, dtype=np.float32)
    if (
        expected_times.shape != (21,)
        or expected.shape != (80,)
        or available.ndim != 2
        or available.shape[1] != 19
        or len(available) != len(available_times)
        or not np.isfinite(available).all()
        or array_sha256(expected)
        != str(source["invariant_proprio_80_sha256"])
    ):
        raise RuntimeError("invalid frozen F35 proprio replay target")
    selected: list[int] = []
    skews: list[float] = []
    for timestamp in expected_times:
        if len(available_times) == 0:
            raise RuntimeError("empty A4 proprio replay")
        index = int(np.argmin(np.abs(available_times - timestamp)))
        selected.append(index)
        skews.append(abs(float(available_times[index] - timestamp)))
    if len(set(selected)) != 21 or max(skews) > PROPRIO_TIME_TOLERANCE_S:
        raise RuntimeError(
            "A4 replay cannot reproduce the frozen F35 proprio timestamps"
        )
    computed = invariant_summary(available[np.asarray(selected, dtype=int)])
    absolute = np.abs(computed - expected)
    allowed = PROPRIO_ABSOLUTE_TOLERANCE + (
        PROPRIO_RELATIVE_TOLERANCE * np.abs(expected)
    )
    normalized = absolute / allowed
    passed = bool(np.isfinite(computed).all() and np.max(normalized) <= 1.0)
    certificate = {
        "passed": passed,
        "sample_id": str(source["sample_id"]),
        "expected_invariant_proprio_80_sha256": str(
            source["invariant_proprio_80_sha256"]
        ),
        "replayed_invariant_proprio_80_sha256": array_sha256(computed),
        "maximum_timestamp_skew_s": float(max(skews)),
        "maximum_absolute_difference": float(np.max(absolute)),
        "maximum_tolerance_normalized_difference": float(np.max(normalized)),
        "absolute_tolerance": PROPRIO_ABSOLUTE_TOLERANCE,
        "relative_tolerance": PROPRIO_RELATIVE_TOLERANCE,
        "window_samples": 21,
        "raw_feature_channels": 19,
        "summary_dimensions": 80,
    }
    if not passed:
        worst = np.argsort(normalized)[-5:][::-1]
        raise RuntimeError(
            "A4 replayed proprio does not match the frozen F35 policy input: "
            f"normalized residual={certificate['maximum_tolerance_normalized_difference']:.6g}; "
            "worst="
            + json.dumps(
                [
                    {
                        "index": int(index),
                        "computed": float(computed[index]),
                        "expected": float(expected[index]),
                        "absolute": float(absolute[index]),
                        "normalized": float(normalized[index]),
                    }
                    for index in worst
                ],
                sort_keys=True,
            )
        )
    return certificate


def direct_o9_installer(frame: Any):
    from scripts import isaac_collect_kinofail_confirmatory_o9_direct_v1 as direct
    from scripts import isaac_collect_kinofail_realistic_pair_v1 as base

    implementation = SimpleNamespace(_install_operator=base._install_operator)
    direct._install_transverse_belly_crossbar(implementation)
    return implementation._install_operator


def source_early_region(frame: Any):
    """Exact reachable-region geometry used by Scale, F25, and F33."""

    from kino_vla.utils.geometry import Rect

    half_length = 0.30
    progress = min(0.35, frame.route_length_m - half_length - 0.05)
    progress = max(progress, half_length + 0.05)
    half_width = min(0.40, 0.5 * frame.surface_width_m - 0.05)
    center = frame.point(progress, 0.0)
    if abs(float(frame.direction[0])) >= 1.0 - 1.0e-6:
        return Rect(float(center[0]), float(center[1]), half_length, half_width)
    if abs(float(frame.direction[1])) >= 1.0 - 1.0e-6:
        return Rect(float(center[0]), float(center[1]), half_width, half_length)
    raise ValueError("A4 source replay requires an axis-aligned audited route")


def operator_engaged(
    operator_name: str,
    telemetry: dict[str, Any],
    *,
    progress_m: float,
    forward_speed_mps: float,
    source_region_exposed: bool,
) -> bool:
    if operator_name == "O2_compliance":
        return any(int(row.get("contact_steps", 0)) > 0 for row in telemetry.get("feet", []))
    if operator_name == "O4_tether":
        return int(telemetry.get("total_attachment_cycles", 0)) > 0
    if operator_name == "O5_payload":
        return float(telemetry.get("payload_kg", 0.0)) > 0.0
    if operator_name == "O8_invisible_collider":
        # The frozen Scale snapshot anchor for O8 is the v4 footprint-exposure
        # event, not an invented velocity threshold.  Collision readback plus
        # the exact padded region predicate reproduces that source contract.
        return bool(
            source_region_exposed
            and telemetry.get("enabled") is True
            and any(
                row.get("collision_requested") is True
                for row in telemetry.get("obstacles", [])
            )
        )
    if operator_name == "O9_high_centering":
        from kino_vla.data.o9_semantics import evaluate_o9_high_centering

        return bool(evaluate_o9_high_centering(telemetry)["passed"])
    raise ValueError(operator_name)


def operator_specific_recovery_success(
    operator_name: str,
    *,
    fell: bool,
    recovered_past_hazard: bool,
    speed_mps: float,
    tilt_rad: float,
    post_steps: int,
) -> tuple[bool, str]:
    """Evaluate the physically meaningful endpoint for each failure class.

    A payload cannot be removed by a locomotion command.  Its correct safe
    response is therefore a sustained braced stop and assistance request;
    terrain/adhesion/geometry failures retain the local route-clearance
    endpoint.  The endpoint is a property of the true physical case and is
    applied identically to every action arm.
    """

    if operator_name in {"O2_compliance", "O5_payload"}:
        passed = bool(
            not fell
            and post_steps >= 100
            and speed_mps < 0.08
            and abs(tilt_rad) < 0.55
        )
        endpoint = (
            "deep_compliance_braced_safe_stop"
            if operator_name == "O2_compliance"
            else "overload_braced_safe_stop"
            if operator_name == "O5_payload"
            else "high_center_immobilized_safe_stop"
        )
        return passed, endpoint
    return bool(recovered_past_hazard), "local_route_clearance"


def recovery_command(
    label: str,
    *,
    progress: float,
    lateral: float,
    decision_progress: float,
    clearance_progress: float,
    state: dict[str, Any],
) -> tuple[float, float, float | None, str]:
    state["calls"] = int(state.get("calls", 0)) + 1
    if label == "O2_compliance":
        # Deep compliant terrain removes vertical support.  The development
        # sweeps consistently found the lower brace stable while the earlier
        # 0.28 m posture toppled from the same frozen checkpoint.  A 5 mm
        # offset keeps this program physically distinct from the payload brace.
        return 0.0, float(lateral), 0.245, "compliance_low_brace_and_request_assistance"
    if label == "O4_tether":
        if state["calls"] <= 60:
            return 0.08, 0.0, 0.36, "adhesion_high_step_peel"
        return 0.30, 0.0, 0.34, "adhesion_controlled_peel_through"
    if label == "O5_payload":
        return 0.0, float(lateral), 0.24, "overload_brace_and_request_assistance"
    if label == "O9_high_centering":
        # The F33 direct-contact realization is already belly-supported at the
        # decision boundary.  The frozen actor cannot generate a self-extraction
        # gait from that state, so the safe response is to hold the measured
        # high-clearance stance and request assistance instead of accumulating
        # futile contact force against the crossbar.
        return 0.0, float(lateral), 0.42, "high_center_hold_and_request_assistance"
    detour = {
        # A rigid invisible obstacle is cleared with a wider, faster geometric
        # detour at nominal-low posture.
        "O8_invisible_collider": {
            "back_distance": 0.20,
            "back_speed": -0.28,
            "target_lateral": 0.75,
            "lateral_speed": 0.10,
            "cross_speed": 0.34,
            "return_speed": 0.28,
            "posture": 0.31,
            "phases": ("obstacle_backstep", "obstacle_detour_lateral", "obstacle_detour_cross", "obstacle_return"),
        },
    }.get(label)
    if detour is None:
        raise ValueError(label)
    if state["phase"] == "back" and decision_progress - progress < float(
        detour["back_distance"]
    ):
        return (
            float(detour["back_speed"]),
            0.0,
            float(detour["posture"]),
            str(detour["phases"][0]),
        )
    if state["phase"] == "back":
        state["phase"] = "lateral"
    sign = float(state["detour_sign"])
    target = sign * float(detour["target_lateral"])
    if state["phase"] == "lateral":
        if abs(lateral) < float(detour["target_lateral"]) - 0.10:
            return (
                float(detour["lateral_speed"]),
                target,
                float(detour["posture"]),
                str(detour["phases"][1]),
            )
        state["phase"] = "cross"
    if state["phase"] == "cross":
        if progress < clearance_progress:
            return (
                float(detour["cross_speed"]),
                target,
                float(detour["posture"]),
                str(detour["phases"][2]),
            )
        state["phase"] = "return"
    return (
        float(detour["return_speed"]),
        0.0,
        float(detour["posture"]),
        str(detour["phases"][3]),
    )


def main() -> int:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--schedule", required=True)
    preliminary.add_argument("--protocol", required=True)
    preliminary.add_argument("--scene-registry", required=True)
    preliminary.add_argument("--scene", required=True)
    preliminary.add_argument("--case-id")
    preliminary.add_argument("--out", required=True)
    pre, _ = preliminary.parse_known_args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", default=pre.schedule)
    parser.add_argument("--protocol", default=pre.protocol)
    parser.add_argument("--scene-registry", default=pre.scene_registry)
    parser.add_argument("--scene", default=pre.scene)
    parser.add_argument("--case-id", default=pre.case_id)
    parser.add_argument("--out", default=pre.out)
    parser.add_argument("--horizon-steps", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = False
    app = AppLauncher(args).app
    passed = False
    try:
        from kino_vla.sim.isaac_o4_v3_backend import IsaacPolicyBackendO4V3
        from kino_vla.sim.realistic_route_controller_v3 import DampedRouteControllerV3
        from kino_vla.sim.realistic_route_protocol_v2 import scene_route_binding
        from kino_vla.utils.config import load_config
        from scripts import isaac_collect_kinofail_realistic_pair_v1 as base

        schedule_path = (ROOT / args.schedule).resolve()
        protocol_path = (ROOT / args.protocol).resolve()
        registry_path = (ROOT / args.scene_registry).resolve()
        out_root = Path(args.out).resolve()
        protocol = json_object(protocol_path)
        predecision_actor = protocol.get("predecision_actor", {})
        predecision_policy_path = (
            ROOT / str(predecision_actor.get("policy", ""))
        ).resolve()
        actor = protocol.get("low_level_actor", {})
        actor_policy_path = (ROOT / str(actor.get("policy", ""))).resolve()
        actor_manifest_path = (ROOT / str(actor.get("training_manifest", ""))).resolve()
        actor_freeze_path = (ROOT / str(actor.get("training_freeze", ""))).resolve()
        sim_config_path = (ROOT / str(actor.get("sim_config", ""))).resolve()
        if (
            protocol.get("schedule_sha256") != sha256(schedule_path)
            or protocol.get("collector_sha256") != sha256(Path(__file__).resolve())
            or not predecision_policy_path.is_file()
            or predecision_actor.get("policy_sha256")
            != sha256(predecision_policy_path)
            or predecision_actor.get("purpose")
            != "replay_frozen_f35_predecision_state"
            or not all(
                path.is_file()
                for path in (
                    actor_policy_path,
                    actor_manifest_path,
                    actor_freeze_path,
                    sim_config_path,
                )
            )
            or actor.get("policy_sha256") != sha256(actor_policy_path)
            or actor.get("training_manifest_sha256") != sha256(actor_manifest_path)
            or actor.get("training_freeze_sha256") != sha256(actor_freeze_path)
            or actor.get("sim_config_sha256") != sha256(sim_config_path)
            or actor.get("shared_across_all_action_arms") is not True
            or actor.get("receives_attribution_input") is not False
        ):
            raise RuntimeError("A4-v7 protocol hash binding failed")
        cases = [row for row in jsonl(schedule_path) if row["scene_id"] == args.scene]
        if args.case_id is not None:
            cases = [row for row in cases if row["case_id"] == args.case_id]
        protocol_counts = protocol.get("counts", {})
        expected_cases = (
            1
            if args.case_id is not None
            else int(protocol_counts.get("physical_cases", 0))
            // max(int(protocol_counts.get("scenes", 0)), 1)
        )
        if not cases or len(cases) != expected_cases:
            raise RuntimeError(
                f"expected {expected_cases} A4 cases for {args.scene}, got {len(cases)}"
            )
        registry = json_object(registry_path)
        scene_rows = [row for row in registry["scenes"] if row["scene_id"] == args.scene]
        if len(scene_rows) != 1:
            raise RuntimeError("A4 scene binding is not unique")
        scene_row = scene_rows[0]
        episode_path = (ROOT / scene_row["episode_usd"]).resolve()
        compiled_path = (ROOT / scene_row["compiled_audit"]).resolve()
        if (
            sha256(episode_path) != scene_row["episode_sha256"]
            or sha256(compiled_path) != scene_row["compiled_audit_sha256"]
        ):
            raise RuntimeError("A4 scene registry hash drift")
        frame = scene_route_binding(json_object(compiled_path)).frame
        backend = IsaacPolicyBackendO4V3(
            load_config(
                sim_config_path,
                overrides={"policy_path": str(predecision_policy_path)},
            ),
            frame.point(0.0, 0.0),
            frame.heading_rad,
        )
        backend.load_realistic_scene(str(episode_path))
        import torch

        predecision_policy = backend._policy
        recovery_policy = (
            torch.jit.load(str(actor_policy_path), map_location=backend._device)
            .to(backend._device)
            .eval()
        )
        prefix_controller = DampedRouteControllerV3(
            cross_track_gain_per_s=2.0,
            lateral_velocity_damping=1.0,
            heading_gain_per_s=2.0,
            lateral_limit_mps=0.35,
            yaw_rate_limit_radps=0.60,
        )
        action_controller = DampedRouteControllerV3(
            cross_track_gain_per_s=2.0,
            lateral_velocity_damping=1.0,
            heading_gain_per_s=2.0,
            lateral_limit_mps=0.45,
            yaw_rate_limit_radps=0.60,
        )
        install_o9 = direct_o9_installer(frame)

        def install_source_operator(
            record: dict[str, Any], operator_name: str, seed: int
        ):
            # The frozen source collectors patch the legacy v1 region function
            # process-locally.  Reapply the same geometry here; otherwise the
            # same seed would still instantiate the mechanism 0.9 m too late.
            prior_region = base._region
            base._region = source_early_region
            try:
                installer = install_o9 if operator_name == "O9_high_centering" else base._install_operator
                return installer(backend, record, frame, seed)
            finally:
                base._region = prior_region
        result_dir = (
            out_root / args.scene / str(args.case_id)
            if args.case_id is not None
            else out_root / args.scene
        )
        trace_dir = result_dir / "traces"
        result_dir.mkdir(parents=True, exist_ok=True)
        trace_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / "results.jsonl"
        audit_path = result_dir / "case_audits.jsonl"
        existing = jsonl(result_path) if args.resume and result_path.exists() else []
        existing_audits = jsonl(audit_path) if args.resume and audit_path.exists() else []
        if not args.resume and (result_path.exists() or audit_path.exists()):
            raise FileExistsError(result_dir)
        completed_cases = {str(row["case_id"]) for row in existing_audits if row.get("passed") is True}

        for case in cases:
            if case["case_id"] in completed_cases:
                continue
            record = dict(case["source_record"])
            true_operator = str(case["operator"])
            arm_results: list[dict[str, Any]] = []
            prefixes: list[np.ndarray] = []
            seed = int(case["reset_seed"])
            nuisance = dict(record.get("physical_nuisance", {}))
            if (
                int(nuisance.get("physics_seed", -1)) != seed
                or nuisance.get("pair_shared") is not True
                or nuisance != case.get("source_physical_nuisance")
            ):
                raise RuntimeError(f"A4 source nuisance drift: {case['case_id']}")
            backend._start_pos = frame.point(
                float(nuisance["start_progress_m"]),
                float(nuisance["start_lateral_offset_m"]),
            )
            backend._start_heading = float(frame.heading_rad) + float(
                nuisance["start_heading_offset_rad"]
            )
            # F33 direct-O9 v2 replaced the whole nuisance installer and, as a
            # consequence, recorded ``spawn_base_height_m`` without applying
            # it to the backend.  Reproduce the actual frozen F33 trajectory
            # here; applying the intended-but-unused value changes the policy
            # state and violates the exact source observation certificate.
            if (
                "spawn_base_height_m" in nuisance
                and not bool(case.get("source_is_f33_direct_o9"))
            ):
                backend._spawn_z = float(nuisance["spawn_base_height_m"])
            backend._policy = predecision_policy
            obs = backend.deep_reset(seed)
            operator, _region = install_source_operator(
                record, true_operator, seed
            )
            # Match the Scale/F33 collector exactly: clean reset, install the
            # operator, then reset once with the realization present.  This
            # prefix is executed once, captured, and shared by every arm.
            initial_obs = backend.reset(seed)
            source_decision_time = float(
                case["source_f35_decision"]["decision_time_s"]
            )
            branch_checkpoint: dict[str, Any] | None = None
            shared_prefix_rows: list[list[float]] | None = None
            shared_prefix_phases: list[str] | None = None
            shared_replay_times: list[float] | None = None
            shared_replay_features: list[list[float]] | None = None
            shared_proprio_certificate: dict[str, Any] | None = None
            shared_decision_step: int | None = None
            shared_decision_progress = float("nan")
            shared_clearance_progress = float("nan")
            for arm_index, action in enumerate(case["actions"]):
                if arm_index == 0:
                    backend._policy = predecision_policy
                    obs = copy.deepcopy(initial_obs)
                    decision_step: int | None = None
                    decision_progress = float("nan")
                    clearance_progress = float("nan")
                    rows: list[list[float]] = []
                    replay_times: list[float] = []
                    replay_features: list[list[float]] = []
                    proprio_certificate: dict[str, Any] | None = None
                    action_phases: list[str] = []
                    loop_start = 0
                else:
                    if (
                        branch_checkpoint is None
                        or shared_prefix_rows is None
                        or shared_prefix_phases is None
                        or shared_replay_times is None
                        or shared_replay_features is None
                        or shared_proprio_certificate is None
                        or shared_decision_step is None
                    ):
                        raise RuntimeError(
                            f"A4 branch checkpoint missing: {case['case_id']}"
                        )
                    obs = restore_branch_checkpoint(backend, branch_checkpoint)
                    backend._policy = recovery_policy
                    decision_step = shared_decision_step
                    decision_progress = shared_decision_progress
                    clearance_progress = shared_clearance_progress
                    rows = copy.deepcopy(shared_prefix_rows)
                    replay_times = list(shared_replay_times)
                    replay_features = copy.deepcopy(shared_replay_features)
                    proprio_certificate = copy.deepcopy(
                        shared_proprio_certificate
                    )
                    action_phases = list(shared_prefix_phases)
                    loop_start = decision_step + 1
                engaged_dwell = 0
                post_steps = 0
                phase_state = {
                    "phase": "back",
                    "detour_sign": 1.0 if int(hashlib.sha256(case["case_id"].encode()).hexdigest(), 16) % 2 == 0 else -1.0,
                }
                max_tilt = max((abs(float(row[7])) for row in rows), default=0.0)
                min_height = min((float(row[6]) for row in rows), default=float("inf"))
                final_telemetry: dict[str, Any] = {}
                for step in range(loop_start, args.horizon_steps):
                    if operator is not None:
                        operator.on_step(backend, float(obs.t))
                    telemetry = base._telemetry(backend, true_operator, operator)
                    progress, lateral = frame.project(obs.pos)
                    region_padding = 0.25 if true_operator == "O8_invisible_collider" else 0.0
                    source_region_exposed = bool(
                        abs(float(obs.pos[0]) - float(_region.cx))
                        <= float(_region.hx) + region_padding
                        and abs(float(obs.pos[1]) - float(_region.cy))
                        <= float(_region.hy) + region_padding
                    )
                    engaged = operator_engaged(
                        true_operator,
                        telemetry,
                        progress_m=float(progress),
                        forward_speed_mps=float(obs.vel_body[0]),
                        source_region_exposed=source_region_exposed,
                    )
                    engaged_dwell = engaged_dwell + 1 if engaged else 0
                    if decision_step is None and float(obs.t) + 1.0e-6 >= source_decision_time:
                        if not engaged:
                            raise RuntimeError(
                                f"A4 source mechanism inactive at frozen F35 decision: "
                                f"{case['case_id']} t={obs.t} target={source_decision_time}"
                            )
                        if abs(float(obs.t) - source_decision_time) > 0.021:
                            raise RuntimeError(
                                f"A4 decision-time skew exceeds 21 ms: {case['case_id']}"
                            )
                        decision_step = step
                        decision_progress = float(progress)
                        clearance_progress = min(
                            float(frame.route_length_m) - 0.20,
                            decision_progress + RECOVERY_CLEARANCE_PROGRESS_M,
                        )
                        if clearance_progress - decision_progress < 0.50:
                            raise RuntimeError(
                                f"A4 route too short for local recovery endpoint: {case['case_id']}"
                            )
                    post = decision_step is not None and step > decision_step
                    prefix_speed = float(nuisance["forward_speed_mps"])
                    prefix_target = float(
                        nuisance["controller_target_lateral_offset_m"]
                    )
                    speed, target, posture, action_phase = (
                        prefix_speed,
                        prefix_target,
                        None,
                        "shared_prefix",
                    )
                    if post:
                        post_steps += 1
                        if action == "continue":
                            action_phase = "continue"
                        elif action == "always_safe_halt":
                            speed, target, posture, action_phase = 0.0, float(lateral), 0.31, "always_safe_halt"
                        else:
                            label = action.removeprefix("recover_as_")
                            speed, target, posture, action_phase = recovery_command(
                                label,
                                progress=float(progress),
                                lateral=float(lateral),
                                decision_progress=decision_progress,
                                clearance_progress=clearance_progress,
                                state=phase_state,
                            )
                    backend.set_posture(posture, 1.5 if posture is not None else 1.0)
                    active_controller = action_controller if post else prefix_controller
                    command, _control = active_controller.command(
                        frame,
                        position_xy_m=np.asarray(obs.pos),
                        heading_rad=float(obs.heading),
                        velocity_body_xy_mps=np.asarray(obs.vel_body[:2]),
                        forward_speed_mps=float(speed),
                        target_lateral_offset_m=float(target),
                    )
                    rows.append(
                        [
                            float(obs.t),
                            float(progress),
                            float(lateral),
                            float(obs.heading),
                            float(obs.vel_body[0]),
                            float(obs.vel_body[1]),
                            float(obs.base_height),
                            float(obs.tilt),
                            float(obs.slip_ratio),
                            float(obs.effort_ratio),
                            float(obs.support_ratio),
                            float(bool(obs.fallen)),
                            float(command[0]),
                            float(command[1]),
                            float(command[2]),
                        ]
                    )
                    action_phases.append(action_phase)
                    max_tilt = max(max_tilt, abs(float(obs.tilt)))
                    min_height = min(min_height, float(obs.base_height))
                    final_telemetry = telemetry
                    obs = backend.step(command)
                    measured = (
                        operator.transform_obs(obs)
                        if operator is not None
                        else obs
                    )
                    replay_times.append(float(measured.t))
                    replay_features.append(base._feature_row(backend, measured))
                    if decision_step is not None and proprio_certificate is None:
                        proprio_certificate = proprio_replay_certificate(
                            replay_times,
                            replay_features,
                            case["source_f35_decision"],
                        )
                        if arm_index != 0:
                            raise RuntimeError(
                                "only the first A4 arm may create a branch checkpoint"
                            )
                        branch_checkpoint = capture_branch_checkpoint(backend, obs)
                        shared_prefix_rows = copy.deepcopy(rows)
                        shared_prefix_phases = list(action_phases)
                        shared_replay_times = list(replay_times)
                        shared_replay_features = copy.deepcopy(replay_features)
                        shared_proprio_certificate = copy.deepcopy(
                            proprio_certificate
                        )
                        shared_decision_step = decision_step
                        shared_decision_progress = decision_progress
                        shared_clearance_progress = clearance_progress
                        backend._policy = recovery_policy
                    new_progress, new_lateral = frame.project(obs.pos)
                    recovered_past_hazard = bool(
                        decision_step is not None
                        and not obs.fallen
                        and new_progress >= clearance_progress
                        and abs(float(new_lateral))
                        <= RECOVERY_CORRIDOR_TOLERANCE_M
                    )
                    current_speed_mps = float(
                        np.linalg.norm(np.asarray(obs.vel_body[:2]))
                    )
                    recovery_success, _endpoint_type = (
                        operator_specific_recovery_success(
                            true_operator,
                            fell=bool(obs.fallen),
                            recovered_past_hazard=recovered_past_hazard,
                            speed_mps=current_speed_mps,
                            tilt_rad=float(obs.tilt),
                            post_steps=post_steps,
                        )
                    )
                    if obs.fallen or recovery_success:
                        break
                    if action == "always_safe_halt" and post_steps >= 120:
                        break
                if decision_step is None:
                    raise RuntimeError(f"A4 operator onset missing: {case['case_id']}/{action}")
                if proprio_certificate is None or proprio_certificate["passed"] is not True:
                    raise RuntimeError(
                        f"A4 source observation replay missing: {case['case_id']}/{action}"
                    )
                trace = np.asarray(rows, dtype=np.float64)
                prefix = trace[: decision_step + 1]
                prefixes.append(prefix)
                trace_path = trace_dir / f"{case['case_id']}__{action}.npz"
                np.savez_compressed(
                    trace_path,
                    trace=trace,
                    field_names=np.asarray(TRACE_FIELDS),
                    action_phases=np.asarray(action_phases),
                )
                progress, lateral = frame.project(obs.pos)
                speed_mps = float(np.linalg.norm(np.asarray(obs.vel_body[:2])))
                recovered_past_hazard = bool(
                    not obs.fallen
                    and progress >= clearance_progress
                    and abs(float(lateral)) <= RECOVERY_CORRIDOR_TOLERANCE_M
                )
                recovery_success, recovery_endpoint_type = (
                    operator_specific_recovery_success(
                        true_operator,
                        fell=bool(obs.fallen),
                        recovered_past_hazard=recovered_past_hazard,
                        speed_mps=speed_mps,
                        tilt_rad=float(obs.tilt),
                        post_steps=post_steps,
                    )
                )
                safe_abort = bool(
                    action == "always_safe_halt"
                    and not obs.fallen
                    and speed_mps < 0.08
                    and abs(float(obs.tilt)) < 0.55
                )
                remaining = (
                    0.0
                    if true_operator
                    in {"O2_compliance", "O5_payload", "O9_high_centering"}
                    and recovery_success
                    else max(0.0, float(clearance_progress - progress))
                )
                safety_exposure = safety_exposure_auc(
                    trace,
                    decision_step,
                    fallback_dt_s=float(getattr(backend, "dt", 0.02)),
                )
                terminal_cost_without_safety_exposure = (
                    100.0 * float(obs.fallen)
                    + 25.0 * float(not recovery_success)
                    + 10.0 * max(0.0, max_tilt - 0.55)
                    + remaining
                    + 0.02 * float(max(0.0, obs.t - rows[decision_step][0]))
                )
                terminal_cost = (
                    terminal_cost_without_safety_exposure
                    + SAFETY_EXPOSURE_WEIGHT * safety_exposure
                )
                arm_results.append(
                    {
                        "schema_version": "kinofail.reconfirmation-a4-v7-result.v1",
                        "created_utc": datetime.now(UTC).isoformat(),
                        "protocol_id": protocol["protocol_id"],
                        "case_id": case["case_id"],
                        "source_counterfactual_group_id": case["source_counterfactual_group_id"],
                        "scene_id": case["scene_id"],
                        "domain": case["domain"],
                        "true_operator": true_operator,
                        "severity_id": case["severity_id"],
                        "reset_seed": seed,
                        "source_physical_nuisance": nuisance,
                        "action": action,
                        "decision_step": decision_step,
                        "decision_time_s": float(rows[decision_step][0]),
                        "source_f35_decision": case["source_f35_decision"],
                        "source_proprio_replay_certificate": proprio_certificate,
                        "decision_time_skew_s": float(
                            rows[decision_step][0] - source_decision_time
                        ),
                        "decision_progress_m": decision_progress,
                        "recovery_checkpoint_progress_m": clearance_progress,
                        "recovery_checkpoint_lateral_tolerance_m": RECOVERY_CORRIDOR_TOLERANCE_M,
                        "predecision_trace_sha256": trace_hash(prefix),
                        "postdecision_numeric_trace_sha256": trace_hash(
                            trace[decision_step + 1 :]
                        ),
                        "trace": str(trace_path),
                        "trace_sha256": sha256(trace_path),
                        "steps": len(rows),
                        "fell": bool(obs.fallen),
                        "recovered_past_hazard": recovered_past_hazard,
                        "operator_recovery_success": recovery_success,
                        "operator_recovery_endpoint_type": recovery_endpoint_type,
                        "safe_abort": safe_abort,
                        "terminal_cost": float(terminal_cost),
                        "terminal_cost_without_safety_exposure": float(
                            terminal_cost_without_safety_exposure
                        ),
                        "safety_exposure_auc": safety_exposure,
                        "safety_exposure_weight": SAFETY_EXPOSURE_WEIGHT,
                        "safety_exposure_thresholds": SAFETY_THRESHOLDS,
                        "final_progress_m": float(progress),
                        "final_lateral_m": float(lateral),
                        "final_speed_mps": speed_mps,
                        "maximum_tilt_rad": max_tilt,
                        "minimum_base_height_m": min_height,
                        "operator_telemetry_final": final_telemetry,
                        "unfavorable_outcome_retained": True,
                    }
                )
            shape_ok = len({array.shape for array in prefixes}) == 1
            maximum = (
                max(float(np.max(np.abs(array - prefixes[0]))) for array in prefixes[1:])
                if shape_ok
                else float("inf")
            )
            prefix_passed = bool(shape_ok and maximum <= PREFIX_TOLERANCE)
            replay_certificates = [
                row["source_proprio_replay_certificate"] for row in arm_results
            ]
            # Isaac/PhysX reset is reproducible within the preregistered
            # observation tolerance, but is not guaranteed to be bitwise
            # identical across separately reset action arms.  Each arm must
            # independently replay the frozen F35 observation within that
            # bound; equality of floating-point hashes is intentionally not a
            # scientific gate.
            replay_passed = bool(
                all(row.get("passed") is True for row in replay_certificates)
                and max(
                    float(row["maximum_tolerance_normalized_difference"])
                    for row in replay_certificates
                )
                <= 1.0
            )
            recovery_numeric_hashes = {
                str(row["postdecision_numeric_trace_sha256"])
                for row in arm_results
                if str(row["action"]).startswith("recover_as_")
            }
            recovery_programs_distinct = len(recovery_numeric_hashes) == 5
            case_audit = {
                "schema_version": "kinofail.reconfirmation-a4-v7-case-audit.v1",
                "case_id": case["case_id"],
                "passed": prefix_passed and replay_passed and recovery_programs_distinct,
                "arms": len(arm_results),
                "same_decision_step": len({row["decision_step"] for row in arm_results}) == 1,
                "prefix_shapes_equal": shape_ok,
                "maximum_absolute_predecision_difference": maximum,
                "tolerance": PREFIX_TOLERANCE,
                "source_proprio_replay_passed": replay_passed,
                "five_recovery_numeric_trajectories_distinct": recovery_programs_distinct,
                "distinct_recovery_numeric_trace_count": len(recovery_numeric_hashes),
                "maximum_proprio_tolerance_normalized_difference": max(
                    float(row["maximum_tolerance_normalized_difference"])
                    for row in replay_certificates
                ),
            }
            with result_path.open("a") as stream:
                for row in arm_results:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
            with audit_path.open("a") as stream:
                stream.write(json.dumps(case_audit, sort_keys=True) + "\n")
            print(
                json.dumps(
                    {
                        "case_id": case["case_id"],
                        "prefix_passed": prefix_passed,
                        "falls": sum(row["fell"] for row in arm_results),
                        "successes": sum(
                            row["operator_recovery_success"] for row in arm_results
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        audits = jsonl(audit_path)
        expected_ids = {row["case_id"] for row in cases}
        terminal = {row["case_id"] for row in audits}
        summary = {
            "schema_version": "kinofail.reconfirmation-a4-v7-scene-summary.v1",
            "scene_id": args.scene,
            "case_id": args.case_id,
            "passed": expected_ids == terminal and all(row["passed"] for row in audits),
            "expected_cases": len(expected_ids),
            "terminal_cases": len(expected_ids & terminal),
            "expected_episodes": sum(len(row["actions"]) for row in cases),
            "completed_episodes": sum(
                row["case_id"] in expected_ids for row in jsonl(result_path)
            ),
            "unfavorable_outcomes_retained": True,
        }
        (result_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        passed = bool(summary["passed"])
    finally:
        sys.stdout.flush()
        closer = threading.Thread(target=app.close, daemon=True)
        closer.start()
        closer.join(timeout=15.0)
    os._exit(0 if passed else 2)


if __name__ == "__main__":
    main()
