#!/usr/bin/env python3
"""Collect v35 adaptive-clearance O4 Continue/Backstep consequences.

The adhesive contact law remains command-agnostic and the v32 one-step branch remains frozen.
Backstep adds a scene-identity-agnostic penetration/urgency guard: it begins at the v32 nominal
reverse speed and latches a stronger, still policy-executable reverse command when measured body
state shows either incipient physical failure or excessive forward penetration after the branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_rows_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _event_trace(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "step": int(row["step"]),
            "time_s": float(row["time_s"]),
            "action_phase": row["action_phase"],
            "foot": foot["name"],
            "event": foot["event"],
            "raw_force_n": float(foot["raw_force_n"]),
        }
        for row in rows
        for foot in row["adhesion"]["feet"]
        if foot["event"] != "none"
    ]


def _decision_trigger_reason(
    *,
    elapsed_steps: int,
    urgency_dwell_steps: int,
    minimum_dwell_steps: int,
    maximum_dwell_steps: int,
    required_urgency_dwell_steps: int,
) -> str | None:
    """Return a deterministic trigger without consulting the future action identity."""

    if (
        elapsed_steps >= minimum_dwell_steps
        and urgency_dwell_steps >= required_urgency_dwell_steps
    ):
        return "posture_urgency"
    if elapsed_steps >= maximum_dwell_steps:
        return "maximum_dwell"
    return None


def _next_clearance_dwell(
    current: int,
    *,
    retreat_m: float,
    target_retreat_m: float,
    active_feet: int,
) -> int:
    """Require consecutive retreat-and-release observations before ending Backstep."""

    if retreat_m >= target_retreat_m and active_feet == 0:
        return current + 1
    return 0


def _adaptive_backstep_command(
    *,
    nominal_speed_mps: float,
    emergency_speed_mps: float,
    decision_progress_m: float,
    current_progress_m: float,
    preattachment_base_height_m: float,
    current_base_height_m: float,
    current_tilt_rad: float,
    maximum_forward_penetration_m: float,
    urgency_tilt_rad: float,
    urgency_height_drop_m: float,
    already_escalated: bool,
) -> tuple[float, bool, str | None]:
    """Return a latched physical reverse command without consulting scene identity.

    The guard observes only measurements available before issuing the current command.  Adhesion
    release is never an input or output: peel must still arise from contact unloading and lift.
    """

    forward_penetration = current_progress_m - decision_progress_m
    height_drop = preattachment_base_height_m - current_base_height_m
    tolerance = 1.0e-9
    urgent = (
        current_tilt_rad + tolerance >= urgency_tilt_rad
        and height_drop + tolerance >= urgency_height_drop_m
    )
    penetrated = forward_penetration + tolerance >= maximum_forward_penetration_m
    escalated = already_escalated or urgent or penetrated
    reason = None
    if not already_escalated:
        if urgent:
            reason = "physical_urgency"
        elif penetrated:
            reason = "forward_penetration_guard"
    speed = emergency_speed_mps if escalated else nominal_speed_mps
    return -speed, escalated, reason


def main() -> int:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--episode-usd", type=Path, required=True)
    preliminary.add_argument("--compiled-audit", type=Path, required=True)
    preliminary.add_argument("--out", type=Path, required=True)
    preliminary.add_argument("--material-id", required=True)
    preliminary.add_argument("--camera-profile", default="go2_front_calib_b")
    preliminary.add_argument("--seed", type=int, required=True)
    pre, _ = preliminary.parse_known_args()

    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-usd", type=Path, default=pre.episode_usd)
    parser.add_argument("--compiled-audit", type=Path, default=pre.compiled_audit)
    parser.add_argument("--out", type=Path, default=pre.out)
    parser.add_argument("--material-id", default=pre.material_id)
    parser.add_argument("--camera-profile", default=pre.camera_profile)
    parser.add_argument("--seed", type=int, default=pre.seed)
    parser.add_argument("--horizon-steps", type=int, default=300)
    parser.add_argument("--minimum-decision-dwell-steps", type=int, default=5)
    parser.add_argument("--decision-dwell-steps", type=int, default=25)
    parser.add_argument("--decision-tilt-urgency-rad", type=float, default=0.08)
    parser.add_argument("--decision-height-drop-urgency-m", type=float, default=0.02)
    parser.add_argument("--decision-urgency-dwell-steps", type=int, default=2)
    parser.add_argument("--maximum-backstep-steps", type=int, default=120)
    parser.add_argument("--target-recovery-distance-m", type=float, default=0.20)
    parser.add_argument("--minimum-recovery-distance-m", type=float, default=0.15)
    parser.add_argument("--adhesion-clearance-dwell-steps", type=int, default=5)
    parser.add_argument("--forward-speed-mps", type=float, default=0.32)
    parser.add_argument("--backstep-speed-mps", type=float, default=0.24)
    parser.add_argument("--emergency-backstep-speed-mps", type=float, default=0.48)
    parser.add_argument(
        "--maximum-postdecision-forward-penetration-m", type=float, default=0.025
    )
    parser.add_argument("--max-route-deviation-m", type=float, default=0.30)
    parser.add_argument("--tangential-force-cap-n", type=float, default=18.0)
    parser.add_argument("--normal-force-cap-n", type=float, default=8.0)
    parser.add_argument("--peel-height-m", type=float, default=0.025)
    parser.add_argument("--unload-steps-to-peel", type=int, default=3)
    parser.add_argument("--reattach-cooldown-steps", type=int, default=2)
    parser.add_argument("--max-active-feet", type=int, default=2)
    parser.add_argument("--cross-track-gain-per-s", type=float, default=1.0)
    parser.add_argument("--lateral-velocity-damping", type=float, default=0.0)
    parser.add_argument("--heading-gain-per-s", type=float, default=2.0)
    parser.add_argument("--lateral-limit-mps", type=float, default=0.2)
    parser.add_argument("--yaw-rate-limit-radps", type=float, default=0.6)
    parser.add_argument(
        "--material-lock",
        type=Path,
        default=ROOT / "outputs/assets/terrain_pbr_v1/terrain_assets.lock.json",
    )
    parser.add_argument(
        "--policy-path", type=Path, default=ROOT / "outputs/locomotion/policy.pt"
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite output: {args.out}")
    if (
        args.horizon_steps <= 0
        or args.minimum_decision_dwell_steps < 1
        or args.decision_dwell_steps < args.minimum_decision_dwell_steps
        or args.decision_urgency_dwell_steps < 1
        or args.maximum_backstep_steps < 1
        or args.target_recovery_distance_m < args.minimum_recovery_distance_m
        or args.minimum_recovery_distance_m <= 0.0
        or args.adhesion_clearance_dwell_steps < 1
        or args.backstep_speed_mps <= 0.0
        or args.emergency_backstep_speed_mps < args.backstep_speed_mps
        or args.maximum_postdecision_forward_penetration_m <= 0.0
    ):
        parser.error("invalid horizon/decision/backstep step count")
    args.enable_cameras = True
    app = AppLauncher(args).app

    passed = False
    try:
        import yaml
        from PIL import Image

        from kino_vla.sim.adhesion import IrregularRegion
        from kino_vla.sim.adhesion_v3 import SurfaceAdhesionConfig
        from kino_vla.sim.isaac_o4_v3_backend import IsaacPolicyBackendO4V3
        from kino_vla.sim.realistic_route_controller_v3 import DampedRouteControllerV3
        from kino_vla.sim.realistic_route_protocol_v2 import scene_route_binding
        from kino_vla.sim.terrain_materials import (
            appearance_binding_from_record,
            load_terrain_asset_lock,
        )
        from kino_vla.utils.config import CONFIGS_DIR, load_config
        from scripts.isaac_collect_realistic_route_operator_lane_v6 import _frame_metrics

        output = args.out.resolve()
        output.mkdir(parents=True)
        episode = args.episode_usd.resolve()
        compiled_path = args.compiled_audit.resolve()
        compiled = _json(compiled_path)
        if compiled.get("passed") is not True:
            raise RuntimeError("terrain compiled audit did not pass")
        if compiled.get("files", {}).get(episode.name) != _sha256(episode):
            raise RuntimeError("episode hash differs from terrain compiled audit")
        base_path = Path(compiled["base_compiled_audit"]).resolve()
        if _sha256(base_path) != compiled["base_compiled_audit_sha256"]:
            raise RuntimeError("base compiled audit hash differs")
        base = _json(base_path)
        if base.get("passed") is not True or base.get("operator", {}).get("operator_id") != "O4":
            raise RuntimeError("base scene lacks a valid route-relative O4 region")
        admission_path = episode.parent / "realistic_stack_v16_admission.json"
        admission = _json(admission_path)
        if admission.get("passed") is not True:
            raise RuntimeError("v16 stack admission did not pass")

        benchmark = yaml.safe_load(
            (CONFIGS_DIR / "data/kinofail_realistic.yaml").read_text(encoding="utf-8")
        )
        camera = dict(benchmark["camera_profile_specs"][args.camera_profile])
        mount = np.asarray(camera["mount_xyz_m"], dtype=np.float64)
        policy_path = args.policy_path.resolve()
        sim_cfg = load_config(
            "sim/go2_skeleton.yaml",
            {
                "policy_path": str(policy_path),
                "cam_width": int(camera["width"]),
                "cam_height": int(camera["height"]),
                "front_cam_width": int(camera["width"]),
                "front_cam_height": int(camera["height"]),
                "front_cam_focal_mm": float(camera["focal_length_mm"]),
                "front_cam_aperture_mm": 24.0,
                "front_cam_x_m": float(mount[0]),
                "front_cam_y_m": float(mount[1]),
                "front_cam_z_m": float(mount[2]),
                "front_cam_pitch_down_rad": float(camera["pitch_down_rad"]),
            },
        )
        scene_binding = scene_route_binding(compiled)
        frame = scene_binding.frame
        controller = DampedRouteControllerV3(
            cross_track_gain_per_s=args.cross_track_gain_per_s,
            lateral_velocity_damping=args.lateral_velocity_damping,
            heading_gain_per_s=args.heading_gain_per_s,
            lateral_limit_mps=args.lateral_limit_mps,
            yaw_rate_limit_radps=args.yaw_rate_limit_radps,
        )
        backend = IsaacPolicyBackendO4V3(
            sim_cfg, frame.point(0.0, 0.0), frame.heading_rad, front_cam=True
        )
        scene_prim = backend.load_realistic_scene(str(episode))

        lock_path = args.material_lock.resolve()
        lock = load_terrain_asset_lock(lock_path)
        asset_root = Path(lock["asset_root"])
        if not asset_root.is_absolute():
            asset_root = ROOT / asset_root
        appearance = appearance_binding_from_record(
            {
                "material_family": args.material_id,
                "appearance_id": f"v27-o4-action-{args.material_id}",
                "uv_scale": 1.0,
                "uv_rotation_deg": 0.0,
                "surface_state": "damp",
            },
            lock=lock,
            asset_root=asset_root,
        )
        backend.set_terrain_appearance(appearance)

        operator_record = base["operator"]
        region = IrregularRegion(
            tuple(tuple(float(v) for v in point) for point in operator_record["vertices_xy_m"])
        )
        adhesion_cfg = SurfaceAdhesionConfig(
            region=region,
            surface_z_m=float(operator_record["surface_z_m"]),
            attach_contact_force_n=5.0,
            detach_contact_force_n=2.0,
            attach_height_tolerance_m=0.04,
            tangential_stiffness_n_per_m=150.0,
            tangential_damping_ns_per_m=3.0,
            normal_stiffness_n_per_m=120.0,
            normal_damping_ns_per_m=2.0,
            tangential_force_cap_n=args.tangential_force_cap_n,
            normal_force_cap_n=args.normal_force_cap_n,
            peel_height_m=args.peel_height_m,
            unload_steps_to_peel=args.unload_steps_to_peel,
            reattach_cooldown_steps=args.reattach_cooldown_steps,
            max_active_feet=args.max_active_feet,
            progress_axis_xy=(float(frame.direction[0]), float(frame.direction[1])),
        )

        reference_predecision_step: int | None = None
        summaries: dict[str, dict[str, Any]] = {}
        rows_by_take: dict[str, list[dict[str, Any]]] = {}
        initial_rgbs: dict[str, np.ndarray] = {}
        decision_rgbs: dict[str, np.ndarray] = {}

        for take in ("continue", "backstep"):
            take_dir = output / take
            take_dir.mkdir(parents=True)
            obs = backend.deep_reset(args.seed)
            backend.add_foot_adhesion(adhesion_cfg)
            rows: list[dict[str, Any]] = []
            captures: list[dict[str, Any]] = []
            first_attachment_step: int | None = None
            preattachment_base_height_m: float | None = None
            predecision_step: int | None = None
            decision_trigger_reason: str | None = None
            urgency_dwell = 0
            recovery_complete = False
            recovery_complete_step: int | None = None
            clearance_dwell = 0
            backstep_escalated = False
            backstep_escalation_step: int | None = None
            backstep_escalation_reason: str | None = None

            def capture(label: str, step: int) -> None:
                sensor = backend.capture_front_camera()
                if sensor is None:
                    raise RuntimeError("Go2-front RTX camera returned no frame")
                rgb = np.asarray(sensor["rgb"])[..., :3].astype(np.uint8)
                path = take_dir / f"{label}.png"
                Image.fromarray(rgb).save(path, compress_level=3)
                if label == "initial":
                    initial_rgbs[take] = rgb.copy()
                elif label == "decision":
                    decision_rgbs[take] = rgb.copy()
                progress, lateral = frame.project(obs.pos)
                captures.append(
                    {
                        "label": label,
                        "step": step,
                        "time_s": float(obs.t),
                        "progress_m": progress,
                        "lateral_offset_m": lateral,
                        "fallen": bool(obs.fallen),
                        "image": path.name,
                        "image_sha256": _sha256(path),
                        "metrics": _frame_metrics(rgb),
                    }
                )

            capture("initial", -1)
            for step in range(args.horizon_steps):
                if (
                    take == "backstep"
                    and predecision_step is not None
                    and predecision_step < step
                    and not recovery_complete
                    and step <= predecision_step + args.maximum_backstep_steps
                ):
                    if preattachment_base_height_m is None:
                        raise RuntimeError("missing preattachment height for adaptive Backstep")
                    decision_progress_for_command = float(
                        rows[predecision_step]["progress_m"]
                    )
                    current_progress_for_command, _ = frame.project(obs.pos)
                    speed, escalated, escalation_reason = _adaptive_backstep_command(
                        nominal_speed_mps=args.backstep_speed_mps,
                        emergency_speed_mps=args.emergency_backstep_speed_mps,
                        decision_progress_m=decision_progress_for_command,
                        current_progress_m=float(current_progress_for_command),
                        preattachment_base_height_m=preattachment_base_height_m,
                        current_base_height_m=float(obs.base_height),
                        current_tilt_rad=float(obs.tilt),
                        maximum_forward_penetration_m=(
                            args.maximum_postdecision_forward_penetration_m
                        ),
                        urgency_tilt_rad=args.decision_tilt_urgency_rad,
                        urgency_height_drop_m=args.decision_height_drop_urgency_m,
                        already_escalated=backstep_escalated,
                    )
                    if escalated and not backstep_escalated:
                        backstep_escalation_step = step
                        backstep_escalation_reason = escalation_reason
                    backstep_escalated = escalated
                    action_phase = (
                        "backstep_emergency_clearance"
                        if backstep_escalated
                        else "backstep_nominal_clearance"
                    )
                elif (
                    take == "backstep"
                    and predecision_step is not None
                    and step > predecision_step
                ):
                    speed = 0.0
                    action_phase = "post_backstep_hold"
                else:
                    speed = args.forward_speed_mps
                    action_phase = "matched_forward"
                command, control = controller.command(
                    frame,
                    position_xy_m=obs.pos,
                    heading_rad=float(obs.heading),
                    velocity_body_xy_mps=np.asarray(obs.vel_body[:2], dtype=np.float64),
                    forward_speed_mps=speed,
                    target_lateral_offset_m=0.0,
                )
                obs = backend.step(command)
                progress, lateral = frame.project(obs.pos)
                adhesion = backend.adhesion_telemetry()
                row = {
                    "step": step,
                    "time_s": float(obs.t),
                    "action_phase": action_phase,
                    "pos_xy_m": obs.pos.tolist(),
                    "progress_m": float(progress),
                    "route_lateral_offset_m": float(lateral),
                    "heading_rad": float(obs.heading),
                    "vel_body_mps": obs.vel_body.tolist(),
                    "base_height_m": float(obs.base_height),
                    "tilt_rad": float(obs.tilt),
                    "slip_ratio": float(obs.slip_ratio),
                    "effort_ratio": float(obs.effort_ratio),
                    "support_ratio": float(obs.support_ratio),
                    "fallen": bool(obs.fallen),
                    "command_body": command.tolist(),
                    "route_controller": control,
                    "adhesion": adhesion,
                }
                rows.append(row)
                if first_attachment_step is None and any(
                    foot["event"] == "attached" for foot in adhesion["feet"]
                ):
                    first_attachment_step = step
                    height_rows = rows[max(0, len(rows) - 21) : -1]
                    preattachment_base_height_m = max(
                        (float(item["base_height_m"]) for item in height_rows),
                        default=float(obs.base_height),
                    )
                if first_attachment_step is not None and predecision_step is None:
                    if preattachment_base_height_m is None:
                        raise RuntimeError("missing preattachment height reference")
                    elapsed = step - first_attachment_step
                    urgent_now = (
                        float(obs.tilt) >= args.decision_tilt_urgency_rad
                        and preattachment_base_height_m - float(obs.base_height)
                        >= args.decision_height_drop_urgency_m
                    )
                    urgency_dwell = urgency_dwell + 1 if urgent_now else 0
                    trigger_reason = _decision_trigger_reason(
                        elapsed_steps=elapsed,
                        urgency_dwell_steps=urgency_dwell,
                        minimum_dwell_steps=args.minimum_decision_dwell_steps,
                        maximum_dwell_steps=args.decision_dwell_steps,
                        required_urgency_dwell_steps=args.decision_urgency_dwell_steps,
                    )
                    if trigger_reason is not None:
                        discovered = step
                        decision_trigger_reason = trigger_reason
                        predecision_step = discovered
                        if take == "continue":
                            reference_predecision_step = discovered
                        elif discovered != reference_predecision_step:
                            raise RuntimeError(
                                "backstep decision rule does not reproduce the reference step"
                            )
                        capture("decision", step)
                if take == "backstep" and predecision_step is not None and step > predecision_step:
                    decision_progress = float(rows[predecision_step]["progress_m"])
                    retreat = decision_progress - float(progress)
                    clearance_dwell = _next_clearance_dwell(
                        clearance_dwell,
                        retreat_m=retreat,
                        target_retreat_m=args.target_recovery_distance_m,
                        active_feet=int(adhesion.get("active_feet", -1)),
                    )
                    if (
                        not recovery_complete
                        and clearance_dwell >= args.adhesion_clearance_dwell_steps
                    ):
                        recovery_complete = True
                        recovery_complete_step = step
                if obs.fallen:
                    capture("outcome", step)
                    break
            if not any(row["label"] == "outcome" for row in captures):
                capture("outcome", int(rows[-1]["step"]))
            if first_attachment_step is None or predecision_step is None:
                raise RuntimeError(f"{take} never established an O4 decision state")

            telemetry_path = take_dir / "telemetry.jsonl"
            with telemetry_path.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
            events = _event_trace(rows)
            after_decision = [row for row in rows if int(row["step"]) > predecision_step]
            decision_progress = float(rows[predecision_step]["progress_m"])
            summary = {
                "take": take,
                "seed": args.seed,
                "steps": len(rows),
                "fixed_horizon_steps": args.horizon_steps,
                "fell": any(row["fallen"] for row in rows),
                "first_fall_step": next(
                    (int(row["step"]) for row in rows if row["fallen"]), None
                ),
                "first_attachment_step": first_attachment_step,
                "predecision_step": predecision_step,
                "decision_trigger_reason": decision_trigger_reason,
                "preattachment_base_height_m": preattachment_base_height_m,
                "action_branch_step": predecision_step + 1,
                "recovery_complete": recovery_complete,
                "recovery_complete_step": recovery_complete_step,
                "recovery_clearance_dwell_steps": clearance_dwell,
                "backstep_escalated": backstep_escalated,
                "backstep_escalation_step": backstep_escalation_step,
                "backstep_escalation_reason": backstep_escalation_reason,
                "decision_progress_m": decision_progress,
                "final_progress_m": float(rows[-1]["progress_m"]),
                "minimum_postdecision_progress_m": min(
                    (float(row["progress_m"]) for row in after_decision),
                    default=decision_progress,
                ),
                "backward_recovery_distance_m": max(
                    0.0,
                    decision_progress
                    - min(
                        (float(row["progress_m"]) for row in after_decision),
                        default=decision_progress,
                    ),
                ),
                "maximum_absolute_route_lateral_offset_m": max(
                    abs(float(row["route_lateral_offset_m"])) for row in rows
                ),
                "max_tilt_rad": max(float(row["tilt_rad"]) for row in rows),
                "min_base_height_m": min(float(row["base_height_m"]) for row in rows),
                "attachment_events": sum(e["event"] == "attached" for e in events),
                "peel_events": sum(e["event"] == "peeled" for e in events),
                "postdecision_peel_events": sum(
                    e["event"] == "peeled" and e["step"] > predecision_step for e in events
                ),
                "event_trace": events,
                "final_adhesion_telemetry": rows[-1]["adhesion"],
                "predecision_rows_sha256": _canonical_rows_hash(
                    rows[: predecision_step + 1]
                ),
                "telemetry": str(telemetry_path),
                "telemetry_sha256": _sha256(telemetry_path),
                "captures": captures,
            }
            _write_json(take_dir / "summary.json", summary)
            summaries[take] = summary
            rows_by_take[take] = rows

        if reference_predecision_step is None:
            raise RuntimeError("reference decision step was not established")
        continue_summary = summaries["continue"]
        backstep_summary = summaries["backstep"]
        continue_pre = rows_by_take["continue"][: reference_predecision_step + 1]
        backstep_pre = rows_by_take["backstep"][: reference_predecision_step + 1]
        predecision_hash_equal = (
            continue_summary["predecision_rows_sha256"]
            == backstep_summary["predecision_rows_sha256"]
        )
        initial_rgb_l1 = float(
            np.abs(
                initial_rgbs["continue"].astype(np.float32)
                - initial_rgbs["backstep"].astype(np.float32)
            ).mean()
            / 255.0
        )
        decision_rgb_l1 = float(
            np.abs(
                decision_rgbs["continue"].astype(np.float32)
                - decision_rgbs["backstep"].astype(np.float32)
            ).mean()
            / 255.0
        )
        checks = {
            "compiled_scene_hash_verified": True,
            "v16_stack_admission_verified": True,
            "command_agnostic_operator_readback": all(
                row["adhesion"].get("mode")
                == "command_agnostic_contact_surface_adhesion_v3"
                and row["adhesion"].get("command_conditioned_release") is False
                for take in ("continue", "backstep")
                for row in rows_by_take[take]
            ),
            "same_seed_episode_operator": continue_summary["seed"]
            == backstep_summary["seed"]
            == args.seed,
            "same_first_attachment_step": continue_summary["first_attachment_step"]
            == backstep_summary["first_attachment_step"],
            "same_decision_step": continue_summary["predecision_step"]
            == backstep_summary["predecision_step"],
            "predecision_telemetry_byte_identical": predecision_hash_equal,
            "predecision_row_count_matches": len(continue_pre) == len(backstep_pre),
            "initial_rgb_matched": initial_rgb_l1 <= 0.03,
            "decision_rgb_matched": decision_rgb_l1 <= 0.03,
            "continue_fails_after_decision": continue_summary["fell"] is True
            and int(continue_summary["first_fall_step"])
            > int(continue_summary["predecision_step"]),
            "backstep_completes_without_fall": backstep_summary["fell"] is False
            and backstep_summary["steps"] == args.horizon_steps,
            "backstep_triggers_postdecision_peel": backstep_summary[
                "postdecision_peel_events"
            ]
            >= 1,
            "backstep_closed_loop_exit_complete": backstep_summary[
                "recovery_complete"
            ]
            is True,
            "backstep_retreats_from_hazard": backstep_summary[
                "backward_recovery_distance_m"
            ]
            >= args.minimum_recovery_distance_m,
            "both_lanes_within_route_budget": continue_summary[
                "maximum_absolute_route_lateral_offset_m"
            ]
            <= args.max_route_deviation_m
            and backstep_summary["maximum_absolute_route_lateral_offset_m"]
            <= args.max_route_deviation_m,
            "three_frames_per_lane": all(
                {row["label"] for row in summaries[take]["captures"]}
                == {"initial", "decision", "outcome"}
                for take in ("continue", "backstep")
            ),
            "all_frames_non_degenerate": all(
                frame_row["metrics"]["std_luminance"] >= 0.025
                and frame_row["metrics"]["black_fraction"] < 0.95
                and frame_row["metrics"]["quantized_color_count_5bit"] >= 32
                for take in ("continue", "backstep")
                for frame_row in summaries[take]["captures"]
            ),
        }
        source_manifest = _json(Path(base["source_manifest"]))
        manifest = {
            "schema_version": "kinofail.o4-action-consequence.v35",
            "created_utc": datetime.now(UTC).isoformat(),
            "passed": all(checks.values()),
            "development_only": True,
            "counts_as_a0_a7_evidence": False,
            "realistic_a0_a7_readiness": "0/8",
            "scene_id": compiled["scene_id"],
            "room_family": source_manifest["generation"]["room_type"],
            "seed": args.seed,
            "material_id": args.material_id,
            "camera_profile": args.camera_profile,
            "episode_usd": str(episode),
            "episode_usd_sha256": _sha256(episode),
            "compiled_audit": str(compiled_path),
            "compiled_audit_sha256": _sha256(compiled_path),
            "base_compiled_audit": str(base_path),
            "base_compiled_audit_sha256": _sha256(base_path),
            "stack_admission": str(admission_path),
            "stack_admission_sha256": _sha256(admission_path),
            "operator": {
                "id": "O4",
                "subtype": "adhesive_contact",
                "mode": "command_agnostic_contact_surface_adhesion_v3",
                "parameters": {
                    key: value
                    for key, value in adhesion_cfg.__dict__.items()
                    if key != "region"
                },
                "region_xy_m": [list(point) for point in region.vertices_xy],
            },
            "decision_contract": {
                "trigger": "first named-foot attachment, then posture urgency or maximum dwell",
                "minimum_decision_dwell_steps": args.minimum_decision_dwell_steps,
                "maximum_decision_dwell_steps": args.decision_dwell_steps,
                "decision_tilt_urgency_rad": args.decision_tilt_urgency_rad,
                "decision_height_drop_urgency_m": args.decision_height_drop_urgency_m,
                "decision_urgency_dwell_steps": args.decision_urgency_dwell_steps,
                "predecision_step": reference_predecision_step,
                "action_branch_step": reference_predecision_step + 1,
                "continue_action": f"forward {args.forward_speed_mps} m/s",
                "backstep_action": (
                    f"reverse {args.backstep_speed_mps} m/s, latching "
                    f"{args.emergency_backstep_speed_mps} m/s when physical urgency or "
                    f"{args.maximum_postdecision_forward_penetration_m} m forward penetration "
                    f"is observed, until retreat reaches "
                    f"{args.target_recovery_distance_m} m and active adhesion remains clear for "
                    f"{args.adhesion_clearance_dwell_steps} steps, capped at "
                    f"{args.maximum_backstep_steps} steps, then hold"
                ),
                "predecision_hash_equal": predecision_hash_equal,
                "initial_rgb_mean_absolute_difference": initial_rgb_l1,
                "decision_rgb_mean_absolute_difference": decision_rgb_l1,
                "adaptive_backstep_guard": {
                    "emergency_backstep_speed_mps": args.emergency_backstep_speed_mps,
                    "maximum_postdecision_forward_penetration_m": (
                        args.maximum_postdecision_forward_penetration_m
                    ),
                    "physical_urgency_uses_frozen_decision_thresholds": True,
                    "latches_after_first_trigger": True,
                    "scene_identity_used": False,
                    "command_conditioned_release_used": False,
                },
            },
            "route_controller": {
                "controller_contract": controller.contract(),
                "maximum_route_deviation_m": args.max_route_deviation_m,
            },
            "takes": summaries,
            "checks": checks,
            "scene_prim": scene_prim,
            "provenance": {
                "command": " ".join(sys.argv),
                "python": sys.executable,
                "collector_sha256": _sha256(Path(__file__).resolve()),
                "surface_interface_sha256": _sha256(ROOT / "kino_vla/sim/adhesion_v3.py"),
                "surface_model_sha256": _sha256(ROOT / "kino_vla/sim/adhesion_v2.py"),
                "backend_adapter_sha256": _sha256(
                    ROOT / "kino_vla/sim/isaac_o4_v3_backend.py"
                ),
                "parent_backend_adapter_sha256": _sha256(
                    ROOT / "kino_vla/sim/isaac_o4_v2_backend.py"
                ),
                "base_backend_sha256": _sha256(
                    ROOT / "kino_vla/sim/isaac_policy_backend.py"
                ),
                "policy_sha256": _sha256(policy_path),
                "material_lock_sha256": _sha256(lock_path),
            },
            "remaining_gates": [
                "independent postrun audit",
                "fresh held-out scene confirmation",
                "registry-bound O4 corpus collection",
                "new realistic A0-A7 experiments",
            ],
        }
        manifest_path = output / "pair_manifest.json"
        _write_json(manifest_path, manifest)
        print(
            json.dumps(
                {"manifest": str(manifest_path), "passed": manifest["passed"], "checks": checks},
                indent=2,
            ),
            flush=True,
        )
        passed = bool(manifest["passed"])
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        closer = threading.Thread(target=app.close, daemon=True)
        closer.start()
        closer.join(timeout=10.0)
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
