#!/usr/bin/env python3
"""Collect a paired O4 adhesion calibration on an admitted v16 realistic route."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def _phase(t: float, args: argparse.Namespace) -> tuple[float, str]:
    if t < args.forward_s:
        return args.forward_speed_mps, "forward"
    if t < args.forward_s + args.peel_pulse_s:
        return -args.reverse_speed_mps, "peel_release"
    if t < args.forward_s + args.peel_pulse_s + args.recovery_s:
        return 0.0, "post_peel_recovery"
    if t < args.forward_s + args.peel_pulse_s + args.recovery_s + args.reverse_s:
        return -args.reverse_speed_mps, "reverse_retreat"
    return 0.0, "stop"


def main() -> int:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--episode-usd", type=Path, required=True)
    preliminary.add_argument("--compiled-audit", type=Path, required=True)
    preliminary.add_argument("--out", type=Path, required=True)
    preliminary.add_argument("--camera-profile", default="go2_front_calib_b")
    preliminary.add_argument("--material-id", required=True)
    preliminary.add_argument("--seed", type=int, required=True)
    preliminary.add_argument("--forward-s", type=float, default=5.0)
    preliminary.add_argument("--peel-pulse-s", type=float, default=0.10)
    preliminary.add_argument("--recovery-s", type=float, default=1.0)
    preliminary.add_argument("--reverse-s", type=float, default=2.0)
    preliminary.add_argument("--stop-s", type=float, default=0.5)
    preliminary.add_argument("--forward-speed-mps", type=float, default=0.32)
    preliminary.add_argument("--reverse-speed-mps", type=float, default=0.24)
    preliminary.add_argument("--rgb-fps", type=float, default=5.0)
    preliminary.add_argument("--max-route-deviation-m", type=float, default=0.30)
    pre, _ = preliminary.parse_known_args()

    parser = argparse.ArgumentParser()
    for action in preliminary._actions:
        if action.dest == "help":
            continue
        kwargs: dict[str, Any] = {"default": getattr(pre, action.dest)}
        if action.type is not None:
            kwargs["type"] = action.type
        if action.required:
            kwargs["required"] = False
        parser.add_argument(*action.option_strings, **kwargs)
    parser.add_argument(
        "--material-lock",
        type=Path,
        default=ROOT / "outputs/assets/terrain_pbr_v1/terrain_assets.lock.json",
    )
    parser.add_argument(
        "--policy-path", type=Path, default=ROOT / "outputs/locomotion/policy.pt"
    )
    parser.add_argument("--cross-track-gain-per-s", type=float, default=1.0)
    parser.add_argument("--lateral-velocity-damping", type=float, default=0.0)
    parser.add_argument("--heading-gain-per-s", type=float, default=2.0)
    parser.add_argument("--lateral-limit-mps", type=float, default=0.2)
    parser.add_argument("--yaw-rate-limit-radps", type=float, default=0.6)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite output: {args.out}")
    if min(
        args.forward_s,
        args.peel_pulse_s,
        args.recovery_s,
        args.reverse_s,
        args.stop_s,
        args.forward_speed_mps,
        args.reverse_speed_mps,
        args.rgb_fps,
    ) <= 0.0:
        parser.error("all durations, speeds, and rgb-fps must be positive")
    args.enable_cameras = True
    app = AppLauncher(args).app

    passed = False
    try:
        import yaml
        from PIL import Image

        from kino_vla.sim.adhesion import FootAdhesionConfig, IrregularRegion
        from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
        from kino_vla.sim.realistic_route_controller_v3 import DampedRouteControllerV3
        from kino_vla.sim.realistic_route_protocol_v2 import scene_route_binding
        from kino_vla.sim.terrain_materials import (
            appearance_binding_from_record,
            load_terrain_asset_lock,
        )
        from kino_vla.utils.config import CONFIGS_DIR, load_config
        from scripts.isaac_collect_embodiedgen_o4_pair import (
            _frame_metrics,
            _paired_consequence_metrics,
        )

        output = args.out.resolve()
        output.mkdir(parents=True)
        episode = args.episode_usd.resolve()
        compiled_path = args.compiled_audit.resolve()
        compiled = _json(compiled_path)
        if compiled.get("passed") is not True:
            raise RuntimeError("terrain compiled audit did not pass")
        if compiled.get("files", {}).get(episode.name) != _sha256(episode):
            raise RuntimeError("episode USD hash differs from terrain compiled audit")
        base_path = Path(compiled["base_compiled_audit"]).resolve()
        if _sha256(base_path) != compiled["base_compiled_audit_sha256"]:
            raise RuntimeError("base compiled audit hash differs from terrain audit")
        base = _json(base_path)
        if base.get("passed") is not True or base.get("operator", {}).get("operator_id") != "O4":
            raise RuntimeError("base scene lacks an admitted route-relative O4 region")
        admission_path = episode.parent / "realistic_stack_v16_admission.json"
        admission = _json(admission_path)
        if admission.get("passed") is not True:
            raise RuntimeError("v16 realistic stack admission did not pass")

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
        binding = scene_route_binding(compiled)
        frame = binding.frame
        start = frame.point(0.0, 0.0)
        controller = DampedRouteControllerV3(
            cross_track_gain_per_s=args.cross_track_gain_per_s,
            lateral_velocity_damping=args.lateral_velocity_damping,
            heading_gain_per_s=args.heading_gain_per_s,
            lateral_limit_mps=args.lateral_limit_mps,
            yaw_rate_limit_radps=args.yaw_rate_limit_radps,
        )
        backend = IsaacPolicyBackend(sim_cfg, start, frame.heading_rad, front_cam=True)
        scene_prim = backend.load_realistic_scene(str(episode))

        lock_path = args.material_lock.resolve()
        lock = load_terrain_asset_lock(lock_path)
        asset_root = Path(lock["asset_root"])
        if not asset_root.is_absolute():
            asset_root = ROOT / asset_root
        appearance = appearance_binding_from_record(
            {
                "material_family": args.material_id,
                "appearance_id": f"v16-o4-pair-{args.material_id}",
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
        adhesion_cfg = FootAdhesionConfig(
            region=region,
            surface_z_m=float(operator_record["surface_z_m"]),
            attach_contact_force_n=5.0,
            attach_height_tolerance_m=0.08,
            stiffness_xy_n_per_m=150.0,
            damping_xy_ns_per_m=3.0,
            stiffness_z_n_per_m=120.0,
            damping_z_ns_per_m=2.0,
            force_cap_n=35.0,
            break_force_n=105.0,
            peel_release_force_n=10.0,
            peel_velocity_threshold_mps=0.03,
            progress_axis_xy=(float(frame.direction[0]), float(frame.direction[1])),
            max_active_feet=1,
        )

        total_s = (
            args.forward_s
            + args.peel_pulse_s
            + args.recovery_s
            + args.reverse_s
            + args.stop_s
        )
        total_steps = round(total_s / backend.dt)
        frame_stride = max(1, round(1.0 / (args.rgb_fps * backend.dt)))
        summaries: dict[str, dict[str, Any]] = {}
        rows_by_take: dict[str, list[dict[str, Any]]] = {}
        initial_states: dict[str, dict[str, Any]] = {}
        initial_rgbs: dict[str, np.ndarray] = {}

        for take in ("nominal", "o4_adhesion"):
            take_dir = output / take
            rgb_dir = take_dir / "rgb"
            rgb_dir.mkdir(parents=True)
            obs = backend.deep_reset(args.seed)
            if take == "o4_adhesion":
                backend.add_foot_adhesion(adhesion_cfg)
            initial_states[take] = {
                "pos_xy_m": obs.pos.tolist(),
                "heading_rad": float(obs.heading),
                "vel_body_mps": obs.vel_body.tolist(),
                "base_height_m": float(obs.base_height),
                "tilt_rad": float(obs.tilt),
            }
            rows: list[dict[str, Any]] = []
            frames: list[dict[str, Any]] = []

            def capture(step: int, phase: str) -> None:
                sensor = backend.capture_front_camera()
                if sensor is None:
                    raise RuntimeError("Go2-front RTX camera returned no frame")
                rgb = np.asarray(sensor["rgb"])[..., :3].astype(np.uint8)
                if step == -1:
                    initial_rgbs[take] = rgb.copy()
                path = rgb_dir / f"frame_{len(frames):06d}.png"
                Image.fromarray(rgb).save(path, compress_level=3)
                frames.append(
                    {
                        "frame_id": len(frames),
                        "step": step,
                        "time_s": float(obs.t),
                        "phase": phase,
                        "path": str(path.relative_to(output)),
                        "sha256": _sha256(path),
                        "metrics": _frame_metrics(rgb),
                    }
                )

            capture(-1, "initial")
            first_fall_step: int | None = None
            for step in range(total_steps):
                t = step * backend.dt
                speed, phase = _phase(t, args)
                command, control = controller.command(
                    frame,
                    position_xy_m=obs.pos,
                    heading_rad=float(obs.heading),
                    velocity_body_xy_mps=np.asarray(obs.vel_body[:2]),
                    forward_speed_mps=speed,
                    target_lateral_offset_m=0.0,
                )
                obs = backend.step(command)
                progress, lateral = frame.project(obs.pos)
                adhesion = backend.adhesion_telemetry()
                row = {
                    "step": step,
                    "time_s": float(obs.t),
                    "phase": phase,
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
                if step % frame_stride == 0 or obs.fallen:
                    capture(step, phase)
                if obs.fallen:
                    first_fall_step = step
                    break

            telemetry_path = take_dir / "telemetry.jsonl"
            with telemetry_path.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
            frames_path = take_dir / "frames.json"
            _write_json(frames_path, frames)
            event_trace = [
                {
                    "step": int(row["step"]),
                    "time_s": float(row["time_s"]),
                    "phase": row["phase"],
                    "foot": foot["name"],
                    "event": foot["event"],
                    "raw_force_n": float(foot["raw_force_n"]),
                }
                for row in rows
                for foot in row["adhesion"]["feet"]
                if foot["event"] != "none"
            ]
            summary = {
                "take": take,
                "seed": args.seed,
                "steps": len(rows),
                "fixed_horizon_steps": total_steps,
                "fell": any(row["fallen"] for row in rows),
                "first_fall_step": first_fall_step,
                "max_progress_m": max(float(row["progress_m"]) for row in rows),
                "final_progress_m": float(rows[-1]["progress_m"]),
                "maximum_absolute_route_lateral_offset_m": max(
                    abs(float(row["route_lateral_offset_m"])) for row in rows
                ),
                "min_base_height_m": min(float(row["base_height_m"]) for row in rows),
                "max_tilt_rad": max(float(row["tilt_rad"]) for row in rows),
                "peak_adhesion_force_n": max(
                    float(row["adhesion"]["total_applied_force_n"]) for row in rows
                ),
                "max_active_feet": max(int(row["adhesion"]["active_feet"]) for row in rows),
                "attachment_events": sum(e["event"] == "attached" for e in event_trace),
                "peel_events": sum(e["event"] == "peeled" for e in event_trace),
                "break_events": sum(e["event"] == "broken" for e in event_trace),
                "forward_break_events": sum(
                    e["event"] == "broken" and e["phase"] == "forward" for e in event_trace
                ),
                "reverse_peel_events": sum(
                    e["event"] == "peeled"
                    and e["phase"] in {"peel_release", "reverse_retreat"}
                    for e in event_trace
                ),
                "event_trace": event_trace,
                "telemetry": str(telemetry_path),
                "telemetry_sha256": _sha256(telemetry_path),
                "frames": str(frames_path),
                "frames_sha256": _sha256(frames_path),
                "rgb_frame_count": len(frames),
                "valid_frame_fraction": sum(
                    frame_row["metrics"]["std_luminance"] >= 0.025
                    and frame_row["metrics"]["black_fraction"] < 0.95
                    and frame_row["metrics"]["quantized_color_count_5bit"] >= 32
                    for frame_row in frames
                )
                / len(frames),
            }
            _write_json(take_dir / "summary.json", summary)
            summaries[take] = summary
            rows_by_take[take] = rows

        nominal = summaries["nominal"]
        anomaly = summaries["o4_adhesion"]
        consequence = _paired_consequence_metrics(
            rows_by_take["nominal"], rows_by_take["o4_adhesion"]
        )
        initial_delta = {
            key: float(
                np.max(
                    np.abs(
                        np.asarray(initial_states["nominal"][key], dtype=np.float64)
                        - np.asarray(initial_states["o4_adhesion"][key], dtype=np.float64)
                    )
                )
            )
            for key in initial_states["nominal"]
        }
        initial_rgb_l1 = float(
            np.abs(initial_rgbs["nominal"].astype(np.float32) - initial_rgbs["o4_adhesion"].astype(np.float32)).mean()
            / 255.0
        )
        checks = {
            "compiled_scene_hash_verified": True,
            "v16_stack_admission_verified": True,
            "same_seed_and_episode": nominal["seed"] == anomaly["seed"] == args.seed,
            "same_initial_physical_state": max(initial_delta.values()) <= 1.0e-5,
            "same_initial_visual_state": initial_rgb_l1 <= 0.03,
            "nominal_completed_without_fall": nominal["steps"] == total_steps
            and nominal["fell"] is False,
            "nominal_operator_inactive": nominal["peak_adhesion_force_n"] == 0.0,
            "o4_attached_at_named_foot": anomaly["attachment_events"] >= 1
            and anomaly["max_active_feet"] >= 1,
            "o4_force_observable": anomaly["peak_adhesion_force_n"] >= 10.0,
            "o4_holds_through_forward": anomaly["forward_break_events"] == 0,
            "o4_release_observable": anomaly["peel_events"] + anomaly["break_events"] >= 1,
            "paired_effect_window_available": consequence.get("available") is True,
            "paired_locomotion_consequence": consequence.get("available") is True
            and float(consequence["window_duration_s"]) >= 1.0
            and float(consequence["progress_gain_lag_m"]) >= 0.08,
            "paired_posture_consequence": consequence.get("available") is True
            and (
                float(consequence["max_tilt_increase_rad"]) >= 0.10
                or float(consequence["min_base_height_drop_m"]) >= 0.03
            ),
            "both_sequences_visually_valid": nominal["valid_frame_fraction"] >= 0.90
            and anomaly["valid_frame_fraction"] >= 0.90,
            "both_lanes_within_route_budget": nominal[
                "maximum_absolute_route_lateral_offset_m"
            ]
            <= args.max_route_deviation_m
            and anomaly["maximum_absolute_route_lateral_offset_m"]
            <= args.max_route_deviation_m,
        }
        manifest = {
            "schema_version": "kinofail.realistic-route-o4-pair.v16",
            "created_utc": datetime.now(UTC).isoformat(),
            "passed": all(checks.values()),
            "development_only": True,
            "counts_as_a0_a7_evidence": False,
            "realistic_a0_a7_readiness": "0/8",
            "scene_id": compiled["scene_id"],
            "room_family": _json(Path(base["source_manifest"]))["generation"]["room_type"],
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
            "pairing": {
                "intervention_only_difference": "per-foot world-anchor adhesion enabled",
                "initial_state_max_abs_delta": initial_delta,
                "initial_rgb_mean_absolute_difference": initial_rgb_l1,
            },
            "operator": {
                "id": "O4",
                "name": "foot_adhesion",
                "fidelity": "per-foot world-anchor force constraint",
                "region_xy_m": [list(point) for point in region.vertices_xy],
                "parameters": {
                    key: value for key, value in adhesion_cfg.__dict__.items() if key != "region"
                },
            },
            "route_controller": {
                "frame": {
                    "origin_xy_m": list(frame.origin_xy_m),
                    "direction_xy": list(frame.direction_xy),
                    "left_xy": list(frame.left_xy),
                    "heading_rad": frame.heading_rad,
                    "route_length_m": frame.route_length_m,
                    "surface_width_m": frame.surface_width_m,
                },
                "controller_contract": controller.contract(),
                "maximum_route_deviation_m": args.max_route_deviation_m,
                "forward_speed_mps": args.forward_speed_mps,
                "reverse_speed_mps": args.reverse_speed_mps,
            },
            "protocol": {
                "forward_s": args.forward_s,
                "peel_pulse_s": args.peel_pulse_s,
                "recovery_s": args.recovery_s,
                "reverse_s": args.reverse_s,
                "stop_s": args.stop_s,
                "control_dt_s": backend.dt,
                "rgb_fps": args.rgb_fps,
            },
            "takes": summaries,
            "paired_consequence": consequence,
            "checks": checks,
            "scene_prim": scene_prim,
            "provenance": {
                "command": " ".join(sys.argv),
                "python": sys.executable,
                "collector_sha256": _sha256(Path(__file__).resolve()),
                "legacy_helper_sha256": _sha256(
                    ROOT / "scripts/isaac_collect_embodiedgen_o4_pair.py"
                ),
                "backend_sha256": _sha256(ROOT / "kino_vla/sim/isaac_policy_backend.py"),
                "adhesion_sha256": _sha256(ROOT / "kino_vla/sim/adhesion.py"),
                "policy_sha256": _sha256(policy_path),
                "material_lock_sha256": _sha256(lock_path),
            },
            "remaining_gates": [
                "O4 multi-scene paired recalibration postrun audit",
                "O4 corpus protocol and registry binding",
                "new realistic A0 corpus completion",
                "new realistic A1-A7 reruns",
            ],
        }
        manifest_path = output / "pair_manifest.json"
        _write_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "passed": manifest["passed"],
                    "checks": checks,
                    "manifest": str(manifest_path),
                },
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
