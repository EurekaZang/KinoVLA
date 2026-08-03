#!/usr/bin/env python3
"""Articulated-Go2 physics and body-fixed front-camera gate for a compiled RoomGen scene."""

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

import numpy as np

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _argument_path(flag: str) -> Path | None:
    """Resolve one required path argument without starting a second AppLauncher parser."""
    try:
        index = sys.argv.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(sys.argv):
        return None
    return Path(sys.argv[index + 1]).resolve()


def _write_exception_audit(exc: BaseException) -> Path | None:
    """Persist fail-closed evidence even when Isaac fails before normal QA telemetry exists."""
    output = _argument_path("--out")
    if output is None:
        return None
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "go2_scene_exception_audit.json"
    episode = _argument_path("--episode-usd")
    compiled = _argument_path("--compiled-audit")
    backend = Path(__file__).resolve().parents[1] / "kino_vla/sim/isaac_policy_backend.py"
    payload = {
        "schema_version": "kinofail.embodiedgen-go2-scene-qa-exception.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": False,
        "admission_state": "articulated_go2_scene_qa_exception",
        "exception_type": type(exc).__name__,
        "reason": str(exc),
        "traceback": "".join(traceback.format_exception(exc)),
        "episode_usd": {
            "path": str(episode) if episode is not None else None,
            "sha256": _sha256(episode) if episode is not None and episode.is_file() else None,
        },
        "compiled_audit": {
            "path": str(compiled) if compiled is not None else None,
            "sha256": _sha256(compiled) if compiled is not None and compiled.is_file() else None,
        },
        "provenance": {
            "command": " ".join(sys.argv),
            "python": sys.executable,
            "qa_script_sha256": _sha256(Path(__file__).resolve()),
            "backend_path": str(backend),
            "backend_sha256": _sha256(backend) if backend.is_file() else None,
        },
        "counts_as_a0_a7_evidence": False,
    }
    if not audit_path.exists():
        audit_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return audit_path


def _quat_apply_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w = float(quat[0])
    xyz = np.asarray(quat[1:4], dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    return vector + 2.0 * np.cross(xyz, np.cross(xyz, vector) + w * vector)


def _quat_multiply_wxyz(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = [float(value) for value in first]
    w2, x2, y2, z2 = [float(value) for value in second]
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_angular_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= max(float(np.linalg.norm(first)), 1.0e-12)
    second /= max(float(np.linalg.norm(second)), 1.0e-12)
    cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return 2.0 * math.acos(cosine)


def _frame_metrics(rgb: np.ndarray) -> dict[str, float | int]:
    values = rgb[..., :3].astype(np.float32) / 255.0
    luminance = 0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]
    return {
        "mean_luminance": float(luminance.mean()),
        "std_luminance": float(luminance.std()),
        "black_fraction": float((luminance < 0.02).mean()),
        "quantized_color_count_5bit": int(
            len(np.unique((values * 31).astype(np.uint8).reshape(-1, 3), axis=0))
        ),
    }


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    alpha = float(np.dot(point - start, delta) / max(np.dot(delta, delta), 1.0e-12))
    projection = start + np.clip(alpha, 0.0, 1.0) * delta
    return float(np.linalg.norm(point - projection))


def main() -> int:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--episode-usd", type=Path, required=True)
    preliminary.add_argument("--compiled-audit", type=Path, required=True)
    preliminary.add_argument("--out", type=Path, required=True)
    preliminary.add_argument("--seed", type=int, default=20260721)
    preliminary.add_argument("--steps", type=int, default=140)
    preliminary.add_argument("--target-progress-m", type=float, default=0.50)
    preliminary.add_argument("--command-speed-mps", type=float, default=0.32)
    preliminary.add_argument("--route-tracking", action="store_true")
    preliminary.add_argument("--cross-track-gain", type=float, default=1.0)
    preliminary.add_argument("--heading-gain", type=float, default=2.0)
    preliminary.add_argument("--camera-profile", default="go2_front_calib_c")
    pre, _ = preliminary.parse_known_args()

    parser = argparse.ArgumentParser(description="EmbodiedGen scene articulated-Go2 QA")
    parser.add_argument("--episode-usd", type=Path, default=pre.episode_usd)
    parser.add_argument("--compiled-audit", type=Path, default=pre.compiled_audit)
    parser.add_argument("--out", type=Path, default=pre.out)
    parser.add_argument("--seed", type=int, default=pre.seed)
    parser.add_argument("--steps", type=int, default=pre.steps)
    parser.add_argument("--target-progress-m", type=float, default=pre.target_progress_m)
    parser.add_argument("--command-speed-mps", type=float, default=pre.command_speed_mps)
    parser.add_argument("--route-tracking", action="store_true", default=pre.route_tracking)
    parser.add_argument("--cross-track-gain", type=float, default=pre.cross_track_gain)
    parser.add_argument("--heading-gain", type=float, default=pre.heading_gain)
    parser.add_argument("--camera-profile", default=pre.camera_profile)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    simulation_app = AppLauncher(args).app

    passed = False
    try:
        from PIL import Image
        from pxr import Usd, UsdPhysics

        from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
        import yaml

        from kino_vla.utils.config import CONFIGS_DIR, REPO_ROOT, load_config

        episode = args.episode_usd.resolve()
        compiled_path = args.compiled_audit.resolve()
        compiled = json.loads(compiled_path.read_text())
        if not compiled.get("passed"):
            raise RuntimeError("compiled scene audit has not passed")
        if compiled["files"].get(episode.name) != _sha256(episode):
            raise RuntimeError("episode USD hash differs from compiled audit")
        route = np.asarray(compiled["route"]["waypoints_xy_m"], dtype=np.float64)
        if len(route) < 3:
            raise RuntimeError("route is too short for articulated traversal QA")
        segment_lengths = np.linalg.norm(route[1:] - route[:-1], axis=1)
        segment_index = int(np.argmax(segment_lengths))
        start = route[segment_index]
        target = route[segment_index + 1]
        segment_length = float(np.linalg.norm(target - start))
        if not 0.0 < args.target_progress_m <= segment_length:
            raise RuntimeError(
                f"target progress {args.target_progress_m}m is outside selected "
                f"segment length {segment_length}m"
            )
        heading = math.atan2(float(target[1] - start[1]), float(target[0] - start[0]))
        direction = (target - start) / segment_length
        left = np.asarray([-direction[1], direction[0]], dtype=np.float64)

        benchmark_config = yaml.safe_load(
            (CONFIGS_DIR / "data/kinofail_realistic.yaml").read_text(encoding="utf-8")
        )
        profile_specs = benchmark_config["camera_profile_specs"]
        if args.camera_profile not in profile_specs:
            raise RuntimeError(f"unknown frozen camera profile: {args.camera_profile}")
        camera_profile = dict(profile_specs[args.camera_profile])
        mount = np.asarray(camera_profile["mount_xyz_m"], dtype=np.float64)
        pitch_down_rad = float(camera_profile["pitch_down_rad"])
        sim_cfg = load_config(
            "sim/go2_skeleton.yaml",
            {
                "cam_width": int(camera_profile["width"]),
                "cam_height": int(camera_profile["height"]),
                "front_cam_width": int(camera_profile["width"]),
                "front_cam_height": int(camera_profile["height"]),
                "front_cam_focal_mm": float(camera_profile["focal_length_mm"]),
                "front_cam_aperture_mm": 24.0,
                "front_cam_x_m": float(mount[0]),
                "front_cam_y_m": float(mount[1]),
                "front_cam_z_m": float(mount[2]),
                "front_cam_pitch_down_rad": pitch_down_rad,
            },
        )
        backend = IsaacPolicyBackend(
            sim_cfg,
            start.copy(),
            heading,
            record_cam=False,
            front_cam=True,
        )
        scene_prim = backend.load_realistic_scene(str(episode))
        obs = backend.reset(int(args.seed))

        import omni.usd

        stage = omni.usd.get_context().get_stage()
        custom_floor_path = f"{scene_prim}/Collision/Floor"
        custom_floor = stage.GetPrimAtPath(custom_floor_path)
        custom_floor_enabled = bool(
            custom_floor.IsValid()
            and custom_floor.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(custom_floor).GetCollisionEnabledAttr().Get() is not False
        )
        disabled_default = list(backend._disabled_default_ground_colliders)
        if not disabled_default:
            raise RuntimeError("custom floor loaded without disabling Isaac default ground")

        args.out.mkdir(parents=True, exist_ok=True)
        initial = obs.pos.copy()
        rows = []
        captures = []
        local_pitch = np.asarray(
            [math.cos(0.5 * pitch_down_rad), 0.0, math.sin(0.5 * pitch_down_rad), 0.0],
            dtype=np.float64,
        )
        max_camera_pose_error = 0.0
        max_camera_orientation_error = 0.0
        if not 0.05 <= float(args.command_speed_mps) <= 0.60:
            raise RuntimeError("command speed must be within the frozen QA-safe range [0.05, 0.60]")
        if min(float(args.cross_track_gain), float(args.heading_gain)) < 0.0:
            raise RuntimeError("route-tracking gains must be non-negative")

        def route_command(current_obs: object) -> np.ndarray:
            if not args.route_tracking:
                return np.array(
                    [float(args.command_speed_mps), 0.0, 0.0], dtype=np.float64
                )
            cross_track = float(np.dot(current_obs.pos - start, left))
            heading_error = math.atan2(
                math.sin(heading - float(current_obs.heading)),
                math.cos(heading - float(current_obs.heading)),
            )
            return np.asarray(
                [
                    float(args.command_speed_mps),
                    float(np.clip(-float(args.cross_track_gain) * cross_track, -0.16, 0.16)),
                    float(np.clip(float(args.heading_gain) * heading_error, -0.60, 0.60)),
                ],
                dtype=np.float64,
            )

        def capture_front(step: int) -> None:
            nonlocal max_camera_pose_error, max_camera_orientation_error
            sensor = backend.capture_front_camera()
            if sensor is None:
                raise RuntimeError("body-fixed front camera returned no frame")
            rgb = np.asarray(sensor["rgb"])[..., :3]
            base_pos = backend._robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
            base_quat = (
                backend._robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
            )
            expected_camera = base_pos + _quat_apply_wxyz(base_quat, mount)
            pose_error = float(np.linalg.norm(np.asarray(sensor["pos"]) - expected_camera))
            max_camera_pose_error = max(max_camera_pose_error, pose_error)
            expected_quat = _quat_multiply_wxyz(base_quat, local_pitch)
            orientation_error = _quat_angular_distance_rad(
                np.asarray(sensor["quat_world"], dtype=np.float64), expected_quat
            )
            max_camera_orientation_error = max(
                max_camera_orientation_error, orientation_error
            )
            frame_path = args.out / f"front_step{step:04d}.png"
            Image.fromarray(rgb.astype(np.uint8)).save(frame_path, optimize=True)
            captures.append(
                {
                    "step": step,
                    "path": frame_path.name,
                    "sha256": _sha256(frame_path),
                    "camera_position_world_m": np.asarray(sensor["pos"]).tolist(),
                    "expected_rigid_mount_position_world_m": expected_camera.tolist(),
                    "mount_position_error_m": pose_error,
                    "expected_rigid_mount_quaternion_wxyz": expected_quat.tolist(),
                    "mount_orientation_error_rad": orientation_error,
                    "intrinsic_matrix": np.asarray(sensor["K"]).tolist(),
                    "metrics": _frame_metrics(rgb),
                }
            )

        for step in range(args.steps):
            command = route_command(obs)
            obs = backend.step(command)
            route_deviation = _point_segment_distance(obs.pos, start, target)
            route_progress = float(np.dot(obs.pos - initial, direction))
            rows.append(
                {
                    "step": step,
                    "time_s": float(obs.t),
                    "pos_xy_m": obs.pos.tolist(),
                    "heading_rad": float(obs.heading),
                    "forward_speed_mps": float(obs.vel_body[0]),
                    "base_height_m": float(obs.base_height),
                    "tilt_rad": float(obs.tilt),
                    "support_ratio": float(obs.support_ratio),
                    "fallen": bool(obs.fallen),
                    "command_body_mps": command.tolist(),
                    "route_deviation_m": route_deviation,
                    "route_progress_m": route_progress,
                }
            )
            if not captures or (
                len(captures) == 1 and route_progress >= 0.5 * args.target_progress_m
            ):
                capture_front(step)
            if obs.fallen:
                break
            if route_progress >= args.target_progress_m:
                break

        if len(captures) < 3:
            capture_front(int(rows[-1]["step"]))

        telemetry_path = args.out / "telemetry.jsonl"
        with telemetry_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        final = np.asarray(rows[-1]["pos_xy_m"], dtype=np.float64)
        progress = float(np.dot(final - initial, direction))
        checks = {
            "compiled_episode_hash_verified": True,
            "kino_custom_floor_collision_enabled": custom_floor_enabled,
            "default_ground_collision_disabled": bool(disabled_default),
            "single_ground_contact_authority": custom_floor_enabled and bool(disabled_default),
            "target_progress_reached_within_step_budget": progress >= args.target_progress_m,
            "robot_did_not_fall": not any(row["fallen"] for row in rows),
            "stable_base_height": min(row["base_height_m"] for row in rows) >= 0.20,
            "stable_body_tilt": max(row["tilt_rad"] for row in rows) <= 0.55,
            "positive_route_progress": progress >= args.target_progress_m,
            "route_tracking_within_planned_clearance": max(
                row["route_deviation_m"] for row in rows
            )
            <= compiled["route"]["required_clearance_m"],
            "three_front_frames": len(captures) == 3,
            "front_frames_non_degenerate": all(
                item["metrics"]["std_luminance"] >= 0.035
                and item["metrics"]["black_fraction"] < 0.90
                and item["metrics"]["quantized_color_count_5bit"] >= 64
                for item in captures
            ),
            "front_camera_position_rigid_mount": max_camera_pose_error <= 1.0e-4,
            "front_camera_orientation_rigid_mount": max_camera_orientation_error <= 0.002,
        }
        audit = {
            "schema_version": "kinofail.embodiedgen-articulated-go2-qa.v2",
            "created_utc": datetime.now(UTC).isoformat(),
            "scene_id": compiled["scene_id"],
            "episode_usd": str(episode),
            "episode_usd_sha256": _sha256(episode),
            "compiled_audit": str(compiled_path),
            "compiled_audit_sha256": _sha256(compiled_path),
            "isaac_scene_prim": scene_prim,
            "start_xy_m": start.tolist(),
            "target_xy_m": target.tolist(),
            "selected_segment_length_m": segment_length,
            "target_progress_m": float(args.target_progress_m),
            "start_heading_rad": heading,
            "command_body_mps": [float(args.command_speed_mps), 0.0, 0.0],
            "route_tracking_controller": {
                "enabled": bool(args.route_tracking),
                "cross_track_gain": float(args.cross_track_gain),
                "heading_gain": float(args.heading_gain),
                "lateral_command_limit_mps": 0.16,
                "yaw_rate_command_limit_radps": 0.60,
            },
            "maximum_steps_requested": int(args.steps),
            "steps_completed": len(rows),
            "sim_time_s": float(rows[-1]["time_s"]),
            "route_progress_m": progress,
            "max_route_deviation_m": max(row["route_deviation_m"] for row in rows),
            "min_base_height_m": min(row["base_height_m"] for row in rows),
            "max_tilt_rad": max(row["tilt_rad"] for row in rows),
            "min_support_ratio": min(row["support_ratio"] for row in rows),
            "collision_authority": {
                "custom_floor": custom_floor_path,
                "custom_floor_enabled": custom_floor_enabled,
                "disabled_default_ground_colliders": disabled_default,
            },
            "camera": {
                "role": "actual body-fixed Go2 front RTX RGB",
                "mount_xyz_base_m": mount.tolist(),
                "pitch_down_rad": pitch_down_rad,
                "profile": str(args.camera_profile),
                "calibration_status": "engineering_pending_physical_fixture_calibration",
                "focal_length_mm": float(camera_profile["focal_length_mm"]),
                "resolution": [
                    int(camera_profile["width"]),
                    int(camera_profile["height"]),
                ],
                "max_rigid_mount_position_error_m": max_camera_pose_error,
                "max_rigid_mount_orientation_error_rad": max_camera_orientation_error,
                "captures": captures,
            },
            "telemetry": telemetry_path.name,
            "telemetry_sha256": _sha256(telemetry_path),
            "checks": checks,
            "passed": all(checks.values()),
            "admission_state": (
                "articulated_go2_scene_passed_pending_paired_operator_collection"
                if all(checks.values())
                else "articulated_go2_scene_failed"
            ),
            "provenance": {
                "command": " ".join(sys.argv),
                "python": sys.executable,
                "gpu": str(backend._device),
                "repository": str(REPO_ROOT),
            },
        }
        audit_path = args.out / "go2_scene_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({
            "passed": audit["passed"],
            "checks": checks,
            "route_progress_m": progress,
            "max_route_deviation_m": audit["max_route_deviation_m"],
            "audit": str(audit_path),
        }, indent=2), flush=True)
        passed = bool(audit["passed"])
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        closer = threading.Thread(target=simulation_app.close, daemon=True)
        closer.start()
        closer.join(timeout=8.0)
    return 0 if passed else 2


if __name__ == "__main__":
    exit_code = 2
    try:
        exit_code = main()
    except BaseException as exc:  # Isaac native teardown may otherwise keep the process alive.
        audit = _write_exception_audit(exc)
        traceback.print_exc()
        if audit is not None:
            print(f"[go2-qa] exception audit: {audit}", file=sys.stderr, flush=True)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
