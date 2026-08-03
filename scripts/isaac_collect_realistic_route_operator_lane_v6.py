#!/usr/bin/env python3
"""Collect one damped lane-aware cross-domain realistic O1/O2/O3 lane.

Version 6 adds measured lateral-velocity damping to the shared controller and retains lane-aware coverage: nominal lanes require sustained coverage, while anomaly lanes may terminate early only after measured load-bearing contact.
It statically binds the shared route protocol and supports both the
forest and EmbodiedGen terrain-route scene schemas.  Unlike the earlier
development bridge, it performs no runtime source rewriting.  Its outputs are
operator-admission evidence only until a separately frozen corpus protocol
binds them into a registry and schedule.
"""

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


def _quat_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= max(float(np.linalg.norm(first)), 1.0e-12)
    second /= max(float(np.linalg.norm(second)), 1.0e-12)
    return 2.0 * math.acos(float(np.clip(abs(np.dot(first, second)), 0.0, 1.0)))


def _frame_metrics(rgb: np.ndarray) -> dict[str, float | int]:
    values = rgb.astype(np.float32) / 255.0
    luminance = (
        0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]
    )
    gradient_x = np.abs(np.diff(luminance, axis=1)).reshape(-1)
    gradient_y = np.abs(np.diff(luminance, axis=0)).reshape(-1)
    gradients = np.concatenate((gradient_x, gradient_y))
    return {
        "mean_luminance": float(luminance.mean()),
        "std_luminance": float(luminance.std()),
        "black_fraction": float((luminance < 0.02).mean()),
        "quantized_color_count_5bit": int(
            len(np.unique((values * 31).astype(np.uint8).reshape(-1, 3), axis=0))
        ),
        "mean_abs_luminance_gradient": float(gradients.mean()),
        "edge_fraction_gt_0_02": float((gradients > 0.02).mean()),
    }


def _front_sequence_non_degenerate(captures: list[dict[str, object]]) -> bool:
    """Phase-aware quality gate for context views plus a possibly close terrain consequence."""
    by_label = {str(row["label"]): row for row in captures}
    if set(by_label) != {"pre_entry", "inside_region", "matched_consequence"}:
        return False
    metrics = [row["metrics"] for row in by_label.values()]
    basic_frame_gate = all(
        float(item["std_luminance"]) >= 0.020
        and float(item["black_fraction"]) < 0.90
        and int(item["quantized_color_count_5bit"]) >= 64
        for item in metrics
    )
    context_gate = all(
        float(by_label[label]["metrics"]["std_luminance"]) >= 0.035
        and float(by_label[label]["metrics"]["edge_fraction_gt_0_02"]) >= 0.010
        for label in ("pre_entry", "inside_region")
    )
    consequence = by_label["matched_consequence"]["metrics"]
    close_ground_consequence_gate = (
        float(consequence["std_luminance"]) >= 0.035
        and float(consequence["mean_luminance"]) <= 0.97
        and float(consequence["edge_fraction_gt_0_02"]) >= 0.004
        and int(consequence["quantized_color_count_5bit"]) >= 96
    )
    return basic_frame_gate and context_gate and close_ground_consequence_gate


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def main(
    *,
    controller_mode: str = "damped_v3",
    collector_lifecycle: str = "native_v6_damped_lane_aware_protocol_candidate",
    collector_script: Path | None = None,
) -> int:
    if controller_mode not in {"damped_v3", "settled_lookahead_v4"}:
        raise ValueError(f"unsupported controller mode: {controller_mode}")
    collector_script = (
        Path(__file__).resolve()
        if collector_script is None
        else Path(collector_script).resolve()
    )
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--operator", choices=("o1", "o2", "o3"), default="o2")
    preliminary.add_argument("--lane", choices=("nominal", "anomaly"), required=True)
    preliminary.add_argument("--episode-usd", type=Path, required=True)
    preliminary.add_argument("--compiled-audit", type=Path, required=True)
    preliminary.add_argument("--out", type=Path, required=True)
    preliminary.add_argument("--camera-profile", default="go2_front_calib_b")
    preliminary.add_argument("--material-id", default="train_ground037")
    preliminary.add_argument(
        "--material-lock",
        type=Path,
        default=ROOT / "outputs/assets/terrain_pbr_v1/terrain_assets.lock.json",
    )
    preliminary.add_argument("--seed", type=int, default=20260722)
    preliminary.add_argument("--steps", type=int, default=400)
    preliminary.add_argument("--o1-mu-s", type=float, default=0.04)
    preliminary.add_argument("--o1-mu-d", type=float, default=0.03)
    preliminary.add_argument("--o3-damage-threshold-ns", type=float, default=25.0)
    preliminary.add_argument("--o3-drop-m", type=float, default=0.12)
    preliminary.add_argument("--o3-residual-support", type=float, default=0.18)
    preliminary.add_argument("--region-progress-m", type=float, default=1.5)
    preliminary.add_argument("--region-lateral-offset-m", type=float, default=0.0)
    preliminary.add_argument("--region-half-length-m", type=float, default=0.65)
    preliminary.add_argument("--region-half-width-m", type=float, default=0.60)
    preliminary.add_argument("--start-lateral-offset-m", type=float, default=0.0)
    preliminary.add_argument("--start-heading-offset-rad", type=float, default=0.0)
    preliminary.add_argument("--target-lateral-offset-m", type=float, default=0.0)
    preliminary.add_argument("--forward-speed-mps", type=float, default=0.20)
    preliminary.add_argument("--max-route-deviation-m", type=float, default=-1.0)
    preliminary.add_argument("--min-region-samples", type=int, default=40)
    preliminary.add_argument("--cross-track-gain-per-s", type=float, default=1.0)
    preliminary.add_argument("--lateral-velocity-damping", type=float, default=1.0)
    preliminary.add_argument("--lateral-limit-mps", type=float, default=0.16)
    preliminary.add_argument("--heading-gain-per-s", type=float, default=2.0)
    preliminary.add_argument("--yaw-rate-limit-radps", type=float, default=0.60)
    preliminary.add_argument("--alignment-gain-per-s", type=float, default=0.8)
    preliminary.add_argument("--alignment-lateral-limit-mps", type=float, default=0.12)
    preliminary.add_argument("--alignment-tolerance-m", type=float, default=0.035)
    preliminary.add_argument("--alignment-dwell-steps", type=int, default=10)
    preliminary.add_argument("--lookahead-m", type=float, default=0.45)
    preliminary.add_argument("--matched-consequence-step", type=int, default=-1)
    pre, _ = preliminary.parse_known_args()

    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", choices=("o1", "o2", "o3"), default=pre.operator)
    parser.add_argument("--lane", choices=("nominal", "anomaly"), default=pre.lane)
    parser.add_argument("--episode-usd", type=Path, default=pre.episode_usd)
    parser.add_argument("--compiled-audit", type=Path, default=pre.compiled_audit)
    parser.add_argument("--out", type=Path, default=pre.out)
    parser.add_argument("--camera-profile", default=pre.camera_profile)
    parser.add_argument("--material-id", default=pre.material_id)
    parser.add_argument("--material-lock", type=Path, default=pre.material_lock)
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=ROOT / "outputs/locomotion/policy.pt",
        help="Explicit low-level locomotion policy; its resolved path and hash are recorded.",
    )
    parser.add_argument("--seed", type=int, default=pre.seed)
    parser.add_argument("--steps", type=int, default=pre.steps)
    parser.add_argument("--o1-mu-s", type=float, default=pre.o1_mu_s)
    parser.add_argument("--o1-mu-d", type=float, default=pre.o1_mu_d)
    parser.add_argument(
        "--o3-damage-threshold-ns", type=float, default=pre.o3_damage_threshold_ns
    )
    parser.add_argument("--o3-drop-m", type=float, default=pre.o3_drop_m)
    parser.add_argument("--o3-residual-support", type=float, default=pre.o3_residual_support)
    parser.add_argument("--region-progress-m", type=float, default=pre.region_progress_m)
    parser.add_argument(
        "--region-lateral-offset-m", type=float, default=pre.region_lateral_offset_m
    )
    parser.add_argument("--region-half-length-m", type=float, default=pre.region_half_length_m)
    parser.add_argument("--region-half-width-m", type=float, default=pre.region_half_width_m)
    parser.add_argument(
        "--start-lateral-offset-m", type=float, default=pre.start_lateral_offset_m
    )
    parser.add_argument(
        "--start-heading-offset-rad", type=float, default=pre.start_heading_offset_rad
    )
    parser.add_argument(
        "--target-lateral-offset-m", type=float, default=pre.target_lateral_offset_m
    )
    parser.add_argument("--forward-speed-mps", type=float, default=pre.forward_speed_mps)
    parser.add_argument(
        "--max-route-deviation-m", type=float, default=pre.max_route_deviation_m
    )
    parser.add_argument("--min-region-samples", type=int, default=pre.min_region_samples)
    parser.add_argument(
        "--cross-track-gain-per-s", type=float, default=pre.cross_track_gain_per_s
    )
    parser.add_argument(
        "--lateral-velocity-damping", type=float, default=pre.lateral_velocity_damping
    )
    parser.add_argument("--lateral-limit-mps", type=float, default=pre.lateral_limit_mps)
    parser.add_argument("--heading-gain-per-s", type=float, default=pre.heading_gain_per_s)
    parser.add_argument(
        "--yaw-rate-limit-radps", type=float, default=pre.yaw_rate_limit_radps
    )
    parser.add_argument(
        "--alignment-gain-per-s", type=float, default=pre.alignment_gain_per_s
    )
    parser.add_argument(
        "--alignment-lateral-limit-mps",
        type=float,
        default=pre.alignment_lateral_limit_mps,
    )
    parser.add_argument(
        "--alignment-tolerance-m", type=float, default=pre.alignment_tolerance_m
    )
    parser.add_argument(
        "--alignment-dwell-steps", type=int, default=pre.alignment_dwell_steps
    )
    parser.add_argument("--lookahead-m", type=float, default=pre.lookahead_m)
    parser.add_argument(
        "--matched-consequence-step", type=int, default=pre.matched_consequence_step
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app

    passed = False
    try:
        import yaml
        from PIL import Image
        from pxr import UsdGeom, UsdPhysics

        import omni.usd

        from kino_vla.eval.realistic_route_runtime_gate import audit_route_lane_runtime
        from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
        from kino_vla.sim.realistic_route_controller_v3 import (
            DampedRouteControllerV3,
        )
        from kino_vla.sim.realistic_route_controller_v4 import (
            SettledLookaheadRouteControllerV4,
        )
        from kino_vla.sim.operators import Collapse, ComplianceField, MuField
        from kino_vla.sim.realistic_route_protocol_v2 import RouteFrameV2 as StraightRouteFrame, scene_route_binding
        from kino_vla.sim.terrain_materials import (
            appearance_binding_from_record,
            load_terrain_asset_lock,
        )
        from kino_vla.utils.config import CONFIGS_DIR, load_config
        from kino_vla.utils.geometry import Rect

        output = args.out.resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite terrain lane: {output}")
        output.mkdir(parents=True)
        print(f"[o2-lane] output-created {output}", flush=True)
        episode = args.episode_usd.resolve()
        compiled_path = args.compiled_audit.resolve()
        compiled = json.loads(compiled_path.read_text(encoding="utf-8"))
        if compiled.get("passed") is not True:
            raise RuntimeError("input compiled scene did not pass")
        scene_binding = scene_route_binding(compiled)
        if compiled["files"].get(episode.name) != _sha256(episode):
            raise RuntimeError("episode USD differs from compiled audit")

        benchmark = yaml.safe_load(
            (CONFIGS_DIR / "data/kinofail_realistic.yaml").read_text(encoding="utf-8")
        )
        camera = dict(benchmark["camera_profile_specs"][args.camera_profile])
        mount = np.asarray(camera["mount_xyz_m"], dtype=np.float64)
        pitch = float(camera["pitch_down_rad"])
        policy_path = args.policy_path.resolve()
        if not policy_path.is_file():
            raise FileNotFoundError(f"locomotion policy not found: {policy_path}")
        config = load_config(
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
                "front_cam_pitch_down_rad": pitch,
            },
        )
        route_frame = scene_binding.frame
        admitted_tracking_budget = float(
            compiled.get("route", {}).get("maximum_admissible_tracking_error_m", -1.0)
            if float(args.max_route_deviation_m) < 0.0
            else float(args.max_route_deviation_m)
        )
        if not 0.0 < admitted_tracking_budget <= 0.5 * route_frame.surface_width_m:
            raise ValueError(
                "route tracking budget must be positive and no wider than half the route surface"
            )
        if int(args.min_region_samples) < 1:
            raise ValueError("min-region-samples must be positive")
        if controller_mode == "damped_v3":
            route_controller = DampedRouteControllerV3(
                cross_track_gain_per_s=float(args.cross_track_gain_per_s),
                lateral_velocity_damping=float(args.lateral_velocity_damping),
                heading_gain_per_s=float(args.heading_gain_per_s),
                lateral_limit_mps=float(args.lateral_limit_mps),
                yaw_rate_limit_radps=float(args.yaw_rate_limit_radps),
            )
            route_controller_source = ROOT / "kino_vla/sim/realistic_route_controller_v3.py"
        else:
            route_controller = SettledLookaheadRouteControllerV4(
                alignment_gain_per_s=float(args.alignment_gain_per_s),
                alignment_lateral_limit_mps=float(args.alignment_lateral_limit_mps),
                alignment_tolerance_m=float(args.alignment_tolerance_m),
                alignment_dwell_steps=int(args.alignment_dwell_steps),
                lookahead_m=float(args.lookahead_m),
                heading_gain_per_s=float(args.heading_gain_per_s),
                yaw_rate_limit_radps=float(args.yaw_rate_limit_radps),
            )
            route_controller_source = ROOT / "kino_vla/sim/realistic_route_controller_v4.py"
        if abs(route_frame.direction[0]) >= 1.0 - 1.0e-6:
            region_hx = float(args.region_half_length_m)
            region_hy = float(args.region_half_width_m)
        elif abs(route_frame.direction[1]) >= 1.0 - 1.0e-6:
            region_hx = float(args.region_half_width_m)
            region_hy = float(args.region_half_length_m)
        else:
            raise RuntimeError(
                "terrain operator regions currently require a world-axis-aligned straight route"
            )
        if float(args.region_half_length_m) <= 0.0 or float(args.region_half_width_m) <= 0.0:
            raise ValueError("operator region half sizes must be positive")
        if (
            abs(float(args.region_lateral_offset_m)) + float(args.region_half_width_m)
            > 0.5 * route_frame.surface_width_m + 1.0e-9
        ):
            raise ValueError("operator region must remain inside the metric route width")
        if not (
            float(args.region_half_length_m)
            < float(args.region_progress_m)
            < route_frame.route_length_m - float(args.region_half_length_m)
        ):
            raise ValueError("operator region must remain inside the route endpoints")
        region_center = route_frame.point(
            float(args.region_progress_m), float(args.region_lateral_offset_m)
        )
        region = Rect(
            float(region_center[0]),
            float(region_center[1]),
            region_hx,
            region_hy,
        )
        start_position = route_frame.point(0.0, float(args.start_lateral_offset_m))
        start_heading = route_frame.heading_rad + float(args.start_heading_offset_rad)
        backend = IsaacPolicyBackend(
            config,
            start_position,
            start_heading,
            front_cam=True,
        )
        print("[o2-lane] backend-created", flush=True)
        scene_prim = backend.load_realistic_scene(str(episode))
        print(f"[o2-lane] scene-loaded {scene_prim}", flush=True)
        lock_path = args.material_lock.resolve()
        lock = load_terrain_asset_lock(lock_path)
        asset_root = Path(lock["asset_root"])
        if not asset_root.is_absolute():
            asset_root = ROOT / asset_root
        binding = appearance_binding_from_record(
            {
                "material_family": args.material_id,
                "appearance_id": f"forest-{args.operator}-pair-{args.material_id}",
                "uv_scale": 1.0,
                "uv_rotation_deg": 0.0,
                "surface_state": "damp",
            },
            lock=lock,
            asset_root=asset_root,
        )
        backend.set_terrain_appearance(binding)
        operator = None
        if args.lane == "anomaly" or args.operator in {"o1", "o2", "o3"}:
            if args.operator == "o2":
                operator = ComplianceField(
                    region=region,
                    k_c=240.0,
                    c_c=0.32,
                    d_sink=0.11,
                    realistic_foot_model=True,
                    matched_control=args.lane == "nominal",
                    longitudinal_axis=(
                        "x"
                        if abs(route_frame.direction[0]) >= 1.0 - 1.0e-6
                        else "y"
                    ),
                )
            elif args.operator == "o1":
                nominal_mu_s, nominal_mu_d = backend.nominal_ground_friction()
                operator = MuField(
                    region=region,
                    mu_s=(
                        float(args.o1_mu_s)
                        if args.lane == "anomaly"
                        else float(nominal_mu_s)
                    ),
                    mu_d=(
                        float(args.o1_mu_d)
                        if args.lane == "anomaly"
                        else float(nominal_mu_d)
                    ),
                )
            else:
                operator = Collapse(
                    region=region,
                    mu_intact=0.8,
                    mu_collapsed=0.07,
                    damage_threshold_ns=(
                        float(args.o3_damage_threshold_ns)
                        if args.lane == "anomaly"
                        else 1.0e9
                    ),
                    drop_m=float(args.o3_drop_m),
                    residual_support=float(args.o3_residual_support),
                )
            operator.on_reset(backend)
        obs = backend.reset(int(args.seed))
        print("[o2-lane] backend-reset", flush=True)
        stage = omni.usd.get_context().get_stage()
        custom_floor_path = scene_binding.floor_prim_path(scene_prim)
        route_surface_path = scene_binding.route_surface_prim_path(scene_prim)
        captures = []
        rows = []
        max_position_error = 0.0
        max_orientation_error = 0.0
        local_pitch = np.asarray(
            [math.cos(pitch / 2.0), 0.0, math.sin(pitch / 2.0), 0.0],
            dtype=np.float64,
        )

        def capture(label: str, step: int) -> None:
            nonlocal max_position_error, max_orientation_error
            print(f"[o2-lane] capture-start {label} step={step}", flush=True)
            sensor = backend.capture_front_camera()
            if sensor is None:
                raise RuntimeError("Go2-front RTX camera returned no frame")
            rgb = np.asarray(sensor["rgb"])[..., :3].astype(np.uint8)
            base_pos = backend._robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
            base_quat = (
                backend._robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
            )
            expected_pos = base_pos + _quat_apply_wxyz(base_quat, mount)
            expected_quat = _quat_multiply_wxyz(base_quat, local_pitch)
            position_error = float(np.linalg.norm(np.asarray(sensor["pos"]) - expected_pos))
            orientation_error = _quat_distance(sensor["quat_world"], expected_quat)
            max_position_error = max(max_position_error, position_error)
            max_orientation_error = max(max_orientation_error, orientation_error)
            path = output / f"{label}.png"
            Image.fromarray(rgb).save(path, optimize=True)
            captures.append(
                {
                    "label": label,
                    "step": step,
                    "sim_time_s": float(obs.t),
                    "robot_position_xy_m": obs.pos.tolist(),
                    "robot_route_progress_m": route_frame.project(obs.pos)[0],
                    "robot_route_lateral_offset_m": route_frame.project(obs.pos)[1],
                    "robot_fallen": bool(obs.fallen),
                    "image": path.name,
                    "image_sha256": _sha256(path),
                    "camera_position_world_m": np.asarray(sensor["pos"]).tolist(),
                    "camera_quaternion_world_wxyz": np.asarray(sensor["quat_world"]).tolist(),
                    "mount_position_error_m": position_error,
                    "mount_orientation_error_rad": orientation_error,
                    "intrinsic_matrix": np.asarray(sensor["K"]).tolist(),
                    "metrics": _frame_metrics(rgb),
                }
            )
            print(f"[o2-lane] capture-done {label} step={step}", flush=True)

        capture("pre_entry", 0)
        inside_captured = False
        consequence_captured = False
        region_start_progress = float(args.region_progress_m) - float(
            args.region_half_length_m
        )
        region_end_progress = float(args.region_progress_m) + float(
            args.region_half_length_m
        )
        completion_progress = region_end_progress + 0.25
        for step in range(1, int(args.steps) + 1):
            command, controller_telemetry = route_controller.command(
                route_frame,
                position_xy_m=obs.pos,
                heading_rad=float(obs.heading),
                velocity_body_xy_mps=np.asarray(obs.vel_body[:2], dtype=np.float64),
                forward_speed_mps=float(args.forward_speed_mps),
                target_lateral_offset_m=float(args.target_lateral_offset_m),
            )
            obs = backend.step(command)
            transformed = operator.transform_obs(obs) if operator is not None else obs
            route_progress, route_lateral_offset = route_frame.project(obs.pos)
            foot_contact = backend.foot_contact_snapshot(
                region,
                region_margin_m=0.04,
                load_threshold_n=1.0,
            )
            rows.append(
                {
                    "step": step,
                    "time_s": float(obs.t),
                    "pos_xy_m": obs.pos.tolist(),
                    "route_progress_m": route_progress,
                    "route_lateral_offset_m": route_lateral_offset,
                    "heading_rad": float(obs.heading),
                    "base_height_m": float(obs.base_height),
                    "tilt_rad": float(obs.tilt),
                    "support_ratio": float(obs.support_ratio),
                    "slip_ratio": float(obs.slip_ratio),
                    "speed_mps": float(np.linalg.norm(obs.vel_body)),
                    "fallen": bool(obs.fallen),
                    "command_body_mps": command.tolist(),
                    "route_controller_telemetry": controller_telemetry,
                    "observed_tilt_rad": float(transformed.tilt),
                    "in_operator_region": bool(region.contains(obs.pos)),
                    "operator_region_foot_contact": foot_contact,
                }
            )
            inside_threshold = (
                region_start_progress - 0.20
                if args.operator == "o3"
                else region_start_progress + 0.01
            )
            if not inside_captured and route_progress >= inside_threshold:
                capture("inside_region", step)
                inside_captured = True
            o3_triggered = False
            if args.operator == "o3" and operator is not None:
                collapse_now = backend.collapse_telemetry()
                o3_triggered = bool(
                    collapse_now.get("enabled")
                    and collapse_now["regions"][0].get("collapsed")
                )
            matched_nominal_step = (
                args.operator == "o3"
                and args.lane == "nominal"
                and int(args.matched_consequence_step) > 0
                and step >= int(args.matched_consequence_step)
            )
            matched_position = args.operator != "o3" and route_progress >= (
                region_start_progress + 0.11
            )
            if not consequence_captured and (
                o3_triggered or matched_nominal_step or matched_position
            ):
                capture("matched_consequence", step)
                consequence_captured = True
            if obs.fallen or route_progress >= completion_progress:
                break
        if not consequence_captured:
            capture("matched_consequence", int(rows[-1]["step"]))

        telemetry_path = output / "telemetry.jsonl"
        with telemetry_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        if args.operator == "o2":
            operator_telemetry = backend.foot_compliance_telemetry()
        elif args.operator == "o1":
            operator_telemetry = backend.friction_topology_telemetry()
        else:
            operator_telemetry = backend.collapse_telemetry()
        custom_floor = stage.GetPrimAtPath(custom_floor_path)
        floor_enabled_attr = UsdPhysics.CollisionAPI(custom_floor).GetCollisionEnabledAttr()
        custom_floor_enabled = (
            bool(floor_enabled_attr.Get())
            if floor_enabled_attr.IsValid() and floor_enabled_attr.HasAuthoredValueOpinion()
            else True
        )
        route_surface = stage.GetPrimAtPath(route_surface_path)
        route_surface_visible = route_surface.IsValid() and (
            UsdGeom.Imageable(route_surface).ComputeVisibility() != UsdGeom.Tokens.invisible
        )
        region_rows = [row for row in rows if row["in_operator_region"]]
        median_height = float(
            np.median([row["base_height_m"] for row in region_rows])
        ) if region_rows else float("nan")
        max_sinkage = 0.0
        shear_work = 0.0
        loaded_feet = 0
        if args.operator == "o2" and operator_telemetry.get("enabled"):
            max_sinkage = max(
                float(foot["max_sinkage_m"]) for foot in operator_telemetry["feet"]
            )
            shear_work = sum(
                float(foot["shear_work_j"]) for foot in operator_telemetry["feet"]
            )
            loaded_feet = sum(
                int(foot["contact_steps"]) > 0 for foot in operator_telemetry["feet"]
            )
        max_region_slip = max(
            (float(row["slip_ratio"]) for row in region_rows),
            default=0.0,
        )
        mean_region_slip = float(
            np.mean([float(row["slip_ratio"]) for row in region_rows])
        ) if region_rows else 0.0
        runtime_gate = audit_route_lane_runtime(
            rows,
            lane=args.lane,
            maximum_route_deviation_m=admitted_tracking_budget,
            minimum_nominal_region_samples=int(args.min_region_samples),
        )
        maximum_absolute_route_lateral_offset = float(
            runtime_gate["measurements"]["maximum_absolute_route_lateral_offset_m"]
        )
        common_checks = {
            "compiled_scene_hash_verified": True,
            "three_event_aligned_frames": len(captures) == 3
            and {item["label"] for item in captures}
            == {"pre_entry", "inside_region", "matched_consequence"},
            "front_frames_non_degenerate": _front_sequence_non_degenerate(captures),
            "front_camera_position_rigid_mount": max_position_error <= 1.0e-4,
            "front_camera_orientation_rigid_mount": max_orientation_error <= 0.002,
            **runtime_gate["checks"],
        }
        if args.lane == "nominal":
            if args.operator == "o2":
                control_topology = operator_telemetry.get("collision_topology", {})
                support_material = control_topology.get("support_material_readback", {})
                nominal_material = control_topology.get("nominal_material", {})
                lane_checks = {
                    "nominal_scene_floor_disabled_for_matched_topology": not custom_floor_enabled,
                    "nominal_route_surface_visible": route_surface_visible,
                    "o2_matched_control_enabled": operator_telemetry.get("enabled") is True
                    and operator_telemetry.get("matched_control") is True,
                    "o2_matched_control_continuous_topology": control_topology.get("mode")
                    == "continuous_topology_matched_o2_heightfield_under_continuous_pbr"
                    and int(control_topology.get("collision_segment_count", 0)) == 1
                    and int(control_topology.get("inter_prim_collision_seam_count", -1)) == 0
                    and int(control_topology.get("vertex_count", 0)) >= 36
                    and int(control_topology.get("quad_count", 0)) >= 25,
                    "o2_matched_control_is_flat_nominal_material": control_topology.get(
                        "central_surface_state"
                    )
                    == "flat_nominal_material"
                    and all(
                        abs(
                            float(support_material.get(key, -1.0))
                            - float(nominal_material.get(key, -2.0))
                        )
                        <= 1.0e-6
                        for key in ("static", "dynamic")
                    )
                    and abs(float(control_topology.get("minimum_surface_height_m", -1.0)))
                    <= 1.0e-9,
                    "o2_matched_control_has_no_visual_boundary": control_topology.get(
                        "collision_geometry_visible"
                    )
                    is False
                    and control_topology.get(
                        "operator_region_has_no_precontact_visual_boundary"
                    )
                    is True,
                    "o2_matched_control_applies_no_soil_force_or_footprint": all(
                        float(foot.get("max_applied_force_n", 0.0)) == 0.0
                        for foot in operator_telemetry.get("feet", [])
                    )
                    and int(operator_telemetry.get("visual_footprint_count", 0)) == 0,
                    "nominal_route_completed": route_frame.project(obs.pos)[0]
                    >= completion_progress,
                    "nominal_robot_not_fallen": not obs.fallen,
                }
            elif args.operator == "o3":
                control_region = (
                    operator_telemetry["regions"][0]
                    if operator_telemetry.get("enabled")
                    else {}
                )
                lane_checks = {
                    "nominal_scene_floor_disabled_for_matched_topology": not custom_floor_enabled,
                    "nominal_route_surface_visible": route_surface_visible,
                    "o3_matched_control_topology_enabled": operator_telemetry.get("enabled")
                    is True
                    and control_region.get("mode") == "impulse_triggered_support_topology",
                    "o3_matched_control_never_triggered": control_region.get("collapsed")
                    is False,
                    "o3_matched_control_has_no_geometric_seams": float(
                        control_region.get("support_cell_gap_m", -1.0)
                    )
                    == 0.0,
                    "o3_matched_control_friction_preserved": abs(
                        float(control_region.get("nominal_requested_static_friction", -1.0))
                        - float(control_region.get("nominal_readback_static_friction", -2.0))
                    )
                    <= 1.0e-6
                    and abs(
                        float(control_region.get("nominal_requested_dynamic_friction", -1.0))
                        - float(control_region.get("nominal_readback_dynamic_friction", -2.0))
                    )
                    <= 1.0e-6,
                    "o3_matched_control_surface_not_predeformed": control_region.get(
                        "continuous_visual_surface", {}
                    ).get("surface_predeformed")
                    is False
                    and control_region.get("continuous_visual_surface", {}).get(
                        "visual_triggered"
                    )
                    is False,
                    "nominal_route_completed": route_frame.project(obs.pos)[0]
                    >= completion_progress,
                    "nominal_robot_not_fallen": not obs.fallen,
                }
            elif args.operator == "o1":
                lane_checks = {
                    "nominal_scene_floor_disabled_for_matched_topology": not custom_floor_enabled,
                    "nominal_route_surface_visible": route_surface_visible,
                    "o1_matched_control_topology_enabled": operator_telemetry.get("enabled")
                    is True
                    and operator_telemetry.get("mode")
                    == "coplanar_segmented_collision_under_continuous_pbr",
                    "o1_matched_control_uses_scene_nominal_friction": abs(
                        float(operator_telemetry.get("requested_static_friction", -1.0))
                        - float(operator_telemetry.get("nominal_requested_static_friction", -2.0))
                    )
                    <= 1.0e-6
                    and abs(
                        float(operator_telemetry.get("requested_dynamic_friction", -1.0))
                        - float(operator_telemetry.get("nominal_requested_dynamic_friction", -2.0))
                    )
                    <= 1.0e-6,
                    "o1_matched_control_readback_preserved": abs(
                        float(operator_telemetry.get("readback_static_friction", -1.0))
                        - float(operator_telemetry.get("nominal_readback_static_friction", -2.0))
                    )
                    <= 1.0e-6
                    and abs(
                        float(operator_telemetry.get("readback_dynamic_friction", -1.0))
                        - float(operator_telemetry.get("nominal_readback_dynamic_friction", -2.0))
                    )
                    <= 1.0e-6,
                    "o1_matched_control_collision_segments_complete": int(
                        operator_telemetry.get("collision_segment_count", 0)
                    )
                    == 5,
                    "o1_matched_control_geometry_hidden": operator_telemetry.get(
                        "collision_segment_visuals_hidden"
                    )
                    is True
                    and operator_telemetry.get("operator_collision_geometry_visible") is False,
                    "nominal_route_completed": route_frame.project(obs.pos)[0]
                    >= completion_progress,
                    "nominal_robot_not_fallen": not obs.fallen,
                }
            else:
                lane_checks = {
                    "nominal_scene_floor_enabled": custom_floor_enabled,
                    "nominal_route_surface_visible": route_surface_visible,
                    "no_target_terrain_operator": operator_telemetry.get("enabled") is False,
                    "nominal_route_completed": route_frame.project(obs.pos)[0]
                    >= completion_progress,
                    "nominal_robot_not_fallen": not obs.fallen,
                }
        elif args.operator == "o2":
            continuous_visual = operator_telemetry.get("continuous_visual_surface", {})
            anomaly_topology = operator_telemetry.get("collision_topology", {})
            operator_surface_visualization_valid = (
                (not route_surface_visible)
                or (
                    route_surface_visible
                    and continuous_visual.get("applied") is True
                    and continuous_visual.get("mode")
                    == "contact_local_load_triggered_imprints"
                    and continuous_visual.get("surface_predeformed") is False
                    and continuous_visual.get("operator_region_has_no_visual_boundary")
                    is True
                    and continuous_visual.get("operator_collision_geometry_visible") is False
                    and int(continuous_visual.get("contact_local_imprint_count", 0)) >= 4
                    and int(continuous_visual.get("deformed_vertex_count", 0)) > 0
                    and float(continuous_visual.get("minimum_visual_offset_m", 0.0)) <= -0.07
                )
            )
            lane_checks = {
                "nominal_scene_floor_disabled": not custom_floor_enabled,
                "operator_surface_visualization_valid": operator_surface_visualization_valid,
                "o2_terramechanics_enabled": operator_telemetry.get("enabled") is True,
                "o2_continuous_seam_free_topology": anomaly_topology.get("mode")
                == "continuous_topology_matched_o2_heightfield_under_continuous_pbr"
                and anomaly_topology.get("matched_control") is False
                and int(anomaly_topology.get("collision_segment_count", 0)) == 1
                and int(anomaly_topology.get("inter_prim_collision_seam_count", -1)) == 0
                and int(anomaly_topology.get("vertex_count", 0)) >= 36
                and int(anomaly_topology.get("quad_count", 0)) >= 25
                and float(anomaly_topology.get("minimum_surface_height_m", 0.0)) <= -0.07,
                "o2_nominal_surrounding_friction_preserved": abs(
                    float(
                        operator_telemetry.get("nominal_ground_friction", {}).get(
                            "static", -1.0
                        )
                    )
                    - float(
                        operator_telemetry.get("nominal_ground_friction", {}).get(
                            "readback_static", -2.0
                        )
                    )
                )
                <= 1.0e-6
                and abs(
                    float(
                        operator_telemetry.get("nominal_ground_friction", {}).get(
                            "dynamic", -1.0
                        )
                    )
                    - float(
                        operator_telemetry.get("nominal_ground_friction", {}).get(
                            "readback_dynamic", -2.0
                        )
                    )
                )
                <= 1.0e-6,
                "actual_foot_sinkage": max_sinkage >= 0.07,
                "multiple_loaded_feet": loaded_feet >= 2,
                "nonzero_shear_dissipation": shear_work > 0.02,
                "physical_footprints": operator_telemetry.get("visual_footprint_count", 0) >= 4,
            }
        elif args.operator == "o1":
            lane_checks = {
                "nominal_scene_floor_disabled": not custom_floor_enabled,
                "nominal_route_surface_visible": route_surface_visible,
                "o1_coplanar_topology_enabled": operator_telemetry.get("enabled") is True,
                "o1_coplanar_topology_mode": operator_telemetry.get("mode")
                == "coplanar_segmented_collision_under_continuous_pbr",
                "o1_friction_readback_matches_request": abs(
                    float(operator_telemetry.get("readback_static_friction", -1.0))
                    - float(args.o1_mu_s)
                )
                <= 1.0e-6
                and abs(
                    float(operator_telemetry.get("readback_dynamic_friction", -1.0))
                    - float(args.o1_mu_d)
                )
                <= 1.0e-6,
                "o1_nominal_surrounding_friction_preserved": abs(
                    float(operator_telemetry.get("nominal_requested_static_friction", -1.0))
                    - float(operator_telemetry.get("nominal_readback_static_friction", -2.0))
                )
                <= 1.0e-6
                and abs(
                    float(operator_telemetry.get("nominal_requested_dynamic_friction", -1.0))
                    - float(operator_telemetry.get("nominal_readback_dynamic_friction", -2.0))
                )
                <= 1.0e-6,
                "o1_collision_segments_complete": int(
                    operator_telemetry.get("collision_segment_count", 0)
                )
                == 5,
                "o1_collision_geometry_hidden": operator_telemetry.get(
                    "collision_segment_visuals_hidden"
                )
                is True
                and operator_telemetry.get("operator_collision_geometry_visible") is False,
                "continuous_pbr_surface_preserved": operator_telemetry.get(
                    "continuous_pbr_surface_visible"
                )
                is True
                and operator_telemetry.get("surface_predeformed") is False
                and operator_telemetry.get("operator_region_has_no_visual_boundary") is True,
                "measured_contact_slip": max_region_slip >= 0.05,
            }
        else:
            collapse_region = (
                operator_telemetry["regions"][0]
                if operator_telemetry.get("enabled")
                else {}
            )
            continuous_visual = collapse_region.get("continuous_visual_surface", {})
            damage = collapse_region.get("last_damage_update") or {}
            lane_checks = {
                "nominal_scene_floor_disabled": not custom_floor_enabled,
                "nominal_route_surface_visible": route_surface_visible,
                "o3_impulse_topology_enabled": operator_telemetry.get("enabled") is True
                and collapse_region.get("mode") == "impulse_triggered_support_topology",
                "o3_measured_impulse_triggered": collapse_region.get("collapsed") is True
                and float(damage.get("normal_impulse_ns", 0.0))
                >= float(args.o3_damage_threshold_ns),
                "o3_support_colliders_removed": int(
                    collapse_region.get("disabled_cell_colliders", 0)
                )
                == int(collapse_region.get("failed_support_cells", -1))
                and int(collapse_region.get("failed_support_cells", 0)) > 0,
                "o3_pretrigger_support_has_no_geometric_seams": float(
                    collapse_region.get("support_cell_gap_m", -1.0)
                )
                == 0.0,
                "o3_nominal_pretrigger_friction_preserved": abs(
                    float(collapse_region.get("nominal_requested_static_friction", -1.0))
                    - float(collapse_region.get("nominal_readback_static_friction", -2.0))
                )
                <= 1.0e-6
                and abs(
                    float(collapse_region.get("nominal_requested_dynamic_friction", -1.0))
                    - float(collapse_region.get("nominal_readback_dynamic_friction", -2.0))
                )
                <= 1.0e-6,
                "o3_visual_and_physics_trigger_synchronized": collapse_region.get(
                    "visual_sync_trigger_step"
                )
                == damage.get("trigger_step")
                == continuous_visual.get("visual_trigger_step"),
                "o3_collision_geometry_hidden": collapse_region.get(
                    "collision_geometry_visible"
                )
                is False
                and continuous_visual.get("operator_collision_geometry_visible") is False,
                "o3_continuous_pbr_triggered_deformation": continuous_visual.get(
                    "applied"
                )
                is True
                and continuous_visual.get("mode")
                == "measured_impulse_triggered_irregular_pbr_collapse"
                and continuous_visual.get("visual_triggered") is True
                and continuous_visual.get("surface_predeformed") is False
                and continuous_visual.get(
                    "operator_region_has_no_pretrigger_visual_boundary"
                )
                is True
                and int(continuous_visual.get("deformed_vertex_count", 0)) > 50
                and float(continuous_visual.get("minimum_visual_offset_m", 0.0)) <= -0.10,
            }
        checks = {**common_checks, **lane_checks}
        manifest = {
            "schema_version": (
                f"kinofail.realistic-route-{args.operator}-fresh-app-lane.v6-protocol-candidate"
            ),
            "scene_route_binding": {
                "source_kind": scene_binding.source_kind,
                "floor_prim_suffix": scene_binding.floor_prim_suffix,
                "route_surface_prim_suffix": scene_binding.route_surface_prim_suffix,
                "route_surface_collision_authored": (
                    scene_binding.route_surface_collision_authored
                ),
            },
            "created_utc": datetime.now(UTC).isoformat(),
            "process_id": os.getpid(),
            "development_only": True,
            "collector_lifecycle": collector_lifecycle,
            "lane": args.lane,
            "operator_family": args.operator,
            "scene_id": compiled["scene_id"],
            "episode_usd": str(episode),
            "episode_usd_sha256": _sha256(episode),
            "compiled_audit": str(compiled_path),
            "compiled_audit_sha256": _sha256(compiled_path),
            "seed": int(args.seed),
            "inference_reset_contract": backend.inference_reset_contract(),
            "camera": {
                "role": "actual body-fixed Go2 front RTX RGB",
                "profile": args.camera_profile,
                "calibration_status": "engineering_pending_physical_fixture_calibration",
                "mount_xyz_base_m": mount.tolist(),
                "pitch_down_rad": pitch,
                "focal_length_mm": float(camera["focal_length_mm"]),
                "resolution": [int(camera["width"]), int(camera["height"])],
                "max_mount_position_error_m": max_position_error,
                "max_mount_orientation_error_rad": max_orientation_error,
                "captures": captures,
            },
            "appearance": {
                "material_id": args.material_id,
                "material_lock": str(lock_path),
                "material_lock_sha256": _sha256(lock_path),
                "surface_state": "damp",
                "appearance_id": binding.appearance_id,
                "physics_independent": True,
            },
            "operator": {
                "id": (
                    (
                        "O2_compliance"
                        if args.operator == "o2"
                        else "O1_mu_field" if args.operator == "o1" else "O3_collapse"
                    )
                    if operator is not None
                    else None
                ),
                "active": args.lane == "anomaly" and operator is not None,
                "matched_counterfactual_topology_active": (
                    args.lane == "nominal"
                    and args.operator in {"o1", "o2", "o3"}
                    and operator is not None
                ),
                "runtime_control_parameters": (
                    {"damage_threshold_ns": 1.0e9, "transition_enabled": False}
                    if args.lane == "nominal" and args.operator == "o3"
                    else {
                        "surface_state": "flat_nominal_material",
                        "soil_force_enabled": False,
                        "visual_footprints_enabled": False,
                    }
                    if args.lane == "nominal" and args.operator == "o2"
                    else {
                        "central_patch_static_friction": operator_telemetry.get(
                            "readback_static_friction"
                        ),
                        "central_patch_dynamic_friction": operator_telemetry.get(
                            "readback_dynamic_friction"
                        ),
                        "friction_anomaly_enabled": False,
                    }
                    if args.lane == "nominal" and args.operator == "o1"
                    else {}
                ),
                "parameters": (
                    (
                        {
                            "sink_depth_m": 0.11,
                            "shear_retention": 0.32,
                            "vertical_stiffness_n_per_m": 240.0,
                            "region": {
                                "cx": region.cx,
                                "cy": region.cy,
                                "hx": region.hx,
                                "hy": region.hy,
                            },
                        }
                        if args.operator == "o2"
                        else {
                            "static_friction": float(args.o1_mu_s),
                            "dynamic_friction": float(args.o1_mu_d),
                            "region": {
                                "cx": region.cx,
                                "cy": region.cy,
                                "hx": region.hx,
                                "hy": region.hy,
                            },
                        }
                        if args.operator == "o1"
                        else {
                            "damage_threshold_ns": float(args.o3_damage_threshold_ns),
                            "drop_m": float(args.o3_drop_m),
                            "residual_support": float(args.o3_residual_support),
                            "intact_friction": 0.8,
                            "collapsed_friction": 0.07,
                            "region": {
                                "cx": region.cx,
                                "cy": region.cy,
                                "hx": region.hx,
                                "hy": region.hy,
                            },
                        }
                    )
                    if operator is not None
                    else {}
                ),
                "telemetry": operator_telemetry,
            },
            "route_controller": {
                "frame": {
                    "origin_xy_m": list(route_frame.origin_xy_m),
                    "direction_xy": list(route_frame.direction_xy),
                    "left_xy": list(route_frame.left_xy),
                    "heading_rad": route_frame.heading_rad,
                    "route_length_m": route_frame.route_length_m,
                    "surface_width_m": route_frame.surface_width_m,
                },
                "start_position_xy_m": start_position.tolist(),
                "start_lateral_offset_m": float(args.start_lateral_offset_m),
                "start_heading_rad": start_heading,
                "start_heading_offset_rad": float(args.start_heading_offset_rad),
                "target_lateral_offset_m": float(args.target_lateral_offset_m),
                "forward_speed_mps": float(args.forward_speed_mps),
                "controller_contract": route_controller.contract(),
                "admitted_maximum_route_deviation_m": admitted_tracking_budget,
                "minimum_nominal_operator_region_samples": int(args.min_region_samples),
                "anomaly_region_coverage_rule": (
                    "at_least_one_load_bearing_region_contact"
                ),
                "yaw_rate_limit_radps": float(args.yaw_rate_limit_radps),
                "operator_region_progress_m": float(args.region_progress_m),
                "operator_region_lateral_offset_m": float(
                    args.region_lateral_offset_m
                ),
                "operator_region_half_length_m": float(args.region_half_length_m),
                "operator_region_half_width_m": float(args.region_half_width_m),
                "completion_progress_m": completion_progress,
                "matched_consequence_step_requested": (
                    int(args.matched_consequence_step)
                    if int(args.matched_consequence_step) > 0
                    else None
                ),
            },
            "measurements": {
                "steps": len(rows),
                "final_position_xy_m": obs.pos.tolist(),
                "final_route_progress_m": route_frame.project(obs.pos)[0],
                "final_route_lateral_offset_m": route_frame.project(obs.pos)[1],
                "maximum_absolute_route_lateral_offset_m": (
                    maximum_absolute_route_lateral_offset
                ),
                "fallen": bool(obs.fallen),
                "region_sample_count": len(region_rows),
                "load_bearing_region_sample_count": runtime_gate["measurements"][
                    "load_bearing_operator_region_sample_count"
                ],
                "median_region_base_height_m": median_height,
                "max_foot_sinkage_m": max_sinkage,
                "total_shear_work_j": shear_work,
                "loaded_feet": loaded_feet,
                "max_region_slip_ratio": max_region_slip,
                "mean_region_slip_ratio": mean_region_slip,
                "custom_scene_floor_enabled": custom_floor_enabled,
                "route_surface_visible": route_surface_visible,
                "collapse_triggered": (
                    bool(operator_telemetry["regions"][0]["collapsed"])
                    if args.operator == "o3" and operator_telemetry.get("enabled")
                    else False
                ),
                "collapse_trigger_step": (
                    operator_telemetry["regions"][0]["visual_sync_trigger_step"]
                    if args.operator == "o3" and operator_telemetry.get("enabled")
                    else None
                ),
            },
            "telemetry": telemetry_path.name,
            "telemetry_sha256": _sha256(telemetry_path),
            "checks": checks,
            "passed": all(checks.values()),
            "scene_registry_eligible": False,
            "counts_as_a0_a7_evidence": False,
            "remaining_gates": [
                f"{collector_lifecycle}_source_freeze",
                "paired_lane_audit",
                "three_synchronized_appearance_views",
                "physical_go2_fixture_camera_calibration",
                "independent_human_review",
                "formal_schedule_binding",
            ],
            "provenance": {
                "locomotion_policy": str(policy_path),
                "locomotion_policy_sha256": _sha256(policy_path),
                "script": str(collector_script),
                "script_sha256": _sha256(collector_script),
                "collector_engine": str(Path(__file__).resolve()),
                "collector_engine_sha256": _sha256(Path(__file__).resolve()),
                "route_protocol_sha256": _sha256(
                    ROOT / "kino_vla/sim/realistic_route_protocol_v2.py"
                ),
                "route_controller_sha256": _sha256(
                    route_controller_source
                ),
                "backend_sha256": _sha256(ROOT / "kino_vla/sim/isaac_policy_backend.py"),
                "operator_sha256": _sha256(
                    ROOT
                    / "kino_vla/sim/operators"
                    / (
                        "o2_compliance.py"
                        if args.operator == "o2"
                        else "o1_mu_field.py" if args.operator == "o1" else "o3_collapse.py"
                    )
                ),
                "command": " ".join(sys.argv),
                "python": sys.executable,
            },
        }
        manifest_path = output / "lane_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "passed": manifest["passed"],
                    "lane": args.lane,
                    "checks": checks,
                    "measurements": manifest["measurements"],
                    "manifest": str(manifest_path),
                },
                indent=2,
                default=_json_default,
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
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
