#!/usr/bin/env python3
"""Capture one provenance-registered Fig. 2 case from one frozen Isaac state.

The paper figure must never pair a Go2-front frame with an explanatory image from a
different scene or rollout.  This script therefore uses exactly one RTX sensor:

1. run one operator in one realistic Kino-Fail scene;
2. pause the simulation at a mechanism-active state;
3. render the calibrated body-fixed Go2-front view;
4. move only that camera to two low external contact views;
5. verify that the complete robot state is byte-identical across all renders.

The outputs are publication visualizations, not additional benchmark trials.
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
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OPERATORS = tuple(f"O{index}" for index in range(1, 12))
SCENE_BY_OPERATOR = {
    "O1": "forest_river_walk_metric_g07",
    "O2": "forest_river_walk_metric_g07",
    "O3": "forest_river_walk_metric_g07",
    "O4": "indoor_kitchen_140",
    "O5": "indoor_kitchen_140",
    "O6": "indoor_office_186",
    "O7": "forest_slope_metric_g06",
    "O8": "indoor_kitchen_140",
    "O9": "indoor_office_186",
    "O10": "indoor_office_186",
    "O11": "indoor_office_186",
}
OPERATOR_NAMES = {
    "O1": "O1_mu_field",
    "O2": "O2_compliance",
    "O3": "O3_collapse",
    "O4": "O4_tether",
    "O5": "O5_payload",
    "O6": "O6_push",
    "O7": "O7_visual_remap",
    "O8": "O8_invisible_collider",
    "O9": "O9_high_centering",
    "O10": "O10_effort_decay",
    "O11": "O11_obs_bias",
}
SEVERE_PARAMETERS: dict[str, dict[str, Any]] = {
    "O1": {"mu_s": 0.10, "mu_d": 0.07, "restitution": 0.0},
    "O2": {"stiffness_n_per_m": 240.0, "shear_gain": 0.32, "sink_depth_m": 0.11},
    "O3": {"damage_threshold_ns": 25.0, "drop_m": 0.12, "residual_support": 0.18},
    "O4": {
        "tangential_force_cap_n": 24.0,
        "normal_force_cap_n": 10.0,
        "peel_height_m": 0.025,
        "unload_steps_to_peel": 3,
    },
    "O5": {"mass_kg": 14.0, "com_offset_x_m": 0.12, "com_offset_y_m": 0.06},
    "O6": {
        "impulse_ns": 12.0,
        "duration_s": 0.12,
        "application_point_body_m": [0.0, -0.085, 0.075],
    },
    "O7": {"mu_s": 0.09, "mu_d": 0.06, "depth_bias_m": 0.32},
    "O8": {
        "obstacle_height_m": 0.38,
        "collision_enabled": True,
        "optical_transmission": 0.98,
    },
    "O9": {"residual_support": 0.22, "ridge_height_m": 0.36, "ridge_width_m": 0.14},
    "O10": {"decay_rate_per_s": 0.90, "effort_floor": 0.20, "onset_s": 1.0},
    "O11": {
        "tilt_bias_rad": 0.35,
        "random_walk_rad_sqrt_s": 0.02,
        "latency_s": 0.20,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _region(frame, operator_id: str):
    from kino_vla.utils.geometry import Rect

    # The realistic O2 model replaces the nominal floor with per-foot terrain support.  Its
    # region must therefore include the reset footprint; otherwise a CPU-PhysX visualization
    # rollout falls before the first valid terrain contact.
    half_length = 0.75 if operator_id == "O2" else 0.30
    requested_progress = 0.45 if operator_id == "O2" else 0.72
    progress = min(requested_progress, frame.route_length_m - half_length - 0.10)
    progress = max(progress, half_length + 0.10)
    if operator_id == "O2":
        progress = 0.45
    half_width = min(0.42, 0.5 * frame.surface_width_m - 0.05)
    center = frame.point(progress, 0.0)
    if abs(float(frame.direction[0])) >= 1.0 - 1.0e-6:
        return Rect(float(center[0]), float(center[1]), half_length, half_width)
    if abs(float(frame.direction[1])) >= 1.0 - 1.0e-6:
        return Rect(float(center[0]), float(center[1]), half_width, half_length)
    raise ValueError("Fig. 2 capture requires an axis-aligned registered route")


def _irregular(rect):
    from kino_vla.sim.adhesion import IrregularRegion

    return IrregularRegion(
        (
            (rect.cx - rect.hx, rect.cy - rect.hy),
            (rect.cx + rect.hx, rect.cy - rect.hy),
            (rect.cx + rect.hx, rect.cy + rect.hy),
            (rect.cx - rect.hx, rect.cy + rect.hy),
        )
    )


def _install_adhesion_film_visual(region) -> dict[str, Any]:
    """Author a non-colliding protective-film asset over the physical O4 footprint.

    The film is an explanatory appearance, not the contact implementation: O4
    mechanics remain entirely in ``SurfaceAdhesionConfig``.  Directly authoring
    the asset into the already loaded stage keeps the front, contact, and
    overview renders within the same scene composition and frozen rollout.
    """
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()
    root = "/World/KinoFig2O4Film"
    UsdGeom.Xform.Define(stage, root)
    UsdGeom.Scope.Define(stage, f"{root}/Looks")
    UsdGeom.Xform.Define(stage, f"{root}/Geometry")

    def material(
        name: str,
        color: tuple[float, float, float],
        *,
        roughness: float,
        opacity: float,
        emissive: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        mat = UsdShade.Material.Define(stage, f"{root}/Looks/{name}")
        shader = UsdShade.Shader.Define(stage, f"{root}/Looks/{name}/PBR")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*emissive)
        )
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    film_material = material(
        "PearlProtectiveFilm",
        (0.82, 0.84, 0.78),
        roughness=0.58,
        opacity=0.94,
        emissive=(0.035, 0.038, 0.032),
    )
    edge_material = material(
        "FilmCrease",
        (0.32, 0.35, 0.33),
        roughness=0.76,
        opacity=1.0,
    )

    width = 2.0 * float(region.hx)
    length = 2.0 * float(region.hy)
    # Slightly overlapping, rotated sheets produce a legible thin-film
    # silhouette without introducing a diagnostic colour patch.
    panels = (
        (0.00, 0.00, 0.82, 0.82, -2.5, 0.0110),
        (-0.34, -0.03, 0.28, 0.66, -7.0, 0.0114),
        (0.34, 0.02, 0.27, 0.64, 6.0, 0.0118),
        (0.00, 0.37, 0.62, 0.20, -3.5, 0.0122),
        (0.02, -0.37, 0.58, 0.19, 4.0, 0.0126),
    )
    authored_paths = []
    for index, (ox, oy, sx, sy, yaw, z) in enumerate(panels):
        path = f"{root}/Geometry/FilmPanel_{index}"
        panel = UsdGeom.Cube.Define(stage, path)
        panel.CreateSizeAttr(1.0)
        xform = UsdGeom.XformCommonAPI(panel)
        xform.SetTranslate(
            Gf.Vec3d(
                float(region.cx) + ox * width,
                float(region.cy) + oy * length,
                z,
            )
        )
        xform.SetRotate(Gf.Vec3f(0.0, 0.0, yaw))
        xform.SetScale(Gf.Vec3f(sx * width, sy * length, 0.006))
        UsdShade.MaterialBindingAPI.Apply(panel.GetPrim()).Bind(film_material)
        panel.GetPrim().CreateAttribute(
            "kino:semanticClass", Sdf.ValueTypeNames.String
        ).Set("floor_protective_film")
        authored_paths.append(path)

    for index, (ox, oy, sx, sy, yaw) in enumerate(
        (
            (-0.10, -0.16, 0.66, 0.018, -5.0),
            (0.08, 0.18, 0.76, 0.015, 4.0),
            (0.27, -0.07, 0.36, 0.020, 9.0),
        )
    ):
        path = f"{root}/Geometry/FilmCrease_{index}"
        crease = UsdGeom.Cube.Define(stage, path)
        crease.CreateSizeAttr(1.0)
        xform = UsdGeom.XformCommonAPI(crease)
        xform.SetTranslate(
            Gf.Vec3d(
                float(region.cx) + ox * width,
                float(region.cy) + oy * length,
                0.0160,
            )
        )
        xform.SetRotate(Gf.Vec3f(0.0, 0.0, yaw))
        xform.SetScale(Gf.Vec3f(sx * width, sy * length, 0.0015))
        UsdShade.MaterialBindingAPI.Apply(crease.GetPrim()).Bind(edge_material)
        authored_paths.append(path)

    return {
        "role": "non_colliding_explanatory_appearance",
        "semantic_class": "floor_protective_film",
        "physical_contact_implementation": "SurfaceAdhesionConfig",
        "footprint_bounds_xy_m": [
            float(region.cx - region.hx),
            float(region.cx + region.hx),
            float(region.cy - region.hy),
            float(region.cy + region.hy),
        ],
        "authored_prim_paths": authored_paths,
        "collision_api_authored": False,
    }


def _install_operator(operator_id: str, backend, frame, seed: int):
    from kino_vla.sim.adhesion_v3 import SurfaceAdhesionConfig
    from kino_vla.sim.operators import (
        Collapse,
        ComplianceField,
        EffortDecay,
        HighCentering,
        InvisibleCollider,
        MuField,
        ObsBias,
        Payload,
        Push,
        VisualPhysicsRemap,
    )

    p = SEVERE_PARAMETERS[operator_id]
    region = _region(frame, operator_id)
    operator = None
    if operator_id == "O1":
        operator = MuField(region, p["mu_s"], p["mu_d"], p["restitution"])
    elif operator_id == "O2":
        operator = ComplianceField(
            region,
            p["stiffness_n_per_m"],
            p["shear_gain"],
            p["sink_depth_m"],
            realistic_foot_model=True,
            longitudinal_axis="x" if abs(float(frame.direction[0])) > 0.9 else "y",
        )
    elif operator_id == "O3":
        operator = Collapse(
            region,
            0.07,
            mu_intact=0.8,
            damage_threshold_ns=p["damage_threshold_ns"],
            drop_m=p["drop_m"],
            residual_support=p["residual_support"],
        )
    elif operator_id == "O4":
        backend.add_foot_adhesion(
            SurfaceAdhesionConfig(
                region=_irregular(region),
                surface_z_m=0.0,
                attach_contact_force_n=5.0,
                detach_contact_force_n=2.0,
                attach_height_tolerance_m=0.04,
                tangential_stiffness_n_per_m=120.0,
                tangential_damping_ns_per_m=2.0,
                normal_stiffness_n_per_m=80.0,
                normal_damping_ns_per_m=1.0,
                tangential_force_cap_n=p["tangential_force_cap_n"],
                normal_force_cap_n=p["normal_force_cap_n"],
                peel_height_m=p["peel_height_m"],
                unload_steps_to_peel=int(p["unload_steps_to_peel"]),
                reattach_cooldown_steps=2,
                max_active_feet=2,
                progress_axis_xy=tuple(float(value) for value in frame.direction),
            )
        )
    elif operator_id == "O5":
        operator = Payload(
            p["mass_kg"],
            (p["com_offset_x_m"], p["com_offset_y_m"], 0.14),
        )
    elif operator_id == "O6":
        operator = Push(
            frame.left * float(p["impulse_ns"]),
            2.0,
            duration_s=p["duration_s"],
            application_point_body_m=np.asarray(p["application_point_body_m"]),
        )
    elif operator_id == "O7":
        operator = VisualPhysicsRemap(
            region,
            p["mu_s"],
            p["mu_d"],
            p["depth_bias_m"],
        )
    elif operator_id == "O8":
        operator = InvisibleCollider(
            region,
            height_m=p["obstacle_height_m"],
            collision_enabled=p["collision_enabled"],
            geometry_kind="transparent_acrylic",
            optical_transmission=p["optical_transmission"],
        )
    elif operator_id == "O9":
        operator = HighCentering(
            region,
            p["residual_support"],
            ridge_height_m=p["ridge_height_m"],
            ridge_width_m=p["ridge_width_m"],
            geometry_kind="pallet_edge",
        )
    elif operator_id == "O10":
        operator = EffortDecay(
            p["decay_rate_per_s"],
            p["effort_floor"],
            p["onset_s"],
        )
    elif operator_id == "O11":
        operator = ObsBias(
            {"tilt": p["tilt_bias_rad"]},
            random_walk_per_sqrt_s={"tilt": p["random_walk_rad_sqrt_s"]},
            latency_s=p["latency_s"],
            seed=seed,
        )
    if operator is not None:
        operator.on_reset(backend)
    return operator, region


def _telemetry(operator_id: str, backend, operator) -> dict[str, Any]:
    if operator_id in {"O1", "O7"}:
        return backend.friction_topology_telemetry()
    if operator_id == "O2":
        return backend.foot_compliance_telemetry()
    if operator_id == "O3":
        return backend.collapse_telemetry()
    if operator_id == "O4":
        return backend.adhesion_telemetry()
    if operator_id == "O5":
        return backend.payload_telemetry()
    if operator_id == "O6":
        return backend.push_telemetry()
    if operator_id == "O8":
        return backend.blocking_telemetry()
    if operator_id == "O9":
        return backend.high_centering_telemetry()
    if operator_id == "O10":
        return backend.actuator_telemetry()
    if operator_id == "O11":
        return operator.sensor_telemetry()
    raise ValueError(operator_id)


def _capture_ready(
    operator_id: str,
    *,
    obs,
    progress_m: float,
    region_progress_m: float,
    telemetry: dict[str, Any],
    operator,
) -> bool:
    inside_or_past = progress_m >= region_progress_m - 0.02
    if operator_id in {"O1", "O7"}:
        return inside_or_past and float(obs.t) >= 2.0
    if operator_id == "O2":
        contact_steps = sum(int(row.get("contact_steps", 0)) for row in telemetry.get("feet", []))
        return contact_steps >= 8
    if operator_id == "O3":
        return any(row.get("collapsed") is True for row in telemetry.get("regions", []))
    if operator_id == "O4":
        # A foot can attach at the leading edge of the registered region, before the base reaches
        # its centre.  Freeze that first physically active state instead of waiting until the
        # attachment has already destabilized the body.
        return int(telemetry.get("total_attachment_cycles", 0)) > 0
    if operator_id == "O5":
        # The severe registered payload is immediately active and can topple the robot before
        # 2.5 s.  A 0.6 s state shows the attached mass while the robot is still inspectable.
        return float(obs.t) >= 0.6 and float(telemetry.get("payload_kg", 0.0)) > 0.0
    if operator_id == "O6":
        return bool(operator and operator.fired) and int(telemetry.get("applied_steps", 0)) >= 2
    if operator_id == "O8":
        return (
            float(obs.t) >= 1.5
            and progress_m >= region_progress_m - 0.70
            and telemetry.get("enabled") is True
        )
    if operator_id == "O9":
        from kino_vla.data.o9_semantics import evaluate_o9_high_centering

        route_axis = (
            "x"
            if abs(float(telemetry.get("regions", [{}])[0].get("region", {}).get("hx", 0.0)))
            >= abs(float(telemetry.get("regions", [{}])[0].get("region", {}).get("hy", 0.0)))
            else "y"
        )
        return evaluate_o9_high_centering(
            telemetry,
            robot_position_xy_m=obs.pos,
            route_axis=route_axis,
        )["passed"]
    if operator_id == "O10":
        return float(telemetry.get("effort_scale", 1.0)) <= 0.30
    if operator_id == "O11":
        return float(obs.t) >= 0.8 and telemetry.get("raw_pipeline_active") is True
    return False


def _state_vector(backend) -> np.ndarray:
    tensors = (
        backend._robot.data.root_state_w[0],
        backend._robot.data.joint_pos[0],
        backend._robot.data.joint_vel[0],
    )
    return np.concatenate(
        [tensor.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1) for tensor in tensors]
    )


def _state_sha256(state: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(state, dtype=np.float64).tobytes()).hexdigest()


def _set_front_pose_on_shared_camera(backend) -> dict[str, list[float]]:
    from isaaclab.utils.math import quat_apply, quat_mul

    camera = backend._front_camera
    if camera is None:
        raise RuntimeError("front camera was not initialized")
    base_pos = backend._robot.data.root_pos_w[0:1]
    base_quat = backend._robot.data.root_quat_w[0:1]
    offset = backend._torch.tensor(
        backend._front_camera_offset_m.reshape(1, 3),
        dtype=base_pos.dtype,
        device=backend._device,
    )
    camera_pos = base_pos + quat_apply(base_quat, offset)
    half_pitch = 0.5 * backend._front_camera_pitch_down_rad
    local_pitch = backend._torch.tensor(
        [[math.cos(half_pitch), 0.0, math.sin(half_pitch), 0.0]],
        dtype=base_quat.dtype,
        device=backend._device,
    )
    camera_quat = quat_mul(base_quat, local_pitch)
    camera.set_world_poses(camera_pos, camera_quat, convention="world")
    return {
        "position_xyz_m": camera_pos[0].detach().cpu().numpy().astype(float).tolist(),
        "quaternion_wxyz": camera_quat[0].detach().cpu().numpy().astype(float).tolist(),
    }


def _capture_shared_camera(backend, app, *, minimum_std: float = 8.0) -> np.ndarray:
    for _ in range(4):
        app.update()
    backend._front_camera.update(backend.dt, force_recompute=True)
    rgb = backend._front_camera.data.output["rgb"][0][..., :3]
    if rgb.dtype.is_floating_point:
        rgb = rgb.clamp(0.0, 1.0) * 255.0
    frame = rgb.to(backend._torch.uint8).cpu().numpy()
    if float(frame.std()) < minimum_std:
        raise RuntimeError("invalid low-variance RTX frame")
    return frame


def _set_external_pose(backend, eye_xyz: np.ndarray, target_xyz: np.ndarray) -> None:
    origin = backend._env.scene.env_origins[0]
    eye = backend._torch.tensor(
        [[float(value) for value in eye_xyz]],
        dtype=origin.dtype,
        device=backend._device,
    ) + origin
    target = backend._torch.tensor(
        [[float(value) for value in target_xyz]],
        dtype=origin.dtype,
        device=backend._device,
    ) + origin
    backend._front_camera.set_world_poses_from_view(eye, target)


def main() -> int:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--operator", required=True, choices=OPERATORS)
    preliminary.add_argument(
        "--out",
        default="outputs/kinofail_realistic/fig2_registered_multiview_v1",
    )
    pre, _ = preliminary.parse_known_args()

    parser = argparse.ArgumentParser(description="Capture one registered Fig. 2 multiview case")
    parser.add_argument("--operator", required=True, choices=OPERATORS)
    parser.add_argument("--out", default=pre.out)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app

    passed = False
    try:
        import omni.timeline
        import yaml
        from PIL import Image

        from kino_vla.sim.isaac_o4_v3_backend import IsaacPolicyBackendO4V3
        from kino_vla.sim.realistic_route_controller_v3 import DampedRouteControllerV3
        from kino_vla.sim.realistic_route_protocol_v2 import scene_route_binding
        from kino_vla.utils.config import CONFIGS_DIR, load_config
        from kino_vla.utils.seeding import seed_everything

        operator_id = str(args.operator)
        scene_id = SCENE_BY_OPERATOR[operator_id]
        registry_path = ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json"
        registry = _json(registry_path)
        scene = next(row for row in registry["scenes"] if row["scene_id"] == scene_id)
        episode_path = ROOT / scene["episode_usd"]
        compiled_path = ROOT / scene["compiled_audit"]
        if _sha256(episode_path) != scene["episode_sha256"]:
            raise RuntimeError("scene episode hash mismatch")
        if _sha256(compiled_path) != scene["compiled_audit_sha256"]:
            raise RuntimeError("scene compiled-audit hash mismatch")
        compiled = _json(compiled_path)
        binding = scene_route_binding(compiled)
        frame = binding.frame

        camera_specs = yaml.safe_load(
            (CONFIGS_DIR / "data/kinofail_realistic.yaml").read_text(encoding="utf-8")
        )["camera_profile_specs"]["go2_front_calib_b"]
        mount = camera_specs["mount_xyz_m"]
        sim_cfg = load_config(
            "sim/go2_skeleton.yaml",
            {
                "cam_width": int(args.width),
                "cam_height": int(args.height),
                "device": str(args.device),
                "front_cam_focal_mm": float(camera_specs["focal_length_mm"]),
                "front_cam_aperture_mm": 20.955,
                "front_cam_x_m": float(mount[0]),
                "front_cam_y_m": float(mount[1]),
                "front_cam_z_m": float(mount[2]),
                "front_cam_pitch_down_rad": float(camera_specs["pitch_down_rad"]),
            },
        )
        seed = 202700 + int(operator_id[1:])
        seed_everything(seed)
        start_xy = frame.point(0.0, 0.0)
        if operator_id == "O9":
            o9_region = _region(frame, operator_id)
            start_xy = np.asarray([o9_region.cx, o9_region.cy], dtype=np.float64)
        backend = IsaacPolicyBackendO4V3(
            sim_cfg,
            start_xy,
            frame.heading_rad,
            front_cam=True,
        )
        if operator_id == "O9":
            # O9 attribution starts from an already-beached state.  The matched
            # high-centering physics gate establishes that 0.43 m gives the
            # chassis 13 mm clearance before gravity loads the 0.36 m runner.
            backend._spawn_z = 0.43
        scene_prim = backend.load_realistic_scene(str(episode_path))
        backend.deep_reset(seed)
        operator, region = _install_operator(operator_id, backend, frame, seed)
        visual_explanation = (
            _install_adhesion_film_visual(region) if operator_id == "O4" else None
        )
        obs = backend.reset(seed, preserve_settle_telemetry=(operator_id == "O9"))
        if operator_id == "O11" and operator is not None:
            operator.on_reset(backend)

        controller = DampedRouteControllerV3(
            cross_track_gain_per_s=2.0,
            lateral_velocity_damping=1.0,
            heading_gain_per_s=2.0,
            lateral_limit_mps=0.35,
            yaw_rate_limit_radps=0.6,
        )
        region_progress, _ = frame.project(np.asarray([region.cx, region.cy], dtype=np.float64))
        measured = obs
        telemetry: dict[str, Any] = {}
        capture_reason = ""
        for step in range(460):
            if operator is not None:
                operator.on_step(backend, float(obs.t))
            command, _ = controller.command(
                frame,
                position_xy_m=obs.pos,
                heading_rad=obs.heading,
                velocity_body_xy_mps=obs.vel_body,
                forward_speed_mps=0.26,
                target_lateral_offset_m=0.0,
            )
            obs = backend.step(command)
            measured = operator.transform_obs(obs) if operator is not None else obs
            telemetry = _telemetry(operator_id, backend, operator)
            progress, _ = frame.project(obs.pos)
            if _capture_ready(
                operator_id,
                obs=obs,
                progress_m=progress,
                region_progress_m=region_progress,
                telemetry=telemetry,
                operator=operator,
            ):
                capture_reason = f"mechanism_active_at_step_{step}"
                break
            if obs.fallen:
                raise RuntimeError(f"{operator_id} fell before a registered capture state")
        else:
            raise RuntimeError(f"{operator_id} did not reach a registered capture state")

        semantic_validation = None
        if operator_id == "O9":
            from kino_vla.data.o9_semantics import evaluate_o9_high_centering

            semantic_validation = evaluate_o9_high_centering(
                telemetry,
                robot_position_xy_m=obs.pos,
                route_axis="x" if abs(float(frame.direction[0])) >= 0.9 else "y",
            )
            if not semantic_validation["passed"]:
                raise RuntimeError("O9 capture reached no valid belly-on-ridge state")

        output_dir = (ROOT / str(args.out) / operator_id).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        state_before = _state_vector(backend)
        state_hash = _state_sha256(state_before)

        front_pose = _set_front_pose_on_shared_camera(backend)
        # A body-fixed frame can legitimately be dominated by a plain floor during a severe
        # perturbation.  It remains provenance evidence, while the publication panel uses the
        # registered external views that must pass the stricter variance gate below.
        front = _capture_shared_camera(backend, app, minimum_std=2.0)
        front_path = output_dir / "01_go2_front.png"
        Image.fromarray(front).save(front_path)

        robot_xy = np.asarray(obs.pos, dtype=np.float64)
        views = []
        for name, side in (("02_contact_left", 1.0), ("03_contact_right", -1.0)):
            eye_xy = robot_xy - 0.48 * frame.direction + side * 0.82 * frame.left
            eye = np.asarray([float(eye_xy[0]), float(eye_xy[1]), 0.54], dtype=np.float64)
            target_xy = robot_xy + 0.12 * frame.direction
            target = np.asarray(
                [float(target_xy[0]), float(target_xy[1]), 0.18],
                dtype=np.float64,
            )
            _set_external_pose(backend, eye, target)
            image = _capture_shared_camera(backend, app)
            path = output_dir / f"{name}.png"
            Image.fromarray(image).save(path)
            views.append(
                {
                    "role": "registered_low_contact_view",
                    "path": str(path.relative_to(ROOT)),
                    "sha256": _sha256(path),
                    "eye_xyz_m": eye.tolist(),
                    "target_xyz_m": target.tolist(),
                }
            )

        overview_eye_xy = robot_xy - 1.15 * frame.direction + 1.05 * frame.left
        overview_eye = np.asarray(
            [float(overview_eye_xy[0]), float(overview_eye_xy[1]), 1.20],
            dtype=np.float64,
        )
        overview_target_xy = robot_xy + 0.10 * frame.direction
        overview_target = np.asarray(
            [float(overview_target_xy[0]), float(overview_target_xy[1]), 0.22],
            dtype=np.float64,
        )
        _set_external_pose(backend, overview_eye, overview_target)
        overview = _capture_shared_camera(backend, app)
        overview_path = output_dir / "04_overview.png"
        Image.fromarray(overview).save(overview_path)
        views.append(
            {
                "role": "registered_scene_overview",
                "path": str(overview_path.relative_to(ROOT)),
                "sha256": _sha256(overview_path),
                "eye_xyz_m": overview_eye.tolist(),
                "target_xyz_m": overview_target.tolist(),
            }
        )

        state_after = _state_vector(backend)
        max_state_delta = float(np.max(np.abs(state_after - state_before)))
        after_hash = _state_sha256(state_after)
        if after_hash != state_hash or max_state_delta != 0.0:
            raise RuntimeError(
                f"physics changed during camera-only renders: delta={max_state_delta:g}"
            )
        case_id = f"fig2-{operator_id.lower()}-{scene_id}-seed{seed}"
        manifest = {
            "schema_version": "kinofail.fig2-registered-multiview-case.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "purpose": "publication_visualization_not_additional_benchmark_trial",
            "case_id": case_id,
            "operator": {
                "id": OPERATOR_NAMES[operator_id],
                "short_id": operator_id,
                "severity": "severe",
                "parameters": SEVERE_PARAMETERS[operator_id],
            },
            "visual_explanation": visual_explanation,
            "scene": {
                "scene_id": scene_id,
                "domain": scene["domain"],
                "split": scene["split"],
                "room_family": scene["room_family"],
                "episode_usd": str(episode_path.relative_to(ROOT)),
                "episode_sha256": _sha256(episode_path),
                "compiled_audit": str(compiled_path.relative_to(ROOT)),
                "compiled_audit_sha256": _sha256(compiled_path),
                "loaded_prim": scene_prim,
            },
            "simulation": {
                "seed": seed,
                "timestamp_s": float(obs.t),
                "capture_reason": capture_reason,
                "robot_position_xy_m": [float(value) for value in obs.pos],
                "robot_heading_rad": float(obs.heading),
                "robot_tilt_rad": float(obs.tilt),
                "robot_base_height_m": float(obs.base_height),
                "fallen": bool(obs.fallen),
                "physics_state_sha256_before": state_hash,
                "physics_state_sha256_after": after_hash,
                "max_abs_state_delta_across_views": max_state_delta,
                "state_frozen_across_views": True,
            },
            "camera": {
                "single_sensor_moved_after_physics_pause": True,
                "resolution_px": [int(args.width), int(args.height)],
                "front_profile": "go2_front_calib_b",
                "front_role": "model-facing body-fixed Go2-front RTX RGB",
                "front_pose": front_pose,
                "front": {
                    "path": str(front_path.relative_to(ROOT)),
                    "sha256": _sha256(front_path),
                },
                "registered_external_views": views,
            },
            "operator_telemetry_at_capture": telemetry,
            "operator_semantic_validation": semantic_validation,
            "measured_observation_at_capture": {
                "tilt_rad": float(measured.tilt),
                "truth_tilt_rad": float(obs.tilt),
            },
            "provenance": {
                "capture_script": {
                    "path": str(Path(__file__).resolve().relative_to(ROOT)),
                    "sha256": _sha256(Path(__file__).resolve()),
                },
                "backend": {
                    "path": "kino_vla/sim/isaac_policy_backend.py",
                    "sha256": _sha256(ROOT / "kino_vla/sim/isaac_policy_backend.py"),
                },
                "o4_backend": {
                    "path": "kino_vla/sim/isaac_o4_v3_backend.py",
                    "sha256": _sha256(ROOT / "kino_vla/sim/isaac_o4_v3_backend.py"),
                },
                "scene_registry": {
                    "path": str(registry_path.relative_to(ROOT)),
                    "sha256": _sha256(registry_path),
                },
            },
            "validation": {
                "same_case_id_all_views": True,
                "same_scene_all_views": True,
                "same_seed_all_views": True,
                "same_operator_parameters_all_views": True,
                "same_timestamp_all_views": True,
                "only_camera_extrinsics_changed": True,
                "operator_semantics_passed": (
                    semantic_validation is None or semantic_validation["passed"] is True
                ),
                "passed": (
                    semantic_validation is None or semantic_validation["passed"] is True
                ),
            },
        }
        _write_json(output_dir / "capture_manifest.json", manifest)
        print(
            json.dumps(
                {
                    "operator": operator_id,
                    "case_id": case_id,
                    "timestamp_s": float(obs.t),
                    "state_sha256": state_hash,
                    "output": str(output_dir),
                    "passed": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        passed = True
    finally:
        sys.stdout.flush()
        closer = threading.Thread(target=app.close, daemon=True)
        closer.start()
        closer.join(timeout=15.0)
    os._exit(0 if passed else 2)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Kit can retain non-daemon threads after an initialization error.  Fail closed without
        # leaving a GPU-resident orphan that could interfere with the confirmatory collectors.
        os._exit(1)
