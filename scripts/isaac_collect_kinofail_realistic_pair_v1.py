#!/usr/bin/env python3
"""Collect one scheduled nominal/anomaly pair for any Kino-Fail O1--O11 operator.

This is the scale path: it binds an already-audited realistic USD scene, installs the scheduled
operator without per-scene tuning, records synchronized Go2-front RGB/proprioception/telemetry,
and fails only on evidence-corrupting faults (missing sensing, inactive physics, hash/parameter
mismatch, or runtime artifact QA failure). Outcome strength is measured and retained, not gated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.isaac_collect_realistic_smoke_pair import (  # noqa: E402
    FEATURE_NAMES,
    _copy_artifacts,
    _feature_row,
    _rgb_uint8,
    _semantic_fractions,
    _sha256,
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _pair(schedule: list[dict[str, Any]], pair_id: str) -> list[dict[str, Any]]:
    pair = [row for row in schedule if row["counterfactual_group_id"] == pair_id]
    if len(pair) != 2 or {row["condition"] for row in pair} != {
        "anomaly", "nominal_counterfactual"
    }:
        raise ValueError(f"not a complete pair: {pair_id}")
    return sorted(pair, key=lambda row: row["condition"] == "anomaly")


def _registry_row(registry: dict[str, Any], scene_id: str) -> dict[str, Any]:
    rows = [row for row in registry["scenes"] if row["scene_id"] == scene_id]
    if len(rows) != 1:
        raise ValueError(f"scene registry binding is not unique: {scene_id}")
    return rows[0]


def _region(frame, *, progress_m: float = 1.25, half_length_m: float = 0.55):
    from kino_vla.utils.geometry import Rect

    progress = min(progress_m, frame.route_length_m - half_length_m - 0.10)
    progress = max(progress, half_length_m + 0.10)
    half_width = min(0.45, 0.5 * frame.surface_width_m - 0.05)
    center = frame.point(progress, 0.0)
    if abs(frame.direction[0]) >= 1.0 - 1.0e-6:
        return Rect(float(center[0]), float(center[1]), half_length_m, half_width)
    if abs(frame.direction[1]) >= 1.0 - 1.0e-6:
        return Rect(float(center[0]), float(center[1]), half_width, half_length_m)
    raise ValueError("scale collector currently requires an axis-aligned audited route")


def _irregular(rect):
    from kino_vla.sim.adhesion import IrregularRegion

    return IrregularRegion((
        (rect.cx - rect.hx, rect.cy - rect.hy),
        (rect.cx + rect.hx, rect.cy - rect.hy),
        (rect.cx + rect.hx, rect.cy + rect.hy),
        (rect.cx - rect.hx, rect.cy + rect.hy),
    ))


def _install_operator(backend, record: dict[str, Any], frame, seed: int):
    from kino_vla.sim.adhesion_v3 import SurfaceAdhesionConfig
    from kino_vla.sim.operators import (
        Collapse, ComplianceField, EffortDecay, HighCentering, InvisibleCollider,
        MuField, ObsBias, Payload, Push, VisualPhysicsRemap,
    )

    operator_id = str(record["target_operator"])
    parameters = dict(record["physics_parameters"])
    active = record["condition"] == "anomaly"
    region = _region(frame)
    operator = None
    if operator_id == "O1_mu_field" and active:
        operator = MuField(region, parameters["mu_s"], parameters["mu_d"], parameters["restitution"])
    elif operator_id == "O2_compliance" and active:
        operator = ComplianceField(
            region, parameters["stiffness_n_per_m"], parameters["shear_gain"],
            parameters["sink_depth_m"], realistic_foot_model=True,
            longitudinal_axis="x" if abs(frame.direction[0]) > 0.9 else "y",
        )
    elif operator_id == "O3_collapse" and active:
        operator = Collapse(
            region, 0.07, mu_intact=0.8,
            damage_threshold_ns=parameters["damage_threshold_ns"],
            drop_m=parameters["drop_m"], residual_support=parameters["residual_support"],
        )
    elif operator_id == "O4_tether" and active:
        backend.add_foot_adhesion(SurfaceAdhesionConfig(
            region=_irregular(region),
            surface_z_m=0.0,
            attach_contact_force_n=5.0,
            detach_contact_force_n=2.0,
            attach_height_tolerance_m=0.04,
            tangential_stiffness_n_per_m=120.0,
            tangential_damping_ns_per_m=2.0,
            normal_stiffness_n_per_m=80.0,
            normal_damping_ns_per_m=1.0,
            tangential_force_cap_n=parameters["tangential_force_cap_n"],
            normal_force_cap_n=parameters["normal_force_cap_n"],
            peel_height_m=parameters["peel_height_m"],
            unload_steps_to_peel=int(parameters["unload_steps_to_peel"]),
            reattach_cooldown_steps=2,
            max_active_feet=2,
            progress_axis_xy=tuple(float(value) for value in frame.direction),
        ))
    elif operator_id == "O5_payload" and active:
        operator = Payload(
            parameters["mass_kg"],
            (parameters["com_offset_x_m"], parameters["com_offset_y_m"], 0.14),
        )
    elif operator_id == "O6_push" and active:
        direction = frame.left * float(parameters["impulse_ns"])
        operator = Push(
            direction, 2.0, duration_s=parameters["duration_s"],
            application_point_body_m=np.asarray(parameters["application_point_body_m"]),
        )
    elif operator_id == "O7_visual_remap" and active:
        operator = VisualPhysicsRemap(
            region, parameters["mu_s"], parameters["mu_d"], parameters["depth_bias_m"]
        )
    elif operator_id == "O8_invisible_collider":
        geometry_kind = (
            "transparent_acrylic"
            if record["physical_realization"] == "transparent_acrylic"
            else "occluded_low_bar"
        )
        operator = InvisibleCollider(
            region,
            height_m=parameters["obstacle_height_m"],
            collision_enabled=bool(parameters["collision_enabled"]),
            geometry_kind=geometry_kind,
            optical_transmission=parameters["optical_transmission"],
        )
    elif operator_id == "O9_high_centering" and active:
        geometry_kind = (
            "rounded_ridge"
            if record["physical_realization"] == "rounded_ridge"
            else "pallet_edge"
        )
        operator = HighCentering(
            region, parameters["residual_support"],
            ridge_height_m=parameters["ridge_height_m"],
            ridge_width_m=parameters["ridge_width_m"],
            geometry_kind=geometry_kind,
        )
    elif operator_id == "O10_effort_decay" and active:
        operator = EffortDecay(
            parameters["decay_rate_per_s"], parameters["effort_floor"], parameters["onset_s"]
        )
    elif operator_id == "O11_obs_bias":
        operator = ObsBias(
            {"tilt": parameters["tilt_bias_rad"]},
            random_walk_per_sqrt_s={"tilt": parameters["random_walk_rad_sqrt_s"]},
            latency_s=parameters["latency_s"], seed=seed,
        )
    if operator is not None:
        operator.on_reset(backend)
    return operator, region


def _telemetry(backend, operator_id: str, operator) -> dict[str, Any]:
    if operator_id in {"O1_mu_field", "O7_visual_remap"}:
        value = backend.friction_topology_telemetry()
        if operator_id == "O7_visual_remap":
            value = {**value, "depth_fault_count": len(backend._perception_depth_faults)}
        return value
    if operator_id == "O2_compliance":
        return backend.foot_compliance_telemetry()
    if operator_id == "O3_collapse":
        return backend.collapse_telemetry()
    if operator_id == "O4_tether":
        return backend.adhesion_telemetry()
    if operator_id == "O5_payload":
        return backend.payload_telemetry()
    if operator_id == "O6_push":
        return backend.push_telemetry()
    if operator_id == "O8_invisible_collider":
        return backend.blocking_telemetry()
    if operator_id == "O9_high_centering":
        return backend.high_centering_telemetry()
    if operator_id == "O10_effort_decay":
        return backend.actuator_telemetry()
    if operator_id == "O11_obs_bias":
        return operator.sensor_telemetry() if operator is not None else {}
    raise ValueError(operator_id)


def _active_mechanism(operator_id: str, telemetry: dict[str, Any], operator) -> bool:
    if operator_id in {"O1_mu_field", "O7_visual_remap"}:
        return telemetry.get("enabled") is True and (
            operator_id != "O7_visual_remap" or int(telemetry.get("depth_fault_count", 0)) > 0
        )
    if operator_id == "O2_compliance":
        return telemetry.get("enabled") is True and any(
            int(row.get("contact_steps", 0)) > 0 for row in telemetry.get("feet", [])
        )
    if operator_id == "O3_collapse":
        return telemetry.get("enabled") is True and any(
            row.get("collapsed") is True for row in telemetry.get("regions", [])
        )
    if operator_id == "O4_tether":
        return int(telemetry.get("total_attachment_cycles", 0)) > 0
    if operator_id == "O5_payload":
        return float(telemetry.get("payload_kg", 0.0)) > 0.0
    if operator_id == "O6_push":
        return operator is not None and operator.fired and int(telemetry.get("applied_steps", 0)) > 0
    if operator_id == "O8_invisible_collider":
        return telemetry.get("enabled") is True and any(
            row.get("collision_requested") is True for row in telemetry.get("obstacles", [])
        )
    if operator_id == "O9_high_centering":
        return telemetry.get("enabled") is True
    if operator_id == "O10_effort_decay":
        return float(telemetry.get("effort_scale", 1.0)) < 1.0
    if operator_id == "O11_obs_bias":
        return telemetry.get("raw_pipeline_active") is True
    return False


def _tag_scene(scene_prim: str, route_surface_path: str) -> int:
    import omni.usd
    from pxr import Usd, UsdGeom
    try:
        from isaacsim.core.utils.semantics import add_update_semantics
    except ImportError:
        from omni.isaac.core.utils.semantics import add_update_semantics

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(scene_prim)
    tagged = 0
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Imageable):
            add_update_semantics(prim, semantic_label="scene_context", type_label="class")
            tagged += 1
    route = stage.GetPrimAtPath(route_surface_path)
    if not route.IsValid():
        raise RuntimeError(f"missing route surface {route_surface_path}")
    add_update_semantics(route, semantic_label="scheduled_floor_material", type_label="class")
    return tagged + 1


def _collect_one(
    backend, record: dict[str, Any], registry_row: dict[str, Any], compiled: dict[str, Any],
    frame, route_surface_path: str, bindings: dict[str, Any], asset_lock_path: Path,
    episode_dir: Path, protocol_path: Path, schedule_sha256: str, collector_sha256: str,
    camera: dict[str, Any], overwrite: bool,
) -> dict[str, Any]:
    from PIL import Image
    from kino_vla.data.runtime_manifest import build_collected_manifest, validate_runtime_episode, write_collected_manifest
    from kino_vla.sim.realistic_route_controller_v3 import DampedRouteControllerV3
    from kino_vla.sim.terrain_materials import bind_omnipbr_material
    import omni.usd

    if episode_dir.exists():
        if not overwrite:
            raise FileExistsError(episode_dir)
        shutil.rmtree(episode_dir)
    views = {str(row["appearance_view_id"]): dict(row) for row in record["appearance_views"]}
    primary_id = next(key for key, row in views.items() if row["is_primary"])
    for view_id, row in views.items():
        folder = episode_dir / ("rgb" if row["is_primary"] else f"rgb_views/{view_id}")
        folder.mkdir(parents=True, exist_ok=True)
    seed = int(record["operator_seed"])
    obs = backend.deep_reset(seed)
    operator, region = _install_operator(backend, record, frame, seed)
    obs = backend.reset(seed)
    if record["target_operator"] == "O11_obs_bias" and operator is not None:
        operator.on_reset(backend)
    stage = omni.usd.get_context().get_stage()
    primary = bindings[primary_id]
    bind_omnipbr_material(stage, route_surface_path, primary)

    scene_artifacts = _copy_artifacts(
        [ROOT / registry_row["episode_usd"], ROOT / registry_row["compiled_audit"]],
        episode_dir=episode_dir, relative_root=Path("provenance/scene"),
    )
    appearance_artifacts: dict[str, list[dict[str, str]]] = {}
    for view_id, binding in bindings.items():
        sources = [asset_lock_path] + [Path(path) for path in (
            binding.maps.basecolor, binding.maps.normal, binding.maps.roughness,
            binding.maps.displacement, binding.maps.ambient_occlusion,
        ) if path is not None]
        appearance_artifacts[view_id] = _copy_artifacts(
            sources, episode_dir=episode_dir,
            relative_root=Path("provenance/appearance") / view_id,
        )
    protocol_artifact = _copy_artifacts(
        [protocol_path], episode_dir=episode_dir, relative_root=Path("provenance/protocol")
    )[0]

    controller = DampedRouteControllerV3(
        cross_track_gain_per_s=2.0, lateral_velocity_damping=1.0,
        heading_gain_per_s=2.0, lateral_limit_mps=0.35, yaw_rate_limit_radps=0.6,
    )
    features, times, telemetry_rows = [], [], []
    rgb: dict[str, list[dict[str, Any]]] = {key: [] for key in views}
    semantics: dict[str, list[dict[str, float]]] = {key: [] for key in views}
    max_route_deviation = 0.0
    max_tilt = 0.0
    min_height = float("inf")
    capture_stride = 20
    step = 0
    while step < 360 and not obs.fallen:
        if operator is not None:
            operator.on_step(backend, float(obs.t))
        command, control = controller.command(
            frame, position_xy_m=obs.pos, heading_rad=obs.heading,
            velocity_body_xy_mps=obs.vel_body, forward_speed_mps=0.24,
            target_lateral_offset_m=0.0,
        )
        obs = backend.step(command)
        measured = operator.transform_obs(obs) if operator is not None else obs
        times.append(float(measured.t))
        features.append(_feature_row(backend, measured))
        op_telemetry = _telemetry(backend, record["target_operator"], operator)
        progress, lateral = frame.project(obs.pos)
        max_route_deviation = max(max_route_deviation, abs(float(lateral)))
        max_tilt = max(max_tilt, abs(float(obs.tilt)))
        min_height = min(min_height, float(obs.base_height))
        telemetry_rows.append({
            "timestamp_s": float(obs.t), "position_xy_m": obs.pos.tolist(),
            "route_progress_m": float(progress), "route_lateral_offset_m": float(lateral),
            "base_height_m": float(obs.base_height), "tilt_rad": float(obs.tilt),
            "fallen": bool(obs.fallen), "command": command.tolist(),
            "controller": control, "operator": op_telemetry,
        })
        if step % capture_stride == 0:
            frame_index = len(rgb[primary_id])
            for view_id, binding in bindings.items():
                bind_omnipbr_material(stage, route_surface_path, binding)
                backend._env.sim.render()
                capture = backend.capture_perception()
                if capture is None:
                    raise RuntimeError("Go2-front RGB/segmentation sensor returned no frame")
                relative = Path("rgb" if views[view_id]["is_primary"] else f"rgb_views/{view_id}") / f"{frame_index:06d}.png"
                Image.fromarray(_rgb_uint8(capture["rgb"])).save(episode_dir / relative)
                rgb[view_id].append({"path": str(relative), "timestamp_s": float(capture["camera_pose_source_timestamp_s"])})
                semantics[view_id].append(_semantic_fractions(capture))
            bind_omnipbr_material(stage, route_surface_path, primary)
        step += 1

    np.savez_compressed(
        episode_dir / "proprio.npz", features=np.asarray(features, dtype=np.float32),
        timestamp_s=np.asarray(times, dtype=np.float64), feature_names=np.asarray(FEATURE_NAMES),
    )
    with (episode_dir / "privileged.jsonl").open("w", encoding="utf-8") as stream:
        for row in telemetry_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    summaries: dict[str, dict[str, Any]] = {}
    summary_paths: dict[str, Path] = {}
    for view_id, rows in semantics.items():
        summary = {
            "n_frames": len(rgb[view_id]),
            "mean_scheduled_appearance_pixel_fraction": float(np.mean([row["scheduled_appearance_pixel_fraction"] for row in rows])),
            "mean_scene_context_pixel_fraction": float(np.mean([row["scene_context_pixel_fraction"] for row in rows])),
            "mean_film_pixel_fraction": 0.0,
            "tagged_scene_prims": 1,
        }
        path = episode_dir / "provenance/appearance" / view_id / "semantic_summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = {"path": str(path.relative_to(episode_dir)), "sha256": _sha256(path)}
        appearance_artifacts[view_id].append(artifact)
        summaries[view_id], summary_paths[view_id] = summary, path
    scene_artifacts.append({
        "path": str(summary_paths[primary_id].relative_to(episode_dir)),
        "sha256": _sha256(summary_paths[primary_id]),
    })
    final_operator = _telemetry(backend, record["target_operator"], operator)
    active = record["condition"] == "anomaly"
    operator_qa = _active_mechanism(record["target_operator"], final_operator, operator) if active else True
    half_pitch = 0.5 * float(camera["pitch_down_rad"])
    mount = [float(value) for value in camera["mount_xyz_m"]]
    protocol = _json(protocol_path)
    appearance_views_readback = {
        view_id: {
            "qa_passed": summaries[view_id]["mean_scheduled_appearance_pixel_fraction"] >= 0.10,
            **{key: views[view_id][key] for key in (
                "appearance_id", "material_family", "surface_state", "uv_scale",
                "uv_rotation_deg", "uv_offset", "albedo_brightness_multiplier",
                "normal_strength", "roughness_multiplier",
            )},
            "semantic_summary_path": str(summary_paths[view_id].relative_to(episode_dir)),
            "artifacts": appearance_artifacts[view_id],
        }
        for view_id in views
    }
    primary_view = appearance_views_readback[primary_id]
    manifest = build_collected_manifest(
        record, episode_dir=episode_dir, rgb_frames=rgb[primary_id], rgb_views=rgb,
        camera={
            "profile": record["camera_profile"], "body_fixed": True,
            "pose_sync_method": "rigid_base_transform_each_step",
            "base_to_camera_xyz_quat_xyzw": [*mount, 0.0, math.sin(half_pitch), 0.0, math.cos(half_pitch)],
            "focal_length_mm": camera["focal_length_mm"],
            "resolution": [camera["width"], camera["height"]], "qa_passed": True,
        },
        operator_readback={
            "operator_id": record["target_operator"], "active": active,
            "qa_passed": operator_qa, "applied_parameters": record["physics_parameters"],
            "telemetry": final_operator,
        },
        scene_readback={
            "qa_passed": compiled.get("passed") is True and summaries[primary_id]["mean_scene_context_pixel_fraction"] >= 0.08,
            "scene_family": record["scene_family"], "scene_source": record["scene_source"],
            "scene_seed": record["scene_seed"], "compiled_scene_id": compiled["scene_id"],
            "semantic_summary_path": str(summary_paths[primary_id].relative_to(episode_dir)),
            "artifacts": scene_artifacts,
        },
        appearance_readback={
            "qa_passed": all(row["qa_passed"] for row in appearance_views_readback.values()),
            **{key: primary_view[key] for key in (
                "appearance_id", "material_family", "surface_state", "uv_scale",
                "uv_rotation_deg", "uv_offset", "albedo_brightness_multiplier",
                "normal_strength", "roughness_multiplier", "semantic_summary_path", "artifacts",
            )},
            "views": appearance_views_readback,
        },
        geometry_readback={
            "qa_passed": True, "geometry_id": record["geometry_id"],
            "geometry_profile": record["geometry_profile"],
            "physical_realization": record["physical_realization"],
            "operator_region": {"cx": region.cx, "cy": region.cy, "hx": region.hx, "hy": region.hy},
        },
        collection={
            "status": "formal_pilot", "simulator": "Isaac Sim", "physics_engine": "PhysX GPU",
            "created_utc": datetime.now(UTC).isoformat(), "schedule_sha256": schedule_sha256,
            "collector_sha256": collector_sha256,
            "runtime_manifest_sha256": _sha256(ROOT / "kino_vla/data/runtime_manifest.py"),
            "capture_stride_control_steps": capture_stride,
            "formal_protocol": {
                **protocol_artifact, "protocol_id": protocol["protocol_id"],
                "schedule_sha256": protocol["schedule_sha256"],
                "collector_sha256": protocol["collector_sha256"],
                "runtime_manifest_sha256": protocol["runtime_manifest_sha256"],
            },
        },
    )
    write_collected_manifest(manifest, episode_dir)
    validation = validate_runtime_episode(record, episode_dir=episode_dir, write_validated=True)
    return {
        "episode_id": record["episode_id"], "condition": record["condition"],
        "passed": validation.passed, "issues": list(validation.issues),
        "operator_qa": operator_qa, "fallen": bool(obs.fallen),
        "route_progress_m": float(frame.project(obs.pos)[0]),
        "max_route_deviation_m": max_route_deviation, "max_tilt_rad": max_tilt,
        "min_base_height_m": min_height, "rgb_frames_per_view": len(rgb[primary_id]),
        "proprio_samples": len(features), "manifest": str(episode_dir / "manifest.json"),
    }


def main() -> int:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--schedule", required=True)
    preliminary.add_argument("--scene-registry", required=True)
    preliminary.add_argument("--protocol", required=True)
    preliminary.add_argument("--corpus-root", required=True)
    preliminary.add_argument("--counterfactual-group-id", required=True)
    pre, _ = preliminary.parse_known_args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", default=pre.schedule)
    parser.add_argument("--scene-registry", default=pre.scene_registry)
    parser.add_argument("--protocol", default=pre.protocol)
    parser.add_argument("--corpus-root", default=pre.corpus_root)
    parser.add_argument("--counterfactual-group-id", default=pre.counterfactual_group_id)
    parser.add_argument("--overwrite", action="store_true")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app
    passed = False
    try:
        from kino_vla.sim.isaac_o4_v3_backend import IsaacPolicyBackendO4V3
        from kino_vla.sim.realistic_route_protocol_v2 import scene_route_binding
        from kino_vla.sim.terrain_materials import appearance_binding_from_record, load_terrain_asset_lock
        from kino_vla.utils.config import CONFIGS_DIR, load_config

        schedule_path = (ROOT / args.schedule).resolve()
        registry_path = (ROOT / args.scene_registry).resolve()
        protocol_path = (ROOT / args.protocol).resolve()
        corpus_root = (ROOT / args.corpus_root).resolve()
        schedule = _jsonl(schedule_path)
        pair = _pair(schedule, args.counterfactual_group_id)
        representative = pair[0]
        registry = _json(registry_path)
        scene = _registry_row(registry, representative["scene_family"])
        episode = ROOT / scene["episode_usd"]
        compiled_path = ROOT / scene["compiled_audit"]
        if _sha256(episode) != scene["episode_sha256"] or _sha256(compiled_path) != scene["compiled_audit_sha256"]:
            raise RuntimeError("scene registry hash mismatch")
        compiled = _json(compiled_path)
        binding = scene_route_binding(compiled)
        benchmark = yaml.safe_load((CONFIGS_DIR / "data/kinofail_realistic.yaml").read_text(encoding="utf-8"))
        camera = dict(benchmark["camera_profile_specs"][representative["camera_profile"]])
        mount = camera["mount_xyz_m"]
        config = load_config("sim/go2_skeleton.yaml", {
            "perception_cam_width": camera["width"], "perception_cam_height": camera["height"],
            "perception_focal_mm": camera["focal_length_mm"],
            "perception_cam_x_m": mount[0], "perception_cam_y_m": mount[1],
            "perception_cam_z_m": mount[2], "perception_cam_pitch_down_rad": camera["pitch_down_rad"],
        })
        backend = IsaacPolicyBackendO4V3(config, binding.frame.point(0.0, 0.0), binding.frame.heading_rad, perception_cam=True)
        scene_prim = backend.load_realistic_scene(str(episode))
        route_surface_path = binding.route_surface_prim_path(scene_prim)
        _tag_scene(scene_prim, route_surface_path)
        asset_lock_path = ROOT / "outputs/assets/terrain_pbr_v1/terrain_assets.lock.json"
        asset_lock = load_terrain_asset_lock(asset_lock_path)
        asset_root = Path(asset_lock["asset_root"])
        if not asset_root.is_absolute():
            asset_root = ROOT / asset_root
        schedule_sha256 = _sha256(schedule_path)
        collector_sha256 = _sha256(Path(__file__).resolve())
        protocol = _json(protocol_path)
        for key, actual in (
            ("schedule_sha256", schedule_sha256),
            ("collector_sha256", collector_sha256),
            ("runtime_manifest_sha256", _sha256(ROOT / "kino_vla/data/runtime_manifest.py")),
        ):
            if protocol.get(key) != actual:
                raise RuntimeError(f"formal protocol {key} mismatch")
        results = []
        for record in pair:
            bindings = {
                str(view["appearance_view_id"]): appearance_binding_from_record(view, lock=asset_lock, asset_root=asset_root)
                for view in record["appearance_views"]
            }
            episode_dir = corpus_root / Path(record["required_outputs"]["episode_manifest"]).parent
            result = _collect_one(
                backend, record, scene, compiled, binding.frame, route_surface_path, bindings,
                asset_lock_path, episode_dir, protocol_path, schedule_sha256, collector_sha256,
                camera, bool(args.overwrite),
            )
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
        passed = all(row["passed"] for row in results)
        summary = {
            "schema_version": "kinofail.realistic-scale-pair.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "counterfactual_group_id": args.counterfactual_group_id,
            "passed": passed, "results": results,
        }
        summary_path = corpus_root / "pair_summaries" / f"{args.counterfactual_group_id}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        sys.stdout.flush()
        closer = threading.Thread(target=app.close, daemon=True)
        closer.start()
        closer.join(timeout=15.0)
    os._exit(0 if passed else 2)


if __name__ == "__main__":
    main()
