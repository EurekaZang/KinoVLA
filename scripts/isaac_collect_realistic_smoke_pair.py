#!/usr/bin/env python3
"""Collect and independently validate one realistic O4 counterfactual smoke pair.

This is the publication-pipeline vertical slice, not a replacement for the 396-episode pilot.  It
uses the frozen schedule row, a body-fixed Go2-front RTX camera, copied/hash-locked scene and PBR
provenance, three synchronized terrain appearance views, per-step raw Isaac proprioception, and
the runtime-v5 artifact gate.  The smoke
protocol may pass artifact QA but is deliberately barred from evaluation eligibility.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _select_pair(
    records: list[dict[str, Any]], counterfactual_group_id: str | None
) -> list[dict[str, Any]]:
    if counterfactual_group_id is None:
        candidates = [
            record
            for record in records
            if record["target_operator"] == "O4_tether"
            and record["domain"] == "life"
            and record["scene_family"] == "indoor_workstudio_01"
            and record["physical_realization"] == "adhesive_foot_contact"
            and record["geometry_profile"] == "irregular_protective_film"
        ]
        groups = sorted({str(record["counterfactual_group_id"]) for record in candidates})
        if not groups:
            raise RuntimeError("pilot schedule has no collector-compatible O4 pair")
        counterfactual_group_id = groups[0]
    pair = [
        record for record in records if record["counterfactual_group_id"] == counterfactual_group_id
    ]
    if len(pair) != 2 or {record["condition"] for record in pair} != {
        "anomaly",
        "nominal_counterfactual",
    }:
        raise RuntimeError(f"{counterfactual_group_id} is not one complete pair")
    required = {
        (record["target_operator"], record["physical_realization"], record["geometry_profile"])
        for record in pair
    }
    if required != {("O4_tether", "adhesive_foot_contact", "irregular_protective_film")}:
        raise ValueError("smoke collector currently certifies only the exact O4 vertical slice")
    return sorted(pair, key=lambda record: record["condition"], reverse=True)


def _rgb_uint8(value: np.ndarray) -> np.ndarray:
    rgb = np.asarray(value)[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.clip(rgb, 0.0, 1.0) * 255.0
    return rgb.astype(np.uint8)


def _copy_artifacts(
    sources: list[Path], *, episode_dir: Path, relative_root: Path
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    destination_root = episode_dir / relative_root
    destination_root.mkdir(parents=True, exist_ok=True)
    for source in sources:
        destination = destination_root / source.name
        shutil.copy2(source, destination)
        rows.append(
            {
                "path": str(destination.relative_to(episode_dir)),
                "sha256": _sha256(destination),
            }
        )
    return rows


def _stage_readback(visual_ground_path: str) -> dict[str, Any]:
    import omni.usd
    from pxr import Usd, UsdShade

    stage = omni.usd.get_context().get_stage()
    scene_root = stage.GetPrimAtPath("/World/KinoIndoor")
    scene_id = scene_root.GetAttribute("kino:sceneId").Get() if scene_root.IsValid() else None
    source_kind = scene_root.GetAttribute("kino:sourceKind").Get() if scene_root.IsValid() else None
    operator_prims = []
    film_panels = []
    footprint_vertex_count = 0
    if scene_root.IsValid():
        for prim in Usd.PrimRange(scene_root):
            if prim.GetAttribute("kino:operator").Get() == "O4_tether":
                operator_prims.append(str(prim.GetPath()))
                footprint = prim.GetAttribute("kino:footprintXY").Get()
                if footprint:
                    footprint_vertex_count = max(footprint_vertex_count, len(footprint))
            if prim.GetName().startswith("FilmPanel_"):
                film_panels.append(str(prim.GetPath()))
    visual_ground = stage.GetPrimAtPath(visual_ground_path)
    bound_material = None
    if visual_ground.IsValid():
        material, _ = UsdShade.MaterialBindingAPI(visual_ground).ComputeBoundMaterial()
        if material and material.GetPrim().IsValid():
            bound_material = str(material.GetPath())
    return {
        "scene_id": scene_id,
        "source_kind": source_kind,
        "operator_prim_paths": operator_prims,
        "film_panel_paths": film_panels,
        "footprint_vertex_count": footprint_vertex_count,
        "visual_ground_path": visual_ground_path,
        "bound_material_path": bound_material,
    }


def _prepare_corpus_rendering(visual_ground_path: str) -> int:
    """Expose the scheduled PBR floor and tag actual RTX-visible scene classes."""
    import omni.usd
    from pxr import Usd, UsdGeom

    try:
        from isaacsim.core.utils.semantics import add_update_semantics
    except ImportError:
        from omni.isaac.core.utils.semantics import add_update_semantics

    stage = omni.usd.get_context().get_stage()
    scene_root = stage.GetPrimAtPath("/World/KinoIndoor")
    tagged = 0
    for prim in Usd.PrimRange(scene_root):
        if prim.GetName().startswith("Floor_Plank_") and prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).MakeInvisible()
        semantic = prim.GetAttribute("kino:semanticClass").Get()
        if semantic and prim.IsA(UsdGeom.Imageable):
            add_update_semantics(prim, semantic_label=str(semantic), type_label="class")
            tagged += 1
    ground = stage.GetPrimAtPath(visual_ground_path)
    if not ground.IsValid():
        raise RuntimeError(f"scheduled PBR visual ground is invalid: {visual_ground_path}")
    add_update_semantics(
        ground,
        semantic_label="scheduled_floor_material",
        type_label="class",
    )
    return tagged + 1


def _semantic_fractions(capture: dict[str, Any]) -> dict[str, float]:
    seg = np.asarray(capture["seg"])
    counts: dict[str, int] = {}
    for semantic_id, label in capture["id_to_labels"].items():
        semantic_class = (
            label.get("class") if isinstance(label, dict) else str(label)
        ) or "UNLABELLED"
        counts[semantic_class] = counts.get(semantic_class, 0) + int(
            np.count_nonzero(seg == int(semantic_id))
        )
    total = max(int(seg.size), 1)
    scheduled = counts.get("scheduled_floor_material", 0) / total
    film = counts.get("floor_protective_film", 0) / total
    skipped = {
        "SCHEDULED_FLOOR_MATERIAL",
        "FLOOR_PROTECTIVE_FILM",
        "UNLABELLED",
        "BACKGROUND",
        "BACKGROUND_GROUND_PLANE",
    }
    context = sum(value for key, value in counts.items() if key.upper() not in skipped) / total
    return {
        "scheduled_appearance_pixel_fraction": float(scheduled),
        "film_pixel_fraction": float(film),
        "scene_context_pixel_fraction": float(context),
    }


def _feature_row(backend, obs) -> list[float]:
    packet = backend.raw_proprio_packet()
    return [
        *[float(value) for value in packet.imu_projected_gravity_b],
        *[float(value) for value in packet.imu_ang_vel_b_radps],
        *[float(value) for value in packet.imu_lin_acc_b_mps2],
        *[float(value) for value in packet.odom_pos_xy_m],
        float(packet.odom_heading_rad),
        *[float(value) for value in packet.odom_vel_body_mps],
        float(obs.slip_ratio),
        float(obs.base_height),
        float(obs.tilt),
        float(obs.effort_ratio),
        float(obs.support_ratio),
    ]


FEATURE_NAMES = (
    "imu_gravity_x",
    "imu_gravity_y",
    "imu_gravity_z",
    "imu_angular_velocity_x_radps",
    "imu_angular_velocity_y_radps",
    "imu_angular_velocity_z_radps",
    "imu_linear_acceleration_x_mps2",
    "imu_linear_acceleration_y_mps2",
    "imu_linear_acceleration_z_mps2",
    "odom_x_m",
    "odom_y_m",
    "odom_heading_rad",
    "odom_velocity_x_mps",
    "odom_velocity_y_mps",
    "slip_ratio",
    "base_height_m",
    "tilt_rad",
    "effort_ratio",
    "support_ratio",
)


def _collect_episode(
    backend,
    record: dict[str, Any],
    *,
    episode_dir: Path,
    scene_spec,
    compiled: dict[str, Any],
    appearance_bindings: dict[str, Any],
    asset_lock_path: Path,
    demo_cfg,
    camera_profile: dict[str, Any],
    overwrite: bool,
    collection_status: str = "smoke_pair_not_formal_pilot",
    schedule_sha256: str | None = None,
    collector_sha256: str | None = None,
    formal_protocol_path: Path | None = None,
) -> dict[str, Any]:
    from PIL import Image

    from kino_vla.data.runtime_manifest import (
        build_collected_manifest,
        validate_runtime_episode,
        write_collected_manifest,
    )
    from kino_vla.sim.adhesion import FootAdhesionConfig
    from kino_vla.sim.terrain_materials import bind_omnipbr_material
    from kino_vla.utils.geometry import Rect

    if episode_dir.exists():
        if not overwrite:
            raise FileExistsError(f"episode exists; pass --overwrite to replace: {episode_dir}")
        shutil.rmtree(episode_dir)
    runtime_manifest_sha256 = _sha256(REPO_ROOT / "kino_vla/data/runtime_manifest.py")
    collector_sha256 = collector_sha256 or _sha256(Path(__file__).resolve())
    formal_protocol_reference: dict[str, str] | None = None
    if collection_status == "formal_pilot":
        if formal_protocol_path is None:
            raise ValueError("formal collection requires a frozen protocol file")
        protocol = json.loads(formal_protocol_path.read_text(encoding="utf-8"))
        if schedule_sha256 != protocol.get("schedule_sha256"):
            raise ValueError("live schedule hash does not match the frozen formal protocol")
        if collector_sha256 != protocol.get("collector_sha256"):
            raise ValueError("live collector bundle hash does not match the frozen formal protocol")
        if runtime_manifest_sha256 != protocol.get("runtime_manifest_sha256"):
            raise ValueError("live runtime validator hash does not match the frozen formal protocol")
        copied = _copy_artifacts(
            [formal_protocol_path],
            episode_dir=episode_dir,
            relative_root=Path("provenance/protocol"),
        )[0]
        formal_protocol_reference = {
            **copied,
            "protocol_id": str(protocol["protocol_id"]),
            "schedule_sha256": str(protocol["schedule_sha256"]),
            "collector_sha256": str(protocol["collector_sha256"]),
            "runtime_manifest_sha256": str(protocol["runtime_manifest_sha256"]),
        }
    appearance_views = {
        str(view["appearance_view_id"]): dict(view) for view in record["appearance_views"]
    }
    primary_view_id = next(
        view_id for view_id, view in appearance_views.items() if view["is_primary"]
    )
    if set(appearance_bindings) != set(appearance_views):
        raise ValueError("resolved appearance bindings do not match the scheduled views")
    for view_id, view in appearance_views.items():
        relative_dir = Path("rgb") if view["is_primary"] else Path("rgb_views") / view_id
        (episode_dir / relative_dir).mkdir(parents=True)

    seed = int(record["operator_seed"])
    backend.deep_reset(seed)
    primary_binding = appearance_bindings[primary_view_id]
    backend.set_terrain_appearance(primary_binding)
    visual_ground_path = backend.add_pbr_visual_ground(
        Rect(cx=3.4, cy=0.0, hx=5.0, hy=3.5),
        primary_binding,
        lift_m=0.004,
    )
    tagged_scene_prims = _prepare_corpus_rendering(visual_ground_path)
    parameters = dict(record["physics_parameters"])
    active = record["condition"] == "anomaly"
    fixed = demo_cfg.adhesion.to_dict()
    if active:
        backend.add_foot_adhesion(
            FootAdhesionConfig(
                region=scene_spec.adhesion_region,
                surface_z_m=float(fixed["surface_z_m"]),
                attach_contact_force_n=float(fixed["attach_contact_force_n"]),
                attach_height_tolerance_m=float(fixed["attach_height_tolerance_m"]),
                stiffness_xy_n_per_m=float(fixed["stiffness_xy_n_per_m"]),
                damping_xy_ns_per_m=float(fixed["damping_xy_ns_per_m"]),
                stiffness_z_n_per_m=float(fixed["stiffness_z_n_per_m"]),
                damping_z_ns_per_m=float(fixed["damping_z_ns_per_m"]),
                force_cap_n=float(parameters["force_cap_n"]),
                break_force_n=float(parameters["break_force_n"]),
                peel_release_force_n=float(fixed["peel_release_force_n"]),
                peel_velocity_threshold_mps=float(fixed["peel_velocity_threshold_mps"]),
                progress_axis_xy=tuple(float(value) for value in fixed["progress_axis_xy"]),
                max_active_feet=int(fixed["max_active_feet"]),
            )
        )
    obs = backend.reset(seed)
    stage_readback = _stage_readback(visual_ground_path)

    scene_sources = [Path(compiled["output_dir"]) / name for name in compiled["layers"]]
    scene_artifacts = _copy_artifacts(
        scene_sources,
        episode_dir=episode_dir,
        relative_root=Path("provenance/scene"),
    )
    appearance_artifacts_by_view: dict[str, list[dict[str, str]]] = {}
    for view_id, binding in appearance_bindings.items():
        appearance_sources = [asset_lock_path]
        for path in (
            binding.maps.basecolor,
            binding.maps.normal,
            binding.maps.roughness,
            binding.maps.displacement,
            binding.maps.ambient_occlusion,
        ):
            if path is not None:
                appearance_sources.append(Path(path))
        appearance_artifacts_by_view[view_id] = _copy_artifacts(
            appearance_sources,
            episode_dir=episode_dir,
            relative_root=Path("provenance/appearance") / view_id,
        )

    forward_s, recovery_s, stop_s = 4.7, 2.2, 0.4
    capture_stride = 5
    feature_rows: list[list[float]] = []
    timestamps: list[float] = []
    telemetry_rows: list[dict[str, Any]] = []
    rgb_frames_by_view: dict[str, list[dict[str, Any]]] = {
        view_id: [] for view_id in appearance_views
    }
    attachment_events = 0
    peel_events = 0
    break_events = 0
    max_applied_per_foot_n = 0.0
    max_pose_sync_error_m = 0.0
    max_orientation_sync_error_rad = 0.0
    semantic_rows_by_view: dict[str, list[dict[str, float]]] = {
        view_id: [] for view_id in appearance_views
    }
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    step_index = 0
    total_s = forward_s + recovery_s + stop_s
    while obs.t < total_s and not obs.fallen:
        if obs.t < forward_s:
            command_x = 0.55
        elif obs.t < forward_s + recovery_s:
            command_x = -0.32
        else:
            command_x = 0.0
        obs = backend.step(np.asarray([command_x, 0.0, 0.0], dtype=np.float64))
        timestamps.append(float(obs.t))
        feature_rows.append(_feature_row(backend, obs))
        adhesion = backend.adhesion_telemetry()
        feet = list(adhesion.get("feet", []))
        attachment_events += sum(foot.get("event") == "attached" for foot in feet)
        peel_events += sum(foot.get("event") == "peeled" for foot in feet)
        break_events += sum(foot.get("event") == "broken" for foot in feet)
        max_applied_per_foot_n = max(
            [max_applied_per_foot_n] + [float(foot.get("applied_force_n", 0.0)) for foot in feet]
        )
        telemetry_rows.append(
            {
                "timestamp_s": float(obs.t),
                "position_xy_m": [float(value) for value in obs.pos],
                "base_height_m": float(obs.base_height),
                "tilt_rad": float(obs.tilt),
                "fallen": bool(obs.fallen),
                "command_x_mps": command_x,
                "adhesion": adhesion,
            }
        )
        if step_index % capture_stride == 0:
            frame_index = len(rgb_frames_by_view[primary_view_id])
            for view_id, binding in appearance_bindings.items():
                bind_omnipbr_material(stage, visual_ground_path, binding)
                # A forced camera annotator update alone can reuse the previous Hydra texture
                # cache when several appearance views share one physics step.  render() pauses
                # the simulation timeline and refreshes Fabric/Hydra only, so PhysX, proprio and
                # the camera pose remain byte-identical across these texture interventions.
                backend._env.sim.render()
                capture = backend.capture_perception()
                if capture is None:
                    raise RuntimeError("Go2-front perception camera produced no frame")
                max_pose_sync_error_m = max(
                    max_pose_sync_error_m,
                    float(capture.get("camera_pose_sync_error_m") or 0.0),
                )
                max_orientation_sync_error_rad = max(
                    max_orientation_sync_error_rad,
                    float(capture.get("camera_orientation_sync_error_rad") or 0.0),
                )
                semantic_rows_by_view[view_id].append(_semantic_fractions(capture))
                filename = f"{frame_index:06d}.png"
                relative_dir = (
                    Path("rgb")
                    if appearance_views[view_id]["is_primary"]
                    else Path("rgb_views") / view_id
                )
                relative_path = relative_dir / filename
                Image.fromarray(_rgb_uint8(capture["rgb"])).save(episode_dir / relative_path)
                rgb_frames_by_view[view_id].append(
                    {
                        "path": str(relative_path),
                        "timestamp_s": float(capture["camera_pose_source_timestamp_s"]),
                    }
                )
            bind_omnipbr_material(stage, visual_ground_path, primary_binding)
        step_index += 1

    np.savez_compressed(
        episode_dir / "proprio.npz",
        features=np.asarray(feature_rows, dtype=np.float32),
        timestamp_s=np.asarray(timestamps, dtype=np.float64),
        feature_names=np.asarray(FEATURE_NAMES),
    )
    with (episode_dir / "privileged.jsonl").open("w", encoding="utf-8") as stream:
        for row in telemetry_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    semantic_summaries: dict[str, dict[str, Any]] = {}
    semantic_paths: dict[str, Path] = {}
    for view_id, rows in semantic_rows_by_view.items():
        summary = {
            "n_frames": len(rgb_frames_by_view[view_id]),
            "mean_scheduled_appearance_pixel_fraction": float(
                np.mean([row["scheduled_appearance_pixel_fraction"] for row in rows])
            ),
            "mean_film_pixel_fraction": float(
                np.mean([row["film_pixel_fraction"] for row in rows])
            ),
            "mean_scene_context_pixel_fraction": float(
                np.mean([row["scene_context_pixel_fraction"] for row in rows])
            ),
            "tagged_scene_prims": tagged_scene_prims,
        }
        semantic_summaries[view_id] = summary
        semantic_path = (
            episode_dir / "provenance" / "appearance" / view_id / "semantic_summary.json"
        )
        semantic_path.parent.mkdir(parents=True, exist_ok=True)
        semantic_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        semantic_paths[view_id] = semantic_path
        semantic_artifact = {
            "path": str(semantic_path.relative_to(episode_dir)),
            "sha256": _sha256(semantic_path),
        }
        appearance_artifacts_by_view[view_id].append(semantic_artifact)
        if view_id == primary_view_id:
            scene_artifacts.append(dict(semantic_artifact))
    semantic_summary = semantic_summaries[primary_view_id]
    semantic_path = semantic_paths[primary_view_id]

    geometry_qa = (
        stage_readback["scene_id"] == record["scene_family"]
        and stage_readback["source_kind"] == record["scene_source"]
        and bool(stage_readback["operator_prim_paths"])
        and int(stage_readback["footprint_vertex_count"]) >= 3
        and len(stage_readback["film_panel_paths"]) >= 5
    )
    appearance_stage_by_view: dict[str, dict[str, Any]] = {}
    appearance_qa_by_view: dict[str, bool] = {}
    for view_id, binding in appearance_bindings.items():
        bind_omnipbr_material(stage, visual_ground_path, binding)
        readback = _stage_readback(visual_ground_path)
        appearance_stage_by_view[view_id] = readback
        appearance_qa_by_view[view_id] = (
            bool(readback["bound_material_path"])
            and semantic_summaries[view_id]["mean_scheduled_appearance_pixel_fraction"] >= 0.10
            and all(
                Path(path).is_file()
                for path in (
                    binding.maps.basecolor,
                    binding.maps.normal,
                    binding.maps.roughness,
                )
            )
        )
    bind_omnipbr_material(stage, visual_ground_path, primary_binding)
    appearance_qa = all(appearance_qa_by_view.values())
    if active:
        operator_qa = (
            attachment_events > 0
            and max_applied_per_foot_n > 0.0
            and max_applied_per_foot_n <= float(parameters["force_cap_n"]) + 1.0e-4
            and not backend._resistance
        )
    else:
        operator_qa = (
            attachment_events == peel_events == break_events == 0 and not backend._resistance
        )
    camera_qa = max_pose_sync_error_m < 1.0e-5 and max_orientation_sync_error_rad < 0.002

    half_pitch = 0.5 * float(camera_profile["pitch_down_rad"])
    mount = [float(value) for value in camera_profile["mount_xyz_m"]]
    manifest = build_collected_manifest(
        record,
        episode_dir=episode_dir,
        rgb_frames=rgb_frames_by_view[primary_view_id],
        rgb_views=rgb_frames_by_view,
        camera={
            "profile": record["camera_profile"],
            "body_fixed": True,
            "pose_sync_method": "rigid_base_transform_each_step",
            "base_to_camera_xyz_quat_xyzw": [
                *mount,
                0.0,
                math.sin(half_pitch),
                0.0,
                math.cos(half_pitch),
            ],
            "focal_length_mm": float(camera_profile["focal_length_mm"]),
            "resolution": [int(camera_profile["width"]), int(camera_profile["height"])],
            "max_pose_sync_error_m": max_pose_sync_error_m,
            "max_orientation_sync_error_rad": max_orientation_sync_error_rad,
            "qa_passed": camera_qa,
        },
        operator_readback={
            "operator_id": record["target_operator"],
            "active": active,
            "qa_passed": operator_qa,
            "applied_parameters": parameters,
            "attachment_events": attachment_events,
            "peel_events": peel_events,
            "break_events": break_events,
            "max_applied_per_foot_n": max_applied_per_foot_n,
        },
        scene_readback={
            "qa_passed": bool(compiled["audit"]["passed"])
            and geometry_qa
            and semantic_summary["mean_scene_context_pixel_fraction"] >= 0.08,
            "scene_family": record["scene_family"],
            "scene_source": record["scene_source"],
            "scene_seed": record["scene_seed"],
            "compiled_scene_id": compiled["scene_id"],
            "compiled_source_kind": compiled["source_kind"],
            "stage": stage_readback,
            "semantic_summary_path": str(semantic_path.relative_to(episode_dir)),
            "artifacts": scene_artifacts,
        },
        appearance_readback={
            "qa_passed": appearance_qa,
            "appearance_id": record["appearance_id"],
            "material_family": record["material_family"],
            "surface_state": record["surface_state"],
            "uv_scale": record["uv_scale"],
            "uv_rotation_deg": record["uv_rotation_deg"],
            "uv_offset": record["uv_offset"],
            "albedo_brightness_multiplier": record["albedo_brightness_multiplier"],
            "normal_strength": record["normal_strength"],
            "roughness_multiplier": record["roughness_multiplier"],
            "bound_material_path": appearance_stage_by_view[primary_view_id]["bound_material_path"],
            "omnipbr": primary_binding.omnipbr_parameters(),
            "semantic_summary_path": str(semantic_path.relative_to(episode_dir)),
            "artifacts": appearance_artifacts_by_view[primary_view_id],
            "views": {
                view_id: {
                    "qa_passed": appearance_qa_by_view[view_id],
                    "appearance_id": appearance_views[view_id]["appearance_id"],
                    "material_family": appearance_views[view_id]["material_family"],
                    "surface_state": appearance_views[view_id]["surface_state"],
                    "uv_scale": appearance_views[view_id]["uv_scale"],
                    "uv_rotation_deg": appearance_views[view_id]["uv_rotation_deg"],
                    "uv_offset": appearance_views[view_id]["uv_offset"],
                    "albedo_brightness_multiplier": appearance_views[view_id][
                        "albedo_brightness_multiplier"
                    ],
                    "normal_strength": appearance_views[view_id]["normal_strength"],
                    "roughness_multiplier": appearance_views[view_id]["roughness_multiplier"],
                    "bound_material_path": appearance_stage_by_view[view_id]["bound_material_path"],
                    "omnipbr": appearance_bindings[view_id].omnipbr_parameters(),
                    "semantic_summary_path": str(semantic_paths[view_id].relative_to(episode_dir)),
                    "artifacts": appearance_artifacts_by_view[view_id],
                }
                for view_id in appearance_views
            },
        },
        geometry_readback={
            "qa_passed": geometry_qa,
            "geometry_id": record["geometry_id"],
            "geometry_profile": record["geometry_profile"],
            "physical_realization": record["physical_realization"],
            "operator_prim_paths": stage_readback["operator_prim_paths"],
            "film_panel_paths": stage_readback["film_panel_paths"],
            "footprint_vertex_count": stage_readback["footprint_vertex_count"],
        },
        collection={
            "status": collection_status,
            "simulator": "Isaac Sim",
            "physics_engine": "PhysX GPU",
            "operator_seed": seed,
            "capture_stride_control_steps": capture_stride,
            "created_utc": datetime.now(UTC).isoformat(),
            "schedule_sha256": schedule_sha256,
            "collector_sha256": collector_sha256,
            "runtime_manifest_sha256": runtime_manifest_sha256,
            **(
                {"formal_protocol": formal_protocol_reference}
                if formal_protocol_reference is not None
                else {}
            ),
            "backend_sha256": _sha256(REPO_ROOT / "kino_vla/sim/isaac_policy_backend.py"),
            "policy_sha256": _sha256(REPO_ROOT / "outputs/locomotion/policy.pt"),
        },
    )
    write_collected_manifest(manifest, episode_dir)
    validation = validate_runtime_episode(record, episode_dir=episode_dir, write_validated=True)
    return {
        "episode_id": record["episode_id"],
        "condition": record["condition"],
        "passed": validation.passed,
        "issues": list(validation.issues),
        "episode_dir": str(episode_dir),
        "rgb_frames": len(rgb_frames_by_view[primary_view_id]),
        "appearance_views": len(rgb_frames_by_view),
        "rgb_frames_total": sum(len(frames) for frames in rgb_frames_by_view.values()),
        "proprio_samples": len(feature_rows),
        "attachment_events": attachment_events,
        "peel_events": peel_events,
        "break_events": break_events,
        "max_applied_per_foot_n": max_applied_per_foot_n,
        "fallen": bool(obs.fallen),
        "final_position_xy_m": [float(value) for value in obs.pos],
        "manifest_sha256": _sha256(episode_dir / "manifest.json"),
    }


def main() -> int:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument(
        "--schedule", default="outputs/kinofail_realistic/design_v1/pilot_schedule.jsonl"
    )
    preliminary.add_argument("--corpus-root", default="outputs/kinofail_realistic/corpus_v1")
    pre_args, _ = preliminary.parse_known_args()

    parser = argparse.ArgumentParser(description="Collect one realistic runtime-v5 smoke pair")
    parser.add_argument("--schedule", default=pre_args.schedule)
    parser.add_argument("--corpus-root", default=pre_args.corpus_root)
    parser.add_argument("--counterfactual-group-id")
    parser.add_argument("--overwrite", action="store_true")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app

    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.realistic_scene import (
        build_indoor_adhesion_scene_spec,
        compile_indoor_scene_layers,
    )
    from kino_vla.sim.terrain_materials import (
        appearance_binding_from_record,
        load_terrain_asset_lock,
    )
    from kino_vla.utils.config import CONFIGS_DIR, load_config

    schedule_path = (REPO_ROOT / args.schedule).resolve()
    pair = _select_pair(_read_jsonl(schedule_path), args.counterfactual_group_id)
    representative = pair[0]
    corpus_root = (REPO_ROOT / args.corpus_root).resolve()
    corpus_root.mkdir(parents=True, exist_ok=True)

    data_config = yaml.safe_load(
        (CONFIGS_DIR / "data/kinofail_realistic.yaml").read_text(encoding="utf-8")
    )
    camera_profile = dict(data_config["camera_profile_specs"][representative["camera_profile"]])
    mount = camera_profile["mount_xyz_m"]
    sim_cfg = load_config(
        "sim/go2_skeleton.yaml",
        {
            "perception_cam_width": int(camera_profile["width"]),
            "perception_cam_height": int(camera_profile["height"]),
            "perception_focal_mm": float(camera_profile["focal_length_mm"]),
            "perception_cam_x_m": float(mount[0]),
            "perception_cam_y_m": float(mount[1]),
            "perception_cam_z_m": float(mount[2]),
            "perception_cam_pitch_down_rad": float(camera_profile["pitch_down_rad"]),
        },
    )
    scene_spec = build_indoor_adhesion_scene_spec(
        int(representative["scene_seed"]),
        scene_id=str(representative["scene_family"]),
        source_kind=str(representative["scene_source"]),
        film_color=(0.62, 0.66, 0.64),
        film_emissive=(0.02, 0.025, 0.022),
    )
    scene_cache = (
        corpus_root / "_scene_cache" / f"{scene_spec.scene_id}_{int(representative['scene_seed'])}"
    )
    compiled = compile_indoor_scene_layers(scene_spec, scene_cache)
    if not compiled["audit"]["passed"]:
        raise RuntimeError(f"scene compile audit failed: {compiled['audit']}")

    asset_lock_path = REPO_ROOT / "outputs/assets/terrain_pbr_v1/terrain_assets.lock.json"
    asset_lock = load_terrain_asset_lock(asset_lock_path)
    demo_cfg = load_config("demo/indoor_adhesion_icra.yaml")
    backend = IsaacPolicyBackend(
        sim_cfg,
        np.asarray(demo_cfg.start.pos, dtype=np.float64),
        float(demo_cfg.start.heading),
        perception_cam=True,
    )
    backend.load_realistic_scene(str(compiled["episode_usd"]))

    results = []
    for record in pair:
        manifest_relative = Path(record["required_outputs"]["episode_manifest"])
        episode_dir = corpus_root / manifest_relative.parent
        bindings = {
            str(view["appearance_view_id"]): appearance_binding_from_record(
                view,
                lock=asset_lock,
                asset_root=REPO_ROOT / str(asset_lock["asset_root"]),
            )
            for view in record["appearance_views"]
        }
        result = _collect_episode(
            backend,
            record,
            episode_dir=episode_dir,
            scene_spec=scene_spec,
            compiled=compiled,
            appearance_bindings=bindings,
            asset_lock_path=asset_lock_path,
            demo_cfg=demo_cfg,
            camera_profile=camera_profile,
            overwrite=bool(args.overwrite),
            schedule_sha256=_sha256(schedule_path),
            collector_sha256=_sha256(Path(__file__).resolve()),
        )
        results.append(result)
        print(
            f"[runtime-smoke] {result['condition']}: passed={result['passed']} "
            f"rgb={result['rgb_frames']} proprio={result['proprio_samples']} "
            f"attach={result['attachment_events']}"
        )

    summary = {
        "schema_version": "kinofail.realistic-smoke-collection.v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(bool(result["passed"]) for result in results),
        "publication_status": "collector_smoke_pair_not_formal_pilot",
        "counterfactual_group_id": representative["counterfactual_group_id"],
        "schedule_path": str(schedule_path),
        "schedule_sha256": _sha256(schedule_path),
        "runtime_schema": "kinofail.realistic-runtime.v5",
        "results": results,
        "input_sha256": {
            "collector": _sha256(Path(__file__).resolve()),
            "backend": _sha256(REPO_ROOT / "kino_vla/sim/isaac_policy_backend.py"),
            "runtime_manifest": _sha256(REPO_ROOT / "kino_vla/data/runtime_manifest.py"),
            "scene_compiler": _sha256(REPO_ROOT / "kino_vla/sim/realistic_scene.py"),
            "terrain_materials": _sha256(REPO_ROOT / "kino_vla/sim/terrain_materials.py"),
            "data_config": _sha256(CONFIGS_DIR / "data/kinofail_realistic.yaml"),
            "sim_config": _sha256(CONFIGS_DIR / "sim/go2_skeleton.yaml"),
            "policy": _sha256(REPO_ROOT / "outputs/locomotion/policy.pt"),
        },
    }
    summary_path = corpus_root / "smoke_pair_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    ok = bool(summary["passed"])
    print("PASS: realistic runtime-v5 smoke pair" if ok else "FAIL: runtime-v5 smoke pair")
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
