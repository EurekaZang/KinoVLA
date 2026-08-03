#!/usr/bin/env python3
"""Collect four matched A7 render arms on frozen O5 physical trajectories.

The adapter retains the scale-v8 episode and its ordinary three synchronized PBR views.  At every
capture timestamp it additionally records: the primary realistic/body-fixed view by reference,
the Isaac default grid, a deterministic procedural floor, and the realistic scene from the legacy
fixed world camera.  Rendering does not advance physics, so proprioception and operator state are
identical across the four A7 arms.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import types
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v1.py"
V8 = ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v8.py"
EXPECTED_V1_SHA256 = "3f1b86cfda574dc4185375a6653e7b8ba71be37710d28041033e1643de0f820a"
EXPECTED_V8_SHA256 = "e5b03314e055230d224f7c45e5e0ca91608734bf4c70b39c1395013d71326570"
PROFILE_INDEX = 2
A7_ARMS = (
    "realistic_scene_scanned_pbr_body_fixed",
    "default_grid_body_fixed",
    "realistic_scene_procedural_floor_body_fixed",
    "realistic_scene_scanned_pbr_world_follow",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_v8():
    actual = _sha256(V8)
    if actual != EXPECTED_V8_SHA256:
        raise RuntimeError(f"A7 visual-replay dependency hash mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("kinofail_a7_visual_scale_v8_frozen", V8)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen collector dependency: {V8}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_visual_replay_implementation():
    if _sha256(V1) != EXPECTED_V1_SHA256:
        raise RuntimeError("A7 visual-replay scale-v1 dependency mismatch")
    source = V1.read_text(encoding="utf-8")
    replacements = {
        "    capture_stride = 20\n": "    capture_stride = 5\n",
        (
            '    seed = int(record["operator_seed"])\n'
            "    obs = backend.deep_reset(seed)\n"
        ): (
            "    nuisance = _physical_nuisance(record)\n"
            "    backend._start_pos = frame.point(\n"
            '        float(nuisance["start_progress_m"]),\n'
            '        float(nuisance["start_lateral_offset_m"]),\n'
            "    )\n"
            "    backend._start_heading = (\n"
            '        float(frame.heading_rad) + float(nuisance["start_heading_offset_rad"])\n'
            "    )\n"
            '    seed = int(record["operator_seed"])\n'
            "    obs = backend.deep_reset(seed)\n"
        ),
        "            velocity_body_xy_mps=obs.vel_body, forward_speed_mps=0.24,\n": (
            "            velocity_body_xy_mps=obs.vel_body,\n"
            '            forward_speed_mps=float(nuisance["forward_speed_mps"]),\n'
        ),
        "            target_lateral_offset_m=0.0,\n": (
            '            target_lateral_offset_m=float(\n'
            '                nuisance["controller_target_lateral_offset_m"]\n'
            "            ),\n"
        ),
        (
            '        folder = episode_dir / ("rgb" if row["is_primary"] else '
            'f"rgb_views/{view_id}")\n'
            "        folder.mkdir(parents=True, exist_ok=True)\n"
            "    nuisance = _physical_nuisance(record)\n"
        ): (
            '        folder = episode_dir / ("rgb" if row["is_primary"] else '
            'f"rgb_views/{view_id}")\n'
            "        folder.mkdir(parents=True, exist_ok=True)\n"
            "    a7_render = _prepare_a7_render(\n"
            "        episode_dir, record, frame, route_surface_path\n"
            "    )\n"
            "    nuisance = _physical_nuisance(record)\n"
        ),
        (
            "            bind_omnipbr_material(stage, route_surface_path, primary)\n"
            "        step += 1\n"
        ): (
            "            bind_omnipbr_material(stage, route_surface_path, primary)\n"
            "            _capture_a7_render_arms(\n"
            "                a7_render, backend=backend, stage=stage,\n"
            "                route_surface_path=route_surface_path, primary_binding=primary,\n"
            "                primary_entry=rgb[primary_id][-1], frame_index=frame_index,\n"
            '                timestamp_s=float(rgb[primary_id][-1]["timestamp_s"]),\n'
            "            )\n"
            "        step += 1\n"
        ),
        "    write_collected_manifest(manifest, episode_dir)\n": (
            '    manifest["collection"]["physical_nuisance"] = dict(nuisance)\n'
            '    manifest["a7_matched_render_readback"] = _finalize_a7_render(\n'
            "        a7_render, episode_dir=episode_dir\n"
            "    )\n"
            "    write_collected_manifest(manifest, episode_dir)\n"
        ),
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f"A7 visual-replay patch point is not unique: {old!r}")
        source = source.replace(old, new)
    module = types.ModuleType("kinofail_a7_visual_replay_10hz")
    module.__file__ = str(V1)
    module.__package__ = "scripts"
    exec(compile(source, str(V1), "exec"), module.__dict__)
    return module


def _install_fixed_nuisance_contract(implementation, v8) -> None:
    profile = dict(v8.PHYSICAL_NUISANCE_PROFILES[PROFILE_INDEX])

    def physical_nuisance(record: dict[str, Any]) -> dict[str, Any]:
        if int(record.get("physical_nuisance_profile_index", -1)) != PROFILE_INDEX:
            raise RuntimeError("A7 visual-replay schedule must use nuisance profile 2")
        return {
            **profile,
            "scene_seed": int(record["scene_seed"]),
            "operator_seed": int(record["operator_seed"]),
            "pair_shared": True,
            "derivation": "frozen A7 matched-design physical_nuisance_profile_index=2",
        }

    implementation._physical_nuisance = physical_nuisance


def _visibility_state(prim) -> tuple[bool, Any]:
    from pxr import UsdGeom

    attr = UsdGeom.Imageable(prim).GetVisibilityAttr()
    return bool(attr.HasAuthoredValueOpinion()), attr.Get()


def _restore_visibility(prim, state: tuple[bool, Any]) -> None:
    from pxr import UsdGeom

    attr = UsdGeom.Imageable(prim).GetVisibilityAttr()
    authored, value = state
    if authored:
        attr.Set(value)
    else:
        attr.Clear()


def _imageable_subtree(stage, root_paths: tuple[str, ...]) -> list[Any]:
    from pxr import Usd, UsdGeom

    imageables = []
    for root_path in root_paths:
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            continue
        imageables.extend(
            prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Imageable)
        )
    return imageables


def _create_procedural_floor_material(
    stage, *, texture_path: Path, seed: int
) -> str:
    from PIL import Image
    from pxr import Sdf, UsdShade

    rng = np.random.default_rng(int(seed))
    yy, xx = np.mgrid[:512, :512]
    checker = ((xx // 32 + yy // 32) % 2).astype(np.float32)
    grain = rng.normal(0.0, 0.025, size=(512, 512)).astype(np.float32)
    base = np.where(checker[..., None] > 0.5, 0.68, 0.36)
    colors = np.asarray([0.62, 0.57, 0.48], dtype=np.float32)
    pixels = np.clip((base + grain[..., None]) * colors, 0.0, 1.0)
    texture_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((pixels * 255.0).astype(np.uint8), mode="RGB").save(texture_path)

    material_path = "/World/Looks/A7ProceduralFloor"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PBR")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.82)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    reader = UsdShade.Shader.Define(stage, f"{material_path}/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture = UsdShade.Shader.Define(stage, f"{material_path}/diffuseTexture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture_path))
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material_path


def _bind_material_path(stage, prim_path: str, material_path: str) -> None:
    from pxr import UsdShade

    prim = stage.GetPrimAtPath(prim_path)
    material = UsdShade.Material.Get(stage, material_path)
    if not prim.IsValid() or not material.GetPrim().IsValid():
        raise RuntimeError("A7 procedural material binding target is invalid")
    UsdShade.MaterialBindingAPI(prim).Bind(material)


def _prepare_a7_render(
    episode_dir: Path, record: dict[str, Any], frame, route_surface_path: str
) -> dict[str, Any]:
    del route_surface_path
    import omni.usd

    if record["target_operator"] != "O5_payload":
        raise RuntimeError("A7 visual replay is frozen to O5_payload")
    root = episode_dir / "a7_matched_render"
    for arm in A7_ARMS[1:]:
        (root / arm).mkdir(parents=True, exist_ok=True)
    texture_path = (
        episode_dir / "provenance/a7_matched_render/procedural_floor_checker.png"
    )
    material_path = _create_procedural_floor_material(
        omni.usd.get_context().get_stage(),
        texture_path=texture_path,
        seed=int(record["appearance_seed"]),
    )
    eye_xy = frame.point(-0.80, 0.0)
    target_xy = frame.point(1.25, 0.0)
    return {
        "episode_dir": episode_dir,
        "record": record,
        "root": root,
        "procedural_texture_path": texture_path,
        "procedural_material_path": material_path,
        "world_eye": [float(eye_xy[0]), float(eye_xy[1]), 1.20],
        "world_target": [float(target_xy[0]), float(target_xy[1]), 0.0],
        "frames": {arm: [] for arm in A7_ARMS},
    }


def _save_capture(
    state: dict[str, Any],
    arm: str,
    capture: dict[str, Any],
    *,
    frame_index: int,
    timestamp_s: float,
) -> None:
    from PIL import Image

    relative = (
        Path("a7_matched_render") / arm / f"{frame_index:06d}.png"
    )
    image = np.asarray(capture["rgb"])
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(state["episode_dir"] / relative)
    state["frames"][arm].append(
        {
            "path": str(relative),
            "timestamp_s": float(timestamp_s),
            "body_fixed": bool(capture["body_fixed"]),
            "pose_sync_method": str(capture["pose_sync_method"]),
            "eye": np.asarray(capture["eye"], dtype=float).tolist(),
            "target": np.asarray(capture["target"], dtype=float).tolist(),
        }
    )


def _capture_a7_render_arms(
    state: dict[str, Any],
    *,
    backend,
    stage,
    route_surface_path: str,
    primary_binding,
    primary_entry: dict[str, Any],
    frame_index: int,
    timestamp_s: float,
) -> None:
    from pxr import UsdGeom
    from kino_vla.sim.terrain_materials import bind_omnipbr_material

    base_entry = dict(primary_entry)
    base_entry.update(
        {
            "body_fixed": True,
            "pose_sync_method": "rigid_base_transform_each_capture",
            "referenced_from_primary_runtime_view": True,
        }
    )
    state["frames"][A7_ARMS[0]].append(base_entry)

    scene = stage.GetPrimAtPath("/World/KinoIndoor")
    if not scene.IsValid() or not scene.IsA(UsdGeom.Imageable):
        raise RuntimeError("A7 visual replay requires /World/KinoIndoor")
    scene_visibility = _visibility_state(scene)
    ground_prims = _imageable_subtree(
        stage, ("/World/ground", "/World/defaultGroundPlane")
    )
    if not ground_prims:
        raise RuntimeError("A7 default-grid arm has no Isaac ground visual")
    ground_visibility = [(prim, _visibility_state(prim)) for prim in ground_prims]
    light_prims = [
        stage.GetPrimAtPath(path)
        for path in getattr(backend, "_disabled_default_sky_lights", [])
        if stage.GetPrimAtPath(path).IsValid()
        and stage.GetPrimAtPath(path).IsA(UsdGeom.Imageable)
    ]
    light_visibility = [(prim, _visibility_state(prim)) for prim in light_prims]
    try:
        backend._perception_manual_aim = False
        UsdGeom.Imageable(scene).MakeInvisible()
        for prim in ground_prims:
            UsdGeom.Imageable(prim).MakeVisible()
        for prim in light_prims:
            UsdGeom.Imageable(prim).MakeVisible()
        backend._env.sim.render()
        capture = backend.capture_perception()
        if capture is None:
            raise RuntimeError("A7 default-grid camera returned no frame")
        _save_capture(
            state, A7_ARMS[1], capture,
            frame_index=frame_index, timestamp_s=timestamp_s,
        )
    finally:
        _restore_visibility(scene, scene_visibility)
        for prim, visibility in ground_visibility:
            _restore_visibility(prim, visibility)
        for prim, visibility in light_visibility:
            _restore_visibility(prim, visibility)

    backend._perception_manual_aim = False
    _bind_material_path(
        stage, route_surface_path, state["procedural_material_path"]
    )
    backend._env.sim.render()
    capture = backend.capture_perception()
    if capture is None:
        raise RuntimeError("A7 procedural-floor camera returned no frame")
    _save_capture(
        state, A7_ARMS[2], capture,
        frame_index=frame_index, timestamp_s=timestamp_s,
    )

    bind_omnipbr_material(stage, route_surface_path, primary_binding)
    backend.aim_perception_camera(
        np.asarray(state["world_eye"], dtype=np.float64),
        np.asarray(state["world_target"], dtype=np.float64),
    )
    backend._env.sim.render()
    capture = backend.capture_perception()
    if capture is None:
        raise RuntimeError("A7 world-follow camera returned no frame")
    _save_capture(
        state, A7_ARMS[3], capture,
        frame_index=frame_index, timestamp_s=timestamp_s,
    )
    backend._perception_manual_aim = False
    bind_omnipbr_material(stage, route_surface_path, primary_binding)


def _finalize_a7_render(
    state: dict[str, Any], *, episode_dir: Path
) -> dict[str, Any]:
    arms = {}
    for arm, rows in state["frames"].items():
        entries = []
        for row in rows:
            path = episode_dir / row["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            entries.append({**row, "sha256": _sha256(path)})
        arms[arm] = {
            "frames": entries,
            "frame_count": len(entries),
            "sequence_sha256": hashlib.sha256(
                "\n".join(entry["sha256"] for entry in entries).encode("ascii")
            ).hexdigest(),
        }
    counts = {row["frame_count"] for row in arms.values()}
    timestamps = {
        tuple(float(entry["timestamp_s"]) for entry in row["frames"])
        for row in arms.values()
    }
    result = {
        "schema_version": "kinofail.realistic-a7-matched-render.v1",
        "matched_without_physics_advance": True,
        "render_arms": list(A7_ARMS),
        "all_arm_frame_counts_equal": len(counts) == 1,
        "timestamps_identical_across_arms": len(timestamps) == 1,
        "world_follow_definition": {
            "mode": "fixed_world_look_at",
            "eye_xyz_m": state["world_eye"],
            "target_xyz_m": state["world_target"],
        },
        "procedural_texture": {
            "path": str(state["procedural_texture_path"].relative_to(episode_dir)),
            "sha256": _sha256(state["procedural_texture_path"]),
            "generator": "seeded 32-pixel checker plus Gaussian grain",
        },
        "arms": arms,
    }
    if not result["all_arm_frame_counts_equal"] or not result[
        "timestamps_identical_across_arms"
    ]:
        raise RuntimeError("A7 render arms are not timestamp matched")
    manifest_path = (
        episode_dir / "provenance/a7_matched_render/render_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **result,
        "render_manifest": {
            "path": str(manifest_path.relative_to(episode_dir)),
            "sha256": _sha256(manifest_path),
        },
    }


def main() -> None:
    v8 = _load_v8()
    v4 = v8._load_v4()
    v4._install_runtime_validation_adapter()
    implementation = _load_visual_replay_implementation()
    _install_fixed_nuisance_contract(implementation, v8)
    # ``_collect_one`` is compiled into the isolated dynamic V1 module above, so its
    # inserted helper calls resolve against that module's globals rather than this
    # adapter module.  Bind the three frozen render helpers explicitly before collection.
    implementation._prepare_a7_render = _prepare_a7_render
    implementation._capture_a7_render_arms = _capture_a7_render_arms
    implementation._finalize_a7_render = _finalize_a7_render
    v4._install_reachable_exposure_contract(implementation)
    v8._install_route_aligned_o9(implementation)
    implementation.__file__ = __file__
    implementation.main()


if __name__ == "__main__":
    main()
