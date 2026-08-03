#!/usr/bin/env python3
"""Collect shared-prefix, texture-debound C1 causal RGB/proprio snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
from collections import deque
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
    _feature_row,
    _rgb_uint8,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _registry_row(registry: dict[str, Any], scene: str) -> dict[str, Any]:
    rows = [row for row in registry["scenes"] if row["scene_id"] == scene]
    if len(rows) != 1:
        raise ValueError(f"scene registry binding is not unique: {scene}")
    return rows[0]


def _remove_c1_visuals(stage: Any) -> None:
    for path in (
        "/World/C1CompliantCause",
        "/World/C1AdhesionCause",
        "/World/Looks/C1SharedCauseEdge",
    ):
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(path)


def _bind_view(
    stage: Any,
    route_surface_path: str,
    cause_visual: Any,
    binding: Any,
) -> None:
    from kino_vla.sim.terrain_materials import bind_omnipbr_material

    bind_omnipbr_material(stage, route_surface_path, binding)
    for path in cause_visual.texture_target_paths:
        bind_omnipbr_material(stage, path, binding)


def _capture_case(
    backend: Any,
    *,
    record: dict[str, Any],
    frame: Any,
    route_surface_path: str,
    bindings: dict[str, Any],
    out_dir: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    import omni.usd
    from PIL import Image
    from kino_vla.sim.c1_causal_visuals import (
        author_c1_cause_visuals,
        show_c1_cause,
    )
    from kino_vla.sim.realistic_route_controller_v3 import DampedRouteControllerV3

    stage = omni.usd.get_context().get_stage()
    profile = dict(record["physical_nuisance"])
    backend._start_pos = frame.point(
        0.0, float(profile["start_lateral_offset_m"])
    )
    backend._start_heading = (
        float(frame.heading_rad)
        + float(profile["start_heading_offset_rad"])
    )
    obs = backend.deep_reset(int(record["reset_seed"]))
    _remove_c1_visuals(stage)
    region = dict(record["cause_region"])
    visuals = author_c1_cause_visuals(
        stage,
        frame,
        start_progress_m=float(region["start_progress_m"]),
        length_m=float(region["length_m"]),
        half_width_m=float(region["half_width_m"]),
        base_z_m=float(region["base_z_m"]),
        seed=int(record["cause_visual_seed"]),
    )
    controller = DampedRouteControllerV3(
        cross_track_gain_per_s=2.0,
        lateral_velocity_damping=1.0,
        heading_gain_per_s=2.0,
        lateral_limit_mps=0.28,
        yaw_rate_limit_radps=0.6,
    )
    causes = [str(value["target_operator"]) for value in record["causes"]]
    views = [str(value["appearance_view_id"]) for value in record["appearance_views"]]
    frames: dict[tuple[str, str], deque[tuple[np.ndarray, float]]] = {
        (cause, view): deque(maxlen=5)
        for cause in causes
        for view in views
    }
    features: list[np.ndarray] = []
    proprio_times: list[float] = []
    render_state_max_abs_delta = 0.0
    step = 0
    while step < 140 and not obs.fallen:
        command, _ = controller.command(
            frame,
            position_xy_m=obs.pos,
            heading_rad=obs.heading,
            velocity_body_xy_mps=obs.vel_body,
            forward_speed_mps=float(profile["forward_speed_mps"]),
            target_lateral_offset_m=0.0,
        )
        obs = backend.step(command)
        progress, _ = frame.project(obs.pos)
        features.append(_feature_row(backend, obs))
        proprio_times.append(float(obs.t))
        if step % 5 == 0 and progress >= 0.03:
            before = np.asarray(
                [
                    float(obs.t),
                    *np.asarray(obs.pos, dtype=np.float64).tolist(),
                    float(obs.heading),
                    float(obs.base_height),
                    float(obs.tilt),
                ],
                dtype=np.float64,
            )
            # Counterbalance render order without allowing it to alter the
            # shared physics prefix.
            render_causes = (
                causes
                if (int(record["cause_visual_seed"]) + step) % 2 == 0
                else list(reversed(causes))
            )
            for cause in render_causes:
                visual = show_c1_cause(stage, visuals, cause)
                for view in views:
                    _bind_view(
                        stage,
                        route_surface_path,
                        visual,
                        bindings[view],
                    )
                    # RTX/DLSS retains temporal history across visibility and
                    # material switches.  A single render left a faint image
                    # of the previous counterfactual in the next one.  Flush
                    # that render-only history without stepping physics.
                    for _ in range(4):
                        backend._env.sim.render()
                    capture = backend.capture_perception()
                    if capture is None:
                        raise RuntimeError("C1 causal Go2-front camera returned no frame")
                    frames[(cause, view)].append(
                        (
                            _rgb_uint8(capture["rgb"]),
                            float(capture["camera_pose_source_timestamp_s"]),
                        )
                    )
            after_obs = backend._make_obs()
            after = np.asarray(
                [
                    float(after_obs.t),
                    *np.asarray(after_obs.pos, dtype=np.float64).tolist(),
                    float(after_obs.heading),
                    float(after_obs.base_height),
                    float(after_obs.tilt),
                ],
                dtype=np.float64,
            )
            render_state_max_abs_delta = max(
                render_state_max_abs_delta,
                float(np.max(np.abs(before - after))),
            )
        step += 1
        if (
            progress >= float(record["decision_progress_m"])
            and len(features) >= 21
            and all(len(value) == 5 for value in frames.values())
        ):
            break
    if obs.fallen:
        raise RuntimeError("C1 shared prefix fell before the decision boundary")
    if len(features) < 21 or not all(len(value) == 5 for value in frames.values()):
        raise RuntimeError("C1 shared prefix has an incomplete observable window")
    proprio = np.asarray(features[-21:], dtype=np.float32)
    proprio_time = np.asarray(proprio_times[-21:], dtype=np.float64)
    shared_proprio_sha = _array_sha(proprio)
    arrays: dict[str, np.ndarray] = {}
    samples: list[dict[str, Any]] = []
    image_hashes: dict[str, str] = {}
    for cause_info in record["causes"]:
        cause = str(cause_info["target_operator"])
        category = str(cause_info["attribution_category"])
        for view_info in record["appearance_views"]:
            view = str(view_info["appearance_view_id"])
            sample_id = f"{record['case_id']}__{cause}__{view}"
            rgb = np.stack([value[0] for value in frames[(cause, view)]])
            rgb_time = np.asarray(
                [value[1] for value in frames[(cause, view)]],
                dtype=np.float64,
            )
            arrays[f"{sample_id}__rgb"] = rgb
            arrays[f"{sample_id}__rgb_timestamp_s"] = rgb_time
            arrays[f"{sample_id}__proprio"] = proprio
            arrays[f"{sample_id}__proprio_timestamp_s"] = proprio_time
            folder = out_dir / cause / view
            folder.mkdir(parents=True, exist_ok=True)
            image_paths = []
            for index, image in enumerate(rgb):
                path = folder / f"{index:06d}.png"
                Image.fromarray(image).save(path)
                relative = str(path.relative_to(out_dir))
                image_paths.append(relative)
                image_hashes[relative] = _sha(path)
            samples.append(
                {
                    "sample_id": sample_id,
                    "case_id": record["case_id"],
                    "scene_cluster": record["scene_cluster"],
                    "domain": record["domain"],
                    "split": record["split"],
                    "profile_index": record["profile_index"],
                    "camera_profile": record["camera_profile"],
                    "target_operator": cause,
                    "attribution_category": category,
                    "visible_physical_cue": cause_info["visible_physical_cue"],
                    "appearance_view_id": view,
                    "material_family": view_info["material_family"],
                    "material_asset_id": view_info["material_asset_id"],
                    "shared_across_causes": True,
                    "rgb_shape": list(rgb.shape),
                    "proprio_shape": list(proprio.shape),
                    "shared_proprio_sha256": shared_proprio_sha,
                    "rgb_paths": image_paths,
                }
            )
    arrays_path = out_dir / "observables.npz"
    np.savez_compressed(arrays_path, **arrays)
    per_cause_hashes = {
        cause: {
            sample["shared_proprio_sha256"]
            for sample in samples
            if sample["target_operator"] == cause
        }
        for cause in causes
    }
    checks = {
        "six_samples": len(samples) == 6,
        "five_rgb_frames_each": all(
            sample["rgb_shape"][0] == 5 for sample in samples
        ),
        "twenty_one_proprio_samples_each": all(
            sample["proprio_shape"][0] == 21 for sample in samples
        ),
        "proprio_byte_identical_across_causes": (
            all(len(value) == 1 for value in per_cause_hashes.values())
            and len({next(iter(value)) for value in per_cause_hashes.values()}) == 1
        ),
        "render_did_not_advance_or_perturb_physics": (
            render_state_max_abs_delta == 0.0
        ),
        "same_material_views_across_causes": all(
            len(
                {
                    sample["material_family"]
                    for sample in samples
                    if sample["appearance_view_id"] == view
                }
            )
            == 1
            for view in views
        ),
        "decision_before_cause_region": (
            float(record["decision_progress_m"])
            < float(region["start_progress_m"])
        ),
    }
    manifest = {
        "schema_version": "kinofail.realistic-c1-causal-case.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "case_id": record["case_id"],
        "passed": all(checks.values()),
        "checks": checks,
        "protocol_id": protocol["protocol_id"],
        "scene_cluster": record["scene_cluster"],
        "domain": record["domain"],
        "split": record["split"],
        "shared_prefix": {
            "reset_seed": record["reset_seed"],
            "decision_progress_m": record["decision_progress_m"],
            "cause_region_start_progress_m": region["start_progress_m"],
            "render_state_max_abs_delta": render_state_max_abs_delta,
            "proprio_sha256": shared_proprio_sha,
            "proprio_feature_names": list(FEATURE_NAMES),
            "physics_rollouts": 1,
            "visual_counterfactual_causes": 2,
        },
        "cause_visuals": {
            "O2_compliance": visuals.compliant.cue_kind,
            "O4_tether": visuals.adhesion.cue_kind,
            "base_material_assignment": "identical_across_causes",
            "collision_authored": False,
        },
        "samples": samples,
        "artifacts": {
            "observables": {
                "path": str(arrays_path.relative_to(out_dir)),
                "sha256": _sha(arrays_path),
            },
            "image_sha256": image_hashes,
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "case_id": record["case_id"],
        "scene_cluster": record["scene_cluster"],
        "passed": manifest["passed"],
        "render_state_max_abs_delta": render_state_max_abs_delta,
        "shared_proprio_sha256": shared_proprio_sha,
        "manifest": str(manifest_path),
    }


def main() -> int:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--schedule", required=True)
    preliminary.add_argument("--scene-registry", required=True)
    preliminary.add_argument("--asset-lock", required=True)
    preliminary.add_argument("--protocol", required=True)
    preliminary.add_argument("--out", required=True)
    preliminary.add_argument("--scene", required=True)
    pre, _ = preliminary.parse_known_args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", default=pre.schedule)
    parser.add_argument("--scene-registry", default=pre.scene_registry)
    parser.add_argument("--asset-lock", default=pre.asset_lock)
    parser.add_argument("--protocol", default=pre.protocol)
    parser.add_argument("--out", default=pre.out)
    parser.add_argument("--scene", default=pre.scene)
    parser.add_argument("--resume", action="store_true")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app
    passed = False
    try:
        from kino_vla.sim.isaac_o4_v3_backend import IsaacPolicyBackendO4V3
        from kino_vla.sim.realistic_route_protocol_v2 import scene_route_binding
        from kino_vla.sim.terrain_materials import (
            appearance_binding_from_record,
            load_terrain_asset_lock,
        )
        from kino_vla.utils.config import CONFIGS_DIR, load_config

        schedule_path = (ROOT / args.schedule).resolve()
        registry_path = (ROOT / args.scene_registry).resolve()
        lock_path = (ROOT / args.asset_lock).resolve()
        protocol_path = (ROOT / args.protocol).resolve()
        out_root = (ROOT / args.out).resolve()
        schedule = [
            row
            for row in _jsonl(schedule_path)
            if row["scene_cluster"] == args.scene
        ]
        if not schedule:
            raise RuntimeError(f"no C1 cases for scene {args.scene}")
        protocol = _json(protocol_path)
        for key, actual in (
            ("schedule_sha256", _sha(schedule_path)),
            ("design_audit_sha256", _sha(schedule_path.parent / "audit.json")),
            ("scene_registry_sha256", _sha(registry_path)),
            ("asset_lock_sha256", _sha(lock_path)),
            ("collector_sha256", _sha(Path(__file__).resolve())),
            (
                "visual_helper_sha256",
                _sha(ROOT / "kino_vla/sim/c1_causal_visuals.py"),
            ),
        ):
            if protocol.get(key) != actual:
                raise RuntimeError(f"C1 formal protocol {key} mismatch")
        if {
            str(row["benchmark_id"]) for row in schedule
        } != {str(protocol["design_tag"])}:
            raise RuntimeError("C1 schedule design tag does not match the protocol")
        registry = _json(registry_path)
        scene = _registry_row(registry, args.scene)
        episode = ROOT / scene["episode_usd"]
        compiled_path = ROOT / scene["compiled_audit"]
        if (
            _sha(episode) != scene["episode_sha256"]
            or _sha(compiled_path) != scene["compiled_audit_sha256"]
        ):
            raise RuntimeError("C1 scene registry hash mismatch")
        compiled = _json(compiled_path)
        scene_binding = scene_route_binding(compiled)
        benchmark = yaml.safe_load(
            (CONFIGS_DIR / "data/kinofail_realistic.yaml").read_text(
                encoding="utf-8"
            )
        )
        camera_profiles = {row["camera_profile"] for row in schedule}
        if len(camera_profiles) != 1:
            raise RuntimeError("C1 shared-scene collection needs one camera profile")
        camera = dict(
            benchmark["camera_profile_specs"][next(iter(camera_profiles))]
        )
        mount = camera["mount_xyz_m"]
        config = load_config(
            "sim/go2_skeleton.yaml",
            {
                "perception_cam_width": camera["width"],
                "perception_cam_height": camera["height"],
                "perception_focal_mm": camera["focal_length_mm"],
                "perception_cam_x_m": mount[0],
                "perception_cam_y_m": mount[1],
                "perception_cam_z_m": mount[2],
                "perception_cam_pitch_down_rad": camera["pitch_down_rad"],
            },
        )
        backend = IsaacPolicyBackendO4V3(
            config,
            scene_binding.frame.point(0.0, 0.0),
            scene_binding.frame.heading_rad,
            perception_cam=True,
        )
        scene_prim = backend.load_realistic_scene(str(episode))
        route_surface_path = scene_binding.route_surface_prim_path(scene_prim)
        lock = load_terrain_asset_lock(lock_path)
        asset_root = Path(lock["asset_root"])
        if not asset_root.is_absolute():
            asset_root = ROOT / asset_root
        results = []
        for record in schedule:
            case_dir = out_root / str(record["case_id"])
            manifest_path = case_dir / "manifest.json"
            if args.resume and manifest_path.exists():
                manifest = _json(manifest_path)
                if manifest.get("passed") is True:
                    results.append(
                        {
                            "case_id": record["case_id"],
                            "scene_cluster": record["scene_cluster"],
                            "passed": True,
                            "resumed": True,
                        }
                    )
                    continue
            if case_dir.exists():
                shutil.rmtree(case_dir)
            case_dir.mkdir(parents=True)
            bindings = {
                str(view["appearance_view_id"]): appearance_binding_from_record(
                    view, lock=lock, asset_root=asset_root
                )
                for view in record["appearance_views"]
            }
            result = _capture_case(
                backend,
                record=record,
                frame=scene_binding.frame,
                route_surface_path=route_surface_path,
                bindings=bindings,
                out_dir=case_dir,
                protocol=protocol,
            )
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
        passed = len(results) == len(schedule) and all(
            row["passed"] for row in results
        )
        summary = {
            "schema_version": "kinofail.realistic-c1-causal-scene.v1",
            "scene_cluster": args.scene,
            "passed": passed,
            "case_count": len(results),
            "results": results,
        }
        summary_dir = out_root / "scene_summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        (summary_dir / f"{args.scene}.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        sys.stdout.flush()
        closer = threading.Thread(target=app.close, daemon=True)
        closer.start()
        closer.join(timeout=15.0)
    os._exit(0 if passed else 2)


if __name__ == "__main__":
    main()
