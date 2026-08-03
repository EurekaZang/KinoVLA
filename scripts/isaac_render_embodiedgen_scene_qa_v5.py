#!/usr/bin/env python3
"""Phase-aware RTX QA with calibrated near clipping for lighting-v4 scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from isaac_render_embodiedgen_scene_qa import (
    _contact_sheet,
    _image_metrics,
    _look_at_quat_wxyz,
)
from kino_vla.eval.embodiedgen_visual_contract_v5 import (
    FAR_CLIP_M,
    NEAR_CLIP_M,
    SCHEMA_VERSION as VIEW_SCHEMA_VERSION,
    evaluate_phase_aware_scene_views,
)
from kino_vla.sim.embodiedgen_scene_v3 import pitch_aware_camera_views
from kino_vla.sim.embodiedgen_scene_v4 import LIGHTING_SCHEMA, ROUTE_SURFACE_SCHEMA


os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-usd", type=Path, required=True)
    parser.add_argument("--compiled-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--warmup-frames", type=int, default=36)
    args, _ = parser.parse_known_args()

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "enable_cameras": True,
            "width": args.width,
            "height": args.height,
        }
    )
    passed = False
    try:
        from isaacsim.core.api import World
        from isaacsim.sensors.camera import Camera
        from PIL import Image
        from pxr import Usd, UsdPhysics

        episode = args.episode_usd.resolve()
        compiled_path = args.compiled_audit.resolve()
        compiled = json.loads(compiled_path.read_text(encoding="utf-8"))
        if (
            compiled.get("passed") is not True
            or compiled.get("schema_version")
            != "kinofail.embodiedgen-compiled-scene.v4-development"
        ):
            raise RuntimeError("compiled scene is not a passed v4 development scene")
        if compiled["files"].get(episode.name) != _sha256(episode):
            raise RuntimeError("episode USD no longer matches compiled-scene audit")

        appearance_path = compiled_path.parent / "appearance_v4.usda"
        appearance = Usd.Stage.Open(str(appearance_path))
        surface_prim = appearance.GetPrimAtPath(
            "/KinoScene/AppearanceV4/RenderOnlyRouteSurface"
        )
        maps = compiled["appearance_contract"]["material"]["maps"]
        structural_checks = {
            "appearance_schema": compiled["appearance_contract"].get("schema")
            == ROUTE_SURFACE_SCHEMA,
            "lighting_schema": compiled["lighting_contract"].get("schema")
            == LIGHTING_SCHEMA,
            "route_surface_visual_only": compiled["appearance_contract"].get(
                "visual_intervention_only"
            )
            is True,
            "route_surface_declares_no_collision": compiled["appearance_contract"].get(
                "collision_authored"
            )
            is False,
            "route_surface_prim_present": surface_prim.IsValid(),
            "route_surface_has_no_collision_api": surface_prim.IsValid()
            and not surface_prim.HasAPI(UsdPhysics.CollisionAPI),
            "seven_route_envelope_panels": compiled["lighting_contract"]
            .get("route_envelope_panels", {})
            .get("count")
            == 7,
            "material_maps_hash_verified": all(
                Path(spec["absolute_path"]).is_file()
                and _sha256(Path(spec["absolute_path"])) == spec["sha256"]
                for spec in maps.values()
            ),
        }
        if not all(structural_checks.values()):
            raise RuntimeError(f"v5 structural checks failed: {structural_checks}")

        views = pitch_aware_camera_views(
            compiled["route"]["waypoints_xy_m"],
            float(compiled["visual_metrics"]["floor_z_m"]),
        )
        args.out.mkdir(parents=True, exist_ok=True)
        world = World(stage_units_in_meters=1.0)
        reference = world.stage.DefinePrim("/World/EmbodiedGenEpisode", "Xform")
        reference.GetReferences().AddReference(str(episode))
        first_eye = np.asarray(views[0]["eye_xyz_m"], dtype=np.float64)
        camera = Camera(
            prim_path="/World/KinoPhaseAwareQACamera",
            position=first_eye,
            resolution=(args.width, args.height),
        )
        world.reset()
        camera.initialize()
        camera.set_focal_length(19.0)
        camera.set_horizontal_aperture(24.0)
        default_clipping_range = [float(value) for value in camera.get_clipping_range()]
        camera.set_clipping_range(
            near_distance=NEAR_CLIP_M, far_distance=FAR_CLIP_M
        )
        clipping_range = [float(value) for value in camera.get_clipping_range()]
        for _ in range(args.warmup_frames):
            world.step(render=True)

        records = []
        arrays: dict[str, np.ndarray] = {}
        paths: list[Path] = []
        for view in views:
            eye = np.asarray(view["eye_xyz_m"], dtype=np.float64)
            target = np.asarray(view["target_xyz_m"], dtype=np.float64)
            quaternion = _look_at_quat_wxyz(eye, target)
            camera.set_world_pose(eye, quaternion, camera_axes="world")
            for _ in range(args.warmup_frames):
                world.step(render=True)
            rgba = np.asarray(camera.get_rgba())
            if rgba.shape[:2] != (args.height, args.width) or rgba.shape[-1] < 3:
                raise RuntimeError(f"invalid RTX frame for {view['name']}: {rgba.shape}")
            rgb = rgba[..., :3].astype(np.float32)
            if rgb.max() > 1.5:
                rgb /= 255.0
            rgb = np.clip(rgb, 0.0, 1.0)
            image_path = args.out / f"{view['name']}.png"
            Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8)).save(
                image_path, optimize=True
            )
            metrics = _image_metrics(rgb)
            record = {
                **view,
                "quaternion_wxyz": quaternion.tolist(),
                "image": image_path.name,
                "image_sha256": _sha256(image_path),
                "metrics": metrics,
            }
            records.append(record)
            arrays[view["name"]] = rgb
            paths.append(image_path)
            print(f"[rtx-v5] {view['name']}: {metrics}", flush=True)

        canonical = [
            record for record in records if record["kind"] != "go2_body_pitch_stress"
        ]
        canonical_pairwise_l1 = {}
        for index, first in enumerate(canonical):
            for second in canonical[index + 1 :]:
                key = f"{first['name']}__{second['name']}"
                canonical_pairwise_l1[key] = float(
                    np.abs(arrays[first["name"]] - arrays[second["name"]]).mean()
                )
        visual_checks = evaluate_phase_aware_scene_views(
            records, canonical_pairwise_l1, clipping_range
        )
        sheet_path = args.out / "phase_aware_route_surface_contact_sheet.png"
        _contact_sheet(paths, sheet_path)
        audit = {
            "schema_version": "kinofail.embodiedgen-rtx-scene-qa.v5-development",
            "view_contract_schema": VIEW_SCHEMA_VERSION,
            "created_utc": datetime.now(UTC).isoformat(),
            "renderer": "Isaac Sim RTX Camera",
            "episode_usd": str(episode),
            "episode_usd_sha256": _sha256(episode),
            "compiled_audit": str(compiled_path),
            "compiled_audit_sha256": _sha256(compiled_path),
            "resolution": [args.width, args.height],
            "camera_intrinsics": {
                "focal_length_mm": 19.0,
                "horizontal_aperture_mm": 24.0,
                "default_clipping_range_m": default_clipping_range,
                "calibrated_clipping_range_m": clipping_range,
            },
            "appearance": compiled["appearance_contract"],
            "lighting": compiled["lighting_contract"],
            "views": records,
            "canonical_pairwise_mean_absolute_rgb": canonical_pairwise_l1,
            "structural_checks": structural_checks,
            "visual_checks": visual_checks,
            "passed": all(structural_checks.values()) and all(visual_checks.values()),
            "admission_state": (
                "phase_aware_route_surface_v5_passed_development_only"
                if all(structural_checks.values()) and all(visual_checks.values())
                else "phase_aware_route_surface_v5_failed_development"
            ),
            "contact_sheet": sheet_path.name,
            "contact_sheet_sha256": _sha256(sheet_path),
        }
        audit_path = args.out / "rtx_scene_audit.json"
        audit_path.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "passed": audit["passed"],
                    "structural_checks": structural_checks,
                    "visual_checks": visual_checks,
                    "audit": str(audit_path),
                },
                indent=2,
            )
        )
        passed = bool(audit["passed"])
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        closer = threading.Thread(target=app.close, daemon=True)
        closer.start()
        closer.join(timeout=4.0)
    return 0 if passed else 2


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
