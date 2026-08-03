#!/usr/bin/env python3
"""Render an operator-blind Go2-front proxy sweep along a compiled route."""

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
import yaml

from isaac_render_embodiedgen_scene_qa import (
    _contact_sheet,
    _image_metrics,
    _look_at_quat_wxyz,
)
from kino_vla.eval.embodiedgen_motion_view_contract_v1 import (
    PROGRESS_M,
    SCHEMA_VERSION as VIEW_SCHEMA_VERSION,
    NOMINAL_STANDING_BASE_HEIGHT_M,
    evaluate_motion_views,
)
from kino_vla.eval.embodiedgen_visual_contract_v5 import FAR_CLIP_M, NEAR_CLIP_M


os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-usd", type=Path, required=True)
    parser.add_argument("--compiled-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--warmup-frames", type=int, default=24)
    parser.add_argument("--camera-profile", default="go2_front_calib_c")
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

        route = np.asarray(compiled["route"]["waypoints_xy_m"], dtype=float)
        segment_lengths = np.linalg.norm(route[1:] - route[:-1], axis=1)
        segment_index = int(np.argmax(segment_lengths))
        start = route[segment_index]
        direction = route[segment_index + 1] - start
        direction /= np.linalg.norm(direction)
        left = np.asarray([-direction[1], direction[0]], dtype=float)
        floor_z = float(compiled["visual_metrics"]["floor_z_m"])
        benchmark_config = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "configs/data/kinofail_realistic.yaml")
            .read_text(encoding="utf-8")
        )
        profile_specs = benchmark_config["camera_profile_specs"]
        if args.camera_profile not in profile_specs:
            raise RuntimeError(f"unknown frozen camera profile: {args.camera_profile}")
        camera_profile = dict(profile_specs[args.camera_profile])
        mount = np.asarray(camera_profile["mount_xyz_m"], dtype=float)
        pitch_down_rad = float(camera_profile["pitch_down_rad"])
        focal_length_mm = float(camera_profile["focal_length_mm"])
        camera_height_m = NOMINAL_STANDING_BASE_HEIGHT_M + float(mount[2])
        lookahead_m = 1.0

        args.out.mkdir(parents=True, exist_ok=True)
        world = World(stage_units_in_meters=1.0)
        reference = world.stage.DefinePrim("/World/EmbodiedGenEpisode", "Xform")
        reference.GetReferences().AddReference(str(episode))
        first_xy = (
            start
            + direction * (PROGRESS_M[0] + float(mount[0]))
            + left * float(mount[1])
        )
        camera = Camera(
            prim_path="/World/KinoGo2FrontMotionProxyCamera",
            position=np.asarray([first_xy[0], first_xy[1], floor_z + camera_height_m]),
            resolution=(args.width, args.height),
        )
        world.reset()
        camera.initialize()
        camera.set_focal_length(focal_length_mm)
        camera.set_horizontal_aperture(24.0)
        camera.set_clipping_range(near_distance=NEAR_CLIP_M, far_distance=FAR_CLIP_M)
        for _ in range(args.warmup_frames):
            world.step(render=True)

        records = []
        paths = []
        for index, progress_m in enumerate(PROGRESS_M):
            # The frozen progress values describe the Go2 base, while the rendered
            # eye must include the actual forward/lateral body-frame camera mount.
            xy = (
                start
                + direction * (progress_m + float(mount[0]))
                + left * float(mount[1])
            )
            eye = np.asarray([xy[0], xy[1], floor_z + camera_height_m], dtype=float)
            target = np.asarray(
                [
                    xy[0] + direction[0] * lookahead_m,
                    xy[1] + direction[1] * lookahead_m,
                    floor_z + camera_height_m - np.tan(pitch_down_rad) * lookahead_m,
                ],
                dtype=float,
            )
            quaternion = _look_at_quat_wxyz(eye, target)
            camera.set_world_pose(eye, quaternion, camera_axes="world")
            for _ in range(args.warmup_frames):
                world.step(render=True)
            rgba = np.asarray(camera.get_rgba())
            if rgba.shape[:2] != (args.height, args.width) or rgba.shape[-1] < 3:
                raise RuntimeError(f"invalid motion-proxy frame {index}: {rgba.shape}")
            rgb = rgba[..., :3].astype(np.float32)
            if rgb.max() > 1.5:
                rgb /= 255.0
            rgb = np.clip(rgb, 0.0, 1.0)
            image_path = args.out / f"front_progress_{progress_m:.2f}m.png"
            Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8)).save(
                image_path, optimize=True
            )
            records.append(
                {
                    "progress_m": progress_m,
                    "eye_xyz_m": eye.tolist(),
                    "target_xyz_m": target.tolist(),
                    "quaternion_wxyz": quaternion.tolist(),
                    "image": image_path.name,
                    "image_sha256": _sha256(image_path),
                    "metrics": _image_metrics(rgb),
                }
            )
            paths.append(image_path)

        checks = evaluate_motion_views(records)
        sheet_path = args.out / "go2_front_motion_proxy_contact_sheet.png"
        _contact_sheet(paths, sheet_path)
        audit = {
            "schema_version": "kinofail.embodiedgen-route-motion-rtx-qa.v1-development",
            "view_contract_schema": VIEW_SCHEMA_VERSION,
            "created_utc": datetime.now(UTC).isoformat(),
            "scene_id": compiled.get("scene_id"),
            "operator_blind": True,
            "episode_usd": str(episode),
            "episode_usd_sha256": _sha256(episode),
            "compiled_audit": str(compiled_path),
            "compiled_audit_sha256": _sha256(compiled_path),
            "resolution": [args.width, args.height],
            "camera_proxy": {
                "camera_profile": args.camera_profile,
                "mount_xyz_base_m": mount.tolist(),
                "nominal_standing_base_height_m": NOMINAL_STANDING_BASE_HEIGHT_M,
                "camera_height_above_floor_m": camera_height_m,
                "pitch_down_rad": pitch_down_rad,
                "focal_length_mm": focal_length_mm,
                "horizontal_aperture_mm": 24.0,
                "calibration_status": "matches_go2_front_calib_c_engineering_proxy",
            },
            "route_start_xy_m": start.tolist(),
            "route_direction_xy": direction.tolist(),
            "views": records,
            "checks": checks,
            "passed": all(checks.values()),
            "admission_state": (
                "route_motion_visual_proxy_passed_pending_articulated_go2_qa"
                if all(checks.values())
                else "route_motion_visual_proxy_failed"
            ),
            "contact_sheet": sheet_path.name,
            "contact_sheet_sha256": _sha256(sheet_path),
            "counts_as_a0_a7_evidence": False,
        }
        audit_path = args.out / "route_motion_rtx_audit.json"
        audit_path.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {"passed": audit["passed"], "checks": checks, "audit": str(audit_path)},
                indent=2,
            ),
            flush=True,
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
