#!/usr/bin/env python3
"""Measure frame-by-frame RTX convergence for one downward pitch view."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
from pathlib import Path

import numpy as np

from isaac_render_embodiedgen_scene_qa import _image_metrics, _look_at_quat_wxyz


os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-usd", type=Path, required=True)
    parser.add_argument("--compiled-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pitch-deg", type=float, default=70.0)
    args, _ = parser.parse_known_args()

    from isaacsim import SimulationApp

    app = SimulationApp(
        {"headless": True, "enable_cameras": True, "width": 960, "height": 540}
    )
    try:
        from isaacsim.core.api import World
        from isaacsim.sensors.camera import Camera
        from PIL import Image

        compiled = json.loads(args.compiled_audit.read_text(encoding="utf-8"))
        route = compiled["route"]["waypoints_xy_m"]
        middle = np.asarray(route[len(route) // 2], dtype=np.float64)
        following = np.asarray(route[len(route) // 2 + 1], dtype=np.float64)
        direction = following - middle
        direction /= np.linalg.norm(direction)
        floor_z = float(compiled["visual_metrics"]["floor_z_m"])
        eye = np.asarray([middle[0], middle[1], floor_z + 0.42])
        distance = 0.42 / math.tan(math.radians(args.pitch_deg))
        target = np.asarray(
            [
                middle[0] + distance * direction[0],
                middle[1] + distance * direction[1],
                floor_z,
            ]
        )

        args.out.mkdir(parents=True, exist_ok=True)
        world = World(stage_units_in_meters=1.0)
        reference = world.stage.DefinePrim("/World/ConvergenceEpisode", "Xform")
        reference.GetReferences().AddReference(str(args.episode_usd.resolve()))
        camera = Camera(
            prim_path="/World/PitchConvergenceCamera",
            position=eye,
            resolution=(960, 540),
        )
        world.reset()
        camera.initialize()
        camera.set_focal_length(19.0)
        camera.set_horizontal_aperture(24.0)
        default_clipping_range = list(camera.get_clipping_range())
        camera.set_clipping_range(near_distance=0.01, far_distance=1000.0)
        calibrated_clipping_range = list(camera.get_clipping_range())
        camera.set_world_pose(
            eye, _look_at_quat_wxyz(eye, target), camera_axes="world"
        )
        checkpoints = {1, 6, 12, 24, 36, 60, 90, 120, 180}
        records = []
        for frame in range(1, max(checkpoints) + 1):
            world.step(render=True)
            if frame not in checkpoints:
                continue
            rgba = np.asarray(camera.get_rgba())
            if rgba.size == 0 or rgba.ndim != 3 or rgba.shape[-1] < 3:
                records.append(
                    {
                        "frame": frame,
                        "image": None,
                        "buffer_ready": False,
                        "shape": list(rgba.shape),
                        "metrics": None,
                    }
                )
                continue
            rgb = rgba[..., :3].astype(np.float32)
            if rgb.max() > 1.5:
                rgb /= 255.0
            rgb = np.clip(rgb, 0.0, 1.0)
            image_path = args.out / f"frame_{frame:03d}.png"
            Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8)).save(
                image_path, optimize=True
            )
            records.append(
                {
                    "frame": frame,
                    "image": image_path.name,
                    "buffer_ready": True,
                    "metrics": _image_metrics(rgb),
                }
            )
        payload = {
            "schema_version": "kinofail.isaac-camera-pitch-convergence.v1",
            "scientific_role": "development_diagnostic_not_benchmark_evidence",
            "pitch_deg": args.pitch_deg,
            "eye_xyz_m": eye.tolist(),
            "target_xyz_m": target.tolist(),
            "default_clipping_range_m": default_clipping_range,
            "calibrated_clipping_range_m": calibrated_clipping_range,
            "records": records,
        }
        (args.out / "convergence.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, indent=2))
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        closer = threading.Thread(target=app.close, daemon=True)
        closer.start()
        closer.join(timeout=4.0)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
