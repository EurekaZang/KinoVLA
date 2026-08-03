#!/usr/bin/env python3
"""Development-only calibration of Isaac Camera pitch image convention."""

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


def _forward_from_quaternion(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion_wxyz
    return np.asarray(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-usd", type=Path, required=True)
    parser.add_argument("--compiled-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
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

        args.out.mkdir(parents=True, exist_ok=True)
        world = World(stage_units_in_meters=1.0)
        reference = world.stage.DefinePrim("/World/DiagnosticEpisode", "Xform")
        reference.GetReferences().AddReference(str(args.episode_usd.resolve()))
        camera = Camera(
            prim_path="/World/PitchConventionDiagnosticCamera",
            position=eye,
            resolution=(960, 540),
        )
        world.reset()
        camera.initialize()
        camera.set_focal_length(19.0)
        camera.set_horizontal_aperture(24.0)
        records = []
        for sign_name, z_sign in (("requested_down", -1.0), ("mirrored_up", 1.0)):
            for pitch_deg in (45.0, 70.0, 80.0):
                horizontal = 0.42 / math.tan(math.radians(pitch_deg))
                target = np.asarray(
                    [
                        middle[0] + horizontal * direction[0],
                        middle[1] + horizontal * direction[1],
                        eye[2] + z_sign * 0.42,
                    ]
                )
                quaternion = _look_at_quat_wxyz(eye, target)
                camera.set_world_pose(eye, quaternion, camera_axes="world")
                for _ in range(36):
                    world.step(render=True)
                returned_position, returned_quaternion = camera.get_world_pose(
                    camera_axes="world"
                )
                rgba = np.asarray(camera.get_rgba())
                rgb = rgba[..., :3].astype(np.float32)
                if rgb.max() > 1.5:
                    rgb /= 255.0
                rgb = np.clip(rgb, 0.0, 1.0)
                name = f"{sign_name}_pitch{int(pitch_deg)}"
                Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8)).save(
                    args.out / f"{name}.png", optimize=True
                )
                requested_forward = target - eye
                requested_forward /= np.linalg.norm(requested_forward)
                returned_forward = _forward_from_quaternion(
                    np.asarray(returned_quaternion, dtype=np.float64)
                )
                returned_forward /= np.linalg.norm(returned_forward)
                records.append(
                    {
                        "name": name,
                        "eye_xyz_m": eye.tolist(),
                        "target_xyz_m": target.tolist(),
                        "input_quaternion_wxyz": quaternion.tolist(),
                        "returned_position_xyz_m": np.asarray(returned_position).tolist(),
                        "returned_quaternion_wxyz": np.asarray(
                            returned_quaternion
                        ).tolist(),
                        "requested_returned_forward_dot": float(
                            requested_forward @ returned_forward
                        ),
                        "metrics": _image_metrics(rgb),
                    }
                )
        payload = {
            "schema_version": "kinofail.isaac-camera-pitch-diagnostic.v1",
            "scientific_role": "development_diagnostic_not_benchmark_evidence",
            "episode_usd": str(args.episode_usd.resolve()),
            "compiled_audit": str(args.compiled_audit.resolve()),
            "records": records,
        }
        out_path = args.out / "diagnostic.json"
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
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
