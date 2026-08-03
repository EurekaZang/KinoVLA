#!/usr/bin/env python3
"""RTX-render an admitted EmbodiedGen/Kino-Fail episode from route and Go2-height views."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _look_at_quat_wxyz(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World quaternion for a camera whose local +X is forward and local +Z is up."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up_hint = np.array([0.0, 0.0, 1.0])
    up = up_hint - np.dot(up_hint, forward) * forward
    if np.linalg.norm(up) < 1.0e-6:
        up_hint = np.array([0.0, 1.0, 0.0])
        up = up_hint - np.dot(up_hint, forward) * forward
    up = up / np.linalg.norm(up)
    left = np.cross(up, forward)
    left = left / np.linalg.norm(left)
    rotation = np.column_stack((forward, left, up))
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / s,
                    0.25 * s,
                    (rotation[0, 1] + rotation[1, 0]) / s,
                    (rotation[0, 2] + rotation[2, 0]) / s,
                ]
            )
        elif index == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / s,
                    (rotation[0, 1] + rotation[1, 0]) / s,
                    0.25 * s,
                    (rotation[1, 2] + rotation[2, 1]) / s,
                ]
            )
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / s,
                    (rotation[0, 2] + rotation[2, 0]) / s,
                    (rotation[1, 2] + rotation[2, 1]) / s,
                    0.25 * s,
                ]
            )
    return quat / np.linalg.norm(quat)


def _image_metrics(rgb_float: np.ndarray) -> dict[str, float | int]:
    luminance = (
        0.2126 * rgb_float[..., 0]
        + 0.7152 * rgb_float[..., 1]
        + 0.0722 * rgb_float[..., 2]
    )
    quantized = (np.clip(rgb_float, 0.0, 1.0) * 31).astype(np.uint8)
    colors = np.unique(quantized.reshape(-1, 3), axis=0)
    return {
        "mean_luminance": float(luminance.mean()),
        "std_luminance": float(luminance.std()),
        "p01_luminance": float(np.quantile(luminance, 0.01)),
        "p99_luminance": float(np.quantile(luminance, 0.99)),
        "black_fraction": float((luminance < 0.02).mean()),
        "white_fraction": float((luminance > 0.98).mean()),
        "quantized_color_count_5bit": int(len(colors)),
    }


def _contact_sheet(paths: list[Path], output: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    images = [Image.open(path).convert("RGB") for path in paths]
    canvas = Image.new("RGB", (sum(image.width for image in images), images[0].height), "white")
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, 22) if Path(font_path).is_file() else ImageFont.load_default()
    x = 0
    for path, image in zip(paths, images):
        canvas.paste(image, (x, 0))
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        label = path.stem.replace("_", " ")
        bbox = draw.textbbox((0, 0), label, font=font)
        width = bbox[2] - bbox[0] + 28
        height = bbox[3] - bbox[1] + 18
        draw.rounded_rectangle((10, 10, 10 + width, 10 + height), 7, fill=(8, 14, 20, 205))
        draw.text((24, 17 - bbox[1]), label, font=font, fill="white")
        canvas.paste(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"), (x, 0))
        x += image.width
    canvas.save(output, optimize=True)


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

        episode = args.episode_usd.resolve()
        compiled_path = args.compiled_audit.resolve()
        compiled = json.loads(compiled_path.read_text())
        if not compiled.get("passed"):
            raise RuntimeError("compiled scene audit has not passed")
        expected_hash = compiled["files"].get(episode.name)
        if expected_hash != _sha256(episode):
            raise RuntimeError("episode USD no longer matches compiled-scene audit")
        floor_z = float(compiled["visual_metrics"]["floor_z_m"])

        args.out.mkdir(parents=True, exist_ok=True)
        world = World(stage_units_in_meters=1.0)
        reference = world.stage.DefinePrim("/World/EmbodiedGenEpisode", "Xform")
        reference.GetReferences().AddReference(str(episode))
        route = np.asarray(compiled["route"]["waypoints_xy_m"], dtype=np.float64)
        if len(route) < 2:
            raise RuntimeError("compiled scene route has fewer than two waypoints")
        start = route[0]
        second = route[1]
        middle_index = len(route) // 2
        middle = route[middle_index]
        middle_next = route[min(len(route) - 1, middle_index + 1)]
        reverse_origin = route[-2]
        reverse_target = route[-3]

        def direction_and_left(origin: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            direction = target - origin
            direction = direction / np.linalg.norm(direction)
            return direction, np.array([-direction[1], direction[0]], dtype=np.float64)

        entry_direction, entry_left = direction_and_left(start, second)
        middle_direction, middle_left = direction_and_left(middle, middle_next)
        reverse_direction, reverse_left = direction_and_left(reverse_origin, reverse_target)
        views = [
            (
                "01_entry_oblique",
                np.array(
                    [
                        start[0] + 0.42 * entry_left[0],
                        start[1] + 0.42 * entry_left[1],
                        floor_z + 1.30,
                    ]
                ),
                np.array(
                    [
                        start[0] + 1.40 * entry_direction[0],
                        start[1] + 1.40 * entry_direction[1],
                        floor_z + 0.62,
                    ]
                ),
                "review",
            ),
            (
                "02_route_mid",
                np.array(
                    [
                        middle[0] - 0.30 * middle_direction[0] - 0.42 * middle_left[0],
                        middle[1] - 0.30 * middle_direction[1] - 0.42 * middle_left[1],
                        floor_z + 1.25,
                    ]
                ),
                np.array(
                    [
                        middle[0] + 1.25 * middle_direction[0],
                        middle[1] + 1.25 * middle_direction[1],
                        floor_z + 0.58,
                    ]
                ),
                "review",
            ),
            (
                "03_reverse",
                np.array(
                    [
                        reverse_origin[0] + 0.36 * reverse_left[0],
                        reverse_origin[1] + 0.36 * reverse_left[1],
                        floor_z + 1.30,
                    ]
                ),
                np.array(
                    [
                        reverse_origin[0] + 1.25 * reverse_direction[0],
                        reverse_origin[1] + 1.25 * reverse_direction[1],
                        floor_z + 0.60,
                    ]
                ),
                "review",
            ),
            (
                "04_go2_front_height",
                np.array([middle[0], middle[1], floor_z + 0.42]),
                np.array([middle_next[0], middle_next[1], floor_z + 0.34]),
                "go2_front_height_proxy",
            ),
        ]

        cam = Camera(
            prim_path="/World/KinoQACamera",
            position=views[0][1],
            resolution=(args.width, args.height),
        )
        world.reset()
        cam.initialize()
        cam.set_focal_length(19.0)
        cam.set_horizontal_aperture(24.0)
        for _ in range(args.warmup_frames):
            world.step(render=True)

        records = []
        arrays: list[np.ndarray] = []
        paths: list[Path] = []
        for name, eye, target, kind in views:
            quat = _look_at_quat_wxyz(eye, target)
            cam.set_world_pose(eye, quat, camera_axes="world")
            for _ in range(args.warmup_frames):
                world.step(render=True)
            rgba = np.asarray(cam.get_rgba())
            if rgba.shape[:2] != (args.height, args.width) or rgba.shape[-1] < 3:
                raise RuntimeError(f"invalid RTX frame for {name}: {rgba.shape}")
            rgb = rgba[..., :3].astype(np.float32)
            if rgb.max() > 1.5:
                rgb /= 255.0
            rgb = np.clip(rgb, 0.0, 1.0)
            image_path = args.out / f"{name}.png"
            Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8)).save(image_path, optimize=True)
            metrics = _image_metrics(rgb)
            records.append(
                {
                    "name": name,
                    "kind": kind,
                    "eye_xyz_m": eye.tolist(),
                    "target_xyz_m": target.tolist(),
                    "quaternion_wxyz": quat.tolist(),
                    "image": image_path.name,
                    "image_sha256": _sha256(image_path),
                    "metrics": metrics,
                }
            )
            arrays.append(rgb)
            paths.append(image_path)
            print(f"[rtx] {name}: {metrics}", flush=True)

        pairwise_l1 = {}
        for i in range(len(arrays)):
            for j in range(i + 1, len(arrays)):
                pairwise_l1[f"{records[i]['name']}__{records[j]['name']}"] = float(
                    np.abs(arrays[i] - arrays[j]).mean()
                )
        checks = {
            "four_views_captured": len(records) == 4,
            "non_degenerate_luminance": all(
                record["metrics"]["std_luminance"] >= 0.035 for record in records
            ),
            "usable_dynamic_range": all(
                record["metrics"]["p99_luminance"]
                - record["metrics"]["p01_luminance"]
                >= 0.12
                for record in records
            ),
            "not_black_or_white": all(
                record["metrics"]["black_fraction"] < 0.90
                and record["metrics"]["white_fraction"] < 0.90
                for record in records
            ),
            "appearance_complexity": all(
                record["metrics"]["quantized_color_count_5bit"] >= 64 for record in records
            ),
            "viewpoints_not_stale": min(pairwise_l1.values()) >= 0.01,
            "go2_height_view_present": any(
                record["kind"] == "go2_front_height_proxy" for record in records
            ),
        }
        sheet_path = args.out / "multiview_contact_sheet.png"
        _contact_sheet(paths, sheet_path)
        audit = {
            "schema_version": "kinofail.embodiedgen-rtx-scene-qa.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "renderer": "Isaac Sim RTX Camera",
            "episode_usd": str(episode),
            "episode_usd_sha256": _sha256(episode),
            "compiled_audit": str(compiled_path),
            "compiled_audit_sha256": _sha256(compiled_path),
            "resolution": [args.width, args.height],
            "camera_intrinsics": {"focal_length_mm": 19.0, "horizontal_aperture_mm": 24.0},
            "lighting": compiled["lighting_contract"],
            "views": records,
            "pairwise_mean_absolute_rgb": pairwise_l1,
            "checks": checks,
            "passed": all(checks.values()),
            "admission_state": (
                "rtx_scene_passed_pending_articulated_go2_physics_qa"
                if all(checks.values())
                else "rtx_scene_failed"
            ),
            "contact_sheet": sheet_path.name,
            "contact_sheet_sha256": _sha256(sheet_path),
        }
        audit_path = args.out / "rtx_scene_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"passed": audit["passed"], "checks": checks, "audit": str(audit_path)}, indent=2))
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
