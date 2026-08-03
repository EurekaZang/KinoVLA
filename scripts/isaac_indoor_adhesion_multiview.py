"""Capture publication-review camera angles from one frozen Isaac adhesion state.

The Go2 executes the real indoor adhesion case until the requested time.  Physics is then held
fixed while one RTX camera is moved around the unchanged stage, so every image shows the exact
same robot/operator state rather than loosely matched replays.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

import numpy as np


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _contact_sheet(frame_paths: list[Path], output_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    thumb_w, thumb_h = 480, 270
    margin, label_h, columns = 18, 36, 2
    rows = (len(frame_paths) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (
            columns * thumb_w + (columns + 1) * margin,
            rows * (thumb_h + label_h) + (rows + 1) * margin,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    for index, path in enumerate(frame_paths):
        row, column = divmod(index, columns)
        x = margin + column * (thumb_w + margin)
        y = margin + row * (thumb_h + label_h + margin)
        image = Image.open(path).convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        canvas.paste(image, (x, y))
        draw.text((x, y + thumb_h + 7), path.stem, fill=(24, 30, 34), font=font)
    canvas.save(output_path, quality=95)


def main() -> None:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", default="demo/indoor_adhesion_icra.yaml")
    preliminary.add_argument("--out", default="outputs/demos/indoor_adhesion_icra/multiview_peak")
    preliminary.add_argument("--capture-time", type=float, default=4.82)
    pre_args, _ = preliminary.parse_known_args()

    parser = argparse.ArgumentParser(description="Frozen-state indoor adhesion multiview capture")
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--out", default=pre_args.out)
    parser.add_argument("--capture-time", type=float, default=pre_args.capture_time)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    simulation_app = AppLauncher(args).app

    import omni.timeline
    import torch
    from PIL import Image

    from kino_vla.sim.adhesion import FootAdhesionConfig
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.realistic_scene import (
        build_indoor_adhesion_scene_spec,
        compile_indoor_scene_layers,
    )
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.utils.seeding import seed_everything

    cfg = load_config(args.config)
    seed = int(cfg.scene.seed)
    seed_everything(seed)
    output_dir = REPO_ROOT / args.out
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = build_indoor_adhesion_scene_spec(seed)
    scene_dir = REPO_ROOT / "outputs/demos/indoor_adhesion_icra/scene_usd"
    compiled = compile_indoor_scene_layers(spec, scene_dir)
    sim_cfg = load_config(
        "sim/go2_skeleton.yaml",
        {
            "cam_width": 1280,
            "cam_height": 720,
        },
    )
    backend = IsaacPolicyBackend(
        sim_cfg,
        np.asarray(cfg.start.pos, dtype=np.float64),
        float(cfg.start.heading),
        record_cam=True,
    )
    backend.load_realistic_scene(str(compiled["episode_usd"]))
    adhesion_cfg = FootAdhesionConfig(
        region=spec.adhesion_region,
        surface_z_m=float(cfg.adhesion.surface_z_m),
        attach_contact_force_n=float(cfg.adhesion.attach_contact_force_n),
        attach_height_tolerance_m=float(cfg.adhesion.attach_height_tolerance_m),
        stiffness_xy_n_per_m=float(cfg.adhesion.stiffness_xy_n_per_m),
        damping_xy_ns_per_m=float(cfg.adhesion.damping_xy_ns_per_m),
        stiffness_z_n_per_m=float(cfg.adhesion.stiffness_z_n_per_m),
        damping_z_ns_per_m=float(cfg.adhesion.damping_z_ns_per_m),
        force_cap_n=float(cfg.adhesion.force_cap_n),
        break_force_n=float(cfg.adhesion.break_force_n),
        peel_release_force_n=float(cfg.adhesion.peel_release_force_n),
        peel_velocity_threshold_mps=float(cfg.adhesion.peel_velocity_threshold_mps),
        progress_axis_xy=tuple(float(value) for value in cfg.adhesion.progress_axis_xy),
        max_active_feet=int(cfg.adhesion.max_active_feet),
    )
    backend.add_foot_adhesion(adhesion_cfg)
    obs = backend.reset(seed)
    while obs.t + 1.0e-9 < float(args.capture_time) and not obs.fallen:
        command_vx = (
            float(cfg.motion.forward_command_mps)
            if obs.t < float(cfg.motion.adhesion_forward_s)
            else float(cfg.motion.reverse_command_mps)
        )
        obs = backend.step(np.asarray([command_vx, 0.0, 0.0], dtype=np.float64))
    if obs.fallen:
        raise RuntimeError(f"robot fell before multiview capture at t={obs.t:.3f}s")

    target_robot = [float(obs.pos[0]), float(obs.pos[1]), 0.27]
    target_scene = [2.75, 0.0, 0.24]
    views = (
        ("01_room_overview_left", [1.25, -3.05, 1.85], target_scene),
        ("02_room_overview_right", [3.80, 2.10, 1.65], target_scene),
        ("03_left_side_low", [2.45, -1.55, 0.68], target_robot),
        ("04_right_side_low", [2.55, 2.30, 0.78], target_robot),
        ("05_rear_three_quarter", [0.35, -1.65, 1.18], target_robot),
        ("06_front_three_quarter", [4.85, -1.65, 1.18], target_robot),
        ("07_top_down", [2.75, 1.35, 5.60], target_scene),
        ("08_room_wide", [-0.75, -3.05, 2.35], [3.15, 0.0, 0.55]),
    )
    paths: list[Path] = []
    view_records: list[dict[str, Any]] = []
    # Keep physics frozen while allowing Kit/RTX to consume each new camera transform.  A sensor
    # force-recompute alone can still return the first Replicator frame at an unchanged timeline
    # timestamp.
    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()
    for name, eye, target in views:
        backend.aim_record_camera(np.asarray(eye), np.asarray(target))
        for _ in range(3):
            simulation_app.update()
        frame = backend.capture_rgb(force_recompute=True)
        if frame is None or float(frame.std()) < 8.0:
            raise RuntimeError(f"invalid RTX frame for {name}")
        path = output_dir / f"{name}.png"
        Image.fromarray(frame).save(path)
        paths.append(path)
        view_records.append({"name": name, "eye_xyz_m": eye, "target_xyz_m": target})

    _contact_sheet(paths, output_dir / "contact_sheet.jpg")
    telemetry = backend.adhesion_telemetry()
    _write_json(
        output_dir / "multiview_manifest.json",
        {
            "scene_id": spec.scene_id,
            "seed": seed,
            "requested_capture_time_s": float(args.capture_time),
            "actual_capture_time_s": float(obs.t),
            "robot_position_xyz_m": [float(value) for value in obs.pos],
            "robot_tilt_rad": float(obs.tilt),
            "adhesion": telemetry,
            "state_frozen_across_views": True,
            "camera_resolution": [1280, 720],
            "views": view_records,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA",
        },
    )
    print(f"PASS: wrote {len(paths)} frozen-state RTX views to {output_dir}")
    sys.stdout.flush()
    closer = threading.Thread(target=simulation_app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
