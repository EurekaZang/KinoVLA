"""Render the ICRA-review indoor + foot-local adhesion vertical slice in full Isaac Sim.

Outputs two matched takes (nominal and adhesion+peel), body-fixed/front and third-person videos,
keyframes, synchronized telemetry, a diagnostic plot, the composed USD layers, and a provenance
manifest.  This is a scene/operator engineering gate; it does not alter frozen A0--A7 evidence.

Usage:
    env -u PYTHONPATH HF_HUB_OFFLINE=1 OMNI_KIT_ACCEPT_EULA=YES \
      ~/miniconda3/envs/kinovla/bin/python scripts/isaac_indoor_adhesion_demo.py --headless
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state(repo_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _make_contact_sheet(frame_paths: list[Path], output_path: Path, title: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if not frame_paths:
        return
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    target_w = 480
    resized = []
    for image in images:
        target_h = round(image.height * target_w / image.width)
        resized.append(image.resize((target_w, target_h), Image.Resampling.LANCZOS))
    margin, label_h = 18, 48
    canvas = Image.new(
        "RGB",
        (
            target_w * len(resized) + margin * (len(resized) + 1),
            resized[0].height + 2 * margin + label_h,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    draw.text((margin, 12), title, fill=(24, 30, 34), font=font)
    for index, (path, image) in enumerate(zip(frame_paths, resized, strict=True)):
        x = margin + index * (target_w + margin)
        canvas.paste(image, (x, margin + label_h))
        draw.text((x + 8, margin + label_h + 8), path.stem, fill=(245, 245, 245), font=font)
    canvas.save(output_path, quality=95)


def _plot_telemetry(
    rows: list[dict[str, Any]], output_path: Path, take: str, max_tilt_rad: float
) -> None:
    import matplotlib.pyplot as plt

    t = np.asarray([row["t_s"] for row in rows], dtype=np.float64)
    x = np.asarray([row["x_m"] for row in rows], dtype=np.float64)
    vx = np.asarray([row["vx_mps"] for row in rows], dtype=np.float64)
    cmd = np.asarray([row["cmd_vx_mps"] for row in rows], dtype=np.float64)
    force = np.asarray([row["adhesion"]["total_applied_force_n"] for row in rows])
    active = np.asarray([row["adhesion"]["active_feet"] for row in rows])
    tilt = np.asarray([row["tilt_rad"] for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(4, 1, figsize=(9.2, 8.4), sharex=True, facecolor="white")
    axes[0].plot(t, x, color="#176B87", linewidth=2.0, label="Go2 x")
    axes[0].set_ylabel("position [m]")
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].plot(t, cmd, color="#6B7280", linestyle="--", linewidth=1.6, label="command")
    axes[1].plot(t, vx, color="#0F766E", linewidth=1.8, label="measured")
    axes[1].set_ylabel("forward speed [m/s]")
    axes[1].legend(frameon=False, ncol=2, loc="upper left")
    axes[2].plot(t, force, color="#B45309", linewidth=1.8, label="foot-local adhesion force")
    axes[2].fill_between(t, 0.0, active, color="#F59E0B", alpha=0.16, label="attached feet")
    axes[2].set_ylabel("force [N]")
    axes[2].legend(frameon=False, ncol=2, loc="upper left")
    axes[3].plot(t, tilt, color="#7C3AED", linewidth=1.8, label="body tilt")
    axes[3].axhline(
        max_tilt_rad,
        color="#DC2626",
        linestyle="--",
        linewidth=1.3,
        label="fall threshold",
    )
    axes[3].set_ylabel("tilt [rad]")
    axes[3].set_xlabel("simulation time [s]")
    axes[3].legend(frameon=False, ncol=2, loc="upper left")
    for axis in axes:
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Indoor adhesion demo — {take}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", default="demo/indoor_adhesion_icra.yaml")
    preliminary.add_argument("--out", default="outputs/demos/indoor_adhesion_icra")
    preliminary.add_argument("--takes", default="nominal,adhesion_peel")
    preliminary.add_argument("--view", choices=("front", "review"), default="front")
    preliminary.add_argument("--seed", type=int, default=None)
    pre_args, _ = preliminary.parse_known_args()

    parser = argparse.ArgumentParser(description="ICRA-review indoor Go2 adhesion demo")
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--out", default=pre_args.out)
    parser.add_argument("--takes", default=pre_args.takes)
    parser.add_argument("--view", choices=("front", "review"), default=pre_args.view)
    parser.add_argument("--seed", type=int, default=pre_args.seed)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    simulation_app = AppLauncher(args).app

    import imageio.v2 as imageio
    import torch
    import yaml
    from PIL import Image

    from kino_vla.sim.adhesion import FootAdhesionConfig
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.realistic_scene import (
        build_indoor_adhesion_scene_spec,
        compile_indoor_scene_layers,
    )
    from kino_vla.utils.config import CONFIGS_DIR, REPO_ROOT, load_config
    from kino_vla.utils.seeding import seed_everything

    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.scene.seed)
    seed_everything(seed)
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_dir = out_dir / str(cfg.scene.usd_dir)

    spec = build_indoor_adhesion_scene_spec(seed)
    compiled = compile_indoor_scene_layers(spec, scene_dir)
    print(
        f"[scene] compiled {compiled['scene_id']}: "
        f"{compiled['audit']['n_primitives']} prims, audit={compiled['audit']['passed']}"
    )

    cam_mount = cfg.camera.mount_xyz_m
    sim_cfg = load_config(
        "sim/go2_skeleton.yaml",
        {
            "cam_width": int(
                cfg.camera.front_width if args.view == "front" else cfg.camera.review_width
            ),
            "cam_height": int(
                cfg.camera.front_height if args.view == "front" else cfg.camera.review_height
            ),
            "front_cam_width": int(cfg.camera.front_width),
            "front_cam_height": int(cfg.camera.front_height),
            "front_cam_focal_mm": float(cfg.camera.front_focal_mm),
            "front_cam_aperture_mm": float(cfg.camera.front_aperture_mm),
            "front_cam_x_m": float(cam_mount[0]),
            "front_cam_y_m": float(cam_mount[1]),
            "front_cam_z_m": float(cam_mount[2]),
        },
    )
    backend = IsaacPolicyBackend(
        sim_cfg,
        np.asarray(cfg.start.pos, dtype=np.float64),
        float(cfg.start.heading),
        record_cam=args.view == "review",
        front_cam=args.view == "front",
    )
    backend.load_realistic_scene(str(compiled["episode_usd"]))
    if args.view == "review":
        backend.aim_record_camera(
            np.asarray(cfg.camera.review_eye_xyz, dtype=np.float64),
            np.asarray(cfg.camera.review_target_xyz, dtype=np.float64),
        )

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

    requested = [value.strip() for value in args.takes.split(",") if value.strip()]
    allowed = {"nominal", "adhesion_peel"}
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError(f"unknown takes {unknown}; choose from {sorted(allowed)}")
    summaries: dict[str, dict[str, Any]] = {}

    for take in requested:
        take_dir = out_dir / take
        take_dir.mkdir(parents=True, exist_ok=True)
        backend.clear_foot_adhesion()
        if take == "adhesion_peel":
            backend.add_foot_adhesion(adhesion_cfg)
        # Paired takes must share the exact seed; the operator is the only intervention.
        obs = backend.reset(seed)
        if args.view == "review":
            backend.aim_record_camera(
                np.asarray(cfg.camera.review_eye_xyz, dtype=np.float64),
                np.asarray(cfg.camera.review_target_xyz, dtype=np.float64),
            )
        video_name = "front_body_fixed.mp4" if args.view == "front" else "review_third_person.mp4"
        writer = imageio.get_writer(
            str(take_dir / video_name),
            fps=int(cfg.camera.fps),
            codec="libx264",
            quality=9,
            macro_block_size=None,
        )
        rows: list[dict[str, Any]] = []
        keyframe_paths: list[Path] = []
        keyframe_times = [float(value) for value in cfg.review.keyframe_times_s]
        saved_keyframes: set[float] = set()
        next_frame_t = 0.0
        n_frames = 0
        initial_x = float(obs.pos[0])
        end_time = (
            float(cfg.motion.adhesion_forward_s)
            + float(cfg.motion.peel_duration_s)
            + float(cfg.motion.stop_duration_s)
        )
        while obs.t < end_time and not obs.fallen:
            if obs.t < float(cfg.motion.adhesion_forward_s):
                command_vx = float(cfg.motion.forward_command_mps)
            elif obs.t < float(cfg.motion.adhesion_forward_s) + float(cfg.motion.peel_duration_s):
                command_vx = float(cfg.motion.reverse_command_mps)
            else:
                command_vx = 0.0
            obs = backend.step(np.array([command_vx, 0.0, 0.0], dtype=np.float64))
            adhesion = backend.adhesion_telemetry()
            rows.append(
                {
                    "take": take,
                    "seed": seed,
                    "t_s": float(obs.t),
                    "x_m": float(obs.pos[0]),
                    "y_m": float(obs.pos[1]),
                    "vx_mps": float(obs.vel_body[0]),
                    "vy_mps": float(obs.vel_body[1]),
                    "base_height_m": float(obs.base_height),
                    "tilt_rad": float(obs.tilt),
                    "fallen": bool(obs.fallen),
                    "cmd_vx_mps": command_vx,
                    "adhesion": adhesion,
                }
            )
            if obs.t + 1.0e-9 >= next_frame_t:
                if args.view == "front":
                    sensor = backend.capture_front_camera()
                    if sensor is None:
                        raise RuntimeError("requested body-fixed RTX camera returned no frame")
                    frame = sensor["rgb"]
                else:
                    frame = backend.capture_rgb()
                    sensor = None
                    if frame is None:
                        raise RuntimeError("requested review RTX camera returned no frame")
                writer.append_data(frame)
                n_frames += 1
                next_frame_t += 1.0 / float(cfg.camera.fps)
                for key_time in keyframe_times:
                    if key_time not in saved_keyframes and obs.t >= key_time:
                        frame_path = take_dir / f"{args.view}_t{key_time:04.1f}s.png"
                        Image.fromarray(frame).save(frame_path)
                        if sensor is not None:
                            np.savez_compressed(
                                take_dir / f"front_sensor_t{key_time:04.1f}s.npz",
                                depth=(
                                    sensor["depth"]
                                    if sensor["depth"] is not None
                                    else np.empty((0,), dtype=np.float32)
                                ),
                                K=sensor["K"],
                                position=sensor["pos"],
                                quaternion_ros=sensor["quat_ros"],
                                timestamp_s=np.asarray(obs.t),
                            )
                        keyframe_paths.append(frame_path)
                        saved_keyframes.add(key_time)
        writer.close()
        telemetry_path = take_dir / f"telemetry_{args.view}.jsonl"
        with telemetry_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        _plot_telemetry(
            rows,
            take_dir / f"telemetry_{args.view}.png",
            take,
            float(sim_cfg.fall_check.max_tilt_rad),
        )
        _make_contact_sheet(
            keyframe_paths, take_dir / f"{args.view}_contact_sheet.jpg", f"{take} | {args.view}"
        )
        attach_events = sum(
            foot["event"] == "attached" for row in rows for foot in row["adhesion"]["feet"]
        )
        peel_events = sum(
            foot["event"] == "peeled" for row in rows for foot in row["adhesion"]["feet"]
        )
        break_events = sum(
            foot["event"] == "broken" for row in rows for foot in row["adhesion"]["feet"]
        )
        summary = {
            "take": take,
            "view": args.view,
            "seed": seed,
            "frames": n_frames,
            "sim_time_s": float(rows[-1]["t_s"] if rows else 0.0),
            "initial_x_m": initial_x,
            "final_x_m": float(rows[-1]["x_m"] if rows else initial_x),
            "net_progress_m": float(rows[-1]["x_m"] - initial_x if rows else 0.0),
            "min_forward_speed_mps": float(min(row["vx_mps"] for row in rows)),
            "peak_adhesion_force_n": float(
                max(row["adhesion"]["total_applied_force_n"] for row in rows)
            ),
            "max_active_feet": int(max(row["adhesion"]["active_feet"] for row in rows)),
            "attachment_events": int(attach_events),
            "peel_events": int(peel_events),
            "break_events": int(break_events),
            "fell": bool(rows[-1]["fallen"] if rows else False),
            "video": str(take_dir / video_name),
            "telemetry": str(telemetry_path),
        }
        _write_json(take_dir / f"summary_{args.view}.json", summary)
        summaries[take] = summary
        print(
            f"[take] {take}: progress={summary['net_progress_m']:.2f}m "
            f"peak_force={summary['peak_adhesion_force_n']:.1f}N "
            f"attach={attach_events} peel={peel_events} fallen={summary['fell']}"
        )

    config_path = CONFIGS_DIR / args.config
    commit, dirty = _git_state(REPO_ROOT)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA"
    reproduction_prefix = (
        "env -u PYTHONPATH HF_HUB_OFFLINE=1 OMNI_KIT_ACCEPT_EULA=YES "
        f"{sys.executable} scripts/isaac_indoor_adhesion_demo.py "
        f"--config {args.config} --takes nominal,adhesion_peel --out {args.out} --headless"
    )
    replay_summaries: dict[str, dict[str, Any]] = {}
    for take in allowed:
        take_dir = out_dir / take
        views: dict[str, Any] = {}
        for path in sorted(take_dir.glob("summary_*.json")):
            with path.open(encoding="utf-8") as stream:
                record = json.load(stream)
            views[str(record["view"])] = record
        if views:
            replay_summaries[take] = views
    manifest = {
        "schema_version": "kino.realistic_episode.v1",
        "scene": compiled,
        "scene_source_statement": {
            "type": spec.source_kind,
            "embodiedgen_generated": False,
            "reason": (
                "First compiler/physics vertical slice uses authored USD-native assets; "
                "EmbodiedGen visual.usd is a drop-in replacement after dependency/asset audit."
            ),
        },
        "operator": {
            "canonical": str(cfg.review.operator_canonical),
            "subtype": str(cfg.review.operator_subtype),
            "fidelity_level": str(cfg.review.fidelity_level),
            "parameters": cfg.adhesion.to_dict(),
            "footprint_xy_m": [list(value) for value in spec.adhesion_region.vertices_xy],
        },
        "camera": {
            "role": "Go2-rigid front RTX RGB; full 6-DoF base transform; no auto-aim",
            "implementation": (
                "World camera receives the exact rigid base-frame transform each step; Isaac 5.1 "
                "Replicator failed when the camera prim was parented below the articulation."
            ),
            "depth": "not collected; no synthetic zero channel",
            "parameters": cfg.camera.to_dict(),
        },
        "takes": replay_summaries,
        "render_protocol": (
            "Body-fixed and third-person views are same-seed independent replays because "
            "Isaac Sim 5.1 multi-Camera Replicator initialization is unstable on this host."
        ),
        "paper_status": str(cfg.review.paper_status),
        "provenance": {
            "git_commit": commit,
            "git_dirty": dirty,
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "scene_seed": seed,
            "torch": str(torch.__version__),
            "gpu": device_name,
            "command": " ".join(sys.argv),
            "reproduction_commands": {
                "front": f"{reproduction_prefix} --view front",
                "review": f"{reproduction_prefix} --view review",
            },
        },
    }
    _write_json(out_dir / "manifest.json", manifest)
    with (out_dir / "manifest.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False, allow_unicode=True)
    _write_json(
        out_dir / "review_summary.json",
        {"scene_audit": compiled["audit"], "takes": replay_summaries},
    )
    print(f"PASS: ICRA indoor adhesion review bundle written to {out_dir}")
    sys.stdout.flush()

    closer = threading.Thread(target=simulation_app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Kit may keep CUDA/RTX worker threads alive after a Python exception.
        # Preserve the traceback, then terminate this isolated demo process so a
        # failed smoke test cannot leak several gigabytes of GPU memory.
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
