"""Record the walking-skeleton demo: per-step telemetry + (Isaac) a 3D camera video.

Runs one walking-skeleton episode while logging every control step (pose, speed,
velocity command, the three Kino-Monitor channels + thresholds, monitor events, FSM
recovery phase / avoid regions / waypoints, fall state) to ``outputs/recording/``,
and — on the Isaac backend — streams a fixed wide-shot RGB render of the real Go2 to
``isaac_raw.mp4``. ``scripts/render_demo_video.py`` turns these into the final
annotated MP4.

Usage:
    python scripts/record_demo.py --backend isaac --headless
    python scripts/record_demo.py --backend surrogate
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys

import numpy as np


def _isaac_available() -> bool:
    return importlib.util.find_spec("isaaclab") is not None


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--backend", choices=["surrogate", "isaac"], default="isaac")
    pre_args, _ = pre.parse_known_args()
    backend = pre_args.backend
    if backend == "isaac" and not _isaac_available():
        backend = "surrogate"

    parser = argparse.ArgumentParser(description="Record the walking-skeleton demo")
    parser.add_argument("--backend", choices=["surrogate", "isaac"], default="isaac")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="outputs/recording")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--cam",
        action="store_true",
        help="capture the Isaac 3D camera (RTX render; unstable headless on some GPUs)",
    )

    simulation_app = None
    if backend == "isaac":
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
        if args.cam:
            args.enable_cameras = True  # RTX rendering, only when capturing the camera
        simulation_app = AppLauncher(args).app
    else:
        parser.add_argument("--headless", action="store_true")
        args = parser.parse_args()

    import imageio.v2 as imageio

    from kino_vla.loop import run_episode
    from kino_vla.skeleton import build_walking_skeleton
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.utils.seeding import seed_everything

    seed = args.seed if args.seed is not None else int(load_config("default.yaml").seed)
    seed_everything(seed)

    use_cam = bool(args.cam) and backend == "isaac"
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    skeleton = build_walking_skeleton(seed, backend, record_cam=use_cam)
    monitor, policy = skeleton.monitor, skeleton.policy
    goal = np.asarray(skeleton.demo_cfg.goal.pos, dtype=np.float64)

    rows: list[dict] = []
    avoid_per_step: list[list[list[float]]] = []
    wp_per_step: list[list[list[float]]] = []
    writer = None
    if use_cam:
        writer = imageio.get_writer(
            str(out_dir / "isaac_raw.mp4"),
            fps=args.fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )

    def on_step(obs, event, cmd, decision) -> None:
        emas = monitor.channel_emas
        track_raw = float(np.linalg.norm(obs.cmd_prev[:2] - obs.vel_body))
        rows.append(
            {
                "t": obs.t,
                "x": float(obs.pos[0]),
                "y": float(obs.pos[1]),
                "heading": obs.heading,
                "vx": float(obs.vel_body[0]),
                "vy": float(obs.vel_body[1]),
                "speed": float(np.linalg.norm(obs.vel_body)),
                "cmd_vx": float(cmd[0]),
                "cmd_wz": float(cmd[2]),
                "cmd_speed": float(np.linalg.norm(np.asarray(cmd[:2]))),
                "slip_raw": obs.slip_ratio,
                "slip_ema": emas["slip_ratio"],
                "track_raw": track_raw,
                "track_ema": emas["tracking_error"],
                "effort_raw": obs.effort_ratio,
                "effort_ema": emas["effort_ratio"],
                "fired": 1 if event is not None else 0,
                "fired_channel": (event.channel if event is not None else ""),
                "base_height": obs.base_height,
                "tilt": obs.tilt,
                "fallen": 1 if obs.fallen else 0,
                "phase": policy.phase.value,
            }
        )
        avoid_per_step.append([[c.center[0], c.center[1], c.radius] for c in policy.avoid_circles])
        wp_per_step.append([[float(w[0]), float(w[1])] for w in policy.waypoints])
        if writer is not None:
            frame = skeleton.backend.capture_rgb()
            if frame is not None:
                writer.append_data(frame)

    result = run_episode(
        skeleton.backend,
        skeleton.operators,
        monitor,
        policy,
        skeleton.shield,
        seed=seed,
        goal_xy=goal,
        goal_tol_m=float(skeleton.demo_cfg.goal.tol_m),
        max_time_s=float(skeleton.demo_cfg.max_time_s),
        on_step=on_step,
    )
    if writer is not None:
        writer.close()

    keys = [k for k in rows[0] if k not in ("fired_channel", "phase")]
    arrays = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in keys}
    arrays["fired_channel"] = np.array([r["fired_channel"] for r in rows])
    arrays["phase"] = np.array([r["phase"] for r in rows])
    arrays["avoid"] = np.array(avoid_per_step, dtype=object)
    arrays["waypoints"] = np.array(wp_per_step, dtype=object)
    np.savez(out_dir / "telemetry.npz", **arrays, allow_pickle=True)

    patch = skeleton.terrain.hazard_patch
    meta = {
        "backend": backend,
        "seed": seed,
        "fps": args.fps,
        "ice": {
            "cx": patch.cx,
            "cy": patch.cy,
            "hx": patch.hx,
            "hy": patch.hy,
            "mu_d": float(skeleton.demo_cfg.ice.mu_d),
        },
        "goal": {
            "x": float(goal[0]),
            "y": float(goal[1]),
            "tol": float(skeleton.demo_cfg.goal.tol_m),
        },
        "start": {
            "x": float(skeleton.demo_cfg.start.pos[0]),
            "y": float(skeleton.demo_cfg.start.pos[1]),
        },
        "terrain_extent": list(skeleton.terrain.extent),
        "thresholds": load_config(
            "monitor/rule_v0_isaac.yaml" if backend == "isaac" else "monitor/rule_v0.yaml"
        ).thresholds.to_dict(),
        "result": {
            "monitor_fired": result.monitor_fired,
            "fell": result.fell,
            "goal_reached": result.goal_reached,
            "final_dist_m": result.final_dist_m,
            "sim_time_s": result.sim_time_s,
            "n_steps": result.n_steps,
        },
        "transitions": list(policy.transitions),
        "events": [
            {
                "t": e.t,
                "x": float(e.pos[0]),
                "y": float(e.pos[1]),
                "channel": e.channel,
                "value": e.value,
                "threshold": e.threshold,
            }
            for e in result.events
        ],
        "has_video": use_cam,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"recorded {len(rows)} steps to {out_dir}")
    print(
        f"  monitor_fired={result.monitor_fired} fell={result.fell} "
        f"goal_reached={result.goal_reached}"
    )
    print(f"  telemetry: {out_dir / 'telemetry.npz'}")
    if use_cam:
        print(f"  isaac video: {out_dir / 'isaac_raw.mp4'}")
    print("PASS: record_demo")

    sys.stdout.flush()
    if simulation_app is not None:
        import os
        import threading

        closer = threading.Thread(target=simulation_app.close, daemon=True)
        closer.start()
        closer.join(timeout=15.0)
        os._exit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
