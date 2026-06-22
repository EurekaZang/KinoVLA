#!/usr/bin/env python
"""Generate the RTX nav-SFT dataset — route-around turn/waypoint decisions on REAL camera frames.

ALL on the GPU (user directive 2026-06-21: no CPU render in any experiment path). For each sampled
robot pose around the hazard patch we teleport the real Go2 there, render the LIVE Isaac RTX body
camera (the very frames the deployed VLA reads, kino_vla.sim.live_camera), and label the correct
next nav action GEOMETRICALLY (kino_vla.vla.nav_teacher): a ``Replan_Waypoint`` pixel that skirts
the patch toward the goal, or a ``Turn`` when the clear way is out of frame. The label pixel is the
projection of the geometric look-ahead THROUGH the same real camera, so it back-projects exactly to
the intended waypoint at deploy. Co-train this with the recovery data so the VLA OWNS the routing
(no geometry in the loop) — train_vla_sft.py --nav-dataset.

    python scripts/build_nav_dataset.py --headless --out outputs/vla/nav_data --seed 7
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def _sample_poses(patch, goal, *, exclude_margin: float = 0.15) -> list[tuple[float, float, float]]:
    """Diverse route-around poses (robot OUTSIDE the patch): backout-facing-the-patch (turn-heavy),
    skirting the top/bottom (waypoint-heavy), approaching the corners, and past the patch."""

    def to(tx: float, ty: float, x: float, y: float) -> float:
        return math.atan2(ty - y, tx - x)

    gx, gy = float(goal[0]), float(goal[1])
    poses: list[tuple[float, float, float]] = []
    # 1) backout, facing the patch (the dog just reversed out → must turn or pick a side waypoint)
    for x in np.linspace(0.2, 1.15, 5):
        for y in np.linspace(-1.6, 1.6, 7):
            poses.append((x, y, 0.0))  # facing +x straight into the patch
            poses.append((x, y, to(patch.cx, patch.cy, x, y)))  # facing the patch centre
    # 2) skirting the top and bottom edges, facing roughly the goal (route-around waypoints)
    for x in np.linspace(-0.5, 6.3, 11):
        for y in (2.55, 3.0, 3.45, -2.55, -3.0, -3.45):
            hd = to(gx, gy, x, y)
            poses.append((x, y, hd))
            poses.append((x, y, hd + math.radians(18)))
    # 3) approaching the corners from the start side (mix of turn/waypoint)
    for x in np.linspace(-0.9, 1.0, 4):
        for y in np.linspace(-3.0, 3.0, 7):
            poses.append((x, y, to(gx, gy, x, y)))
    # 4) past the patch, heading to the goal
    for x in np.linspace(5.6, 6.7, 3):
        for y in np.linspace(-3.2, 3.2, 6):
            poses.append((x, y, to(gx, gy, x, y)))
    keep = []
    for x, y, hd in poses:
        on_patch = (
            abs(x - patch.cx) <= patch.hx + exclude_margin
            and abs(y - patch.cy) <= patch.hy + exclude_margin
        )
        if not on_patch:
            keep.append((float(x), float(y), float(hd)))
    return keep


def _target_for_label(label: dict) -> str:
    """The <Thought>/<Action> training target for a nav label (loss_span='action')."""
    if label["kind"] == "waypoint":
        thought = "Route around the hazard patch toward the goal on clear ground."
        action = {
            "attribution": "nominal",
            "primitive": "Replan_Waypoint",
            "params": {"point_px": label["point_px"]},
        }
    else:
        thought = (
            "No clear ground straight ahead; turn to bring the way around the patch into view."
        )
        action = {
            "attribution": "nominal",
            "primitive": "Turn",
            "params": {"yaw_deg": label["yaw_deg"]},
        }
    return f"<Thought>{thought}</Thought>\n<Action>{json.dumps(action)}</Action>"


def _cruise_proprio_pool(backend, cfg, seed: int, n_windows: int = 10) -> list[np.ndarray]:
    """Capture REAL cruise proprioception windows on clean ground (the deploy-time nav regime).

    The nav decision is driven by the RGB; the proprio is the latent-route side channel. We cruise
    the Go2 forward in a clean lane (away from the patch) and snapshot the rolling window so every
    nav example carries a real cruise window (NOT a synthesised one)."""
    from kino_vla.tokens.window import RollingWindow, window_length

    snap = cfg.snapshot
    win = RollingWindow(window_length(float(snap.window_ms), float(snap.control_hz)))
    backend._start_pos = np.array([0.0, -5.5])  # a clean lane well clear of the patch
    backend._start_heading = 0.0
    backend.reset(seed)
    pool: list[np.ndarray] = []
    wl = window_length(float(snap.window_ms), float(snap.control_hz))
    for k in range(120):
        obs = backend.step(np.array([0.5, 0.0, 0.0]))
        win.push(obs)
        if k >= wl and k % 8 == 0:
            pool.append(win.window().copy())
        if obs.base_height < 0.15:  # fell — stop cruising
            break
    if not pool:
        pool.append(win.window().copy())
    return pool


def main() -> int:
    ap = argparse.ArgumentParser(description="RTX nav-SFT dataset generator (route-around labels)")
    ap.add_argument("--out", default="outputs/vla/nav_data")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-poses", type=int, default=10_000)

    # Launch the Isaac app FIRST (the demo's pattern): isaaclab.sim only imports once the app is up,
    # and --enable_cameras is required for the RTX perception camera. AppLauncher adds --headless.
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    args.enable_cameras = True
    simulation_app = AppLauncher(args).app  # noqa: F841  (keep the app alive for the whole run)

    from kino_vla.sim.live_camera import LiveRtxCamera
    from kino_vla.sim.operators import Tether
    from kino_vla.skeleton import build_walking_skeleton
    from kino_vla.utils.config import load_config
    from kino_vla.utils.geometry import wrap_angle
    from kino_vla.vla.nav_teacher import next_nav_label
    from kino_vla.vla.prompt import PlannerContext, build_nav_messages, format_map_note

    def factory(rect):  # operator_factory(hazard_patch rect) -> (operator, scene_region)
        op = Tether(rect, k=80.0, d=6.0, l0=0.0, f_break=1.0e6, peel_factor=0.0)
        return op, op.scene_region()

    overrides = {
        "max_time_s": 60,
        "terrain.hazard_patch.center": [3.5, 0.0],
        "terrain.hazard_patch.half_size": [2.236, 2.236],
        "goal.pos": [7.5, 0.0],
    }
    print("[nav-data] building the tether skeleton (RTX camera + textured patch) …", flush=True)
    sk = build_walking_skeleton(
        args.seed,
        "isaac",
        demo_overrides=overrides,
        record_cam=True,
        operator_factory=factory,
        live_perception=True,
    )
    backend = sk.backend
    cam = LiveRtxCamera(backend)
    goal = np.asarray(sk.demo_cfg.goal.pos, dtype=np.float64)
    patch = sk.nav_map.scene[0].rect
    pcfg = load_config("data/hindsight.yaml")

    pool = _cruise_proprio_pool(backend, pcfg, args.seed)
    print(f"[nav-data] captured {len(pool)} real cruise proprio windows", flush=True)

    poses = _sample_poses(patch, goal)[: args.max_poses]
    print(f"[nav-data] {len(poses)} candidate poses", flush=True)

    records: list[dict] = []
    frames: dict[str, np.ndarray] = {}
    n_turn = n_wp = n_skip = 0
    for i, (x, y, hd) in enumerate(poses):
        backend._start_pos = np.array([x, y])
        backend._start_heading = hd
        obs = backend.reset(args.seed)
        if obs.base_height < 0.18 or obs.tilt > 0.6:  # fell / unstable after teleport+settle
            n_skip += 1
            continue
        pose = obs.pos.astype(np.float64)
        heading = float(obs.heading)
        # never label from inside the perceived patch (the dog should be routing AROUND it)
        if abs(pose[0] - patch.cx) <= patch.hx and abs(pose[1] - patch.cy) <= patch.hy:
            n_skip += 1
            continue
        rgb = cam.snapshot_rgb()  # captures + caches the cap (pixel_from_world reuses it below)
        if rgb is None:
            n_skip += 1
            continue
        label = next_nav_label(pose, heading, patch, goal, cam.pixel_from_world, margin=0.5)
        map_note = format_map_note(pose, heading, [patch])
        ctx = PlannerContext(
            monitor_channel="clock",
            pose_xy=(float(pose[0]), float(pose[1])),
            prior_outputs=[],
            map_note=map_note,
        )
        goal_bearing = math.degrees(
            wrap_angle(math.atan2(goal[1] - pose[1], goal[0] - pose[0]) - heading)
        )
        messages = build_nav_messages(ctx, pcfg, goal_bearing, route="latent", n_images=1)
        sid = f"nav_{i:04d}"
        records.append(
            {
                "sample_id": sid,
                "messages": messages,
                "target_text": _target_for_label(label),
                "kind": label["kind"],
                "pose_xy": [round(float(pose[0]), 3), round(float(pose[1]), 3)],
                "heading_deg": round(math.degrees(heading), 1),
                "goal_bearing_deg": round(goal_bearing, 1),
                "wp": [round(v, 2) for v in label["wp"]],
            }
        )
        frames[f"{sid}__rgb"] = rgb[None].astype(np.float32)  # (1, H, W, 3)
        frames[f"{sid}__proprio"] = pool[i % len(pool)].astype(np.float32)
        n_turn += label["kind"] == "turn"
        n_wp += label["kind"] == "waypoint"
        if (i + 1) % 25 == 0:
            print(f"[nav-data] {i + 1}/{len(poses)} (wp={n_wp} turn={n_turn})", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "nav_meta.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    np.savez_compressed(out / "nav_frames.npz", **frames)
    summary = {
        "n_examples": len(records),
        "n_waypoint": n_wp,
        "n_turn": n_turn,
        "n_skipped": n_skip,
        "patch": [patch.cx, patch.cy, patch.hx, patch.hy],
        "goal": [float(goal[0]), float(goal[1])],
        "n_proprio_windows": len(pool),
    }
    (out / "nav_card.json").write_text(json.dumps(summary, indent=2))
    print(f"[nav-data] DONE → {out}: {json.dumps(summary)}", flush=True)
    os._exit(0)  # Isaac SimulationApp.close() busy-spins (§6 #6); the data is flushed, force-exit
    return 0  # unreachable; keeps the type checker happy


if __name__ == "__main__":
    raise SystemExit(main())
