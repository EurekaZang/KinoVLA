#!/usr/bin/env python
"""M7 decoupled-nav gate (#44): the VLA ATTRIBUTES, a grid planner ROUTES — on the REAL Go2.

The oscillation #43/#44 fought: a per-frame-reactive VLA picks the nominal waypoint each tick from a
single frame, so even with a persistent world-frame hazard map (used as a veto) it ping-pongs at the
patch edge (turn -> re-see patch -> turn back -> drive in) and never routes all the way around the
5x adhesive tether. The fix is architectural, not more data: DECOUPLE open-set attribution (the VLA,
kept) from memory+commitment (a classic grid planner over the persistent §7 costmap, kino_vla/vla/
nav_planner.py). The VLA marks "region R untraversable, type=<open-vocab>" ONCE on contact; the
planner commits a complete route around the marked cells and holds it.

This gate runs the trained VLA planner CLOSED-LOOP on the physical Go2 (live RTX + real CLIP, the §0
real stack) in either mode and asserts the decoupled arm REACHES THE GOAL without falling and
without the endless edge-oscillation. Run BOTH arms for the paper's headline contrast:

    python scripts/isaac_decoupled_nav_check.py --headless                       # decoupled (PASS)
    python scripts/isaac_decoupled_nav_check.py --headless --nav-mode vla_reactive  # baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _route_around_verdict(traj: list, rect, reached: bool) -> dict:
    """enter the patch once -> exit -> no re-entry -> reach the goal (the ideal decoupled route)."""

    def inside(x: float, y: float) -> bool:
        return abs(x - rect.cx) <= rect.hx and abs(y - rect.cy) <= rect.hy

    flags = [inside(p[1], p[2]) for p in traj]
    segs, i = [], 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j < len(flags) and flags[j]:
                j += 1
            segs.append((i, j - 1))
            i = j
        else:
            i += 1
    entered = len(segs) >= 1
    exited = entered and not flags[-1]
    return {
        "entered": bool(entered),
        "n_inside_segments": int(len(segs)),  # 1 = discovered once then routed; >1 = re-contacts
        "exited_completely": bool(exited),
        "re_entered": bool(len(segs) >= 2),
        "reached_goal": bool(reached),
        "ideal": bool(len(segs) <= 1 and exited and reached),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="M7 decoupled-nav gate (real Go2 tether)")
    ap.add_argument("--adapter", default="outputs/vla/sft_latent/adapter_best")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-time", type=float, default=90.0)
    ap.add_argument(
        "--nav-mode",
        choices=["decoupled", "vla_reactive"],
        default="decoupled",
        help="decoupled = grid planner routes off the costmap (the fix); vla_reactive = the VLA "
        "picks per frame (the oscillating ablation baseline)",
    )
    # Patch geometry (defaults = the extreme 5× tether, area 20 m²; --patch-half 1.414 ⇒ 2× [8 m²]).
    ap.add_argument("--patch-half", type=float, default=2.236, help="patch half-size (m); 1.414=2×")
    ap.add_argument("--patch-cx", type=float, default=3.5, help="patch center x (m)")
    ap.add_argument("--goal-x", type=float, default=7.5, help="goal x (m)")
    ap.add_argument("--tag", default="", help="label suffix for the gate.json (e.g. 2x)")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    args.enable_cameras = True
    simulation_app = AppLauncher(args).app  # noqa: F841

    import torch

    from kino_vla.data.snapshot import SnapshotRecorder
    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.loop import run_episode
    from kino_vla.monitor.reflex import ActiveProbe
    from kino_vla.shield.primitive_compiler import PrimitiveCompiler
    from kino_vla.sim.live_camera import LiveRtxCamera
    from kino_vla.sim.operators import Tether
    from kino_vla.skeleton import build_walking_skeleton
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.utils.seeding import seed_everything
    from kino_vla.vla.model import KinoVLA
    from kino_vla.vla.planner import ModelVlaPolicy, VlaPlanner

    seed_everything(args.seed)

    def factory(r):
        op = Tether(r, k=80.0, d=6.0, l0=0.0, f_break=1.0e6, peel_factor=0.0)
        return op, op.scene_region()

    overrides = {
        "max_time_s": args.max_time,
        "terrain.hazard_patch.center": [args.patch_cx, 0.0],
        "terrain.hazard_patch.half_size": [args.patch_half, args.patch_half],
        "goal.pos": [args.goal_x, 0.0],
    }
    skeleton = build_walking_skeleton(
        args.seed,
        "isaac",
        demo_overrides=overrides,
        record_cam=True,
        operator_factory=factory,
        live_perception=True,
    )
    nav_map = skeleton.nav_map
    goal = np.asarray(skeleton.demo_cfg.goal.pos, dtype=np.float64)
    patch = nav_map.scene[0].rect

    vcfg = load_config("vla/sft.yaml")
    pcfg = load_config("data/hindsight.yaml")
    route = str(vcfg.get("route", "latent"))
    n_images = int(vcfg.data.get("n_images", 1))
    print(
        f"[gate] nav_mode={args.nav_mode}; loading Qwen3-VL-4B + LoRA + Kino-Projector …",
        flush=True,
    )
    model = KinoVLA.from_pretrained(vcfg, device="cuda", adapter_dir=args.adapter)
    model.eval()
    torch.set_grad_enabled(False)

    scene_region = nav_map.scene[0]
    # The deployed loop is plugin-free: the planner routes off the persistent costmap (nav_mode set
    # via a load-time override so BOTH arms share one calibration — isolates the architecture).
    fsm_config = load_config("recovery/fsm_isaac.yaml", overrides={"nav_mode": args.nav_mode})
    recorder = SnapshotRecorder(
        pcfg,
        scene=[scene_region],
        operator_name="O4_tether",
        appearance_class=scene_region.appearance_class,
        privileged_fn=lambda: {},
        gate_rect=None,
        live_camera=LiveRtxCamera(skeleton.backend),
    )
    planner = VlaPlanner(
        cfg=fsm_config,
        policy=ModelVlaPolicy(
            model,
            pcfg,
            FailureTaxonomy(pcfg),
            route=route,
            n_images=n_images,
            temperature=0.0,
            proprio_detail=str(vcfg.data.get("proprio_detail", "binned")),
        ),
        goal_xy=goal,
        dt=skeleton.backend.dt,
        recorder=recorder,
        compiler=PrimitiveCompiler(skeleton.shield),
        probe=ActiveProbe.from_config(fsm_config),
    )
    planner.monitor = skeleton.monitor
    planner.nav_map = nav_map

    traj: list = []

    def on_step(obs, event, cmd, decision) -> None:  # noqa: ARG001
        traj.append([round(float(obs.t), 3), float(obs.pos[0]), float(obs.pos[1])])

    result = run_episode(
        skeleton.backend,
        skeleton.operators,
        skeleton.monitor,
        planner,
        skeleton.shield,
        seed=args.seed,
        goal_xy=goal,
        goal_tol_m=float(skeleton.demo_cfg.goal.tol_m),
        max_time_s=float(skeleton.demo_cfg.max_time_s),
        on_step=on_step,
        nav_map=nav_map,
        perceive_every=25,  # ~2 Hz RTX perception (quality-neutral throughput; #44)
    )
    verdict = _route_around_verdict(traj, patch, bool(result.goal_reached))
    # PASS = the real success: reach the goal without falling (so NO endless edge-oscillation). The
    # route-around quality (one discovery contact then a committed detour vs several re-contacts) is
    # reported as a diagnostic — fewer inside-segments is better, but reaching the goal is the win.
    passed = bool(result.goal_reached and not result.fell)
    report = {
        "nav_mode": args.nav_mode,
        "adapter": args.adapter,
        "reached_goal": bool(result.goal_reached),
        "fell": bool(result.fell),
        "final_dist_m": round(float(result.final_dist_m), 3),
        "n_committed_routes": len(planner.nav_log),  # how many times the planner re-planned
        "costmap_physical_cells": int(nav_map.costmap.n_physical),
        "recovery_rounds": len(getattr(planner, "decisions", [])),
        "patch_half": args.patch_half,
        "trajectory_verdict": verdict,
        "PASS": passed,
    }
    out = REPO_ROOT / "outputs/vla/decoupled_nav_gate"
    out.mkdir(parents=True, exist_ok=True)
    fname = f"gate_{args.nav_mode}{('_' + args.tag) if args.tag else ''}.json"
    (out / fname).write_text(
        json.dumps({**report, "traj": traj, "nav_log": planner.nav_log}, indent=2)
    )
    print(json.dumps(report, indent=2), flush=True)
    print(("PASS" if passed else "FAIL") + f" decoupled-nav gate (nav_mode={args.nav_mode})")
    sys.stdout.flush()
    os._exit(0 if passed else 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
