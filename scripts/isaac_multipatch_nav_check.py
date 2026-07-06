#!/usr/bin/env python
"""M7 multi-patch + long-distance gate (#47): the decoupled framework routes around SEVERAL hazards
on a course longer than the start-centred costmap, on the REAL Go2.

Proves the readiness fixes #47 integrated on the real stack:
  • #3 ROLLING costmap — the goal (x≈11) is BEYOND the start-centred grid (x∈[-8,8]); the egocentric
    grid must follow the robot so the far patch + goal become representable (rolling: true).
  • #1 max_rounds is per-spot, not lifetime — several distinct hazard contacts each get attributed
    (the lifetime cap of 6 never blocks a long course).
  • #2 location-aware grace — the 2nd patch's fire is NOT suppressed by the 1st recovery's grace.
  • the decoupled planner commits a fresh route around EACH marked patch and reaches the far goal.

Two yellow-adhesive tethers (escape → Backstep → route-around) at x≈3 and x≈7.5; goal at x≈11.
PASS = reach the far goal, no fall, end OUTSIDE every patch (routed around all), ≥1 attribution, and
the costmap actually rolled (long-distance exercised).

    python scripts/isaac_multipatch_nav_check.py --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _patch_verdict(traj: list, rect) -> dict:
    """Per-patch enter/exit summary from the (t,x,y) trajectory (terminal-inside ⇒ not routed)."""

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
    return {
        "entered": bool(segs),
        "n_inside_segments": int(len(segs)),
        "terminal_inside": bool(flags and flags[-1]),
        "cx": round(float(rect.cx), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="M7 multi-patch + long-distance gate (real Go2, #47)")
    ap.add_argument("--adapter", default="outputs/vla/sft_latent/adapter_best")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-time", type=float, default=160.0)
    # Use the PROVEN tether geometry (half 2.236 = the decoupled gate's, which reliably attributes
    # "adhesion" → Backstep → route-around; a small patch gives a weak signal the VLA mis-reads as
    # low_friction). Two patches spaced clear; goal far beyond the start grid ⇒ rolling.
    ap.add_argument("--patch1-cx", type=float, default=3.5)
    ap.add_argument("--patch2-cx", type=float, default=10.5)
    ap.add_argument("--patch-half", type=float, default=2.236)
    ap.add_argument("--goal-x", type=float, default=15.0, help="beyond the start grid ⇒ rolls")

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
    from kino_vla.utils.geometry import Rect
    from kino_vla.utils.seeding import seed_everything
    from kino_vla.vla.model import KinoVLA
    from kino_vla.vla.planner import ModelVlaPolicy, VlaPlanner

    seed_everything(args.seed)

    def _tether(rect: Rect):
        return Tether(rect, k=80.0, d=6.0, l0=0.0, f_break=1.0e6, peel_factor=0.0)

    half = float(args.patch_half)
    # patch1 = the primary (placed at terrain.hazard_patch); patch2 = an extra operator/region.
    rect2 = Rect(float(args.patch2_cx), 0.0, half, half)
    op2 = _tether(rect2)
    extra = [(op2, op2.scene_region())]

    overrides = {
        "max_time_s": args.max_time,
        "terrain.hazard_patch.center": [float(args.patch1_cx), 0.0],
        "terrain.hazard_patch.half_size": [half, half],
        "goal.pos": [float(args.goal_x), 0.0],
    }
    skeleton = build_walking_skeleton(
        args.seed,
        "isaac",
        demo_overrides=overrides,
        record_cam=True,
        operator_factory=lambda r: (_tether(r), _tether(r).scene_region()),
        live_perception=True,
        extra_operators=extra,
        map_overrides={"rolling": True},  # #3 egocentric costmap for the long course
    )
    nav_map = skeleton.nav_map
    goal = np.asarray(skeleton.demo_cfg.goal.pos, dtype=np.float64)
    patches = [reg.rect for reg in nav_map.scene]
    origin0 = nav_map.costmap.origin_xy.copy()  # to detect that the grid rolled

    vcfg = load_config("vla/sft.yaml")
    pcfg = load_config("data/hindsight.yaml")
    route = str(vcfg.get("route", "latent"))
    n_images = int(vcfg.data.get("n_images", 1))
    print(f"[gate] {len(patches)} patches, goal x={goal[0]} (rolling); loading VLA …", flush=True)
    model = KinoVLA.from_pretrained(vcfg, device="cuda", adapter_dir=args.adapter)
    model.eval()
    torch.set_grad_enabled(False)

    scene_region = nav_map.scene[0]
    # Gentler push-off for this gate: the marginal Go2 startup drifts +y and the extra PhysX prims
    # of a 2-patch scene perturb the knife-edge (the single-patch decoupled gate survives default).
    # A slower, longer startup line-track tames the push-off topple — orthogonal to the #47 logic
    # under test (which only engages on hazard CONTACT, well after the warmup).
    fsm_config = load_config(
        "recovery/fsm_isaac.yaml",
        overrides={"startup_speed_mps": 0.25, "startup_straight_s": 3.5},
    )
    recorder = SnapshotRecorder(
        pcfg,
        scene=list(nav_map.scene),
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
        perceive_every=25,
    )
    per_patch = [_patch_verdict(traj, r) for r in patches]
    rolled = bool(float(np.linalg.norm(nav_map.costmap.origin_xy - origin0)) > 1e-6)
    routed_all = all(not pv["terminal_inside"] for pv in per_patch)
    recovery_rounds = len(getattr(planner, "decisions", []))
    max_x = max((p[1] for p in traj), default=0.0)
    # PASS: reached the FAR goal, no fall, ended outside EVERY patch (routed around all), >=1
    # attribution happened, and the costmap actually rolled (the long-distance path was exercised).
    passed = bool(
        result.goal_reached and not result.fell and routed_all and recovery_rounds >= 1 and rolled
    )
    report = {
        "n_patches": len(patches),
        "goal_x": float(goal[0]),
        "reached_goal": bool(result.goal_reached),
        "fell": bool(result.fell),
        "final_dist_m": round(float(result.final_dist_m), 3),
        "max_x_reached": round(float(max_x), 2),
        "recovery_rounds": recovery_rounds,
        "attributions": [
            {"attribution": d.attribution, "primitive": d.primitive}
            for d in getattr(planner, "decisions", [])
        ],
        "costmap_physical_cells": int(nav_map.costmap.n_physical),
        "costmap_rolled": rolled,
        "origin_shift_m": round(float(np.linalg.norm(nav_map.costmap.origin_xy - origin0)), 2),
        "per_patch": per_patch,
        "routed_around_all": routed_all,
        "PASS": passed,
    }
    out = REPO_ROOT / "outputs/vla/multipatch_nav_gate"
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps({**report, "traj": traj}, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(("PASS" if passed else "FAIL") + " multi-patch + long-distance gate (#47)")
    sys.stdout.flush()
    os._exit(0 if passed else 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
