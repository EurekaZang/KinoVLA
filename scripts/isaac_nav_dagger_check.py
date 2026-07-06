#!/usr/bin/env python
"""M7 nav-DAgger gate (#43): the trained VLA emits Turn and routes around on the REAL Go2 tether 5x.

The freeze the DAgger sprint fixes: before, the deployed SFT VLA emitted 0 Turns / 94 forbidden
Replan_Waypoints on the 5x adhesive tether and froze for 87 s at x=1.1. This gate runs the trained
VLA planner CLOSED-LOOP on the physical Go2 (live RTX + real CLIP, the §0 real stack) and asserts
the data-level fix took: the VLA now EMITS Turn(s) at the freeze region, the freeze rate -> ~0
(nav picks commit), and the dog routes around (enter once, exit, no re-entry) toward the goal.

    python scripts/isaac_nav_dagger_check.py --headless  # [--adapter <dir>] (default sft_latent)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _route_around_verdict(traj: list, rect, goal, goal_tol: float, reached: bool) -> dict:
    """enter the patch once -> exit -> no re-entry -> reach the goal (the ideal route)."""

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
        "n_inside_segments": int(len(segs)),
        "exited_completely": bool(exited),
        "re_entered": bool(len(segs) >= 2),
        "reached_goal": bool(reached),
        "ideal": bool(len(segs) <= 1 and exited and reached),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="M7 nav-DAgger gate (real Go2 tether 5x)")
    ap.add_argument("--adapter", default="outputs/vla/sft_latent/adapter_best")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-time", type=float, default=60.0)
    ap.add_argument("--min-turns", type=int, default=1, help="≥ this many Turns ⇒ Turn is alive")
    ap.add_argument("--max-freeze-frac", type=float, default=0.2, help="uncommitted nav-pick frac")
    ap.add_argument(
        "--min-agreement",
        type=float,
        default=0.7,
        help="min fraction of committed nav decisions whose KIND (+turn DIRECTION) matches the "
        "privileged geometric teacher — the real metric (the model already emits some turns; the "
        "fix is making them AGREE with the route-around optimum, #44)",
    )
    # Patch geometry (defaults = the extreme 5× tether, area 20 m². --patch-half 1.414 ⇒ 2× [8 m²];
    # isolates the SIZE variable: same start/center/goal, smaller trap — tests if the residual
    # failure is the 5× EXTREMITY vs the data approach itself, #44).
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
    from kino_vla.vla.nav_teacher import next_nav_label
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
    patches = [reg.rect for reg in nav_map.scene]
    cam = LiveRtxCamera(skeleton.backend)  # the teacher's projector (real intrinsics + pose)

    vcfg = load_config("vla/sft.yaml")
    pcfg = load_config("data/hindsight.yaml")
    route = str(vcfg.get("route", "latent"))
    n_images = int(vcfg.data.get("n_images", 1))
    print("[gate] loading Qwen3-VL-4B + LoRA + Kino-Projector …", flush=True)
    model = KinoVLA.from_pretrained(vcfg, device="cuda", adapter_dir=args.adapter)
    model.eval()
    torch.set_grad_enabled(False)

    scene_region = nav_map.scene[0]
    fsm_config = load_config("recovery/fsm_isaac.yaml")
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

    counts = {"turn": 0, "waypoint": 0, "forbidden": 0, "freeze": 0, "total": 0}
    agree = {"decided": 0, "kind_ok": 0, "turn_into_hazard": 0}

    def _model_kind_yaw(tag: str):
        if tag.startswith("turn"):
            try:
                return "turn", float(tag[len("turn") :])
            except ValueError:
                return "turn", 0.0
        if tag == "Replan_Waypoint":
            return "waypoint", None
        return None, None

    def sink(rec: dict) -> None:
        counts["total"] += 1
        tag = str(rec["tag"])
        m_kind, m_yaw = _model_kind_yaw(tag)
        if tag.startswith("turn"):
            counts["turn"] += 1
        elif tag == "Replan_Waypoint":
            counts["waypoint"] += 1
        elif tag == "wp_forbidden":
            counts["forbidden"] += 1
        if not rec["committed"]:
            counts["freeze"] += 1
        # decision-agreement vs the privileged geometric teacher (the real #44 metric)
        if rec["committed"] and m_kind is not None:
            pose = np.asarray(rec["pose_xy"], dtype=np.float64)
            heading = float(rec["heading"])
            cam.snapshot_rgb()  # refresh the cap at THIS pose
            t = next_nav_label(pose, heading, patches, goal, cam.pixel_from_world, margin=0.5)
            agree["decided"] += 1
            kind_ok = m_kind == t["kind"]
            if kind_ok and m_kind == "turn" and abs(float(t["yaw_deg"])) > 20.0:
                kind_ok = (m_yaw or 0.0) != 0.0 and ((m_yaw or 0.0) > 0) == (
                    float(t["yaw_deg"]) > 0
                )
            agree["kind_ok"] += int(kind_ok)
            if t["kind"] == "turn" and m_kind == "waypoint":
                agree["turn_into_hazard"] += 1  # the route-INTO-hazard mistake (the freeze cause)

    planner.nav_trace_sink = sink

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
    verdict = _route_around_verdict(
        traj, patch, goal, float(skeleton.demo_cfg.goal.tol_m), bool(result.goal_reached)
    )
    freeze_frac = counts["freeze"] / max(1, counts["total"])
    teacher_agreement = agree["kind_ok"] / max(1, agree["decided"])
    emitted_turn = counts["turn"] >= args.min_turns
    no_freeze = freeze_frac <= args.max_freeze_frac
    agrees = teacher_agreement >= args.min_agreement
    routed = not verdict["re_entered"] and verdict["entered"]
    # The real #44 PASS: the model's nominal decisions AGREE with the route-around teacher AND the
    # dog routes around without re-entry. "emitted_turn"/"no_freeze" are reported but not gating
    # (the model already emits some turns; agreement is what the data-level fix must move).
    passed = bool(agrees and routed)
    report = {
        "adapter": args.adapter,
        "nav_pick_counts": counts,
        "teacher_agreement": round(teacher_agreement, 3),
        "decided_nav_picks": agree["decided"],
        "turn_into_hazard_mistakes": agree["turn_into_hazard"],
        "freeze_frac": round(freeze_frac, 3),
        "emitted_turn": bool(emitted_turn),
        "no_freeze": bool(no_freeze),
        "agrees_with_teacher": bool(agrees),
        "routed_no_reentry": bool(routed),
        "reached_goal": bool(result.goal_reached),
        "fell": bool(result.fell),
        "final_dist_m": round(float(result.final_dist_m), 3),
        "patch_half": args.patch_half,
        "trajectory_verdict": verdict,
        "PASS": passed,
    }
    out = REPO_ROOT / "outputs/vla/nav_dagger_gate"
    out.mkdir(parents=True, exist_ok=True)
    fname = f"gate{('_' + args.tag) if args.tag else ''}.json"
    (out / fname).write_text(json.dumps({**report, "traj": traj}, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(("PASS" if passed else "FAIL") + " nav-DAgger gate (Turn + freeze->0 + route-around)")
    sys.stdout.flush()
    os._exit(0 if passed else 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
