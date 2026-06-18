"""Per-operator closed-loop chase-cam recordings on the Isaac Go2 (one operator per run).

For a single Kino-Fail operator (--operator), drops it on the walking-skeleton path as the
hazard, runs the SAME closed loop the ice demo uses (Kino-Monitor → FSM recovery → CBF shield
→ semantic map), and records a robot-FOLLOWING chase camera (re-aimed every control step) to
``outputs/operators/<NAME>/chase.mp4``. Reports the honest closed-loop outcome — whether the
monitor fired, the robot recovered, fell, and reached the goal. The policy is zero-shot on the
operator and recovery is the FSM stub (real planner = M7), so falls/non-recoveries are expected
and recorded as-is — that is the benchmark's point.

One operator per Isaac process (Go2 prims persist across resets, so destabilising operators
can't share a forward course — CLAUDE.md #21); a separate process per operator sidesteps that.

Usage:
    python scripts/record_operators.py --operator O9_high_centering --headless
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading

import numpy as np


def _factories():
    """name -> (patch Rect) -> (operator, scene_region|None). Each operator is placed at the
    walking-skeleton path hazard so the robot walks into it; globals/events ignore the rect."""
    from kino_vla.map import SemanticRegion
    from kino_vla.sim.operators import (
        Collapse,
        ComplianceField,
        EffortDecay,
        HighCentering,
        InvisibleCollider,
        MuField,
        Payload,
        Push,
        Tether,
        VisualPhysicsRemap,
    )
    from kino_vla.utils.geometry import Rect

    def ice_region(r: Rect, cls: str) -> SemanticRegion:
        return SemanticRegion(Rect(r.cx, r.cy, r.hx, r.hy), cls)

    return {
        # O1 μ-field: low-friction ice sheet (the pinned demo hazard).
        "O1_mu_field": lambda r: (MuField(r, mu_s=0.10, mu_d=0.10), ice_region(r, "ice_sheet")),
        # O2 compliance-field: soft mud that resists + sinks the chassis.
        "O2_compliance": lambda r: (
            (op := ComplianceField(r, k_c=15.0, c_c=8.0, d_sink=0.05)),
            op.scene_region(),
        ),
        # O3 collapse: intact ground that collapses to ice after a dwell.
        "O3_collapse": lambda r: (
            Collapse(r, mu_collapsed=0.08, trigger_dwell_s=0.4, mu_intact=0.8),
            ice_region(r, "ice_sheet"),
        ),
        # O4 tether/adhesion: a sticky board that pulls back until it breaks.
        "O4_tether": lambda r: (
            (op := Tether(r, k=150.0, d=6.0, l0=0.0, f_break=22.0)),
            op.scene_region(),
        ),
        # O5 payload: a heavy load on the trunk (global; no region).
        "O5_payload": lambda r: (Payload(mass_kg=6.0, com_offset_m=(0.0, 0.0)), None),
        # O6 push: a lateral impulse mid-crossing (event; no region).
        "O6_push": lambda r: (Push(impulse_xy_ns=np.array([0.0, 22.0]), t_push_s=4.0), None),
        # O7 visual-physics remap: looks like solid ground, is low-μ ice (deceptive).
        "O7_visual_remap": lambda r: (
            (op := VisualPhysicsRemap(r, mu_s=0.10, mu_d=0.08, depth_bias_m=0.6)),
            op.scene_region(),
        ),
        # O8 invisible collider: an unmodelled wall on the path (no visual signature).
        "O8_invisible_collider": lambda r: (InvisibleCollider(r), None),
        # O9 high-centering: a ridge that unloads the feet.
        "O9_high_centering": lambda r: (HighCentering(r, residual_support=0.3), None),
        # O10 effort-decay: actuator heat-derating to a hard floor (global).
        "O10_effort_decay": lambda r: (
            EffortDecay(decay_rate_per_s=0.6, floor=0.2, t_start_s=2.0),
            None,
        ),
    }


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--operator", required=True)
    pre.add_argument("--out", default="outputs/operators")
    pre.add_argument("--fps", type=int, default=50)
    pre.add_argument("--seed", type=int, default=None)
    pre_args, _ = pre.parse_known_args()

    parser = argparse.ArgumentParser(description="Per-operator closed-loop chase-cam recording")
    parser.add_argument("--operator", required=True)
    parser.add_argument("--out", default="outputs/operators")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True  # RTX render for the chase camera
    simulation_app = AppLauncher(args).app

    import imageio.v2 as imageio

    from kino_vla.loop import run_episode
    from kino_vla.skeleton import build_walking_skeleton
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.utils.seeding import seed_everything

    factories = _factories()
    if args.operator not in factories:
        raise SystemExit(f"unknown operator {args.operator!r}; choices: {sorted(factories)}")

    seed = args.seed if args.seed is not None else int(load_config("default.yaml").seed)
    seed_everything(seed)

    skeleton = build_walking_skeleton(
        seed, "isaac", record_cam=True, operator_factory=factories[args.operator]
    )
    goal = np.asarray(skeleton.demo_cfg.goal.pos, dtype=np.float64)
    out_dir = REPO_ROOT / args.out / args.operator
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_dir / "chase.mp4"), fps=args.fps, codec="libx264", quality=8, macro_block_size=None
    )

    n_frames = 0

    def on_step(obs, event, cmd, decision) -> None:
        nonlocal n_frames
        # Chase camera rigidly trailing the robot: 2.8 m behind + 1.6 m up, looking just ahead.
        c, s = math.cos(obs.heading), math.sin(obs.heading)
        eye = np.array([obs.pos[0] - 2.8 * c, obs.pos[1] - 2.8 * s, 1.6])
        target = np.array([obs.pos[0] + 0.5 * c, obs.pos[1] + 0.5 * s, 0.25])
        skeleton.backend.aim_record_camera(eye, target)
        frame = skeleton.backend.capture_rgb()
        if frame is not None:
            writer.append_data(frame)
            n_frames += 1

    result = run_episode(
        skeleton.backend,
        skeleton.operators,
        skeleton.monitor,
        skeleton.policy,
        skeleton.shield,
        seed=seed,
        goal_xy=goal,
        goal_tol_m=float(skeleton.demo_cfg.goal.tol_m),
        max_time_s=float(skeleton.demo_cfg.max_time_s),
        on_step=on_step,
        nav_map=skeleton.nav_map,
    )
    writer.close()

    outcome = {
        "operator": args.operator,
        "seed": seed,
        "frames": n_frames,
        "monitor_fired": result.monitor_fired,
        "recovered": len(result.events) > 0 and not result.fell,
        "fell": result.fell,
        "goal_reached": result.goal_reached,
        "final_dist_m": round(result.final_dist_m, 3),
        "sim_time_s": round(result.sim_time_s, 2),
        "n_events": len(result.events),
        "video": str(out_dir / "chase.mp4"),
    }
    (out_dir / "outcome.json").write_text(json.dumps(outcome, indent=2))
    print(
        f"[op] {args.operator}: fired={result.monitor_fired} fell={result.fell} "
        f"goal={result.goal_reached} dist={result.final_dist_m:.2f}m events={len(result.events)} "
        f"frames={n_frames}"
    )
    print(f"PASS: recorded {args.operator}")

    sys.stdout.flush()
    closer = threading.Thread(target=simulation_app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0)


if __name__ == "__main__":
    main()
