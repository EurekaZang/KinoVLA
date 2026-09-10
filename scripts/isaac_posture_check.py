#!/usr/bin/env python
"""M7 #41 — the BODY-POSTURE controller on the real Isaac Go2 (closes the §6 #40 residual).

Every recovery primitive must land a real, distinct, precisely-executed low-level response (user
goal). Switch_Gait / Adjust_Posture / Set_Constraint used to change only the CBF shield's support
polygon + a speed cap — the dog's body never moved (``set_reflex`` was a no-op on Isaac). This gate
drives the physically-simulated Go2 with the new closed-loop posture controller
(kino_vla/sim/isaac_policy_backend.py: a leg-extension residual blended onto the policy action,
I-controlled on the MEASURED trunk height) and verifies the body height is tracked:

  PASS criteria (on the real Go2, while walking forward):
    1. a crawl (low z_c) command lowers the trunk meaningfully below the nominal trot;
    2. a high_step (high z_c) command raises it above the crawl height;
    3. the two gaits are DISTINCT (height gap ≥ gap_tol) and correctly ORDERED;
    4. an Adjust_Posture(height) command tracks its target within a tolerance;
    5. releasing posture returns the trunk to ~nominal;
    6. NO fall throughout.

The targets are the shield modes' z_c (what the planner emits). A diagnostic line flags an inverted
flex sign (height moved the wrong way) so configs/sim/go2_skeleton.yaml posture.sign can be flipped.

Run:  python scripts/isaac_posture_check.py --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="M7 #41 Isaac body-posture controller gate")
    parser.add_argument("--fwd", type=float, default=0.3, help="forward walk speed [m/s]")
    parser.add_argument("--settle-s", type=float, default=2.0, help="settle/measure-nominal window")
    parser.add_argument("--hold-s", type=float, default=4.0, help="per-posture hold window")
    parser.add_argument(
        "--gap-tol", type=float, default=0.03, help="min crawl↔high_step height gap"
    )
    parser.add_argument("--track-tol", type=float, default=0.04, help="Adjust_Posture tracking tol")
    parser.add_argument(
        "--return-tol",
        type=float,
        default=0.08,
        help="release band: the trunk must come back UP to within this of nominal (the free trot "
        "height has ~0.04 m gait-phase/settle variance, so this is one-sided, not a tight abs tol)",
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    from kino_vla.shield.cbf_shield import CbfShield
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import load_config

    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)
    dt = backend.dt
    modes = CbfShield(load_config("shield/cbf_v0.yaml")).modes_table()
    crawl_z, high_z = float(modes["crawl"].z_c), float(modes["high_step"].z_c)
    fwd = np.array([float(args.fwd), 0.0, 0.0])

    def run(steps: int) -> float:
        """Step the backend `steps` times at the forward command; return mean base_height of the
        trailing half (steady-state), or NaN if the dog fell."""
        heights: list[float] = []
        for k in range(steps):
            obs = backend.step(fwd)
            if obs.fallen:
                return float("nan")
            if k >= steps // 2:
                heights.append(obs.base_height)
        return float(np.mean(heights)) if heights else float("nan")

    backend.reset(0)
    n_settle = int(args.settle_s / dt)
    n_hold = int(args.hold_s / dt)

    backend.set_posture(None)
    nominal = run(n_settle)
    print(f"[posture] nominal trot base_height = {nominal:.3f} m")

    backend.set_posture(crawl_z, stiffness=1.0)
    crawl_h = run(n_hold)
    print(f"[posture] Switch_Gait(crawl)  target z_c={crawl_z:.3f} -> achieved {crawl_h:.3f} m")

    backend.set_posture(high_z, stiffness=1.0)
    high_h = run(n_hold)
    print(f"[posture] Switch_Gait(high_step) target z_c={high_z:.3f} -> achieved {high_h:.3f} m")

    adj_target = float(np.clip(0.5 * (crawl_z + nominal), 0.10, 0.45))
    backend.set_posture(adj_target, stiffness=1.0)
    adj_h = run(n_hold)
    print(f"[posture] Adjust_Posture target {adj_target:.3f} -> achieved {adj_h:.3f} m")

    backend.set_posture(None)
    released = run(n_settle)
    print(f"[posture] released -> base_height {released:.3f} m (nominal {nominal:.3f})")

    fell = any(np.isnan(x) for x in (nominal, crawl_h, high_h, adj_h, released))
    gap = (high_h - crawl_h) if not fell else 0.0
    ordered = (not fell) and crawl_h < high_h
    distinct = (not fell) and abs(gap) >= args.gap_tol
    crouched = (not fell) and crawl_h < nominal - 0.01
    tracks_adj = (not fell) and abs(adj_h - adj_target) <= args.track_tol
    # One-sided: the trunk must come back UP to ~nominal (not stay stuck in the crouch). Being at or
    # above nominal is fine — the free trot height oscillates with the gait phase (~0.04 m), so a
    # tight two-sided abs tol gives false negatives; this still catches a "stuck crouched" release.
    returns = (not fell) and released >= nominal - args.return_tol

    if not fell and crawl_h > nominal + 0.01 and high_h < nominal:
        print("[posture] DIAGNOSTIC: response INVERTED — flip posture.sign in go2_skeleton.yaml")

    print(
        f"[posture] gap(high-crawl)={gap:+.3f} ordered={ordered} distinct={distinct} "
        f"crouched={crouched} tracks_adj={tracks_adj} returns={returns} fell={fell}"
    )
    ok = ordered and distinct and crouched and tracks_adj and returns and not fell
    print("PASS: posture controller" if ok else "FAIL: posture controller")
    # FLUSH before os._exit — os._exit bypasses stdio flushing, so under a pipe (the sim-gate's
    # capture) the block-buffered prints are lost (gate sees returncode 0 but no PASS).
    sys.stdout.flush()
    # Isaac's app.close() busy-spins; close it on a daemon thread and hard-exit so
    # the process actually terminates (a plain close()+return hangs the gate to its 1800 s timeout).
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    raise SystemExit(main())
