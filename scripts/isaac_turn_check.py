#!/usr/bin/env python
"""IN-PLACE TURN gate — the low-level policy executes a commanded yaw velocity (user directive).

The Turn primitive is a PURE in-place rotation: the planner commands a yaw velocity (vx = vy = 0)
and the dog must rotate in place and physically track that yaw command. This gate drives the
physically-simulated Go2 with a pure yaw command and verifies, on the REAL policy:

  PASS criteria (per commanded yaw rate, both signs):
    1. the achieved steady-state yaw rate tracks the command within ``rate_tol`` and has the
       correct sign (the policy actually turns at the commanded rate);
    2. the body stays roughly in place — net translation drift over the window < ``drift_tol``
       (a Turn re-aims the camera, it must not walk off);
    3. NO fall throughout.

It tests wz = ±1.0 rad/s (the planner's turn_rate_radps) and reports wz = ±1.5 (the backend clamp,
max_yaw_rate_radps) as a diagnostic of the trained yaw-command range — if 1.0 passes but 1.5 tracks
poorly, the yaw-emphasis retrain (configs/locomotion/go2_flat_ppo.yaml command.ang_vel_z) is what
widens it. Mirrors scripts/isaac_posture_check.py (real Go2, headless, os._exit gate convention).

Run:  python scripts/isaac_turn_check.py --headless
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="In-place yaw-turn policy gate (real Go2)")
    parser.add_argument("--settle-s", type=float, default=1.5, help="settle window before a turn")
    parser.add_argument("--turn-s", type=float, default=3.0, help="per-turn measurement window")
    parser.add_argument(
        "--rate-tol",
        type=float,
        default=0.3,
        help="max |achieved steady yaw rate − commanded| [rad/s] for the PASS rates (±1.0)",
    )
    parser.add_argument(
        "--drift-tol",
        type=float,
        default=0.7,
        help="max net XY translation [m] over a turn window (in-place ⇒ small)",
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import load_config

    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)
    dt = backend.dt
    n_settle = int(args.settle_s / dt)
    n_turn = int(args.turn_s / dt)

    def settle() -> None:
        """Walk-free settle at a zero command (no posture hold) so each turn starts from rest."""
        backend.set_posture(None)
        for _ in range(n_settle):
            backend.step(np.zeros(3))

    def turn(wz: float) -> dict:
        """Command a pure yaw velocity [0,0,wz]; return achieved steady yaw rate, drift, fall, and
        the net heading change (unwrapped)."""
        start = backend.step(np.array([0.0, 0.0, wz]))  # one step to read the start pose
        p0 = start.pos.copy()
        h_prev = float(start.heading)
        unwrapped = 0.0
        rates: list[float] = []
        max_drift = 0.0
        fell = False
        for k in range(n_turn):
            obs = backend.step(np.array([0.0, 0.0, wz]))
            if obs.fallen:
                fell = True
                break
            dh = math.atan2(
                math.sin(obs.heading - h_prev), math.cos(obs.heading - h_prev)
            )  # wrapped step delta
            unwrapped += dh
            h_prev = float(obs.heading)
            max_drift = max(max_drift, float(np.linalg.norm(obs.pos - p0)))
            if k >= n_turn // 2:  # steady-state half (skip the spin-up transient)
                rates.append(float(obs.yaw_rate))
        return {
            "wz": wz,
            "achieved_rate": float(np.mean(rates)) if rates else float("nan"),
            "heading_change": unwrapped,
            "drift_m": max_drift,
            "fell": fell,
        }

    backend.reset(0)

    results = {}
    for wz in (1.0, -1.0, 1.5, -1.5):
        settle()
        r = turn(wz)
        results[wz] = r
        print(
            f"[turn] cmd wz={wz:+.2f} rad/s -> achieved {r['achieved_rate']:+.2f} rad/s  "
            f"Δheading={math.degrees(r['heading_change']):+6.1f}°  drift={r['drift_m']:.2f} m  "
            f"fell={r['fell']}"
        )

    # PASS bar: the ±1.0 rad/s turns track within tol, correct sign, in place, no fall.
    def good(wz: float) -> bool:
        r = results[wz]
        if r["fell"] or math.isnan(r["achieved_rate"]):
            return False
        tracks = abs(r["achieved_rate"] - wz) <= args.rate_tol
        signed = (r["achieved_rate"] > 0) == (wz > 0)
        in_place = r["drift_m"] <= args.drift_tol
        return tracks and signed and in_place

    ok = good(1.0) and good(-1.0)
    # Diagnostic: does the policy also track the backend-max ±1.5 (the trained-range question)?
    wide_ok = all(
        (not results[w]["fell"])
        and abs(results[w]["achieved_rate"] - w) <= args.rate_tol
        and results[w]["drift_m"] <= args.drift_tol
        for w in (1.5, -1.5)
    )
    print(f"[turn] ±1.0 tracked-in-place = {ok}   ±1.5 (diagnostic, wider range) = {wide_ok}")
    print("PASS: in-place turn policy" if ok else "FAIL: in-place turn policy")

    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    raise SystemExit(main())
