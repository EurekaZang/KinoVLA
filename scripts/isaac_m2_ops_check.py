#!/usr/bin/env python
"""M2 operator GPU gate (spec §8.2) — runs O3/O5/O8/O9/O10 on the Isaac Go2.

Backfills the M2 operators that were surrogate-only (`_isaac_deferred`) so every
benchmark operator now has a real PhysX path on the physically-simulated Go2.

Each destabilising operator (O3 ice, O8 wall, O9 high-centering) gets its own lateral
LANE so a topple/stall on one cannot starve the others; the Go2 is re-teleported to each
lane and driven straight through it. O5 (mass) and O10 (effort) are geometry-free.

Run:  python scripts/isaac_m2_ops_check.py --headless   (GPU machine)
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

import numpy as np

CMD = np.array([0.6, 0.0, 0.0])


def _drive_lane(backend, y: float, n: int, record):
    """Teleport to lane (0, y) and drive +x for n steps, calling record(obs) each step."""
    backend._start_pos = np.array([0.0, y])
    backend._start_heading = 0.0
    backend.reset(42)
    for _ in range(n):
        obs = backend.step(CMD)
        record(obs)
        if obs.fallen:
            break
    return obs


def main() -> int:
    parser = argparse.ArgumentParser(description="M2 Isaac operator gate")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.operators import Collapse, HighCentering, InvisibleCollider, Payload
    from kino_vla.utils.config import load_config
    from kino_vla.utils.geometry import Rect

    cfg = load_config("sim/go2_skeleton.yaml")
    backend = IsaacPolicyBackend(cfg, np.array([0.0, 0.0]), 0.0)
    results: dict[str, bool] = {}

    # Spawn each geometry operator in its own lateral lane (y = 0 / +4 / -4).
    o3 = Collapse(region=Rect(1.6, 0.0, 0.6, 0.8), mu_collapsed=0.10, trigger_dwell_s=0.2)
    o8 = InvisibleCollider(region=Rect(2.0, 4.0, 0.15, 1.5))
    o9 = HighCentering(region=Rect(1.6, -4.0, 0.5, 0.6), residual_support=0.3)
    for op in (o3, o8, o9):
        op.on_reset(backend)

    # --- O3 collapse (lane y=0): intact μ swaps to collapsed on dwell ⇒ feet slip ----------
    mu_intact = backend.friction_at(np.array([1.6, 0.0]))
    slip_o3 = []

    def rec_o3(obs):
        if o3.region.contains(obs.pos):
            slip_o3.append(obs.slip_ratio)

    _drive_lane(backend, 0.0, 200, rec_o3)
    mu_after = backend.friction_at(np.array([1.6, 0.0]))
    results[f"O3 collapse swaps mu ({mu_intact:.2f}->{mu_after:.2f}) on dwell"] = (
        mu_intact > 0.5 and mu_after < 0.2
    )
    max_slip = max(slip_o3) if slip_o3 else 0.0
    results[f"O3 collapsed plate makes feet slip (slip={max_slip:.2f})"] = max_slip > 0.2

    # --- O8 invisible collider (lane y=+4): the Go2 cannot cross the wall -----------------
    max_x = [0.0]

    def rec_o8(obs):
        max_x[0] = max(max_x[0], float(obs.pos[0]))

    _drive_lane(backend, 4.0, 250, rec_o8)
    results[f"O8 wall blocks the Go2 (max x={max_x[0]:.2f} < 1.9 wall edge)"] = max_x[0] < 1.9

    # --- O9 high-centering (lane y=-4): the ridge unloads feet / lifts the chassis --------
    base_o9, support_o9 = [], []

    def rec_o9(obs):
        if o9.region.contains(obs.pos):
            base_o9.append(obs.base_height)
            support_o9.append(obs.support_ratio)

    _drive_lane(backend, -4.0, 200, rec_o9)
    min_support = min(support_o9) if support_o9 else 1.0
    max_base = max(base_o9) if base_o9 else 0.35
    results[f"O9 ridge high-centers (support {min_support:.2f}, base max {max_base:.2f})"] = (
        min_support < 0.9
    )

    # --- O10 effort-decay (clean lane y=+8): a hard effort cut starves the gait -----------
    sp_nom, sp_dec = [], []
    backend._start_pos = np.array([0.0, 8.0])
    backend.reset(42)
    for _ in range(45):  # settle to steady cruise BEFORE recording the nominal speed
        obs = backend.step(CMD)
    for _ in range(20):
        obs = backend.step(CMD)
        sp_nom.append(float(np.linalg.norm(obs.vel_body)))
    backend.set_effort_scale(0.2)  # hard heat-derating cut (effort_limit + saturation_effort)
    fell = False
    for _ in range(40):
        obs = backend.step(CMD)
        sp_dec.append(float(np.linalg.norm(obs.vel_body)))
        if obs.fallen:
            fell = True
            break
    backend.set_effort_scale(1.0)
    v0 = float(np.mean(sp_nom)) if sp_nom else 0.0
    v1 = float(np.mean(sp_dec[-15:])) if sp_dec else 0.0
    # Starved: the Go2 cannot hold the commanded speed (drop) or it collapses outright.
    results[f"O10 effort-decay starves the gait (speed {v0:.2f}->{v1:.2f} m/s, fell={fell})"] = (
        v0 > 0.2 and (v1 < v0 - 0.1 or fell)
    )

    # --- O5 payload: trunk mass increases via the PhysX mass API (readback) ---------------
    mass0 = backend.mass_kg
    Payload(mass_kg=6.0, com_offset_m=np.array([0.0, 0.0])).on_reset(backend)
    results[f"O5 payload mass +6kg ({mass0:.2f}->{backend.mass_kg:.2f} kg)"] = (
        backend.mass_kg > mass0 + 5.0
    )

    print("[isaac_m2_ops_check] results:")
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    passed = all(results.values())
    print("PASS: M2 Isaac operator gate" if passed else "FAIL: M2 Isaac operator gate")

    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if passed else 1)


if __name__ == "__main__":
    main()
