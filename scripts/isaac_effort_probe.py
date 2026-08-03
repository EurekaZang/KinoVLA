#!/usr/bin/env python
"""Diagnostic: does the real-Go2 effort_ratio channel ever bind for O5/O10? (CLAUDE.md #15/#31).

The M6 dataset reads effort_ratio≈0 for O5 (payload) and O10 (effort-decay). This drives each
in its own lane with the bang-bang excitation and logs the effort_ratio trajectory, to decide
whether effort=0 is a calibration/timing issue (fixable) or the genuine observability floor.
Run:  python scripts/isaac_effort_probe.py --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-Go2 effort observability probe")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import load_config

    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)
    dt = backend.dt

    def bangbang(k: int, hi: float = 1.4, lo: float = 0.1) -> float:
        return hi if (((k * dt) / 1.0) % 1.0) < 0.5 else lo

    def drive(label: str, y: float, seed: int, *, floor=None, payload=None, com=0.0) -> None:
        backend._start_pos = np.array([0.0, y])
        backend._start_heading = 0.0
        obs = backend.reset(seed)
        if floor is not None:
            backend.set_effort_scale(1.0)
        for _ in range(25):
            obs = backend.step(np.array([bangbang(0), 0.0, 0.0]))
        if floor is not None:
            backend.set_effort_scale(float(floor))
        if payload is not None:
            backend.add_payload(float(payload), np.array([float(com), 0.0]))
        eff = []
        for k in range(250):
            if obs.fallen:
                break
            eff.append(float(obs.effort_ratio))
            obs = backend.step(np.array([bangbang(k), 0.0, 0.0]))
        e = np.asarray(eff)
        binds = float((e > 0.05).mean()) if e.size else 0.0
        print(
            f"[effort] {label:28s} steps={e.size:3d} fell={obs.fallen} "
            f"eff_max={e.max(initial=0):.2f} eff_mean={e.mean() if e.size else 0:.2f} "
            f"frac_binding={binds:.2f}"
        )
        if floor is not None:
            backend.set_effort_scale(1.0)

    print("== O10 effort-decay: sweep decay floor (lower floor => smaller cap => binds easier) ==")
    for i, fl in enumerate([0.30, 0.20, 0.15, 0.10, 0.07]):
        drive(f"O10 floor={fl}", y=3.0 * i, seed=10 + i, floor=fl)
    print("== O5 payload: sweep mass + CoM offset (off-centre loads specific legs harder) ==")
    for i, (m, c) in enumerate([(6.0, 0.0), (10.0, 0.0), (8.0, 0.15)]):
        drive(f"O5 mass={m} com={c}", y=-3.0 * (i + 1), seed=50 + i, payload=m, com=c)

    print("PASS: effort probe complete")
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0)


if __name__ == "__main__":
    main()
