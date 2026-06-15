#!/usr/bin/env python
"""M5 operator GPU gate (spec §8.2 Axis I/II/III) — runs O2/O4/O7 on the Isaac Go2.

One Isaac process, one physically-simulated Go2 episode. Verifies the M5 operators'
PhysX paths on GPU (not the surrogate):

  - O2 Compliance-Field: a tangential-resistance external wrench on the trunk measurably
    slows the Go2 inside the region vs. its pre-region cruise (θ-application, QA 5.2a).
  - O4 Tether: shares the exact wrench code; a low break-force tether snaps after loading
    (the `broken` flag flips), so resistance ceases — verified via the backend state.
  - O7 Visual-Physics Remap: its physics is a low-μ PhysX plate; μ reads back at the set
    value and the real feet slip on it (measured slip rises).

Run:  python scripts/isaac_m5_check.py --headless   (GPU machine)
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="M5 Isaac operator gate")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.operators import ComplianceField, Tether, VisualPhysicsRemap
    from kino_vla.utils.config import load_config
    from kino_vla.utils.geometry import Rect

    cfg = load_config("sim/go2_skeleton.yaml")
    backend = IsaacPolicyBackend(cfg, np.array([0.0, 0.0]), 0.0)

    # Non-overlapping regions on the forward path, sized so the Go2 still traverses all
    # three: O2 resistance first (moderate, μ nominal), a low-break O4 tether next, then
    # the O7 low-μ deceptive patch.
    o2 = ComplianceField(Rect(2.6, 0.0, 0.7, 1.2), k_c=15.0, c_c=8.0, d_sink=0.05)
    o4 = Tether(Rect(4.2, 0.0, 0.4, 1.2), k=150.0, d=6.0, l0=0.0, f_break=22.0)
    o7 = VisualPhysicsRemap(Rect(5.6, 0.0, 0.6, 1.2), mu_s=0.10, mu_d=0.08, depth_bias_m=0.6)
    for op in (o2, o4, o7):
        op.on_reset(backend)

    backend.reset(42)
    speed_pre, speed_in_o2, slip_o7 = [], [], []
    o4_state = backend._resistance[1]  # the Tether region state
    for _ in range(700):
        obs = backend.step(np.array([0.7, 0.0, 0.0]))
        x = float(obs.pos[0])
        sp = float(np.linalg.norm(obs.vel_body))
        if 1.2 <= x <= 1.8:
            speed_pre.append(sp)
        if 2.3 <= x <= 3.0:
            speed_in_o2.append(sp)
        if o7.region.contains(obs.pos):
            slip_o7.append(obs.slip_ratio)
        if obs.fallen or x > 6.0:
            break
    print(f"[isaac_m5_check] final x={float(obs.pos[0]):.2f} fallen={obs.fallen}")

    mu_o7 = backend.friction_at(np.array([5.5, 0.0]))
    pre = float(np.mean(speed_pre)) if speed_pre else 0.0
    in_o2 = float(np.mean(speed_in_o2)) if speed_in_o2 else 0.0
    slip = float(np.max(slip_o7)) if slip_o7 else 0.0

    checks = {
        f"O2 resistance slows the Go2 (pre {pre:.3f} -> in {in_o2:.3f} m/s)": (
            pre > 0.05 and in_o2 < pre - 0.05
        ),
        f"O4 tether snapped under load (broken={o4_state['broken']})": o4_state["broken"],
        f"O7 mu readback == 0.08 (got {mu_o7:.3f})": abs(mu_o7 - 0.08) <= 0.02,
        f"O7 feet slip on the deceptive patch (slip={slip:.2f})": slip > 0.2,
    }
    print("[isaac_m5_check] results:")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    passed = all(checks.values())
    print("PASS: M5 Isaac operator gate" if passed else "FAIL: M5 Isaac operator gate")

    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if passed else 1)


if __name__ == "__main__":
    main()
