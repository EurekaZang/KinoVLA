#!/usr/bin/env python
"""Probe the O5/O10 collection regime (#31 fix): are the real-Go2 snapshots identifiable?

Drives the real ``collect_lane`` for several O10 (effort-decay) and O5 (overload) lanes and prints
the EXACT proprio window the Oracle will read (slip/effort/base_height/tracking traces + means) +
privileged θ + fell. We want, with NO weakened filter and NO θ leak:

  O10 => effort_trace steps up to a SUSTAINED saturation, slip LOW, base_height NORMAL.
  O5  => base_height_trace steps DOWN (a sag), slip LOW, effort un-saturated.

so a properly-conditioned Oracle attributes effort_decay / overload (not region_collapse). Use it
to pick a decay floor + payload mass + sustained speed that produce a clean window without toppling.

Run:  python scripts/isaac_embodiment_probe.py --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="O5/O10 embodiment-signature probe")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    from kino_vla.data.isaac_rollout import collect_lane
    from kino_vla.data.oracle import _proprio_summary
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import load_config

    cfg = load_config("data/hindsight.yaml")
    monitor_cfg = load_config(str(cfg.drive.monitor))
    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)

    lanes = [
        {"op": "O10_effort_decay", "y": 0.0, "appearance": "solid_ground", "floor": 0.18},
        {"op": "O10_effort_decay", "y": 3.0, "appearance": "solid_ground", "floor": 0.15},
        {"op": "O10_effort_decay", "y": 6.0, "appearance": "solid_ground", "floor": 0.13},
        # O5: sweep heavier + an off-centre CoM to cross the observability floor (find a sag/tilt
        # without an instant topple). clear_payload makes each lane's mass absolute.
        {"op": "O5_payload", "y": -3.0, "appearance": "solid_ground", "mass": 9.0},
        {"op": "O5_payload", "y": -6.0, "appearance": "solid_ground", "mass": 12.0},
        {"op": "O5_payload", "y": -9.0, "appearance": "solid_ground", "mass": 15.0},
        {"op": "O5_payload", "y": -12.0, "appearance": "solid_ground", "mass": 18.0},
    ]
    for i, lane in enumerate(lanes):
        snap, _theta, fell = collect_lane(backend, cfg, monitor_cfg, lane, seed=300 + i)
        param = lane.get("floor", lane.get("mass"))
        if snap is None:
            print(f"{lane['op']:18s} param={param} -> NO SNAPSHOT (fell={fell})")
            continue
        s = _proprio_summary(snap)
        th = snap.privileged_theta
        print(
            f"{lane['op']:18s} param={param} fell={fell} ch={snap.monitor_channel} "
            f"θ(pay={th['payload_kg']:.1f} eff={th['effort_scale']:.2f})"
        )
        print(f"    slip   {s['slip_trace']} (mean {s['slip_mean']})")
        print(f"    effort {s['effort_trace']} (mean {s['effort_mean']})")
        print(f"    height {s['base_height_trace']} (mean {s['base_height_mean']})")
        print(f"    track  {s['tracking_trace']} (mean {s['tracking_mean']}) tilt={s['tilt_peak']}")

    print("PASS: embodiment probe complete")
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0)


if __name__ == "__main__":
    main()
