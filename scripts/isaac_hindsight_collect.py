#!/usr/bin/env python
"""Collect REAL-Go2 (Isaac) Hindsight-CoT snapshots for the dataset (spec §10 PHASE 1/2).

The spec builds the data pipeline on Isaac (§1/§8.1), so the dataset's snapshots must come from
the physically-simulated Go2 — real proprioception, real slip/effort/contact, real privileged θ —
NOT the surrogate point-robot. This drives the real Go2 into each operator's failure across
lateral lanes (Isaac is one-episode/process, #21a) with the M4 bang-bang excitation, intercepts
with the high-recall collection monitor, and SAVES the snapshots. The external Oracle (gpt-5.5)
then annotates them offline + the truth-consistency filter runs — the decoupled GPU→CPU pipeline
(CLAUDE.md #23):

    python scripts/isaac_hindsight_collect.py --headless           # GPU: collect + save snapshots
    python scripts/build_hindsight_dataset.py --oracle api \
        --from-snapshots outputs/hindsight_isaac/snapshots.npz \
        --out outputs/hindsight_isaac                              # CPU: real-LLM annotate + filter
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect real-Go2 Hindsight-CoT snapshots")
    parser.add_argument("--out", default="outputs/hindsight_isaac/snapshots.npz")
    parser.add_argument("--random", action="store_true", help="randomized-θ lanes (scale mode)")
    parser.add_argument("--n-lanes", type=int, default=50, help="number of lanes in --random mode")
    parser.add_argument("--seed", type=int, default=0, help="shard seed (--random mode)")
    parser.add_argument("--min-keep", type=int, default=10, help="PASS if >= this many intercepted")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    from kino_vla.data.dataset import save_snapshots
    from kino_vla.data.isaac_rollout import collect_lane, random_lanes
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import REPO_ROOT, load_config

    cfg = load_config("data/hindsight.yaml")
    monitor_cfg = load_config(str(cfg.drive.monitor))
    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)

    if args.random:
        lanes = random_lanes(cfg, args.seed, args.n_lanes)
    else:
        lanes = list(cfg.isaac_collect.dataset_lanes)
    items: list = []
    base_seed = 200 + args.seed * 100000
    for i, lane in enumerate(lanes):
        snap, op_theta, fell = collect_lane(backend, cfg, monitor_cfg, lane, seed=base_seed + i)
        ok = snap is not None
        th = (
            f"mu={snap.privileged_theta['mu']:.2f} pay={snap.privileged_theta['payload_kg']:.1f} "
            f"eff={snap.privileged_theta['effort_scale']:.2f} ch={snap.monitor_channel}"
            if ok
            else "—"
        )
        print(
            f"[collect] {lane['op']:22s} y={float(lane['y']):+5.1f} intercept={ok} {th} fell={fell}"
        )
        if ok:
            items.append((snap, op_theta))

    out = REPO_ROOT / args.out
    save_snapshots(out, items)
    print(f"[collect] saved {len(items)}/{len(lanes)} real-Go2 snapshots → {out}")
    ok = len(items) >= int(args.min_keep)
    print("PASS: collected real-Go2 Hindsight snapshots" if ok else "FAIL: too few snapshots")

    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
