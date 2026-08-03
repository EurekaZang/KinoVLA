#!/usr/bin/env python
"""Collect REAL-Go2 (Isaac Lab) labelled proprioception for the learning-based Kino-Monitor.

Drives the trained policy through every operator scenario (kino_vla.monitor.hazard_lab) at the
deployment operating point (steady cruise; ``maneuver`` lanes add turns/accel on clean ground) and
logs the per-step measured-proprioception feature vector + the ground-truth hazard label + the
operator class. The trainer (scripts/train_learned_monitor.py) slices these into 500 ms windows.

This is the deployment-distribution training data the learned monitor needs (windows captured at
cruise, NOT bang-bang — closing the #34c train/deploy proprio shift at the monitor level). One lane
per (op × seed) in its own y-lane (Isaac is one-episode/process, #21a).

Run:  python scripts/isaac_monitor_data_collect.py --headless --seeds 10 --tag train
      python scripts/isaac_monitor_data_collect.py --headless --seeds 4 --seed-base 500 --tag test
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

import numpy as np

LANE_SPACING_M = 4.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect real-Go2 labelled data for the monitor")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--steps", type=int, default=460)  # ~9.2 s at 50 Hz
    parser.add_argument("--ops", type=str, default="all")
    parser.add_argument("--tag", type=str, default="train")
    parser.add_argument("--out", type=str, default="outputs/monitor_learned")
    args = parser.parse_args()
    app = AppLauncher(args).app

    from kino_vla.monitor import hazard_lab as HL
    from kino_vla.monitor.hazard_lab import MON_FEATURE_SCHEMA, monitor_features
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import load_config

    cfg = load_config("sim/go2_skeleton.yaml")
    backend = IsaacPolicyBackend(cfg, np.array([0.0, 0.0]), 0.0)
    dt = backend.dt
    op_ids = list(HL.ALL_IDS) if args.ops == "all" else [s.strip() for s in args.ops.split(",")]
    onset_step = int(round(HL.ONSET_T / dt))
    print(
        f"[collect] tag={args.tag} dt={dt:.4f} ops={op_ids} seeds={args.seeds} "
        f"(base {args.seed_base}) steps={args.steps} features={MON_FEATURE_SCHEMA}",
        flush=True,
    )

    feats_all, haz_all, cls_all, lane_all, t_all = [], [], [], [], []
    lane_i = 0
    total = len(op_ids) * args.seeds
    for op_id in op_ids:
        for s in range(args.seeds):
            seed = args.seed_base + s
            y = LANE_SPACING_M * (lane_i + 1)
            lane_i += 1
            rng = np.random.default_rng(seed * 131 + HL.ALL_IDS.index(op_id) * 9973)
            theta = HL.sample_theta(op_id, rng)
            sc = HL.build_scenario(op_id, y, theta)

            backend._start_pos = np.array([0.0, y])
            backend._start_heading = 0.0
            obs = backend.reset(seed)
            backend.clear_payload()
            backend.set_effort_scale(1.0)
            HL.install(sc, backend)

            payload_added = False
            t_in_region = 0.0
            n_pos = 0
            fell = False
            for k in range(args.steps):
                obs_m = sc.operator.transform_obs(obs) if sc.onset_kind == "obsbias" else obs
                if sc.rect is not None and sc.rect.contains(obs.pos):
                    t_in_region += dt
                haz = HL.hazard_label(sc, obs.pos, float(obs.t), t_in_region)
                feats_all.append(monitor_features(obs_m).astype(np.float32))
                haz_all.append(haz)
                cls_all.append(HL.CLASS_OF[op_id] if haz == 1 else 0)
                lane_all.append(lane_i)
                t_all.append(float(obs.t))
                n_pos += haz

                if sc.onset_kind == "payload" and k == onset_step and not payload_added:
                    backend.add_payload(float(sc.theta["mass"]), np.zeros(2))
                    payload_added = True
                if sc.operator is not None and sc.onset_kind in ("effort", "push"):
                    sc.operator.on_step(backend, float(obs.t))

                obs = backend.step(HL.drive_cmd(sc, k, dt, rng))
                if obs.fallen:
                    fell = True
                    break
            backend.clear_payload()
            backend.set_effort_scale(1.0)
            th_s = {kk: round(v, 3) for kk, v in theta.items()}
            print(
                f"[{lane_i}/{total}] {op_id:>8} seed{seed} y={y:.0f}: "
                f"steps={k + 1} pos={n_pos} fell={fell} theta={th_s}",
                flush=True,
            )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"windows_{args.tag}.npz"
    np.savez_compressed(
        out,
        feats=np.asarray(feats_all, dtype=np.float32),
        hazard=np.asarray(haz_all, dtype=np.int8),
        cls=np.asarray(cls_all, dtype=np.int8),
        lane=np.asarray(lane_all, dtype=np.int32),
        t=np.asarray(t_all, dtype=np.float32),
        feature_schema=np.asarray(list(MON_FEATURE_SCHEMA)),
        dt=np.float32(dt),
        op_ids=np.asarray(op_ids),
    )
    n = len(haz_all)
    n_pos = int(np.sum(haz_all))
    print(
        f"[collect] wrote {out}  rows={n}  pos={n_pos} ({n_pos / max(1, n):.2%})  lanes={lane_i}",
        flush=True,
    )

    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0)


if __name__ == "__main__":
    main()
