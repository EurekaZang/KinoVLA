#!/usr/bin/env python
"""Collect A-CLASS NEGATIVE lanes for the A/B-aware Kino-Monitor retrain (E4 caliper fix).

The deployed monitor false-fires on A-class perturbations because its negative set is only
clean + maneuver (no operator) — a mild-but-real perturbation is out-of-distribution on the
negative side, so the detector saturates at hazard=1 (outputs/eval/e4/monitor_probe.json). The fix
(A实验/E4.md §6.1): add A-class perturbation lanes — a real operator at a GENTLE θ the low-level
policy absorbs — labelled NORMAL (hazard=0, cls=0), so the detector learns the observable-signal
MAGNITUDE boundary between "mild perturbation low-level absorbs" (quiet) and "real hazard" (fire).

RED LINE (held): the label "A-class ⇒ no-fire" is grounded in an OBSERVABLE OUTCOME — the base
policy crosses the patch WITHOUT FALLING at that gentle θ (nominal cruise, this same pass records
``fell``; fallen lanes are DROPPED because a topple means low-level did NOT absorb it). The monitor
consumes only ``monitor_features`` (observable proprio) + a fixed threshold; θ is NEVER a monitor
input. The A-class θ ranges below sit strictly milder than the B-class ``_THETA_RANGES`` (physics/
spec A/B thresholds, configs/data/hindsight.yaml) — not tuned to any θ*.

Reuses hazard_lab.build_scenario / install / drive_cmd exactly (same lane geometry + deployment
cruise as the positive collection), so the windows are drawn from the same distribution — only the
θ magnitude and the (all-zero) label differ.

Run:  python scripts/isaac_monitor_aclass_collect.py --headless --seeds 6 --tag aclass_train
      python scripts/isaac_monitor_aclass_collect.py --headless --seeds 3 --seed-base 700 \
          --tag aclass_test
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

import numpy as np

LANE_SPACING_M = 4.0

# A-class θ ranges: strictly MILDER than the B-class hazard_lab._THETA_RANGES (a notch on the benign
# side of the spec A/B thresholds). O1/O7 = high friction (mild ice), O2 = shallow soft mud, O5 =
# light payload, O6 = sub-threshold push, O10 = mild effort derate, O11 = small IMU tilt. O3/O8/O9
# are excluded — a collapse/wall/beaching has no "mild that low-level absorbs" regime (genuine
# hazards). Grounding is empirical: any lane that FALLS at these θ is dropped (not A-class).
_ACLASS_THETA: dict[str, dict[str, tuple[float, float]]] = {
    "O1": {"mu": (0.30, 0.45)},
    "O2": {"k": (4.0, 8.0), "c": (2.0, 4.0), "d_sink": (0.02, 0.04)},
    "O5": {"mass": (2.0, 4.0)},
    "O6": {"impulse": (4.0, 9.0)},
    "O7": {"mu": (0.30, 0.45)},
    "O10": {"floor": (0.75, 0.95)},
    "O11": {"tilt_bias": (0.05, 0.15)},
}
_ACLASS_OPS: tuple[str, ...] = tuple(_ACLASS_THETA.keys())


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect A-class NEGATIVE lanes for the monitor")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--steps", type=int, default=460)  # ~9.2 s at 50 Hz (match positive collect)
    ap.add_argument("--ops", type=str, default="all")
    ap.add_argument("--tag", type=str, default="aclass_train")
    ap.add_argument("--out", type=str, default="outputs/monitor_learned")
    args = ap.parse_args()
    app = AppLauncher(args).app

    from kino_vla.monitor import hazard_lab as HL
    from kino_vla.monitor.hazard_lab import MON_FEATURE_SCHEMA, monitor_features
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import load_config

    cfg = load_config("sim/go2_skeleton.yaml")
    backend = IsaacPolicyBackend(cfg, np.array([0.0, 0.0]), 0.0)
    dt = backend.dt
    ops = list(_ACLASS_OPS) if args.ops == "all" else [s.strip() for s in args.ops.split(",")]
    onset_step = int(round(HL.ONSET_T / dt))
    print(f"[aclass] tag={args.tag} dt={dt:.4f} ops={ops} seeds={args.seeds} "
          f"(base {args.seed_base}) steps={args.steps}", flush=True)

    def sample_aclass(op_id: str, rng: np.random.Generator) -> dict[str, float]:
        return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in _ACLASS_THETA[op_id].items()}

    feats_all, haz_all, cls_all, lane_all, t_all = [], [], [], [], []
    lane_i, kept, dropped = 0, 0, 0
    total = len(ops) * args.seeds
    for op_id in ops:
        for s in range(args.seeds):
            seed = args.seed_base + s
            y = LANE_SPACING_M * (lane_i + 1)
            lane_i += 1
            rng = np.random.default_rng(seed * 149 + HL.ALL_IDS.index(op_id) * 7919 + 5)
            theta = sample_aclass(op_id, rng)
            sc = HL.build_scenario(op_id, y, theta)

            backend._start_pos = np.array([0.0, y])
            backend._start_heading = 0.0
            obs = backend.reset(seed)
            backend.clear_payload()
            backend.set_effort_scale(1.0)
            HL.install(sc, backend)

            payload_added = False
            lane_feats, lane_t = [], []
            fell = False
            k = 0
            for k in range(args.steps):
                obs_m = sc.operator.transform_obs(obs) if sc.onset_kind == "obsbias" else obs
                lane_feats.append(monitor_features(obs_m).astype(np.float32))
                lane_t.append(float(obs.t))
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
            final_x = float(obs.pos[0])
            crossed = final_x > 4.0  # traversed the patch (x∈[2,4]) at this gentle θ
            th_s = {kk: round(v, 3) for kk, v in theta.items()}
            # A-class ⇒ low-level absorbs it: keep ONLY non-fallen lanes that crossed the patch.
            if fell or not crossed:
                dropped += 1
                print(f"[{lane_i}/{total}] {op_id:>4} seed{seed}: DROP "
                      f"(fell={fell} final_x={final_x:.2f}) theta={th_s}", flush=True)
                continue
            kept += 1
            for f, tt in zip(lane_feats, lane_t, strict=True):
                feats_all.append(f)
                haz_all.append(0)   # A-class = NORMAL (no-fire): low-level absorbs it
                cls_all.append(0)   # attribution class 0 = normal
                lane_all.append(lane_i)
                t_all.append(tt)
            print(f"[{lane_i}/{total}] {op_id:>4} seed{seed}: KEEP "
                  f"steps={len(lane_feats)} final_x={final_x:.2f} theta={th_s}", flush=True)

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
        op_ids=np.asarray(ops),
    )
    print(f"[aclass] wrote {out}  rows={len(haz_all)}  kept_lanes={kept} dropped={dropped}",
          flush=True)
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0)


if __name__ == "__main__":
    main()
