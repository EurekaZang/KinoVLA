#!/usr/bin/env python
"""Online ~50 Hz eval of the learning-based Kino-Monitor on the REAL Go2 (Isaac Lab).

Runs the trained :class:`LearnedMonitor` exactly as it would in deployment — pushing each control
step's measured proprioception into the rolling window, firing on the hazard probability — across
all 11 operator scenarios + clean/maneuver negatives. Records, per lane: whether it fired while the
hazard was active (TRUE POSITIVE) and when (latency from the ground-truth hazard onset), the
attributed cause (the event channel), and on clean/maneuver lanes whether it fired at all (FALSE
POSITIVE). This is the deployment metric (online, debounced, cooled-down) — distinct from the
trainer's per-window TPR/FPR.

Run:  python scripts/isaac_monitor_learned_eval.py --headless \
          --model outputs/monitor_learned/monitor --threshold 0.5 --seeds 4 --seed-base 900
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np

LANE_SPACING_M = 4.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Online real-Go2 eval of the learned monitor")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--model", default="outputs/monitor_learned/monitor")
    parser.add_argument("--threshold", type=float, default=-1.0)  # <0 ⇒ use the report's chosen
    parser.add_argument("--debounce", type=int, default=5)
    parser.add_argument("--ema-alpha", type=float, default=0.4)
    parser.add_argument("--arm-s", type=float, default=2.5)
    parser.add_argument("--cooldown-s", type=float, default=6.0)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=900)
    parser.add_argument("--steps", type=int, default=460)
    parser.add_argument("--ops", type=str, default="all")
    parser.add_argument("--out", type=str, default="outputs/monitor_learned")
    args = parser.parse_args()
    app = AppLauncher(args).app

    from kino_vla.monitor import hazard_lab as HL
    from kino_vla.monitor.learned_monitor import LearnedMonitor, LearnedMonitorModel
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import load_config

    threshold = args.threshold
    if threshold < 0:
        rep = json.loads(Path(args.model).with_suffix(".report.json").read_text())
        threshold = float(rep["chosen_threshold"])
    model = LearnedMonitorModel.load(args.model, device="cuda")

    cfg = load_config("sim/go2_skeleton.yaml")
    backend = IsaacPolicyBackend(cfg, np.array([0.0, 0.0]), 0.0)
    dt = backend.dt
    op_ids = list(HL.ALL_IDS) if args.ops == "all" else [s.strip() for s in args.ops.split(",")]
    onset_step = int(round(HL.ONSET_T / dt))
    print(
        f"[learned-eval] threshold={threshold:.3f} dt={dt:.4f} ops={op_ids} seeds={args.seeds}",
        flush=True,
    )

    rows = []
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
            monitor = LearnedMonitor(
                model, dt=dt, threshold=threshold, debounce_steps=args.debounce,
                arm_s=args.arm_s, cooldown_s=args.cooldown_s, ema_alpha=args.ema_alpha,
            )

            payload_added = False
            t_in_region = 0.0
            first_hazard_t = None
            first_fire = None
            n_fire_normal = 0  # fires while ground-truth normal (premature / spurious)
            fell = False
            prob_tr, haz_tr, t_tr, attr_tr = [], [], [], []  # per-step traces for the offline sweep
            for k in range(args.steps):
                obs_m = sc.operator.transform_obs(obs) if sc.onset_kind == "obsbias" else obs
                if sc.rect is not None and sc.rect.contains(obs.pos):
                    t_in_region += dt
                haz = HL.hazard_label(sc, obs.pos, float(obs.t), t_in_region)
                if haz == 1 and first_hazard_t is None:
                    first_hazard_t = float(obs.t)
                ev = monitor.step(obs_m)
                prob_tr.append(round(monitor.raw_prob, 4))  # raw per-window hazard prob
                haz_tr.append(int(haz))
                t_tr.append(round(float(obs.t), 3))
                attr_tr.append(int(np.argmax(monitor.attr_probs)))
                if ev is not None:
                    if first_fire is None:
                        first_fire = {"t": ev.t, "channel": ev.channel, "value": ev.value}
                    if haz == 0:
                        n_fire_normal += 1

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

            gt_positive = op_id in HL.OP_IDS
            fired = first_fire is not None
            latency = None
            if fired and first_hazard_t is not None:
                latency = round(first_fire["t"] - first_hazard_t, 3)
            attr_ok = bool(fired and gt_positive and first_fire["channel"] == op_id)
            rows.append({
                "op": op_id, "seed": seed, "y": y, "gt_positive": gt_positive,
                "fired": fired, "first_fire": first_fire, "latency_s": latency,
                "attr_ok": attr_ok, "first_hazard_t": first_hazard_t,
                "n_fire_normal": n_fire_normal, "fell": fell,
                "prob_trace": prob_tr, "haz_trace": haz_tr, "t_trace": t_tr, "attr_trace": attr_tr,
            })
            tag = (
                f"FIRE {first_fire['channel']}@{first_fire['t']:.1f}s p={first_fire['value']:.2f} "
                f"lat={latency} attr_ok={attr_ok}"
                if fired else "NO-FIRE"
            )
            print(f"[{lane_i}/{total}] {op_id:>8} seed{seed}: {tag} fell={fell}", flush=True)

    # Aggregate: per-op detection (gt+), FPR (gt-), attribution, latency.
    summary = {}
    for op in op_ids:
        rs = [r for r in rows if r["op"] == op]
        n = len(rs)
        fired = [r for r in rs if r["fired"]]
        if op in HL.OP_IDS:
            summary[op] = {
                "metric": "detection_rate", "n": n,
                "rate": round(len(fired) / n, 3),
                "attr_correct_rate": round(sum(r["attr_ok"] for r in rs) / n, 3),
                "median_latency_s": round(float(np.median([r["latency_s"] for r in fired
                                          if r["latency_s"] is not None])), 3) if fired else None,
                "fell_rate": round(sum(r["fell"] for r in rs) / n, 3),
            }
        else:
            summary[op] = {
                "metric": "false_positive_rate", "n": n,
                "rate": round(len(fired) / n, 3),
            }
    pos_ops = [o for o in op_ids if o in HL.OP_IDS]
    neg_ops = [o for o in op_ids if o not in HL.OP_IDS]
    tpr_min = min((summary[o]["rate"] for o in pos_ops), default=None)
    fpr_max = max((summary[o]["rate"] for o in neg_ops), default=None)
    print("\n[learned-eval] per-operator:")
    for op in op_ids:
        s = summary[op]
        extra = (
            f" attr_ok={s.get('attr_correct_rate')} lat={s.get('median_latency_s')}"
            f" fell={s.get('fell_rate')}"
            if op in HL.OP_IDS
            else ""
        )
        print(f"    {op:>8} {s['metric']}={s['rate']}{extra}", flush=True)
    print(
        f"[learned-eval] TPR_min(over 11 ops)={tpr_min}  FPR_max(clean/maneuver)={fpr_max}",
        flush=True,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model, "threshold": threshold, "tpr_min": tpr_min, "fpr_max": fpr_max,
        "summary": summary, "rows": rows,
    }
    (out_dir / "learned_eval.json").write_text(json.dumps(payload, indent=2))
    print(f"[learned-eval] wrote {out_dir}/learned_eval.json", flush=True)

    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0)


if __name__ == "__main__":
    main()
