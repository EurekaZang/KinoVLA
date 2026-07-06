#!/usr/bin/env python
"""Offline operating-point sweep for the online learning-based monitor.

Replays the EMA + debounce + cooldown firing logic of :class:`LearnedMonitor` over the per-step
hazard-probability traces logged by ``scripts/isaac_monitor_learned_eval.py`` (the ``learned_eval``
json), for a grid of (threshold, ema_alpha, debounce). One real-Go2 eval run thus yields the online
ROC — per-operator TPR + clean/maneuver FPR at every operating point — without re-running Isaac.
Recommends the point with the largest threshold margin at TPR_min == 1.0 and FPR == 0.

Run:  python scripts/sweep_online_monitor.py --eval outputs/monitor_learned/learned_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kino_vla.monitor.hazard_lab import OP_IDS


def replay_fire(prob, t, attr, threshold, ema_alpha, debounce, arm_s, cooldown_s):
    """Replay LearnedMonitor firing (EMA + det/attr-agreement gate + debounce) over a trace.

    ``attr`` is the per-step attribution argmax; firing requires the EMA prob over threshold AND the
    attribution != normal (class 0) — the same agreement gate the online monitor uses.
    """
    ema = 0.0
    above = 0
    cooldown_until = -1.0
    for i, (p, ti, a) in enumerate(zip(prob, t, attr, strict=True)):
        ema += ema_alpha * (p - ema)
        if ti < arm_s or ti < cooldown_until:
            above = 0
            continue
        above = above + 1 if (ema >= threshold and int(a) != 0) else 0
        if above >= debounce:
            return i
    return None


def first_hazard_idx(haz):
    nz = np.nonzero(np.asarray(haz))[0]
    return int(nz[0]) if nz.size else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Online monitor operating-point sweep")
    ap.add_argument("--eval", default="outputs/monitor_learned/learned_eval.json")
    ap.add_argument("--arm-s", type=float, default=2.5)
    ap.add_argument("--cooldown-s", type=float, default=6.0)
    ap.add_argument("--out", default="outputs/monitor_learned/online_sweep.json")
    args = ap.parse_args()

    rows = json.loads(Path(args.eval).read_text())["rows"]
    pos = [r for r in rows if r["gt_positive"]]
    neg = [r for r in rows if not r["gt_positive"]]
    n_neg = len(neg)
    print(f"[sweep] {len(pos)} positive lanes, {n_neg} negative lanes from {args.eval}", flush=True)

    grid_thr = [round(x, 2) for x in np.arange(0.2, 0.96, 0.1)]
    grid_alpha = [0.3, 0.5]
    grid_deb = [3, 5, 8]
    results = []
    for alpha in grid_alpha:
        for deb in grid_deb:
            for thr in grid_thr:
                # per-op detection
                tpr = {}
                for op in OP_IDS:
                    lanes = [r for r in pos if r["op"] == op]
                    if not lanes:
                        continue
                    det = 0
                    for r in lanes:
                        fi = replay_fire(
                            r["prob_trace"], r["t_trace"], r["attr_trace"], thr, alpha, deb,
                            args.arm_s, args.cooldown_s,
                        )
                        hi = first_hazard_idx(r["haz_trace"])
                        # detected = fires, and (if a hazard onset exists) not LONG before it
                        ok = fi is not None and (hi is None or fi >= hi - deb)
                        det += int(ok)
                    tpr[op] = det / len(lanes)
                fp = 0
                for r in neg:
                    fi = replay_fire(
                        r["prob_trace"], r["t_trace"], r["attr_trace"], thr, alpha, deb,
                        args.arm_s, args.cooldown_s,
                    )
                    fp += int(fi is not None)
                tpr_min = min(tpr.values()) if tpr else 0.0
                fpr = fp / max(1, n_neg)
                results.append({
                    "threshold": thr, "ema_alpha": alpha, "debounce": deb,
                    "tpr_min": round(tpr_min, 3),
                    "tpr_mean": round(float(np.mean(list(tpr.values()))), 3),
                    "fpr": round(fpr, 3), "tpr_per_op": {k: round(v, 3) for k, v in tpr.items()},
                })

    # Recommend: TPR_min == 1.0 and FPR == 0, maximise threshold (margin), then larger debounce.
    perfect = [r for r in results if r["tpr_min"] >= 1.0 and r["fpr"] <= 0.0]
    rec = (
        max(perfect, key=lambda r: (r["threshold"], r["debounce"]))
        if perfect
        else max(results, key=lambda r: (r["tpr_min"], -r["fpr"]))
    )
    print("\n[sweep] operating points with TPR_min=1.0 AND FPR=0:")
    for r in sorted(perfect, key=lambda r: (r["threshold"], r["debounce"])):
        print(f"    thr={r['threshold']} alpha={r['ema_alpha']} deb={r['debounce']} "
              f"-> TPR_min={r['tpr_min']} FPR={r['fpr']}", flush=True)
    if not perfect:
        print("    (none) — best by (TPR_min, -FPR):")
        for r in sorted(results, key=lambda r: (-r["tpr_min"], r["fpr"]))[:8]:
            print(f"    thr={r['threshold']} alpha={r['ema_alpha']} deb={r['debounce']} "
                  f"-> TPR_min={r['tpr_min']} FPR={r['fpr']}", flush=True)
    print(f"\n[sweep] RECOMMENDED: {rec}", flush=True)
    Path(args.out).write_text(json.dumps({"recommended": rec, "grid": results}, indent=2))
    print(f"[sweep] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
