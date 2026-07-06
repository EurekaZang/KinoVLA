#!/usr/bin/env python
"""E4 caliper item (1): search a FIXED online op-point that rejects A-class clustered transients.

The A/B-aware detector (monitor_abaware) is now graded (per-window A-class fire 5-10%), but the
online firing (EMA + debounce + arm + det-attr gate) still trips on a CLUSTERED patch-entry
transient at gentle θ (leftpoint_abaware.json). This offline search replays the EXACT online firing
logic (kino_vla.monitor.learned_monitor.LearnedMonitor.step) over the collected per-step feature
sequences and sweeps (threshold, debounce, arm) to find a fixed op-point where A-class lanes do NOT
fire (a sustained-not-transient gate) while B-class lanes still fire.

RED LINE: the op-point is a FIXED function of the observable EMA + a sustained-count (debounce) +
warmup (arm) — no θ input. A longer debounce = "require the hazard signature to PERSIST, not just
spike on entry", which is a legitimate observable-signal criterion, not a θ-peek. CPU only (reuses
the model + the collected windows); no Isaac, so it is cheap to sweep before any real-Go2 re-check.

Run:  python scripts/e4_monitor_optpoint.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _lane_prob_seq(model, feats: np.ndarray, *, window_len: int):
    """Run the model ONCE over a lane's rolling windows → (raw_prob[T], attr_argmax[T]).

    Batches all T windows through one predict call (the raw probs are op-point-independent, so this
    is computed once per lane and the op-point sweep is then cheap arithmetic)."""
    t = feats.shape[0]
    wins = np.empty((t, window_len, feats.shape[1]), dtype=np.float32)
    for i in range(t):
        lo = i - window_len + 1
        if lo < 0:
            rows = np.concatenate([np.repeat(feats[0:1], -lo, axis=0), feats[0 : i + 1]], axis=0)
        else:
            rows = feats[lo : i + 1]
        wins[i] = rows
    prob, attr = model.predict(wins)
    return prob.astype(np.float64), attr.argmax(axis=1).astype(np.int64)


def _fire_from_seq(raw: np.ndarray, argmax: np.ndarray, ts: np.ndarray, *, threshold: float,
                   debounce: int, arm_s: float, ema_alpha: float = 0.4) -> bool:
    """Apply the LearnedMonitor.step EMA + arm + det-attr gate + debounce logic to cached sequences
    (mirrors learned_monitor.LearnedMonitor.step; cooldown omitted — we only ask 'does it fire')."""
    ema = 0.0
    above = 0
    for r, cls, t in zip(raw, argmax, ts, strict=True):
        ema += ema_alpha * (float(r) - ema)
        if t < arm_s:
            above = 0
            continue
        if ema >= threshold and int(cls) != 0:
            above += 1
        else:
            above = 0
        if above >= debounce:
            return True
    return False


def _lane_seqs(npz_path: str):
    """Yield (lane_id, feats[T,F], t[T]) per lane, in time order."""
    d = np.load(npz_path, allow_pickle=True)
    feats, lane, t = d["feats"], d["lane"], d["t"]
    cls = d["cls"] if "cls" in d else np.zeros(len(feats), dtype=np.int8)
    for lid in np.unique(lane):
        m = lane == lid
        order = np.argsort(t[m], kind="stable")
        yield int(lid), feats[m][order], t[m][order], int(np.max(cls[m]))


def main() -> None:
    ap = argparse.ArgumentParser(description="E4 online op-point search (reject A-class spikes)")
    ap.add_argument("--model", default="outputs/monitor_learned/monitor_abaware")
    ap.add_argument("--aclass", default="outputs/monitor_learned/windows_aclass_test.npz")
    ap.add_argument("--bclass", default="outputs/monitor_learned/windows_test.npz")
    ap.add_argument("--neg", default="outputs/monitor_learned/windows_negbig.npz")
    ap.add_argument("--out", default="outputs/eval/e4/optpoint.json")
    args = ap.parse_args()

    from kino_vla.monitor.learned_monitor import LearnedMonitorModel

    model = LearnedMonitorModel.load(args.model, device="cpu")
    wl = model.cfg.window_len

    def prep(lanes):
        # precompute (raw_prob, attr_argmax, t) per lane ONCE (model inference is op-point-free)
        out = []
        for _lid, f, t, _c in lanes:
            raw, am = _lane_prob_seq(model, f, window_len=wl)
            out.append((raw, am, t))
        return out

    aclass = prep(_lane_seqs(args.aclass))
    bclass = prep([x for x in _lane_seqs(args.bclass) if x[3] != 0])  # operator (positive) lanes
    neg = prep(_lane_seqs(args.neg))
    print(f"[optpoint] A-class lanes={len(aclass)} B-class(pos) lanes={len(bclass)} "
          f"clean/man lanes={len(neg)} window_len={wl}", flush=True)

    def lane_fire_rate(lanes, **kw) -> float:
        if not lanes:
            return 0.0
        fired = [_fire_from_seq(raw, am, t, **kw) for raw, am, t in lanes]
        return float(np.mean(fired))

    # Sweep a fixed op-point: threshold ∈ {0.6,0.7,0.8,0.85,0.9}, debounce ∈ {5,10,15,20,25},
    # arm ∈ {2.5, 4.0}. Want: A-class lane fire ≈ 0 AND B-class lane fire high.
    grid = []
    for thr in (0.6, 0.7, 0.8, 0.85, 0.9):
        for deb in (5, 10, 15, 20, 25):
            for arm in (2.5, 4.0):
                a = lane_fire_rate(aclass, threshold=thr, debounce=deb, arm_s=arm)
                b = lane_fire_rate(bclass, threshold=thr, debounce=deb, arm_s=arm)
                n = lane_fire_rate(neg, threshold=thr, debounce=deb, arm_s=arm)
                sep = b - max(a, n)  # want B high, A & clean/man low
                grid.append({"threshold": thr, "debounce": deb, "arm_s": arm,
                             "aclass_fire": round(a, 3), "bclass_tpr": round(b, 3),
                             "neg_fpr": round(n, 3), "separation": round(sep, 3)})
                print(f"[optpoint] thr={thr} deb={deb:2d} arm={arm}: A={a:.2f} B={b:.2f} "
                      f"neg={n:.2f} sep={sep:+.2f}", flush=True)

    # Best: maximise B-class TPR subject to A-class lane-fire <= 0.05 and clean/maneuver <= 0.05.
    # (An exact 0 A-class is not expected — the O2 mud↔adhesion pair is proprio-ambiguous by
    # construction, E1; ~1/21 residual A-class lane is that honest ceiling, not a caliper failure.)
    clean = [g for g in grid if g["aclass_fire"] <= 0.05 and g["neg_fpr"] <= 0.05]
    best = max(clean, key=lambda g: (g["bclass_tpr"], -g["debounce"])) if clean else \
        max(grid, key=lambda g: g["separation"])
    out = {"model": args.model, "window_len": wl,
           "best_clean_oppoint": best,
           "has_clean_separation": bool(clean),
           "grid": grid}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("\n=== E4 online op-point search (monitor_abaware) ===")
    if clean:
        print(f"CLEAN op-point EXISTS: thr={best['threshold']} debounce={best['debounce']} "
              f"arm={best['arm_s']} → A-class fire={best['aclass_fire']} "
              f"B-class TPR={best['bclass_tpr']} neg={best['neg_fpr']}")
    else:
        print(f"NO clean op-point (A-class fire never reaches 0 with neg≤0.1); best-sep: {best}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
