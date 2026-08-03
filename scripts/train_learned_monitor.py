#!/usr/bin/env python
"""Train + evaluate the learning-based Kino-Monitor on real-Go2 windows.

Loads the per-step labelled rollouts from ``scripts/isaac_monitor_data_collect.py``, slices them
into 500 ms windows (last-step label), trains :class:`MonitorNet` (BCE hazard + CE attribution),
then evaluates per-operator TPR + overall FPR on the held-out test set and calibrates the firing
threshold to the target FPR. Saves the checkpoint + a metrics report.

Pure CPU/GPU torch (no Isaac). Run after collecting ``windows_train.npz`` + ``windows_test.npz``:
    python scripts/train_learned_monitor.py --train outputs/monitor_learned/windows_train.npz \
        --test outputs/monitor_learned/windows_test.npz --epochs 120 --target-fpr 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from kino_vla.monitor.hazard_lab import N_CLASSES, OP_IDS
from kino_vla.monitor.learned_monitor import CLASS_NAMES, LearnedMonitorModel, MonitorNetConfig
from kino_vla.tokens.features import Standardizer


def slice_windows(npz_path: str, window_len: int, stride: int = 1):
    """Slice per-step rows into (windows, hazard, cls) using each window's LAST-step label."""
    d = np.load(npz_path, allow_pickle=True)
    feats, hazard, cls, lane = d["feats"], d["hazard"], d["cls"], d["lane"]
    X, yh, yc = [], [], []
    for lane_id in np.unique(lane):
        m = lane == lane_id
        f = feats[m]
        h = hazard[m]
        c = cls[m]
        if f.shape[0] < window_len:
            continue
        for s in range(0, f.shape[0] - window_len + 1, stride):
            X.append(f[s : s + window_len])
            yh.append(int(h[s + window_len - 1]))
            yc.append(int(c[s + window_len - 1]))
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(yh, dtype=np.int64),
        np.asarray(yc, dtype=np.int64),
    )


def evaluate(model, Xte, yh, yc, thresholds):
    """Per-threshold overall FPR + per-operator TPR + attribution accuracy on hazard windows."""
    prob, attr = model.predict(Xte)
    pred_cls = attr.argmax(axis=1)
    pos = yh == 1
    neg = yh == 0
    out = {}
    for thr in thresholds:
        fired = prob >= thr
        fpr = float(fired[neg].mean()) if neg.any() else 0.0
        tpr_per_op = {}
        for k, op in enumerate(OP_IDS, start=1):
            mk = yc == k
            tpr_per_op[op] = float(fired[mk].mean()) if mk.any() else None
        valid = [v for v in tpr_per_op.values() if v is not None]
        out[round(float(thr), 3)] = {
            "fpr": round(fpr, 4),
            "tpr_min": round(min(valid), 4) if valid else None,
            "tpr_mean": round(float(np.mean(valid)), 4) if valid else None,
            "tpr_per_op": {
                k: (round(v, 4) if v is not None else None) for k, v in tpr_per_op.items()
            },
        }
    attr_acc = float((pred_cls[pos] == yc[pos]).mean()) if pos.any() else 0.0
    return out, round(attr_acc, 4)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train + eval the learning-based Kino-Monitor")
    ap.add_argument("--train", default="outputs/monitor_learned/windows_train.npz")
    ap.add_argument("--test", default="outputs/monitor_learned/windows_test.npz")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window-len", type=int, default=25)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--w-attr", type=float, default=0.7)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="outputs/monitor_learned/monitor")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    train_paths = [p.strip() for p in args.train.split(",") if p.strip()]
    parts = [slice_windows(p, args.window_len, args.stride) for p in train_paths]
    Xtr = np.concatenate([p[0] for p in parts], axis=0)
    yhtr = np.concatenate([p[1] for p in parts], axis=0)
    yctr = np.concatenate([p[2] for p in parts], axis=0)
    # --test accepts comma-separated shards too (backward-compatible with a single path) so the
    # held-out FPR can include A-class negative lanes (the E4 caliper-fix retrain).
    test_paths = [p.strip() for p in args.test.split(",") if p.strip()]
    te_parts = [slice_windows(p, args.window_len, args.stride) for p in test_paths]
    Xte = np.concatenate([p[0] for p in te_parts], axis=0)
    yhte = np.concatenate([p[1] for p in te_parts], axis=0)
    ycte = np.concatenate([p[2] for p in te_parts], axis=0)
    print(
        f"[train] train windows={Xtr.shape} pos={yhtr.mean():.3f} | "
        f"test windows={Xte.shape} pos={yhte.mean():.3f}",
        flush=True,
    )

    cfg = MonitorNetConfig(window_len=args.window_len)
    std = Standardizer.fit(Xtr.reshape(-1, Xtr.shape[-1]), eps=1e-3)
    model = LearnedMonitorModel(cfg, std, device=args.device)
    net = model.net

    x = torch.tensor(std.transform(Xtr), dtype=torch.float32)
    yh = torch.tensor(yhtr, dtype=torch.float32)
    yc = torch.tensor(yctr, dtype=torch.long)
    # Per-class balance: upweight RARE operators (O6 push, O9 high-centering have few hazard
    # windows — they fall early) so the detector is confident on them at a high firing threshold.
    counts = np.bincount(yctr, minlength=N_CLASSES).astype(np.float64)
    pos_counts = counts[1:]
    pos_w_cls = np.clip(pos_counts.mean() / np.maximum(pos_counts, 1.0), 0.3, 6.0)  # over O1..O11
    cls_w = np.ones(N_CLASSES, dtype=np.float64)
    cls_w[1:] = pos_w_cls
    cls_w_t = torch.tensor(cls_w, dtype=torch.float32, device=args.device)
    # detection per-sample weight: negatives weight 1, positives weighted by their op rarity
    samp_w_np = np.where(yhtr == 1, cls_w[yctr], 1.0)
    samp_w = torch.tensor(samp_w_np, dtype=torch.float32)
    pos_w = torch.tensor([float((yhtr == 0).sum() / max(1, (yhtr == 1).sum()))])
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(args.device), reduction="none")
    ce = nn.CrossEntropyLoss(weight=cls_w_t)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    n = x.shape[0]
    gen = torch.Generator().manual_seed(args.seed)
    net.train()
    for epoch in range(args.epochs):
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for s in range(0, n, args.batch):
            idx = perm[s : s + args.batch]
            xb = x[idx].to(args.device)
            out = net(xb)
            det = bce(out["det"], yh[idx].to(args.device))
            det = (det * samp_w[idx].to(args.device)).mean()
            loss = det + args.w_attr * ce(out["attr"], yc[idx].to(args.device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * xb.shape[0]
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  loss={tot / n:.4f}", flush=True)

    thresholds = np.round(np.arange(0.1, 0.96, 0.05), 3)
    table, attr_acc = evaluate(model, Xte, yhte, ycte, thresholds)
    # Pick the threshold that maximises min-per-op TPR subject to FPR <= target.
    feasible = {
        t: v
        for t, v in table.items()
        if v["fpr"] <= args.target_fpr and v["tpr_min"] is not None
    }
    if feasible:
        best_t = max(feasible, key=lambda t: feasible[t]["tpr_min"])
    else:
        best_t = min(table, key=lambda t: table[t]["fpr"])  # none meet target ⇒ lowest-FPR point
    best = table[best_t]
    print(f"\n[eval] attribution_acc(on hazard windows)={attr_acc}")
    print(
        f"[eval] chosen threshold={best_t}  FPR={best['fpr']}  "
        f"TPR_min={best['tpr_min']}  TPR_mean={best['tpr_mean']}"
    )
    print("[eval] per-operator TPR @ chosen threshold:")
    for op in OP_IDS:
        print(f"    {op:>4}: {best['tpr_per_op'][op]}")

    model.save(args.out)
    report = {
        "train": args.train,
        "test": args.test,
        "n_train_windows": int(Xtr.shape[0]),
        "n_test_windows": int(Xte.shape[0]),
        "attribution_acc": attr_acc,
        "target_fpr": args.target_fpr,
        "chosen_threshold": float(best_t),
        "chosen": best,
        "roc_table": table,
        "class_names": list(CLASS_NAMES),
    }
    rp = Path(args.out).with_suffix(".report.json")
    rp.write_text(json.dumps(report, indent=2))
    print(f"\n[train] saved model -> {args.out}.pt ; report -> {rp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
