"""Kino-Monitor ROC plot + metrics (M2 exit criterion, spec §12 metric).

Runs labeled nominal/failure rollouts on the surrogate and renders the monitor's
detection ROC (TPR vs FPR) with its AUC, plus a metrics JSON. Reproducible from a
fixed base seed; artifacts land in ``outputs/`` (gitignored, per QA 5.4).

Usage:
    python scripts/monitor_roc.py --n 60
"""

from __future__ import annotations

import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

from kino_vla.monitor.roc import compute_monitor_roc  # noqa: E402
from kino_vla.utils.config import REPO_ROOT  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Kino-Monitor ROC over labeled rollouts")
    parser.add_argument("--n", type=int, default=60, help="episodes per class")
    parser.add_argument("--seed", type=int, default=1000, help="base seed")
    parser.add_argument("--out", default="outputs/monitor_roc", help="output path stem")
    args = parser.parse_args()

    roc = compute_monitor_roc(n_per_class=args.n, base_seed=args.seed)
    out_stem = REPO_ROOT / args.out
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(roc.fpr, roc.tpr, "-", color="C0", lw=2, label=f"Kino-Monitor (AUC={roc.auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="chance")
    ax.set_xlabel("False positive rate (nominal false alarms)")
    ax.set_ylabel("True positive rate (failures detected)")
    ax.set_title(f"Kino-Monitor ROC — {args.n} nominal vs {args.n} failure rollouts")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png_path = out_stem.with_suffix(".png")
    fig.savefig(png_path, dpi=120)

    metrics = {
        "auc": roc.auc,
        "n_per_class": args.n,
        "base_seed": args.seed,
        "n_channels": 3,
        "failure_score_min": float(roc.scores[roc.labels == 1].min()),
        "nominal_score_max": float(roc.scores[roc.labels == 0].max()),
    }
    json_path = out_stem.with_suffix(".json")
    json_path.write_text(json.dumps(metrics, indent=2))

    print(f"AUC = {roc.auc:.4f}")
    print(
        f"failure score min = {metrics['failure_score_min']:.3f}, "
        f"nominal score max = {metrics['nominal_score_max']:.3f}"
    )
    print(f"ROC plot:    {png_path}")
    print(f"ROC metrics: {json_path}")
    print("PASS: monitor ROC" if roc.auc >= 0.9 else "FAIL: monitor ROC (AUC < 0.9)")
    return 0 if roc.auc >= 0.9 else 1


if __name__ == "__main__":
    sys.exit(main())
