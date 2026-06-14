"""Reflex T_safe + push-recovery evaluation (M2 exit criteria, spec §6.9).

Quantifies, on the surrogate, (1) the survival-time extension the Reflex stance buys
under a repeated-push protocol and (2) the largest single impulse recovered with vs
without Reflex (operator O6). Writes a metrics JSON and a survival bar plot to
``outputs/`` (gitignored).

Usage:
    python scripts/reflex_eval.py
"""

from __future__ import annotations

import argparse
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kino_vla.monitor.reflex import run_push_recovery_sweep, run_survival_episode  # noqa: E402
from kino_vla.utils.config import REPO_ROOT, load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Reflex survival + push-recovery eval")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", default="outputs/reflex_eval", help="output path stem")
    args = parser.parse_args()

    sim_cfg = load_config("sim/surrogate.yaml")
    reflex_cfg = load_config("recovery/reflex_v0.yaml")

    no_reflex = [
        run_survival_episode(sim_cfg, reflex_cfg, seed=s, use_reflex=False)
        for s in range(args.seeds)
    ]
    with_reflex = [
        run_survival_episode(sim_cfg, reflex_cfg, seed=s, use_reflex=True)
        for s in range(args.seeds)
    ]
    nr_mean = float(np.mean([r.survived_s for r in no_reflex]))
    rx_mean = float(np.mean([r.survived_s for r in with_reflex]))

    pr_no = run_push_recovery_sweep(sim_cfg, reflex_cfg, seed=0, use_reflex=False)
    pr_rx = run_push_recovery_sweep(sim_cfg, reflex_cfg, seed=0, use_reflex=True)

    out_stem = REPO_ROOT / args.out
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["no Reflex", "Reflex"], [nr_mean, rx_mean], color=["C3", "C0"])
    ax.set_ylabel("survival time under repeated pushes [s]")
    ax.set_title("Reflex extends T_safe (spec §6.9)")
    for i, v in enumerate([nr_mean, rx_mean]):
        ax.text(i, v, f"{v:.1f}s", ha="center", va="bottom")
    fig.tight_layout()
    png_path = out_stem.with_suffix(".png")
    fig.savefig(png_path, dpi=120)

    metrics = {
        "survival_s_no_reflex_mean": nr_mean,
        "survival_s_reflex_mean": rx_mean,
        "survival_extension_factor": rx_mean / max(nr_mean, 1e-9),
        "push_recovery_max_ns_no_reflex": pr_no.max_recovered_ns,
        "push_recovery_max_ns_reflex": pr_rx.max_recovered_ns,
        "n_seeds": args.seeds,
    }
    json_path = out_stem.with_suffix(".json")
    json_path.write_text(json.dumps(metrics, indent=2))

    print(
        f"survival: no-Reflex {nr_mean:.2f}s -> Reflex {rx_mean:.2f}s "
        f"({metrics['survival_extension_factor']:.1f}x)"
    )
    print(
        f"push-recovery max: no-Reflex {pr_no.max_recovered_ns:.1f} Ns -> "
        f"Reflex {pr_rx.max_recovered_ns:.1f} Ns"
    )
    print(f"plot:    {png_path}")
    print(f"metrics: {json_path}")
    ok = rx_mean > nr_mean and pr_rx.max_recovered_ns >= pr_no.max_recovered_ns
    print("PASS: reflex eval" if ok else "FAIL: reflex eval")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
