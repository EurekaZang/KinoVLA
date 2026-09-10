#!/usr/bin/env python
"""Offline CPU train+gate for the M4 Isaac Kino-Tokens rollouts (spec §4).

Loads the rollouts saved by ``scripts/isaac_tokens_check.py --collect-only`` and runs the
strict four-channel gate (kino_vla/tokens/isaac_gate.py) without touching the GPU — so the
expensive Isaac collection runs once and the gate iterates cheaply here. Bound BLAS threads
(``OMP_NUM_THREADS=MKL_NUM_THREADS=4``) before launching on this RAM-limited box.

Run:  python scripts/isaac_tokens_gate.py [--npz outputs/tokens/isaac_rollouts.npz]
"""

from __future__ import annotations

import argparse
import sys

from kino_vla.tokens.isaac_gate import load_rollouts, print_report, train_and_gate
from kino_vla.utils.config import REPO_ROOT, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="M4 offline Isaac Kino-Tokens gate")
    parser.add_argument("--npz", default="outputs/tokens/isaac_rollouts.npz")
    args = parser.parse_args()

    rollouts = load_rollouts(REPO_ROOT / args.npz)
    counts: dict[str, int] = {}
    for r in rollouts:
        counts[r.channel] = counts.get(r.channel, 0) + len(r.feats)
    print(f"[isaac_tokens_gate] {len(rollouts)} rollouts, steps/channel = {counts}")

    report = train_and_gate(
        rollouts, load_config("tokens/extractor_v0.yaml"), load_config("tolerances.yaml")
    )
    print_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
