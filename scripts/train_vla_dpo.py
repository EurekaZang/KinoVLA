#!/usr/bin/env python
"""Embodied DPO: DPO-train the SFT planner on physical-outcome preference pairs (spec §11 Stage 2).

Usage:
    python scripts/train_vla_dpo.py --pairs outputs/vla/dpo_pairs \
        --sft-adapter outputs/vla/sft/adapter_best --out outputs/vla/dpo

Loads the preference pairs (built by build_dpo_pairs.py), initializes the policy from the SFT
adapter, and runs DPO with the adapter-toggle reference (CLAUDE.md §6 #33). The real GPU step;
metrics in ``outputs/vla/dpo/dpo_metrics.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kino_vla.data.schema import Snapshot
from kino_vla.utils.config import load_config
from kino_vla.vla.dpo import PreferencePair, train_dpo


def load_pairs(pairs_dir: str | Path) -> list[PreferencePair]:
    """Reload preference pairs saved by :func:`kino_vla.vla.dpo.save_pairs`."""
    d = Path(pairs_dir)
    npz = np.load(d / "pairs.npz")
    manifest = [json.loads(line) for line in (d / "pairs.jsonl").read_text().splitlines() if line]
    pairs: list[PreferencePair] = []
    for i, m in enumerate(manifest):
        snap_meta = m["snapshot"]
        snap = Snapshot(
            operator_name=snap_meta["operator_name"],
            appearance_class=snap_meta["appearance_class"],
            t=float(snap_meta["t"]),
            pose_xy=np.asarray(snap_meta["pose_xy"], dtype=np.float64),
            heading=float(snap_meta["heading"]),
            rgb=npz[f"{i}__rgb"],
            depth=np.zeros((npz[f"{i}__rgb"].shape[0], 1, 1), dtype=np.float32),
            proprio_window=npz[f"{i}__proprio"],
            prior_outputs=list(snap_meta["prior_outputs"]),
            privileged_theta={},
            monitor_channel=snap_meta["monitor_channel"],
        )
        pairs.append(
            PreferencePair(
                snapshot=snap,
                chosen_text=m["chosen"],
                rejected_text=m["rejected"],
                chosen_attr=m.get("chosen_attr"),
                rejected_attr=m.get("rejected_attr"),
                reason=m.get("reason", "outcome"),
            )
        )
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Embodied DPO training")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--config", default="vla/dpo.yaml")
    ap.add_argument("--out", default="outputs/vla/dpo")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None, help="override (lower=less VRAM)")
    args = ap.parse_args()

    overrides: dict = {}
    if args.epochs is not None:
        overrides["train.epochs"] = args.epochs
    if args.grad_accum is not None:
        overrides["train.grad_accum"] = args.grad_accum
    cfg = load_config(args.config, overrides)
    pairs = load_pairs(args.pairs)
    if not pairs:
        raise SystemExit("no preference pairs found — run build_dpo_pairs.py first")
    metrics = train_dpo(cfg, pairs, args.sft_adapter, args.out)
    print(json.dumps({"n_pairs": metrics["n_pairs"], "final": metrics["history"][-1]}, indent=2))


if __name__ == "__main__":
    main()
