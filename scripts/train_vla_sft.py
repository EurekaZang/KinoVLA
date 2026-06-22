#!/usr/bin/env python
"""Kino-SFT: LoRA fine-tune Qwen3-VL-4B + Kino-Projector on Hindsight-CoT (spec §11 Stage 1).

Usage (on the GPU box):
    python scripts/train_vla_sft.py [--config vla/sft.yaml] [--dataset outputs/hindsight_isaac]
                                    [--out outputs/vla/sft] [--route latent|text]
                                    [--epochs N] [--limit K] [--set k=v ...]

A documented manual GPU run (the real-dependency component, §0 hard rule): the SFT model is the M7
deliverable, trained on the real-Go2 + gpt-5.5 + filter dataset. Reproducible from this config +
seed; metrics land in ``outputs/vla/sft/sft_metrics.json``.
"""

from __future__ import annotations

import argparse
import json

from kino_vla.utils.config import load_config
from kino_vla.vla.sft import train_sft


def main() -> None:
    ap = argparse.ArgumentParser(description="Kino-SFT (Qwen3-VL-4B + LoRA + Kino-Projector)")
    ap.add_argument("--config", default="vla/sft.yaml")
    ap.add_argument("--dataset", default=None, help="override data.dataset_dir")
    ap.add_argument(
        "--nav-dataset",
        default=None,
        help="co-train the RTX nav-SFT examples in this dir (route-around turn/waypoint decisions)",
    )
    ap.add_argument("--out", default="outputs/vla/sft")
    ap.add_argument("--route", default=None, choices=["latent", "text"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap #train examples (smoke runs)")
    ap.add_argument("--set", nargs="*", default=[], help="dotted config overrides k=v")
    args = ap.parse_args()

    overrides: dict = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = v
    if args.route:
        overrides["route"] = args.route
    if args.epochs is not None:
        overrides["train.epochs"] = args.epochs
    if args.dataset:
        overrides["data.dataset_dir"] = args.dataset
    if args.limit is not None:
        overrides["data.limit"] = args.limit

    cfg = load_config(args.config, overrides)
    dataset_dir = args.dataset or str(cfg.data.dataset_dir)
    metrics = train_sft(cfg, dataset_dir, args.out, nav_dataset_dir=args.nav_dataset)
    print(
        json.dumps(
            {k: metrics[k] for k in ("best_val_loss", "n_train", "n_val", "n_test", "wall_time_s")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
