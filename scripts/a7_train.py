#!/usr/bin/env python
# ruff: noqa: E501
"""Thin A7 trainer wrapper around the canonical Kino-SFT trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kino_vla.eval.a7_ablation import artifact_meta, load_yaml, write_json
from kino_vla.utils.config import load_config
from kino_vla.vla.sft import train_sft


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "A7 VLA training requires CUDA, but torch.cuda.is_available() is False. "
            "Check nvidia-smi and resolve driver/library mismatches before rerunning."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one A7 SFT arm via kino_vla.vla.sft.train_sft")
    ap.add_argument("--config", default="configs/eval/a7.yaml")
    ap.add_argument("--sft-config", default="vla/sft.yaml")
    ap.add_argument("--arm", required=True, choices=["text_schema", "conflict_dose", "cot_filter", "encoder"])
    ap.add_argument("--schema", default="latent")
    ap.add_argument("--dose", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    _require_cuda()
    a7 = load_yaml(args.config)
    overrides: dict[str, object] = {"train.seed": args.seed}
    if args.epochs is not None:
        overrides["train.epochs"] = args.epochs
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = v

    dataset = args.dataset
    out = args.out
    if args.arm == "text_schema":
        schema = a7["text_schemas"][args.schema]
        overrides["route"] = schema["route"]
        overrides["data.proprio_detail"] = schema["proprio_detail"]
        dataset = dataset or a7["sources"]["a3_conflict_dataset"]
        out = out or f"{a7['output_dir']}/adapters/text_schema/seed{args.seed}/{args.schema}"
    elif args.arm == "conflict_dose":
        if args.dose is None:
            raise SystemExit("--dose is required for --arm conflict_dose")
        overrides["route"] = "latent"
        dataset = dataset or f"{a7['output_dir']}/datasets/conflict_dose/dose_{args.dose:02d}"
        out = out or f"{a7['output_dir']}/adapters/conflict_dose/seed{args.seed}/dose_{args.dose:02d}"
    elif args.arm == "cot_filter":
        overrides["route"] = "latent"
        dataset = dataset or a7["sources"]["hindsight_filtered"]
        out = out or f"{a7['output_dir']}/adapters/cot_filter/seed{args.seed}/truth_filtered"
    elif args.arm == "encoder":
        overrides["route"] = "latent"
        dataset = dataset or a7["sources"]["a3_conflict_dataset"]
        out = out or f"{a7['output_dir']}/adapters/encoder/seed{args.seed}/default"

    cfg = load_config(args.sft_config, overrides)
    nav_dataset_dir = cfg.data.get("nav_dataset_dir")
    metrics = train_sft(cfg, dataset, out, nav_dataset_dir=nav_dataset_dir)
    meta = {
        **artifact_meta(args.config, sources={"dataset": dataset}),
        "arm": args.arm,
        "schema": args.schema,
        "dose": args.dose,
        "seed": args.seed,
        "dataset": dataset,
        "out": out,
        "metrics": metrics,
    }
    write_json(Path(out) / "a7_train_meta.json", meta)
    print(json.dumps({"out": out, "best_val_loss": metrics.get("best_val_loss")}, indent=2))


if __name__ == "__main__":
    main()
