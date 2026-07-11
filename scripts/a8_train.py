#!/usr/bin/env python
# ruff: noqa: E501
"""A8 same-backbone LoRA trainer wrapper around kino_vla.vla.sft.train_sft."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kino_vla.eval.a7_ablation import artifact_meta, load_yaml, repo_path, write_json
from kino_vla.utils.config import load_config
from kino_vla.vla.sft import train_sft

A8A_ARMS = ("failcot_sft", "kino_general")
A8B_ARMS = ("v_only", "p_only", "concat", "text", "latent", "latent_conflict")
ALL_ARMS = A8A_ARMS + A8B_ARMS


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "A8 VLA training requires CUDA, but torch.cuda.is_available() is False."
        )


def resolve_arm(
    cfg: dict,
    arm: str,
    *,
    seed: int,
    dataset: str | None,
    out: str | None,
) -> tuple[str, dict[str, object], str, str]:
    """Return (dataset_dir, overrides, out_dir, route_label)."""
    out_root = repo_path(cfg["output_dir"])
    overrides: dict[str, object] = {
        "train.seed": seed,
        "model.model_id": cfg["model"]["model_id"],
        "model.lora.r": cfg["model"]["lora"]["r"],
        "model.lora.alpha": cfg["model"]["lora"]["alpha"],
        "model.lora.dropout": cfg["model"]["lora"]["dropout"],
        "train.epochs": cfg["model"]["train"]["epochs"],
        "train.lr": cfg["model"]["train"]["lr"],
        "data.nav_dataset_dir": None,  # A8 external track: no Go2 nav co-train
    }

    if arm == "failcot_sft":
        # Prefer a combined train mix if present; else first available a8a train split.
        ds = dataset or str(out_root / "datasets/a8a/train_mix")
        if not Path(ds).exists():
            # fall back to any a8a dataset with 'train' in name
            cand = sorted((out_root / "datasets/a8a").glob("*train*"))
            if cand:
                ds = str(cand[0])
        out_dir = out or str(out_root / f"adapters/failcot_sft/seed{seed}")
        overrides["route"] = "text"
        overrides["data.proprio_detail"] = "none"
        return ds, overrides, out_dir, "vision_only"

    if arm == "kino_general":
        ds = dataset or str(out_root / "datasets/a8a/train_mix")
        if not Path(ds).exists():
            cand = sorted((out_root / "datasets/a8a").glob("*train*"))
            if cand:
                ds = str(cand[0])
        out_dir = out or str(out_root / f"adapters/kino_general/seed{seed}")
        overrides["route"] = "latent"
        overrides["data.proprio_detail"] = "binned"
        return ds, overrides, out_dir, "latent_general"

    # A8b arms
    ds = dataset or str(out_root / "datasets/a8b/reflect_multisensory")
    out_dir = out or str(out_root / f"adapters/a8b_{arm}/seed{seed}")
    if arm == "v_only":
        overrides["route"] = "text"
        overrides["data.proprio_detail"] = "none"
    elif arm == "p_only":
        # vision still present in dataset frames; eval masks RGB separately.
        overrides["route"] = "latent"
        overrides["data.proprio_detail"] = "binned"
    elif arm == "concat":
        overrides["route"] = "text"
        overrides["data.proprio_detail"] = "scalar"
    elif arm == "text":
        overrides["route"] = "text"
        overrides["data.proprio_detail"] = "rich"
    elif arm in ("latent", "latent_conflict"):
        overrides["route"] = "latent"
        overrides["data.proprio_detail"] = "binned"
    else:
        raise SystemExit(f"unknown arm: {arm}")
    return ds, overrides, out_dir, arm


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one A8 LoRA arm")
    ap.add_argument("--config", default="configs/eval/a8.yaml")
    ap.add_argument("--sft-config", default="vla/sft.yaml")
    ap.add_argument("--arm", required=True, choices=list(ALL_ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke-run example cap")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    _require_cuda()
    cfg = load_yaml(args.config)
    dataset, overrides, out_dir, route_label = resolve_arm(
        cfg, args.arm, seed=args.seed, dataset=args.dataset, out=args.out
    )
    if args.epochs is not None:
        overrides["train.epochs"] = args.epochs
    if args.limit is not None:
        overrides["data.limit"] = args.limit
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = v

    if not Path(dataset).exists():
        raise SystemExit(
            f"dataset missing for arm={args.arm}: {dataset}. "
            "Run scripts/a8_download.py then scripts/a8_build_datasets.py first."
        )

    # A8b conflict arm requires admission pass
    if args.arm in A8B_ARMS:
        audit_path = repo_path(cfg["a8b"]["admission_gate"])
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            if not audit.get("pass"):
                raise SystemExit(f"A8b admission gate failed: {audit.get('reason')}")

    sft_cfg = load_config(args.sft_config, overrides)
    metrics = train_sft(sft_cfg, dataset, out_dir, nav_dataset_dir=None)
    meta = {
        **artifact_meta(args.config, sources={"dataset": dataset}),
        "arm": args.arm,
        "route_label": route_label,
        "seed": args.seed,
        "dataset": dataset,
        "out": out_dir,
        "metrics": metrics,
    }
    write_json(Path(out_dir) / "a8_train_meta.json", meta)
    print(json.dumps({"out": out_dir, "best_val_loss": metrics.get("best_val_loss")}, indent=2))


if __name__ == "__main__":
    main()
