#!/usr/bin/env python
"""On-policy Embodied-DPO: sample the SFT model's OWN outputs at the failure nodes (spec §11).

The literal §11 procedure: at each failure state node, sample multiple rollouts from the current
policy at temperature; the physical outcome labels them Chosen (correct attribution+recovery) vs
Rejected (the model's actual wrong-sibling choice). Unlike the canonical (templated) pairs, these
are ON-POLICY — they target the model's real confusions — so DPO measurably reduces them. The
outcome oracle is the privileged truth (the M6 anchor): a sample whose attribution matches the
node's privileged category is Chosen; a wrong one is Rejected.

Builds the pairs from the in-distribution train Suite-Sem snapshots (so the model is not OOD), then
DPO-trains and re-evaluates at temperature (exit-3: held-out test attribution DPO > SFT).

Usage:  python scripts/dpo_onpolicy.py --sft outputs/vla/sft_latent/adapter_best \
            --out outputs/vla/dpo_onpolicy --sample-temp 1.0 --n-samples 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.eval.suite_sem import load_suite_sem
from kino_vla.utils.config import load_config
from kino_vla.vla.dpo import PreferencePair, pair_stats, save_pairs
from kino_vla.vla.prompt import format_target
from kino_vla.vla.sft import load_split_for_eval


def main() -> None:
    ap = argparse.ArgumentParser(description="On-policy DPO pair collection + save")
    ap.add_argument("--sft", required=True, help="SFT adapter to sample from")
    ap.add_argument("--config", default="vla/sft.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default="outputs/vla/dpo_onpolicy_pairs")
    ap.add_argument("--sample-temp", type=float, default=1.0)
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--max-pairs-per-item", type=int, default=2)
    ap.add_argument("--max-items", type=int, default=0, help="cap sampled items (0=all)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    dataset_dir = args.dataset or str(cfg.data.dataset_dir)

    import torch

    from kino_vla.vla.model import KinoVLA
    from kino_vla.vla.planner import ModelVlaPolicy

    model = KinoVLA.from_pretrained(cfg, device="cuda", adapter_dir=args.sft)
    model.eval()
    torch.set_grad_enabled(False)
    policy = ModelVlaPolicy(
        model,
        pcfg,
        tax,
        route=str(cfg.get("route", "latent")),
        n_images=int(cfg.data.get("n_images", 1)),
        temperature=args.sample_temp,
    )

    split = load_split_for_eval(cfg, dataset_dir)
    ids = [e.sample_id for e in split.train]
    items = load_suite_sem(dataset_dir, ids, ambiguity_only=True)
    if args.max_items:
        items = items[: args.max_items]

    pairs: list[PreferencePair] = []
    n_items_with_pairs = 0
    for it in items:
        correct, wrong = [], []
        for _ in range(args.n_samples):
            d = policy.decide(it.snapshot)
            if not d.ok or d.annotation is None:
                continue
            (correct if d.attribution == it.attribution_truth else wrong).append(d.annotation)
        made = 0
        # pair the model's own correct vs its own wrong output (truncate to the shorter pool)
        for c, w in zip(correct, wrong, strict=False):
            pairs.append(
                PreferencePair(
                    snapshot=it.snapshot,
                    chosen_text=format_target(c),
                    rejected_text=format_target(w),
                    chosen_attr=c.attribution,
                    rejected_attr=w.attribution,
                    reason="on_policy",
                )
            )
            made += 1
            if made >= args.max_pairs_per_item:
                break
        if made:
            n_items_with_pairs += 1

    save_pairs(pairs, args.out)
    stats = pair_stats(pairs)
    stats["n_items_with_pairs"] = n_items_with_pairs
    stats["n_items_seen"] = len(items)
    stats["sample_temp"] = args.sample_temp
    Path(f"{args.out}/pair_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
