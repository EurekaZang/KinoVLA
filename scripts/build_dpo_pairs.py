#!/usr/bin/env python
"""Build Embodied-DPO preference pairs from closed-loop rollouts (spec §11 Stage 2).

Usage:
    # surrogate ablation / CI (stub or a trained model):
    python scripts/build_dpo_pairs.py --policy stub  --out outputs/vla/dpo_pairs_surrogate
    python scripts/build_dpo_pairs.py --policy model --adapter outputs/vla/sft/adapter_best \
        --backend surrogate --out outputs/vla/dpo_pairs

At each Suite-Sem failure node, multiple temperatures are sampled; the physical outcome (reach vs
fall/dead-loop) labels them; successes become Chosen, failures Rejected (spec §11). For ``stub``,
the temperature switches the policy between correct and sibling-mistake (the controllable surrogate
that exercises the construction without a model); for ``model`` it is the real sampling temperature.
"""

from __future__ import annotations

import argparse
import json

from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.utils.config import load_config
from kino_vla.vla import scenarios as S
from kino_vla.vla.dpo import build_preference_pairs, pair_stats, save_pairs
from kino_vla.vla.planner import StubVlaPolicy
from kino_vla.vla.rollout import run_vla_rollout


def main() -> None:
    ap = argparse.ArgumentParser(description="Build DPO preference pairs from closed-loop rollouts")
    ap.add_argument("--policy", choices=["stub", "model"], default="stub")
    ap.add_argument("--adapter", default=None, help="SFT adapter dir (for --policy model)")
    ap.add_argument("--config", default="vla/dpo.yaml")
    ap.add_argument("--backend", default="surrogate", choices=["surrogate", "isaac"])
    ap.add_argument("--temperatures", default="0.0,0.7,1.0,1.2")
    ap.add_argument("--repeats", type=int, default=1, help="seeds per (scenario, temperature)")
    ap.add_argument("--out", default="outputs/vla/dpo_pairs")
    args = ap.parse_args()

    cfg = load_config(args.config)  # vla/dpo.yaml — model + rollout knobs
    pcfg = load_config("data/hindsight.yaml")  # §5 vocab + taxonomy for the policies/prompt
    tax = FailureTaxonomy(pcfg)
    temps = [float(t) for t in args.temperatures.split(",")]

    model = None
    if args.policy == "model":
        import torch

        from kino_vla.vla.model import KinoVLA

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = KinoVLA.from_pretrained(cfg, device=device, adapter_dir=args.adapter)
        model.eval()

    def make_policy(temp: float):
        if args.policy == "stub":
            return StubVlaPolicy(pcfg, tax, error_mode="sibling" if temp > 0 else None)
        from kino_vla.vla.planner import ModelVlaPolicy

        return ModelVlaPolicy(
            model,
            pcfg,
            tax,
            route=str(cfg.get("route", "latent")),
            n_images=int(cfg.data.get("n_images", 1)),
            temperature=temp,
        )

    all_pairs = []
    node_summaries = []
    scenarios = S.surrogate_scenarios() if args.backend == "surrogate" else S.all_scenarios()
    for scn in scenarios:
        for rep in range(args.repeats):
            results = [
                run_vla_rollout(scn, make_policy(t), seed=rep * 100 + i, backend=args.backend)
                for i, t in enumerate(temps)
            ]
            pairs = build_preference_pairs(results, max_pairs=int(cfg.rollout.max_pairs_per_node))
            all_pairs.extend(pairs)
            node_summaries.append(
                {
                    "scenario": scn.name,
                    "rep": rep,
                    "outcomes": [r.success for r in results],
                    "n_pairs": len(pairs),
                }
            )

    save_pairs(all_pairs, args.out)
    stats = pair_stats(all_pairs)
    stats["nodes"] = node_summaries
    with open(f"{args.out}/pair_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: stats[k] for k in ("n_pairs", "by_operator", "by_reason")}, indent=2))


if __name__ == "__main__":
    main()
