#!/usr/bin/env python
"""Suite-Sem attribution eval: trained VLA vs the FSM/majority baseline (M7 exit criterion 1).

Usage:
    python scripts/eval_vla_suite_sem.py --adapter outputs/vla/sft/adapter_best
                                         [--config vla/sft.yaml] [--dataset outputs/hindsight_isaac]
                                         [--out outputs/vla/suite_sem.json]

Loads the SFT adapter, rebuilds the exact held-out test split (same seed/fractions as training),
runs the trained planner on the Suite-Sem ambiguity-pair snapshots, and reports its attribution
accuracy vs the strongest constant (majority-class) FSM baseline. Also reports the 100%-parse
invariant (M7 exit criterion 2) measured on the model's real generations.
"""

from __future__ import annotations

import argparse
import json

from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.eval.suite_sem import compare_vla_vs_fsm, load_suite_sem
from kino_vla.utils.config import load_config
from kino_vla.vla.planner import ModelVlaPolicy
from kino_vla.vla.sft import load_split_for_eval


def main() -> None:
    ap = argparse.ArgumentParser(description="Suite-Sem attribution eval (VLA vs FSM)")
    ap.add_argument("--adapter", required=True, help="SFT adapter dir (adapter_best/adapter_last)")
    ap.add_argument("--config", default="vla/sft.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default="outputs/vla/suite_sem.json")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--route", default=None, choices=["latent", "text"], help="match the adapter")
    ap.add_argument("--temperature", type=float, default=0.0, help="VLA sampling temperature")
    ap.add_argument("--samples", type=int, default=1, help="VLA samples/item (mean acc @temp>0)")
    args = ap.parse_args()

    cfg = load_config(args.config, {"route": args.route} if args.route else None)
    pcfg = load_config("data/hindsight.yaml")
    dataset_dir = args.dataset or str(cfg.data.dataset_dir)
    tax = FailureTaxonomy(pcfg)

    import torch

    from kino_vla.vla.model import KinoVLA

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KinoVLA.from_pretrained(cfg, device=device, adapter_dir=args.adapter)
    model.eval()
    policy = ModelVlaPolicy(
        model,
        pcfg,
        tax,
        route=str(cfg.get("route", "latent")),
        n_images=int(cfg.data.get("n_images", 1)),
        temperature=args.temperature,
    )

    split = load_split_for_eval(cfg, dataset_dir)
    eval_examples = split.test if args.split == "test" else split.val
    ids = [e.sample_id for e in eval_examples]
    items = load_suite_sem(dataset_dir, ids, ambiguity_only=True)
    train_categories = [e.attribution_truth for e in split.train]
    feasible = {k: set(v) for k, v in tax._feasible.items()}

    # At temperature > 0 average over `samples` runs (the exit-3 metric: DPO sharpens the policy so
    # its sampled attribution accuracy beats SFT, where temp-0 is at ceiling for both).
    cmp = compare_vla_vs_fsm(
        policy, items, cfg=pcfg, train_categories=train_categories, feasible_sets=feasible
    )
    if args.samples > 1:
        from kino_vla.eval.suite_sem import evaluate_attribution

        accs = [cmp["vla"]["attribution_accuracy"]]
        feas = [cmp["vla"]["feasible_recovery_rate"]]
        for _ in range(args.samples - 1):
            e = evaluate_attribution(policy, items, feasible_sets=feasible)
            accs.append(e["attribution_accuracy"])
            feas.append(e["feasible_recovery_rate"])
        import statistics

        cmp["vla_samples"] = {
            "temperature": args.temperature,
            "n_samples": args.samples,
            "attr_acc_mean": statistics.mean(accs),
            "attr_acc_std": statistics.pstdev(accs),
            "feasible_mean": statistics.mean(feas),
            "attr_acc_runs": accs,
        }
        cmp["vla_beats_fsm"] = (
            cmp["vla_samples"]["attr_acc_mean"] > cmp["fsm_baseline"]["attribution_accuracy"]
        )
    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(cmp, f, indent=2)
    print(json.dumps(cmp, indent=2))
    print(
        f"\nEXIT-1 {'PASS' if cmp['vla_beats_fsm'] else 'FAIL'}: "
        f"VLA {cmp['vla']['attribution_accuracy']:.3f} vs FSM "
        f"{cmp['fsm_baseline']['attribution_accuracy']:.3f} (margin {cmp['margin']:+.3f}); "
        f"VLA parse-rate {cmp['vla']['parse_rate']:.3f} (exit-2)"
    )


if __name__ == "__main__":
    main()
