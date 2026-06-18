#!/usr/bin/env python
"""Build §11 ambiguity-pair Embodied-DPO preference pairs from the in-distribution dataset.

The spec §11 v2 supplement: at each ambiguity-pair failure node, the wrong-sibling strategy is a
high-quality Rejected and the correct strategy the Chosen, so DPO directly optimizes attribution
correctness. This builds those pairs deterministically from the held-out *train* Suite-Sem
snapshots + the privileged taxonomy (no model sampling, no runtime closed-loop proprio shift), so
they are θ-grounded (the M6 truth filter would drop every Rejected). DPO then trains on them
(train_vla_dpo.py) and the held-out *test* attribution accuracy @temperature is the exit-3 metric.

Usage:  python scripts/build_dpo_ambiguity.py --out outputs/vla/dpo_pairs_ambiguity
"""

from __future__ import annotations

import argparse
import json

from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.eval.suite_sem import load_suite_sem
from kino_vla.utils.config import load_config
from kino_vla.vla.dpo import build_ambiguity_pairs, pair_stats, save_pairs
from kino_vla.vla.sft import load_split_for_eval


def main() -> None:
    ap = argparse.ArgumentParser(description="Build §11 ambiguity-pair DPO preferences")
    ap.add_argument("--config", default="vla/sft.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--out", default="outputs/vla/dpo_pairs_ambiguity")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    dataset_dir = args.dataset or str(cfg.data.dataset_dir)

    split = load_split_for_eval(cfg, dataset_dir)
    examples = split.train if args.split == "train" else split.val
    ids = [e.sample_id for e in examples]
    items = load_suite_sem(dataset_dir, ids, ambiguity_only=True)
    pairs = build_ambiguity_pairs(items, tax, pcfg)
    save_pairs(pairs, args.out)
    stats = pair_stats(pairs)
    with open(f"{args.out}/pair_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
