#!/usr/bin/env python
"""Quick held-out nav eval: does the co-trained VLA reproduce the geometric teacher (turn vs waypoint)?

Loads the adapter + the held-out nav split (same seed/fractions as training) and compares the model's
parsed nav action KIND to the teacher's label. A high kind-accuracy ⇒ the model learned the routing
(a deploy frame/pose mismatch is then the closed-loop gap); a low one ⇒ the nav data needs scaling.

    python scripts/eval_nav.py --adapter outputs/vla/sft_nav/adapter_best
"""

from __future__ import annotations

import argparse

import torch

from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.utils.config import load_config
from kino_vla.vla.dataset_build import load_nav_examples, split_examples
from kino_vla.vla.model import KinoVLA
from kino_vla.vla.output import parse_nav_decision


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--nav-dataset", default="outputs/vla/nav_data")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--config", default="vla/sft.yaml")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    vcfg = load_config(args.config)
    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    exs = load_nav_examples(args.nav_dataset)
    sp = split_examples(
        exs,
        seed=int(vcfg.data.split_seed),
        val_frac=float(vcfg.data.val_frac),
        test_frac=float(vcfg.data.test_frac),
    )
    items = getattr(sp, args.split)[: args.limit]
    n_images = int(vcfg.data.get("n_images", 1))
    model = KinoVLA.from_pretrained(vcfg, device="cuda", adapter_dir=args.adapter)
    model.eval()
    torch.set_grad_enabled(False)

    correct = 0
    conf = {"turn": {"turn": 0, "waypoint": 0, "other": 0}, "waypoint": {"turn": 0, "waypoint": 0, "other": 0}}
    for ex in items:
        text = model.generate(
            ex.messages,
            list(ex.rgb[-n_images:]),
            proprio_window=ex.proprio_window,
            temperature=args.temperature,
        )
        d = parse_nav_decision(text, synonyms=tax.synonyms, valid_categories=tax.valid_categories)
        if d.ok and d.nav_turn_deg is not None:
            pred = "turn"
        elif d.ok and d.primitive_name == "Replan_Waypoint":
            pred = "waypoint"
        else:
            pred = "other"
        truth = ex.primitive_truth  # "turn" | "waypoint"
        conf.setdefault(truth, {"turn": 0, "waypoint": 0, "other": 0})[pred] += 1
        correct += pred == truth
    n = len(items)
    print(f"\nHELD-OUT NAV ({args.split}, n={n}) kind-accuracy: {correct / max(1, n):.3f}")
    print("confusion (truth -> pred counts):")
    for truth, row in conf.items():
        print(f"  {truth:9s} -> {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
