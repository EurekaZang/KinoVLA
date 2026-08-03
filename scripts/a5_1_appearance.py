#!/usr/bin/env python
"""A5.1 — appearance OOD: per-cell attribution accuracy on train vs held-out (test) appearances.

C4 generalization leg 1. The A0.4 appearance library pre-registered a train/test split per semantic
class (adhesion train {yellow_board, amber} vs held-out {checker, gray_tape, translucent}; etc.).
A2 already reports the appearance-held-out headline; A5.1 gives the PER-CELL deltas for every agent
(frozen-snapshot, offline). Prediction (spec): the closed-set fusion classifier B-F (CLIP⊕proprio)
degrades most on held-out appearances — its CLIP features shift — while the VLM agents (B5-*), which
reason rather than classify, hold up. That separation is the C4 generalization content (a reasoner
beats a classifier exactly on the axis a classifier can't generalize across).

Reads A3 per_item (all agents, 692 snaps) joined to the corpus appearance_split by sample-id.

Run:  ~/miniconda3/envs/kinovla/bin/python scripts/a5_1_appearance.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def load_appearance_split() -> dict[str, str]:
    """sample_id → appearance_split, from the A0.3 corpus + the A3 T3 extension."""
    sid2split: dict[str, str] = {}
    for d in ("outputs/eval/a0/corpus", "outputs/eval/a3/corpus_t3"):
        p = Path(d, "samples.jsonl")
        if not p.exists():
            continue
        for ln in p.read_text().splitlines():
            if ln:
                r = json.loads(ln)
                sid2split[r["sample_id"]] = r.get("appearance_split", "train")
    return sid2split


def main() -> None:
    ap = argparse.ArgumentParser(description="A5.1 appearance-OOD per-cell deltas (offline)")
    ap.add_argument("--a3-dir", default="outputs/eval/a3")
    ap.add_argument("--out", default="outputs/eval/a5")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sid2split = load_appearance_split()
    a3dir = Path(args.a3_dir)
    # agent file → display name
    agents = {
        "B1": "B1",
        "B_F": "B-F",
        "B5_unshaped": "B5-unshaped",
        "B5_conflict": "B5-conflict",
        "B5_conflict_bi": "B5-conflict-bi",
        "B_V": "B-V",
        "B_T": "B-T",
        "zero_shot": "zero-shot",
    }
    cells = ["T1", "T2", "T3", "T4", "T5"]

    result: dict = {"agents": list(agents.values()), "cells": cells, "per_agent": {}}
    print(f"{'agent':<16} {'cell':<4} {'train':>8} {'test':>8} {'d':>6}  n_tr/n_te")
    deltas_by_agent: dict[str, float] = {}
    for fname, agent in agents.items():
        f = a3dir / f"per_item_{fname}.json"
        if not f.exists():
            continue
        rows = json.loads(f.read_text())
        per_cell: dict[str, dict] = {}
        for cell in cells:
            train = [
                r
                for r in rows
                if r.get("cell") == cell and sid2split.get(r["sid"], "train") == "train"
            ]
            test = [
                r
                for r in rows
                if r.get("cell") == cell and sid2split.get(r["sid"], "train") == "test"
            ]
            kt = sum(1 for r in train if r.get("attr_ok"))
            ktst = sum(1 for r in test if r.get("attr_ok"))
            tr_acc = kt / max(1, len(train))
            te_acc = ktst / max(1, len(test))
            delta = te_acc - tr_acc
            per_cell[cell] = {
                "n_train": len(train),
                "n_test": len(test),
                "train_acc": round(tr_acc, 3),
                "train_ci": wilson(kt, len(train)),
                "test_acc": round(te_acc, 3),
                "test_ci": wilson(ktst, len(test)),
                "delta": round(delta, 3),
            }
            print(
                f"{agent:<16} {cell:<4} {tr_acc:>10.3f} {te_acc:>10.3f} {delta:>+14.3f}  "
                f"{len(train)}/{len(test)}"
            )
        # aggregate (weighted by cell n)
        all_train = [r for r in rows if sid2split.get(r["sid"], "train") == "train"]
        all_test = [r for r in rows if sid2split.get(r["sid"], "train") == "test"]
        agg_tr = sum(1 for r in all_train if r.get("attr_ok")) / max(1, len(all_train))
        agg_te = sum(1 for r in all_test if r.get("attr_ok")) / max(1, len(all_test))
        deltas_by_agent[agent] = round(agg_te - agg_tr, 3)
        result["per_agent"][agent] = {
            "per_cell": per_cell,
            "agg_train": round(agg_tr, 3),
            "agg_test": round(agg_te, 3),
            "agg_delta": round(agg_te - agg_tr, 3),
        }

    print("\n=== aggregated appearance-OOD delta (test − train accuracy), per agent ===")
    for agent, d in sorted(deltas_by_agent.items(), key=lambda x: x[1]):
        print(
            f"  {agent:<16} Δ={d:+.3f}  (train {result['per_agent'][agent]['agg_train']:.3f} → "
            f"test {result['per_agent'][agent]['agg_test']:.3f})"
        )

    (out / "a5_1_appearance.json").write_text(json.dumps(result, indent=2))
    print(f"\n[OK] wrote {out / 'a5_1_appearance.json'}")


if __name__ == "__main__":
    main()
