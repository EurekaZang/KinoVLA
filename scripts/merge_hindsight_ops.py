#!/usr/bin/env python
"""Fold a freshly-annotated per-operator dataset into the canonical Hindsight-CoT dataset (§10).

The canonical real-Go2 dataset (outputs/hindsight_isaac) was built before O5 (overload) and O10
(effort-decay) were observable / correctly conditioned (#31/#32). This merges a freshly-annotated
O5/O10 dataset into it WITHOUT re-annotating the (expensive, already-good) other operators. The
merge logic + card recompute lives in kino_vla.data.merge_datasets (tested); this is a thin CLI.

    python scripts/merge_hindsight_ops.py --base outputs/hindsight_isaac \
        --add outputs/hindsight_isaac_ops --ops O5_payload O10_effort_decay
"""

from __future__ import annotations

import argparse

from kino_vla.data import git_commit_sha, merge_datasets
from kino_vla.utils.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge an O5/O10 dataset into the canonical one (§10)")
    ap.add_argument("--base", required=True, help="canonical dataset dir (modified in place)")
    ap.add_argument("--add", required=True, help="freshly-annotated per-op dataset dir")
    ap.add_argument("--ops", nargs="+", default=["O5_payload", "O10_effort_decay"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--oracle", default="api gpt-5.5 over real-Go2 (region xhigh + O5/O10 merge, #31/#32)"
    )
    args = ap.parse_args()

    cfg = load_config("data/hindsight.yaml")
    card = merge_datasets(
        args.base,
        args.add,
        list(args.ops),
        cfg=cfg,
        seed=args.seed,
        oracle_name=args.oracle,
        git_commit=git_commit_sha(),
    )
    s = card["stats"]
    print(f"[merge] added ops {sorted(set(args.ops))}")
    print(
        f"[merge] dataset now kept={s['kept']} filter-dropped={s['dropped']} "
        f"oracle_error={s['oracle_error']} reject={s['reject_rate']:.1%}"
    )
    print(f"[merge] per-op kept: {s['per_operator_kept']}")
    for label, cov in s["ambiguity_coverage"].items():
        print(f"[merge] ambiguity {label}: both_present={cov['both_present']}")
    print(f"[merge] wrote merged dataset → {args.base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
