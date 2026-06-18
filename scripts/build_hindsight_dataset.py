"""Build the Privileged-Grounded Hindsight CoT dataset (spec §10, M6 deliverable).

Runs PHASE 1 (procedural counterfactual maze) → PHASE 2 (failure interception + multimodal
snapshot) → PHASE 3 (Oracle annotation + the truth-consistency filter) and writes the dataset
(JSONL manifest + npz frames), the auto-generated dataset card, and 10 review annotations.

Usage:
    python scripts/build_hindsight_dataset.py --n-cells 200 --seed 1
    python scripts/build_hindsight_dataset.py --oracle api      # external LLM (needs an API key)
"""

from __future__ import annotations

import argparse
import subprocess

from kino_vla.data import (
    ApiOracle,
    FailureTaxonomy,
    ScriptedOracle,
    run_pipeline,
    write_dataset,
)
from kino_vla.utils.config import REPO_ROOT, load_config
from kino_vla.utils.seeding import seed_everything


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Hindsight-CoT dataset (spec §10)")
    parser.add_argument("--n-cells", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="outputs/hindsight")
    parser.add_argument("--oracle", choices=["scripted", "api"], default="scripted")
    parser.add_argument("--confab", type=float, default=None, help="override Oracle confab rate")
    parser.add_argument("--no-frames", action="store_true", help="skip the heavy RGB-D npz")
    parser.add_argument(
        "--from-snapshots",
        default=None,
        help="annotate pre-collected REAL-Go2 snapshots (.npz from isaac_hindsight_collect.py) "
        "instead of generating surrogate ones — the spec's Isaac data path",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1, help="parallel Oracle calls (API is the bottleneck)"
    )
    args = parser.parse_args()

    overrides = {"maze.n_cells": args.n_cells}
    if args.confab is not None:
        overrides["oracle.confab_rate"] = args.confab
    cfg = load_config("data/hindsight.yaml", overrides)
    seed_everything(args.seed)
    taxonomy = FailureTaxonomy(cfg)
    if args.oracle == "api":
        oracle = ApiOracle.from_config(cfg)
    else:
        oracle = ScriptedOracle(cfg, taxonomy, seed=args.seed)

    if args.from_snapshots:
        from kino_vla.data import annotate_snapshots, load_snapshots

        items = load_snapshots(args.from_snapshots)
        print(
            f"[hindsight] annotating {len(items)} REAL-Go2 snapshots from {args.from_snapshots} "
            f"(concurrency={args.concurrency})"
        )
        result = annotate_snapshots(
            cfg, items, oracle=oracle, taxonomy=taxonomy, concurrency=args.concurrency
        )
    else:
        result = run_pipeline(cfg, seed=args.seed, oracle=oracle, taxonomy=taxonomy)
    out_dir = REPO_ROOT / args.out
    oracle_name = args.oracle
    if args.from_snapshots:
        oracle_name = f"{args.oracle} over real-Go2 Isaac snapshots ({args.from_snapshots})"
    card = write_dataset(
        result,
        cfg,
        out_dir,
        seed=args.seed,
        oracle_name=oracle_name,
        git_commit=_git_commit(),
        with_frames=not args.no_frames,
    )

    s = result.stats
    print(
        f"[hindsight] cells={s.n_cells} intercepted={s.n_interceptions} "
        f"kept={s.kept} dropped={s.dropped} reject={s.reject_rate:.1%}"
    )
    print(f"[hindsight] by reason: {s.by_reason}")
    print(
        f"[hindsight] throughput {s.samples_per_hour:.0f}/h "
        f"(kept {s.kept_per_hour:.0f}/h) target≥{cfg.report.target_samples_per_hour} "
        f"→ {'PASS' if card['throughput_ok'] else 'FAIL'}"
    )
    for label, cov in s.ambiguity_coverage.items():
        print(f"[hindsight] ambiguity {label}: both_present={cov['both_present']}")
    print(f"[hindsight] wrote dataset + card → {out_dir}")
    ok = card["throughput_ok"] and s.kept > 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
