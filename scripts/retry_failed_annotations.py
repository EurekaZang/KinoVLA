#!/usr/bin/env python
"""Recover gateway-failed Oracle annotations (re-annotate ONLY the failed snapshots).

A concurrent xhigh annotation can overload the gateway (HTTP 503), which the resilient pipeline
marks ``schema_invalid`` ("oracle call failed"). This reuses the already-good annotations (cache)
and re-annotates ONLY the failed snapshots at a gentler concurrency, then rewrites the dataset.

    python scripts/retry_failed_annotations.py --dir outputs/hindsight_isaac --concurrency 3
"""

from __future__ import annotations

import argparse
import json

from kino_vla.data import (
    ApiOracle,
    FailureTaxonomy,
    annotate_snapshots,
    git_commit_sha,
    load_snapshot_meta,
    load_snapshots,
    write_dataset,
)
from kino_vla.data.schema import ORACLE_ERROR, CoTAnnotation, RecoveryPrimitive, Verdict
from kino_vla.utils.config import REPO_ROOT, load_config


def _reconstruct(rec: dict) -> tuple:
    v = rec["verdict"]
    verdict = Verdict(keep=v["keep"], reason=v["reason"], detail=v.get("detail", ""))
    a = rec["annotation"]
    if a is None:
        return None, verdict
    prim = RecoveryPrimitive(a["action"]["primitive"], a["action"].get("params", {}))
    ann = CoTAnnotation(
        thought=a.get("thought", ""),
        attribution=a["attribution"],
        primitive=prim,
        attribution_raw=a.get("attribution_raw", a["attribution"]),
        raw_text="",
    )
    return ann, verdict


def main() -> int:
    p = argparse.ArgumentParser(description="Re-annotate gateway-failed snapshots only")
    p.add_argument("--dir", default="outputs/hindsight_isaac")
    p.add_argument("--concurrency", type=int, default=3)
    args = p.parse_args()

    d = REPO_ROOT / args.dir
    cfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(cfg)
    items = load_snapshots(d / "snapshots.npz")

    recs = [json.loads(line) for line in (d / "samples.jsonl").read_text().splitlines() if line]
    recs += [json.loads(line) for line in (d / "dropped.jsonl").read_text().splitlines() if line]
    by_id = {r["sample_id"]: r for r in recs}

    cache: dict[int, tuple] = {}
    n_failed = 0
    for sid, r in by_id.items():
        i = int(sid.split("_")[0])
        v = r["verdict"]
        # An Oracle-call failure: the new ORACLE_ERROR status, or a legacy "oracle call failed"
        # record (pre-M1 it was mislabeled schema_invalid). Everything else reuses its annotation.
        is_api_fail = v["reason"] == ORACLE_ERROR or (
            v["reason"] == "schema_invalid" and "oracle call failed" in v.get("detail", "")
        )
        if is_api_fail:
            n_failed += 1  # NOT cached ⇒ re-annotated
        else:
            cache[i] = _reconstruct(r)

    meta = load_snapshot_meta(d / "snapshots.npz")
    n_miss = max(0, int(meta.get("n_attempted", len(items))) - len(items))
    print(f"[retry] re-annotating {n_failed} failed snapshots (concurrency {args.concurrency})")
    oracle = ApiOracle.from_config(cfg)
    result = annotate_snapshots(
        cfg,
        items,
        oracle=oracle,
        taxonomy=tax,
        concurrency=args.concurrency,
        cache=cache,
        n_no_interception=n_miss,
    )
    write_dataset(
        result,
        cfg,
        d,
        seed=1,
        oracle_name="api gpt-5.5 over real-Go2 (retried)",
        git_commit=git_commit_sha(),
    )
    s = result.stats
    print(
        f"[retry] AFTER: kept {s.kept}/{s.n_interceptions} reject {s.reject_rate:.1%} {s.by_reason}"
    )
    print("PASS: retry complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
