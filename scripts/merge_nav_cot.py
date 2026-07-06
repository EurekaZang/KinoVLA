#!/usr/bin/env python
"""Merge nav-CoT datasets into one trainable dir — DAgger round aggregation (#44).

DAgger trains on the AGGREGATE of all rounds' on-policy states (round 1 from sft_latent + round 2
from sft_dagger + ...). Each round's CoT dir (build_nav_cot_dataset.py output: nav_meta.jsonl +
nav_frames.npz) is produced separately (so round 1's Oracle work is reused, not re-paid); this
unions them into one dir that train_vla_sft.py consumes via data.nav_dataset_dir. Sample_ids are
distinct across rounds (different collection seeds), so the union is collision-free.

    python scripts/merge_nav_cot.py --inputs outputs/vla/nav_dagger outputs/vla/nav_dagger_r2 \
        --out outputs/vla/nav_dagger_combined
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from kino_vla.utils.config import REPO_ROOT


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge nav-CoT datasets (DAgger aggregation)")
    ap.add_argument("--inputs", nargs="+", required=True, help="nav-CoT dirs to union")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta: list[str] = []
    frames: dict[str, np.ndarray] = {}
    seen: set[str] = set()
    for d in args.inputs:
        p = REPO_ROOT / d
        n0 = len(meta)
        for line in (p / "nav_meta.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            sid = json.loads(line)["sample_id"]
            if sid in seen:
                raise SystemExit(f"sample_id collision {sid!r} across inputs — use distinct seeds")
            seen.add(sid)
            meta.append(line)
        npz = np.load(p / "nav_frames.npz")
        for k in npz.files:
            frames[k] = npz[k]
        print(f"[merge] {d}: +{len(meta) - n0} samples")

    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "nav_meta.jsonl").write_text("\n".join(meta))
    np.savez_compressed(out / "nav_frames.npz", **frames)
    kinds = Counter(json.loads(line)["kind"] for line in meta)
    card = {"inputs": list(args.inputs), "n_total": len(meta), "by_kind": dict(kinds)}
    (out / "nav_card.json").write_text(json.dumps(card, indent=2))
    print(f"[merge] DONE -> {out}: {json.dumps(card)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
