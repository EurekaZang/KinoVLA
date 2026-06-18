#!/usr/bin/env python
"""Scale driver for the Hindsight-CoT dataset: run M Isaac collect shards + merge (spec §10, #31).

Isaac holds the GPU one process at a time, so the shards run SEQUENTIALLY (each a fresh Isaac with
its own seed ⇒ different randomized-θ lanes over the observable operators). Each shard saves its
own ``.npz``; a crash loses only that shard. The shards are merged into one snapshots file, which
``build_hindsight_dataset.py --from-snapshots --oracle api --concurrency N`` then annotates.

    python scripts/scale_collect.py --shards 10 --n-lanes 50
    python scripts/build_hindsight_dataset.py --oracle api --concurrency 8 \
        --from-snapshots outputs/hindsight_isaac/snapshots.npz --out outputs/hindsight_isaac
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from kino_vla.data import merge_snapshots
from kino_vla.utils.config import REPO_ROOT


def main() -> int:
    p = argparse.ArgumentParser(description="Collect the scaled real-Go2 snapshot set in shards")
    p.add_argument("--shards", type=int, default=10)
    p.add_argument("--n-lanes", type=int, default=50, help="lanes per shard (one Isaac process)")
    p.add_argument("--out", default="outputs/hindsight_isaac/snapshots.npz")
    p.add_argument("--shard-dir", default="outputs/hindsight_isaac/shards")
    args = p.parse_args()

    shard_dir = REPO_ROOT / args.shard_dir
    shard_dir.mkdir(parents=True, exist_ok=True)
    collect = REPO_ROOT / "scripts" / "isaac_hindsight_collect.py"

    paths: list[Path] = []
    for s in range(int(args.shards)):
        sp = shard_dir / f"shard_{s}.npz"
        print(f"=== shard {s + 1}/{args.shards} (seed {s}, {args.n_lanes} lanes) ===", flush=True)
        proc = subprocess.run(
            [
                sys.executable,
                str(collect),
                "--headless",
                "--random",
                "--n-lanes",
                str(args.n_lanes),
                "--seed",
                str(s),
                "--out",
                str(sp),
                "--min-keep",
                "1",
            ],
            cwd=REPO_ROOT,
        )
        if proc.returncode == 0 and sp.exists() and sp.with_suffix(".json").exists():
            paths.append(sp)
        else:
            print(f"[scale] shard {s} failed (rc={proc.returncode}); skipping", flush=True)

    if not paths:
        print("FAIL: no shards collected")
        return 1
    n = merge_snapshots([str(x) for x in paths], REPO_ROOT / args.out)
    print(f"PASS: merged {len(paths)}/{args.shards} shards -> {n} real-Go2 snapshots @ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
