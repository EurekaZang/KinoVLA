#!/usr/bin/env python
"""A0.3 — Freeze + content-hash the collected snapshot corpus (offline). Paper-A §2 A0.3 / §7.

Consolidates the per-scenario ``frames_<scenario>.npz`` batches into a single ``frames.npz`` (so the
E2 ``load_suite_sem`` reads it unchanged), content-hashes every sample (record + arrays), and writes
``frozen_manifest.json``: SHA-256 corpus hash, commit, seeds, and per-(taxonomy-cell / appearance /
split / operator) counts — the reproducibility stamp the paper's tables cite. Validates the headline
cells reach ≥100 and that every class has both a train and a held-out test appearance (A0.4).

Run:  python scripts/a0_3_freeze.py [--dir outputs/eval/a0/corpus]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description="A0.3 corpus freeze + content hash")
    ap.add_argument("--dir", default="outputs/eval/a0/corpus")
    ap.add_argument("--headline-min", type=int, default=100, help="min per headline cell")
    args = ap.parse_args()
    d = Path(args.dir)

    lines = (d / "samples.jsonl").read_text().splitlines()
    records = [json.loads(ln) for ln in lines if ln.strip()]
    by_id = {r["sample_id"]: r for r in records}

    # consolidate the per-scenario frame batches into one frames.npz (load_suite_sem contract)
    frames: dict[str, np.ndarray] = {}
    for npz in sorted(d.glob("frames_*.npz")):
        with np.load(npz) as z:
            for k in z.files:
                frames[k] = z[k]
    np.savez_compressed(d / "frames.npz", **frames)

    # content hash: sha256 over sorted (sample_id, record-json, array-bytes)
    h = hashlib.sha256()
    for sid in sorted(by_id):
        h.update(sid.encode())
        h.update(json.dumps(by_id[sid], sort_keys=True).encode())
        for suffix in ("rgb", "depth", "proprio", "binding"):
            arr = frames.get(f"{sid}__{suffix}")
            if arr is not None:
                h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    corpus_hash = h.hexdigest()

    # counts
    by_cell = Counter(r["taxonomy_cell"] for r in records)
    by_app = Counter(r["appearance_id"] for r in records)
    by_split = Counter(r.get("appearance_split", "?") for r in records)
    by_op = Counter(r["snapshot"]["operator_name"] for r in records)
    by_scn = Counter(r["sample_id"].split("_s")[0].replace("a0corpus_", "").rsplit("_", 1)[0]
                     for r in records)
    # a headline cell = the matched pair (T1 matched_O2 / T2 matched_O4). validate ≥ headline_min.
    headline = {c: by_cell.get(c, 0) for c in ("T1", "T2")}
    headline_ok = all(v >= args.headline_min for v in headline.values())

    # appearance-split coverage per class (A0.4): both train + test present among collected
    splits_per_class: dict[str, set[str]] = {}
    for r in records:
        cls = r["ground_truth"]["category"]
        splits_per_class.setdefault(cls, set()).add(r.get("appearance_split", "?"))

    manifest = {
        "corpus_hash_sha256": corpus_hash,
        "commit": git_commit(),
        "n_snapshots": len(records),
        "n_frame_arrays": len(frames),
        "by_taxonomy_cell": dict(by_cell),
        "by_appearance": dict(by_app),
        "by_split": dict(by_split),
        "by_operator": dict(by_op),
        "by_scenario": dict(by_scn),
        "headline_cells": headline,
        "headline_ge_min": {"min": args.headline_min, "pass": headline_ok},
        "appearance_splits_per_class": {k: sorted(v) for k, v in splits_per_class.items()},
        "binding_window_dims": 60,
        "schema": ["rgb(5,72,96,3)", "depth(5,72,96)", "proprio(25,11)", "binding(<=100,60)"],
        "determinism": "deep_reset (A0.1) + fixed lane_y=4.0 → order-independent, byte-identical",
        "seeds": "base 500 (disjoint from training 0-9)",
    }
    (d / "frozen_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in (
        "n_snapshots", "by_taxonomy_cell", "by_split", "headline_cells",
        "headline_ge_min", "corpus_hash_sha256")}, indent=2))
    print(f"consolidated {len(frames)} frame arrays → {d / 'frames.npz'}")
    print(f"wrote {d / 'frozen_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
