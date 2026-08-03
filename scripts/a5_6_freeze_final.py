#!/usr/bin/env python
"""Freeze and hash an A5.6 final-validation corpus without altering its frame archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def freeze(directory: Path, *, config: Path, gate: Path) -> dict:
    import yaml

    config_data = yaml.safe_load(config.read_text())
    records = [
        json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines() if line
    ]
    by_id = {str(row["sample_id"]): row for row in records}
    if len(by_id) != len(records):
        raise ValueError(f"{directory}: duplicate sample ids")
    with np.load(directory / "frames.npz") as frames:
        digest = hashlib.sha256()
        for sid in sorted(by_id):
            digest.update(sid.encode())
            digest.update(json.dumps(by_id[sid], sort_keys=True).encode())
            for suffix in ("rgb", "depth", "proprio", "binding"):
                key = f"{sid}__{suffix}"
                if key not in frames:
                    raise KeyError(f"{directory}: missing {key}")
                digest.update(np.ascontiguousarray(frames[key], dtype=np.float32).tobytes())
        n_arrays = len(frames.files)
    manifest = {
        "experiment": str(config_data["experiment"]),
        "corpus_hash_sha256": digest.hexdigest(),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "gate_config_sha256": hashlib.sha256(gate.read_bytes()).hexdigest(),
        "n_snapshots": len(records),
        "n_frame_arrays": n_arrays,
        "by_taxonomy_cell": dict(Counter(str(row["taxonomy_cell"]) for row in records)),
        "by_appearance": dict(Counter(str(row["appearance_id"]) for row in records)),
        "by_split": dict(Counter(str(row["appearance_split"]) for row in records)),
        "by_operator": dict(Counter(str(row["snapshot"]["operator_name"]) for row in records)),
        "all_final_split": all(row["appearance_split"] == "final" for row in records),
        "binding_window_dims": 60,
        "determinism": "deep_reset(A0.1) + fixed lane_y",
        "seed_base": int(config_data["seed_base"]),
    }
    (directory / "frozen_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze A5.6 final corpora")
    parser.add_argument(
        "--dirs",
        default="outputs/eval/a5/c4_final_main,outputs/eval/a5/c4_final_t3",
    )
    parser.add_argument("--config", default="configs/eval/c4_final_appearances.yaml")
    parser.add_argument("--gate", default="configs/eval/c4_structured_gate.yaml")
    args = parser.parse_args()
    for value in args.dirs.split(","):
        directory = Path(value)
        result = freeze(directory, config=Path(args.config), gate=Path(args.gate))
        print(directory, json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
