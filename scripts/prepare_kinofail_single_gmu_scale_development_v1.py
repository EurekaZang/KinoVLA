#!/usr/bin/env python3
"""Prepare the retained 11-class Scale corpus for unified GMU development.

The source corpus predates the all-191 confirmation collection and retains all
five robot-front RGB frames plus the registered 21 x 19 proprioceptive window.
This helper reconstructs the 80-D invariant descriptor alongside the original
190-D temporal summary so the unified model uses the same 251-D body contract
as the final benchmark.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from kino_vla.eval.c2_temporal_v5 import invariant_summary


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/eval/realistic_a0_a7_v6"
OUTPUT = ROOT / "outputs/eval/kinofail_single_gmu_scale_development_v1"


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    records_path = SOURCE / "snapshots/snapshot_records.jsonl"
    records = [
        json.loads(line) for line in records_path.read_text().splitlines() if line
    ]
    with np.load(SOURCE / "features/features.npz", allow_pickle=False) as archive:
        sample_ids = archive["sample_ids"].astype(str)
        visual = np.asarray(archive["visual"], dtype=np.float32)
        full_proprio = np.asarray(archive["proprio"], dtype=np.float32)
    record_ids = np.asarray([str(row["sample_id"]) for row in records])
    if not np.array_equal(sample_ids, record_ids):
        raise RuntimeError("retained Scale features and records are misaligned")
    invariant = []
    with np.load(SOURCE / "snapshots/snapshots.npz", allow_pickle=False) as archive:
        for sample_id in sample_ids:
            invariant.append(
                invariant_summary(
                    np.asarray(archive[f"{sample_id}__proprio"], dtype=np.float32)
                )
            )
    invariant_proprio = np.stack(invariant).astype(np.float32)
    if full_proprio.shape != (len(records), 190):
        raise RuntimeError(f"unexpected full proprio shape: {full_proprio.shape}")
    if invariant_proprio.shape != (len(records), 80):
        raise RuntimeError(f"unexpected invariant shape: {invariant_proprio.shape}")
    OUTPUT.mkdir(parents=True)
    np.savez_compressed(
        OUTPUT / "features.npz",
        sample_ids=sample_ids,
        visual=visual,
        full_proprio=full_proprio,
        invariant_proprio=invariant_proprio,
    )
    manifest = {
        "schema_version": "kinofail.single-gmu-scale-development.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "source": str(SOURCE),
        "views": len(records),
        "physical_groups": len(
            {str(row["counterfactual_group_id"]) for row in records}
        ),
        "scenes": len({str(row["scene_family"]) for row in records}),
        "classes": sorted({str(row["attribution_category"]) for row in records}),
        "dimensions": {"visual": 1536, "full_proprio": 190, "invariant": 80},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
