#!/usr/bin/env python3
"""Build the shared visual/invariant-proprio feature contract for unified KINO."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from kino_vla.eval.c2_temporal_v5 import invariant_summary


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v6/snapshots",
    )
    parser.add_argument(
        "--visual-feature-dir",
        type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v6/features",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/eval/unified_moe_v1_features",
    )
    args = parser.parse_args()

    snapshot_dir = args.snapshot_dir.resolve()
    visual_dir = args.visual_feature_dir.resolve()
    output = args.output_dir.resolve()
    records_path = snapshot_dir / "snapshot_records.jsonl"
    snapshots_path = snapshot_dir / "snapshots.npz"
    extraction_audit_path = snapshot_dir / "extraction_audit.json"
    visual_features_path = visual_dir / "features.npz"
    visual_manifest_path = visual_dir / "feature_manifest.json"
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    audit = json.loads(extraction_audit_path.read_text(encoding="utf-8"))
    visual_manifest = json.loads(
        visual_manifest_path.read_text(encoding="utf-8")
    )
    if audit.get("passed") is not True:
        raise RuntimeError("source snapshot audit did not pass")
    if (
        visual_manifest.get("status") != "complete"
        or visual_manifest["output_sha256"]["features"]
        != _sha256(visual_features_path)
    ):
        raise RuntimeError("source visual feature cache is invalid")
    with np.load(visual_features_path, allow_pickle=False) as archive:
        sample_ids = archive["sample_ids"].astype(str)
        visual = np.asarray(archive["visual"], dtype=np.float32)
    expected_ids = np.asarray(
        [str(row["sample_id"]) for row in records], dtype=str
    )
    if not np.array_equal(sample_ids, expected_ids):
        raise RuntimeError("visual features and snapshot records are misaligned")

    proprio = []
    with np.load(snapshots_path, allow_pickle=False) as archive:
        for sample_id in sample_ids:
            proprio.append(
                invariant_summary(
                    np.asarray(
                        archive[f"{sample_id}__proprio"],
                        dtype=np.float32,
                    )
                )
            )
    proprio_array = np.stack(proprio).astype(np.float32)
    if proprio_array.shape != (len(records), 80):
        raise RuntimeError(
            f"unexpected invariant proprio shape: {proprio_array.shape}"
        )
    if not np.isfinite(visual).all() or not np.isfinite(proprio_array).all():
        raise RuntimeError("unified features contain non-finite values")

    output.mkdir(parents=True, exist_ok=False)
    features_path = output / "features.npz"
    np.savez_compressed(
        features_path,
        sample_ids=sample_ids,
        visual=visual,
        proprio=proprio_array,
    )
    manifest = {
        "schema_version": "kinofail.unified-observable-features.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "shared_with_c2_v5": True,
        "deployment_inputs": [
            "five body-fixed RGB frames",
            "21-sample 19-channel proprioception window",
        ],
        "visual_descriptor": (
            "mean, final, and final-minus-first CLIP ViT-B/32 embeddings"
        ),
        "proprio_descriptor": {
            "signals": [
                "angular_speed",
                "linear_acceleration_norm",
                "odom_speed",
                "slip_ratio",
                "base_height",
                "tilt",
                "effort_ratio",
                "support_ratio",
            ],
            "summary": (
                "mean/std/min/max/q25/q75/first/last/delta/slope"
            ),
        },
        "dimensions": {"visual": 1536, "proprio": 80},
        "counts": {
            "samples": len(records),
            "physical_episodes": len(
                {str(row["physical_episode_id"]) for row in records}
            ),
            "counterfactual_pairs": len(
                {str(row["counterfactual_group_id"]) for row in records}
            ),
        },
        "forbidden_deployment_inputs": [
            "scene/operator/material/severity/appearance/split identifiers",
            "ground truth",
            "outcome and cost",
            "privileged simulator telemetry",
        ],
        "source_sha256": {
            "snapshot_records": _sha256(records_path),
            "snapshots": _sha256(snapshots_path),
            "snapshot_audit": _sha256(extraction_audit_path),
            "visual_features": _sha256(visual_features_path),
            "visual_manifest": _sha256(visual_manifest_path),
        },
        "output_sha256": {"features": _sha256(features_path)},
    }
    manifest_path = output / "feature_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

