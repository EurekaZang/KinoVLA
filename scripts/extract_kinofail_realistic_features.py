#!/usr/bin/env python3
"""Extract one hash-bound CLIP/proprio feature cache for all realistic evaluations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.data.runtime_manifest import sha256_file  # noqa: E402
from kino_vla.eval.realistic_multimodal import (  # noqa: E402
    proprio_summary,
    temporal_clip_summary,
)
from kino_vla.map.clip_appearance import ClipAppearanceEncoder  # noqa: E402


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v4/snapshots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v4/features",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    snapshot_dir = args.snapshot_dir.resolve()
    output = args.output_dir.resolve()
    records_path = snapshot_dir / "snapshot_records.jsonl"
    arrays_path = snapshot_dir / "snapshots.npz"
    audit_path = snapshot_dir / "extraction_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True:
        raise RuntimeError("snapshot extraction audit did not pass")
    records = _records(records_path)
    if len(records) != int(audit["counts"]["snapshot_records"]):
        raise RuntimeError("snapshot record count does not match its audit")

    # Prevent an accidental network lookup from changing a frozen experiment environment.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    encoder = ClipAppearanceEncoder()
    snapshot_arrays = np.load(arrays_path, allow_pickle=False)
    sample_ids = [str(row["sample_id"]) for row in records]
    frame_count = int(records[0]["rgb_shape"][0])
    if any(int(row["rgb_shape"][0]) != frame_count for row in records):
        raise RuntimeError("snapshot sequences do not share the frozen frame count")

    frame_embeddings: np.ndarray | None = None
    pending_images: list[np.ndarray] = []
    pending_locations: list[tuple[int, int]] = []

    def flush() -> None:
        nonlocal frame_embeddings, pending_images, pending_locations
        if not pending_images:
            return
        embedded = encoder.embed_batch(pending_images).astype(np.float32)
        if frame_embeddings is None:
            frame_embeddings = np.empty(
                (len(records), frame_count, embedded.shape[1]), dtype=np.float32
            )
        for location, value in zip(pending_locations, embedded, strict=True):
            frame_embeddings[location] = value
        pending_images = []
        pending_locations = []

    proprio = []
    for sample_index, sample_id in enumerate(sample_ids):
        rgb = np.asarray(snapshot_arrays[f"{sample_id}__rgb"], dtype=np.uint8)
        proprio.append(
            proprio_summary(snapshot_arrays[f"{sample_id}__proprio"])
        )
        for frame_index, frame in enumerate(rgb):
            pending_images.append(frame)
            pending_locations.append((sample_index, frame_index))
            if len(pending_images) >= args.batch_size:
                flush()
    flush()
    assert frame_embeddings is not None
    visual = np.stack([temporal_clip_summary(value) for value in frame_embeddings])
    proprio_array = np.stack(proprio).astype(np.float32)

    output.mkdir(parents=True, exist_ok=True)
    features_path = output / "features.npz"
    np.savez_compressed(
        features_path,
        sample_ids=np.asarray(sample_ids),
        visual=visual,
        proprio=proprio_array,
        frame_clip=frame_embeddings,
    )
    manifest = {
        "schema_version": "kinofail.realistic-observable-features.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "deployment_inputs": [
            "five body-fixed RGB frames",
            "21-sample proprioception window",
        ],
        "forbidden_deployment_inputs": [
            "target_operator",
            "scene_family",
            "severity_id",
            "appearance_id",
            "privileged telemetry",
            "ground truth",
        ],
        "encoder": encoder.model_id,
        "temporal_visual_summary": "mean + last + last-minus-first CLIP embedding",
        "temporal_proprio_summary": "mean/std/min/max/q25/q75/first/last/delta/slope",
        "counts": {
            "samples": len(records),
            "physical_episodes": len({row["physical_episode_id"] for row in records}),
            "counterfactual_pairs": len({row["counterfactual_group_id"] for row in records}),
            "frames_per_sample": frame_count,
        },
        "dimensions": {
            "frame_clip": int(frame_embeddings.shape[-1]),
            "visual": int(visual.shape[1]),
            "proprio": int(proprio_array.shape[1]),
        },
        "source_sha256": {
            "snapshot_records": sha256_file(records_path),
            "snapshots": sha256_file(arrays_path),
            "extraction_audit": sha256_file(audit_path),
        },
        "output_sha256": {"features": sha256_file(features_path)},
    }
    manifest_path = output / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**manifest["counts"], **manifest["dimensions"], "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
