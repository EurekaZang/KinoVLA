#!/usr/bin/env python3
"""Extract frozen front-ground CLIP and proprio features for realistic C1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.realistic_multimodal import (  # noqa: E402
    proprio_summary,
    temporal_clip_summary,
)
from kino_vla.map.clip_appearance import ClipAppearanceEncoder  # noqa: E402


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    corpus = args.corpus.resolve()
    output = args.output.resolve()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    manifests = sorted(
        path
        for path in corpus.glob("*/manifest.json")
        if path.parent.name != "scene_summaries"
    )
    if not manifests:
        raise RuntimeError(f"no C1 case manifests under {corpus}")
    cases = [_json(path) for path in manifests]
    if not all(case.get("passed") is True for case in cases):
        raise RuntimeError("C1 feature extraction refuses a failed case manifest")
    protocol_ids = {str(case["protocol_id"]) for case in cases}
    if len(protocol_ids) != 1:
        raise RuntimeError("C1 corpus mixes protocol IDs")

    rows: list[dict[str, Any]] = []
    rgb_sequences: list[np.ndarray] = []
    proprio_features: list[np.ndarray] = []
    source_hashes: dict[str, str] = {}
    for manifest_path, case in zip(manifests, cases, strict=True):
        arrays_path = manifest_path.parent / case["artifacts"]["observables"]["path"]
        if _sha(arrays_path) != case["artifacts"]["observables"]["sha256"]:
            raise RuntimeError(f"C1 observables hash mismatch: {arrays_path}")
        source_hashes[str(manifest_path.relative_to(ROOT))] = _sha(manifest_path)
        source_hashes[str(arrays_path.relative_to(ROOT))] = _sha(arrays_path)
        archive = np.load(arrays_path, allow_pickle=False)
        for sample in case["samples"]:
            sample_id = str(sample["sample_id"])
            rgb = np.asarray(archive[f"{sample_id}__rgb"], dtype=np.uint8)
            proprio = np.asarray(
                archive[f"{sample_id}__proprio"], dtype=np.float32
            )
            if rgb.ndim != 4 or rgb.shape[0] != 5:
                raise RuntimeError(f"invalid C1 RGB sequence: {sample_id} {rgb.shape}")
            crop_start = int(np.floor(0.45 * rgb.shape[1]))
            rgb = np.ascontiguousarray(rgb[:, crop_start:, :, :])
            rgb_sequences.append(rgb)
            proprio_features.append(proprio_summary(proprio))
            rows.append(
                {
                    **sample,
                    "protocol_id": case["protocol_id"],
                    "rgb_crop": {
                        "x_fraction": [0.0, 1.0],
                        "y_fraction": [0.45, 1.0],
                        "pixel_y_start": crop_start,
                    },
                }
            )

    order = sorted(
        range(len(rows)),
        key=lambda index: (
            str(rows[index]["split"]),
            str(rows[index]["scene_cluster"]),
            str(rows[index]["case_id"]),
            str(rows[index]["target_operator"]),
            str(rows[index]["appearance_view_id"]),
        ),
    )
    rows = [rows[index] for index in order]
    rgb_sequences = [rgb_sequences[index] for index in order]
    proprio_array = np.stack([proprio_features[index] for index in order])

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    encoder = ClipAppearanceEncoder()
    frame_count = rgb_sequences[0].shape[0]
    frame_embeddings: np.ndarray | None = None
    images: list[np.ndarray] = []
    locations: list[tuple[int, int]] = []

    def flush() -> None:
        nonlocal frame_embeddings, images, locations
        if not images:
            return
        values = encoder.embed_batch(images).astype(np.float32)
        if frame_embeddings is None:
            frame_embeddings = np.empty(
                (len(rows), frame_count, values.shape[1]), dtype=np.float32
            )
        for location, value in zip(locations, values, strict=True):
            frame_embeddings[location] = value
        images = []
        locations = []

    for sample_index, sequence in enumerate(rgb_sequences):
        for frame_index, frame in enumerate(sequence):
            images.append(frame)
            locations.append((sample_index, frame_index))
            if len(images) >= args.batch_size:
                flush()
    flush()
    if frame_embeddings is None:
        raise RuntimeError("C1 CLIP extraction produced no features")
    visual = np.stack(
        [temporal_clip_summary(value) for value in frame_embeddings]
    ).astype(np.float32)

    case_hash_sets: dict[str, set[str]] = {}
    for row in rows:
        case_hash_sets.setdefault(str(row["case_id"]), set()).add(
            str(row["shared_proprio_sha256"])
        )
    checks = {
        "all_case_manifests_pass": all(case["passed"] for case in cases),
        "one_protocol": len(protocol_ids) == 1,
        "six_samples_per_case": len(rows) == 6 * len(cases),
        "proprio_hash_identical_within_every_case": all(
            len(values) == 1 for values in case_hash_sets.values()
        ),
        "three_views_per_cause": all(
            sum(
                row["case_id"] == case_id
                and row["target_operator"] == cause
                for row in rows
            )
            == 3
            for case_id in case_hash_sets
            for cause in ("O2_compliance", "O4_tether")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"C1 feature extraction audit failed: {checks}")

    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    features_path = output / "features.npz"
    np.savez_compressed(
        features_path,
        sample_ids=np.asarray([row["sample_id"] for row in rows]),
        visual=visual,
        proprio=proprio_array.astype(np.float32),
        frame_clip=frame_embeddings,
    )
    manifest = {
        "schema_version": "kinofail.realistic-c1-causal-features.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "passed": all(checks.values()),
        "checks": checks,
        "protocol_id": next(iter(protocol_ids)),
        "corpus": str(corpus.relative_to(ROOT)),
        "encoder": encoder.model_id,
        "input_contract": {
            "camera": "Go2 front RTX camera",
            "crop_xy_fraction": [0.0, 0.45, 1.0, 1.0],
            "frames": 5,
            "temporal_visual_summary": (
                "mean + last + last-minus-first CLIP embedding"
            ),
            "temporal_proprio_summary": (
                "mean/std/min/max/q25/q75/first/last/delta/slope"
            ),
        },
        "counts": {
            "cases": len(cases),
            "samples": len(rows),
            "scene_clusters": len({row["scene_cluster"] for row in rows}),
            "domains": len({row["domain"] for row in rows}),
        },
        "dimensions": {
            "frame_clip": int(frame_embeddings.shape[-1]),
            "visual": int(visual.shape[1]),
            "proprio": int(proprio_array.shape[1]),
        },
        "source_sha256": source_hashes,
        "output_sha256": {
            "records": _sha(records_path),
            "features": _sha(features_path),
        },
    }
    manifest_path = output / "feature_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                **manifest["counts"],
                **manifest["dimensions"],
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
