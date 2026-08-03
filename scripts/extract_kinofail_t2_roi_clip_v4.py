#!/usr/bin/env python3
"""Extract frozen CLIP features from terrain-focused T2 image crops.

The existing global descriptor preserves the full scene and can dilute the
small film/deformation cue.  This development cache adds two deterministic,
robot-camera-relative crops while retaining the same frozen CLIP backbone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.realistic_multimodal import temporal_clip_summary  # noqa: E402
from kino_vla.map.clip_appearance import ClipAppearanceEncoder  # noqa: E402
from scripts.develop_kinofail_conflict_invariant_kino_v4 import (  # noqa: E402
    F42,
    _jsonl,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _crop(image: np.ndarray, kind: str) -> np.ndarray:
    height, width = image.shape[:2]
    if kind == "ground":
        return image[int(round(0.40 * height)) :, :]
    if kind == "route":
        return image[
            int(round(0.35 * height)) :,
            int(round(0.12 * width)) : int(round(0.88 * width)),
        ]
    raise ValueError(kind)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/eval/kino_t2_roi_clip_v4_development",
    )
    parser.add_argument("--batch-size", type=int, default=192)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    records_path = F42 / "conflict/features/records.jsonl"
    rows = [
        row
        for row in _jsonl(records_path)
        if str(row["cell"]) == "T2_vision_decisive"
    ]
    if not rows:
        raise RuntimeError("no T2 rows found")
    frame_count = len(rows[0]["rgb_paths"])
    if any(len(row["rgb_paths"]) != frame_count for row in rows):
        raise RuntimeError("T2 sequences do not share the same frame count")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    encoder = ClipAppearanceEncoder()
    embeddings: dict[str, np.ndarray | None] = {"ground": None, "route": None}
    pending: list[np.ndarray] = []
    locations: list[tuple[str, int, int]] = []

    def flush() -> None:
        nonlocal pending, locations
        if not pending:
            return
        encoded = encoder.embed_batch(pending).astype(np.float32)
        for (kind, sample_index, frame_index), feature in zip(
            locations, encoded, strict=True
        ):
            if embeddings[kind] is None:
                embeddings[kind] = np.empty(
                    (len(rows), frame_count, len(feature)), dtype=np.float32
                )
            embeddings[kind][sample_index, frame_index] = feature
        pending = []
        locations = []

    for sample_index, row in enumerate(rows):
        source = Path(str(row["source_case_dir"]))
        for frame_index, relative in enumerate(row["rgb_paths"]):
            path = source / str(relative)
            image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            for kind in ("ground", "route"):
                pending.append(_crop(image, kind))
                locations.append((kind, sample_index, frame_index))
            if len(pending) >= args.batch_size:
                flush()
        if (sample_index + 1) % 500 == 0:
            print(
                json.dumps(
                    {"processed_samples": sample_index + 1, "total": len(rows)}
                ),
                flush=True,
            )
    flush()
    if any(value is None for value in embeddings.values()):
        raise RuntimeError("crop feature extraction did not complete")

    summaries = {
        kind: np.stack([temporal_clip_summary(frames) for frames in values])
        for kind, values in embeddings.items()
        if values is not None
    }
    output.mkdir(parents=True, exist_ok=False)
    features_path = output / "features.npz"
    np.savez_compressed(
        features_path,
        sample_ids=np.asarray([str(row["sample_id"]) for row in rows]),
        visual_ground=summaries["ground"].astype(np.float32),
        visual_route=summaries["route"].astype(np.float32),
    )
    manifest = {
        "schema_version": "kinofail.t2-roi-clip-v4-development.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "development_only_complete",
        "confirmatory_evidence": False,
        "encoder": encoder.model_id,
        "deployment_input": "single five-frame robot-front RGB sequence",
        "crops": {
            "ground": {"x_fraction": [0.0, 1.0], "y_fraction": [0.40, 1.0]},
            "route": {"x_fraction": [0.12, 0.88], "y_fraction": [0.35, 1.0]},
        },
        "counts": {"samples": len(rows), "frames_per_sample": frame_count},
        "dimensions": {
            name: int(value.shape[1]) for name, value in summaries.items()
        },
        "source_sha256": {"records": _sha256(records_path)},
        "output_sha256": {"features": _sha256(features_path)},
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
