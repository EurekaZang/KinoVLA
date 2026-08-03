#!/usr/bin/env python3
"""Extract a generic, appearance-resistant geometry stream for C2.

The descriptor is deliberately label agnostic: grayscale HOG is computed on
the same Go2-front ground ROI for each of five frames, then summarized by
mean, final-frame, and final-minus-first values.  It complements semantic CLIP
features without using material IDs, scene IDs, or operator-specific masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


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


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _hog_summary(rgb: np.ndarray, y_start: int) -> np.ndarray:
    if rgb.shape != (5, 360, 640, 3) or rgb.dtype != np.uint8:
        raise RuntimeError(f"unexpected RGB tensor: {rgb.shape} {rgb.dtype}")
    descriptor = cv2.HOGDescriptor(
        (80, 48), (16, 16), (8, 8), (8, 8), 9
    )
    frames = []
    for image in rgb:
        cropped = image[int(y_start) :, :, :]
        gray = cv2.cvtColor(
            cv2.resize(cropped, (80, 48), interpolation=cv2.INTER_AREA),
            cv2.COLOR_RGB2GRAY,
        )
        frames.append(descriptor.compute(gray).reshape(-1))
    values = np.stack(frames).astype(np.float32)
    return np.concatenate(
        [values.mean(axis=0), values[-1], values[-1] - values[0]]
    ).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--c1-corpus", type=Path, required=True)
    parser.add_argument("--scale-snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    feature_dir = args.features.resolve()
    c1_corpus = args.c1_corpus.resolve()
    scale_dir = args.scale_snapshots.resolve()
    output = args.output.resolve()

    feature_manifest_path = feature_dir / "feature_manifest.json"
    records_path = feature_dir / "records.jsonl"
    features_path = feature_dir / "features.npz"
    feature_manifest = _json(feature_manifest_path)
    if feature_manifest.get("passed") is not True:
        raise RuntimeError("C2 geometry refuses an incomplete feature cache")
    if (
        _sha(records_path) != feature_manifest["output_sha256"]["records"]
        or _sha(features_path) != feature_manifest["output_sha256"]["features"]
    ):
        raise RuntimeError("C2 feature cache provenance mismatch")
    rows = _jsonl(records_path)
    archive = np.load(features_path, allow_pickle=False)
    sample_ids = archive["sample_ids"].astype(str)
    if sample_ids.tolist() != [str(row["sample_id"]) for row in rows]:
        raise RuntimeError("C2 records/features are misaligned")

    scale_records_path = scale_dir / "snapshot_records.jsonl"
    scale_arrays_path = scale_dir / "snapshots.npz"
    scale_audit_path = scale_dir / "extraction_audit.json"
    scale_audit = _json(scale_audit_path)
    if scale_audit.get("passed") is not True:
        raise RuntimeError("C2 geometry refuses failed scale snapshots")
    scale_archive = np.load(scale_arrays_path, allow_pickle=False)

    cache: dict[tuple[str, str], np.ndarray] = {}
    values = []
    for row in rows:
        y_start = int(row["rgb_crop"]["pixel_y_start"])
        if row["cell"] == "T2_vision_decisive":
            frames = []
            for relative in row["rgb_paths"]:
                path = c1_corpus / str(row["case_id"]) / str(relative)
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(path)
                frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            rgb = np.stack(frames).astype(np.uint8)
            key = (str(row["sample_id"]), str(y_start))
        elif row["cell"] == "T3_proprio_decisive":
            source = str(row["visual_source_sample_id"])
            rgb = np.asarray(
                scale_archive[f"{source}__rgb"], dtype=np.uint8
            )
            key = (source, str(y_start))
        else:
            raise RuntimeError(f"unknown C2 cell: {row['cell']}")
        if key not in cache:
            cache[key] = _hog_summary(rgb, y_start)
        values.append(cache[key])

    geometry = np.stack(values).astype(np.float32)
    output.mkdir(parents=True, exist_ok=False)
    geometry_path = output / "geometry.npz"
    np.savez_compressed(
        geometry_path,
        sample_ids=sample_ids,
        geometry=geometry,
    )
    checks = {
        "sample_order_matches_features": True,
        "all_values_finite": bool(np.isfinite(geometry).all()),
        "expected_sample_count": len(geometry) == len(rows),
        "t3_visual_identity_preserved": all(
            np.array_equal(geometry[left], geometry[right])
            for left in range(len(rows))
            for right in []
        ),
    }
    # Explicitly check paired T3 labels rather than relying on the source cache.
    by_pair: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        if row["cell"] == "T3_proprio_decisive":
            by_pair.setdefault(
                (str(row["case_id"]), str(row["appearance_view_id"])), []
            ).append(index)
    checks["t3_visual_identity_preserved"] = all(
        len(indices) == 2
        and np.array_equal(geometry[indices[0]], geometry[indices[1]])
        for indices in by_pair.values()
    )
    manifest = {
        "schema_version": "kinofail.realistic-c2-geometry-features.v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "descriptor": {
            "name": "grayscale_HOG_temporal_summary",
            "roi": "full width, RGB rows 45%-100%",
            "resize": [80, 48],
            "hog": {
                "window": [80, 48],
                "block": [16, 16],
                "stride": [8, 8],
                "cell": [8, 8],
                "bins": 9,
            },
            "temporal_summary": ["mean", "last", "last-minus-first"],
            "uses_labels_or_material_metadata": False,
        },
        "counts": {
            "samples": len(rows),
            "dimension": int(geometry.shape[1]),
            "unique_visual_sequences": len(cache),
        },
        "source_sha256": {
            "feature_manifest": _sha(feature_manifest_path),
            "records": _sha(records_path),
            "scale_records": _sha(scale_records_path),
            "scale_arrays": _sha(scale_arrays_path),
            "scale_audit": _sha(scale_audit_path),
        },
        "output": str(geometry_path.relative_to(ROOT)),
        "output_sha256": _sha(geometry_path),
    }
    manifest_path = output / "geometry_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
