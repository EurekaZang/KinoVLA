#!/usr/bin/env python3
"""Extract generic grayscale-HOG temporal geometry from a C1 corpus."""

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


def _summary(images: list[np.ndarray]) -> np.ndarray:
    descriptor = cv2.HOGDescriptor(
        (80, 48), (16, 16), (8, 8), (8, 8), 9
    )
    frames = []
    for image in images:
        gray = cv2.cvtColor(
            cv2.resize(image, (80, 48), interpolation=cv2.INTER_AREA),
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
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    feature_dir = args.features.resolve()
    corpus = args.corpus.resolve()
    output = args.output.resolve()
    manifest_path = feature_dir / "feature_manifest.json"
    records_path = feature_dir / "records.jsonl"
    features_path = feature_dir / "features.npz"
    manifest = _json(manifest_path)
    if manifest.get("passed") is not True:
        raise RuntimeError("C1 geometry refuses an incomplete feature cache")
    if (
        _sha(records_path) != manifest["output_sha256"]["records"]
        or _sha(features_path) != manifest["output_sha256"]["features"]
    ):
        raise RuntimeError("C1 feature cache provenance mismatch")
    rows = _jsonl(records_path)
    archive = np.load(features_path, allow_pickle=False)
    sample_ids = archive["sample_ids"].astype(str)
    if sample_ids.tolist() != [str(row["sample_id"]) for row in rows]:
        raise RuntimeError("C1 records/features are misaligned")

    geometry = []
    for row in rows:
        y_start = int(row["rgb_crop"]["pixel_y_start"])
        images = []
        for relative in row["rgb_paths"]:
            path = corpus / str(row["case_id"]) / str(relative)
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            images.append(rgb[y_start:, :, :])
        if len(images) != 5:
            raise RuntimeError(f"C1 sample does not have five frames: {row['sample_id']}")
        geometry.append(_summary(images))
    values = np.stack(geometry).astype(np.float32)
    checks = {
        "sample_order_matches_features": True,
        "expected_sample_count": len(values) == len(rows),
        "all_values_finite": bool(np.isfinite(values).all()),
    }
    output.mkdir(parents=True, exist_ok=False)
    arrays_path = output / "geometry.npz"
    np.savez_compressed(
        arrays_path,
        sample_ids=sample_ids,
        geometry=values,
    )
    result = {
        "schema_version": "kinofail.realistic-c1-geometry-features.v2",
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
            "dimension": int(values.shape[1]),
        },
        "source_sha256": {
            "feature_manifest": _sha(manifest_path),
            "records": _sha(records_path),
        },
        "output": str(arrays_path.relative_to(ROOT)),
        "output_sha256": _sha(arrays_path),
    }
    result_path = output / "geometry_manifest.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
