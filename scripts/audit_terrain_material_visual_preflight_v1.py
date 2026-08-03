#!/usr/bin/env python3
"""Audit train-split terrain base colours for local and global visual content."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from kino_vla.eval.terrain_material_visual_preflight_v1 import (
    MIN_P01_P99_LUMINANCE_RANGE,
    MIN_PATCH_COLOR_COUNT_P10,
    MIN_PATCH_STD_P10,
    MIN_QUANTIZED_COLOR_COUNT_5BIT,
    SCHEMA_VERSION,
    evaluate_basecolor_metrics,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(path: Path) -> dict[str, Any]:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    quantized = (rgb * 31.0).astype(np.uint8)
    height, width = luminance.shape
    patch_stds: list[float] = []
    patch_colors: list[int] = []
    for row in range(4):
        for column in range(4):
            ys = slice(row * height // 4, (row + 1) * height // 4)
            xs = slice(column * width // 4, (column + 1) * width // 4)
            patch_stds.append(float(luminance[ys, xs].std()))
            patch_colors.append(
                int(len(np.unique(quantized[ys, xs].reshape(-1, 3), axis=0)))
            )
    return {
        "mean_luminance": float(luminance.mean()),
        "std_luminance": float(luminance.std()),
        "p01_p99_luminance_range": float(
            np.quantile(luminance, 0.99) - np.quantile(luminance, 0.01)
        ),
        "quantized_color_count_5bit": int(
            len(np.unique(quantized.reshape(-1, 3), axis=0))
        ),
        "patch_std_p10": float(np.quantile(patch_stds, 0.10)),
        "patch_color_count_5bit_p10": float(np.quantile(patch_colors, 0.10)),
    }


def audit(asset_lock_path: Path, split: str) -> dict[str, Any]:
    asset_lock_path = asset_lock_path.resolve()
    lock = json.loads(asset_lock_path.read_text(encoding="utf-8"))
    asset_root = asset_lock_path.parent
    rows = []
    integrity = True
    for material in lock["materials"]:
        if material["split"] != split:
            continue
        basecolor = material["maps"]["basecolor"]
        path = asset_root / basecolor["path"]
        hash_ok = path.is_file() and _sha256(path) == basecolor["sha256"]
        integrity = integrity and hash_ok
        metrics = _metrics(path) if hash_ok else None
        decision = evaluate_basecolor_metrics(metrics) if metrics is not None else {
            "checks": {},
            "eligible_for_rtx_development": False,
            "failed_checks": ["basecolor_integrity"],
        }
        rows.append(
            {
                "material_id": material["id"],
                "semantic_family": material["semantic_family"],
                "domains": material["domains"],
                "physical_size_m": material["physical_size_m"],
                "basecolor": {
                    "path": str(path),
                    "sha256": basecolor["sha256"],
                    "hash_verified": hash_ok,
                },
                "metrics": metrics,
                **decision,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "asset_lock": {"path": str(asset_lock_path), "sha256": _sha256(asset_lock_path)},
        "split": split,
        "thresholds": {
            "minimum_p01_p99_luminance_range": MIN_P01_P99_LUMINANCE_RANGE,
            "minimum_quantized_color_count_5bit": MIN_QUANTIZED_COLOR_COUNT_5BIT,
            "minimum_patch_std_p10": MIN_PATCH_STD_P10,
            "minimum_patch_color_count_5bit_p10": MIN_PATCH_COLOR_COUNT_P10,
        },
        "materials": rows,
        "selected_for_rtx_development": [
            row["material_id"] for row in rows if row["eligible_for_rtx_development"]
        ],
        "audit_integrity_passed": integrity,
        "passed": integrity,
        "interpretation": (
            "Development-only, data-derived cost screen. Passing does not imply RTX or "
            "benchmark admission; val/test materials are not inspected by a train-split run."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-lock", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = audit(Path(args.asset_lock), args.split)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
