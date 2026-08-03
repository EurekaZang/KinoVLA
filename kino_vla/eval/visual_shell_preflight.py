"""Image-only preflight for development EmbodiedGen panorama visual shells."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image


SCHEMA_VERSION = "kinofail.embodiedgen-visual-shell-preflight.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_visual_shell_panorama(
    image_path: Path,
    *,
    expected_width: int,
    expected_height: int,
    thresholds: Mapping[str, float | int],
) -> dict[str, Any]:
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    height, width, channels = rgb.shape
    luminance = (
        0.2126 * rgb[..., 0].astype(np.float64)
        + 0.7152 * rgb[..., 1].astype(np.float64)
        + 0.0722 * rgb[..., 2].astype(np.float64)
    )
    colors_5bit = (rgb.astype(np.uint16) >> 3).reshape(-1, 3)
    unique_5bit_colors = int(np.unique(colors_5bit, axis=0).shape[0])
    black_fraction = float(np.mean(luminance <= 5.0))
    white_fraction = float(np.mean(luminance >= 250.0))
    seam_mae = float(
        np.mean(
            np.abs(rgb[:, 0, :].astype(np.float64) - rgb[:, -1, :].astype(np.float64))
        )
    )
    metrics = {
        "width": width,
        "height": height,
        "channels": channels,
        "luminance_std": float(np.std(luminance)),
        "unique_5bit_colors": unique_5bit_colors,
        "black_fraction": black_fraction,
        "white_fraction": white_fraction,
        "horizontal_seam_mae": seam_mae,
    }
    checks = {
        "expected_dimensions": width == expected_width and height == expected_height,
        "rgb_channels": channels == 3,
        "luminance_complexity": metrics["luminance_std"]
        >= float(thresholds["minimum_luminance_std"]),
        "color_complexity": unique_5bit_colors
        >= int(thresholds["minimum_5bit_colors"]),
        "not_mostly_black": black_fraction <= float(thresholds["maximum_black_fraction"]),
        "not_mostly_white": white_fraction <= float(thresholds["maximum_white_fraction"]),
        "horizontal_seam": seam_mae
        <= float(thresholds["maximum_horizontal_seam_mae"]),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "image": {"path": str(image_path), "sha256": sha256_file(image_path)},
        "metrics": metrics,
        "thresholds": dict(thresholds),
        "checks": checks,
        "passed": all(checks.values()),
        "interpretation": (
            "Development image gate only; passing does not validate 3D geometry, collision, "
            "Go2 sensing, operator physics, corpus eligibility, or sim-to-real transfer."
        ),
    }

