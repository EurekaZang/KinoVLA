"""Pixel checks for rendered development visual-shell viewpoints."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "kinofail.embodiedgen-visual-shell-novel-view-preflight.v1"


def evaluate_rendered_view(
    rgb: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    expected_width: int,
    expected_height: int,
    thresholds: Mapping[str, float | int],
) -> dict[str, Any]:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape HxWx3")
    if foreground_mask.shape != rgb.shape[:2]:
        raise ValueError("foreground mask must match the RGB spatial dimensions")
    rgb = np.asarray(rgb, dtype=np.uint8)
    mask = np.asarray(foreground_mask, dtype=bool)
    height, width, channels = rgb.shape
    luminance = (
        0.2126 * rgb[..., 0].astype(np.float64)
        + 0.7152 * rgb[..., 1].astype(np.float64)
        + 0.0722 * rgb[..., 2].astype(np.float64)
    )
    colors_5bit = (rgb.astype(np.uint16) >> 3).reshape(-1, 3)
    metrics = {
        "width": width,
        "height": height,
        "channels": channels,
        "foreground_fraction": float(mask.mean()),
        "luminance_std": float(np.std(luminance)),
        "unique_5bit_colors": int(np.unique(colors_5bit, axis=0).shape[0]),
        "black_fraction": float(np.mean(luminance <= 5.0)),
        "white_fraction": float(np.mean(luminance >= 250.0)),
    }
    checks = {
        "expected_dimensions": width == expected_width and height == expected_height,
        "foreground_coverage": metrics["foreground_fraction"]
        >= float(thresholds["minimum_foreground_fraction"]),
        "luminance_complexity": metrics["luminance_std"]
        >= float(thresholds["minimum_luminance_std"]),
        "color_complexity": metrics["unique_5bit_colors"]
        >= int(thresholds["minimum_5bit_colors"]),
        "not_mostly_black": metrics["black_fraction"]
        <= float(thresholds["maximum_black_fraction"]),
        "not_mostly_white": metrics["white_fraction"]
        <= float(thresholds["maximum_white_fraction"]),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "metrics": metrics,
        "thresholds": dict(thresholds),
        "checks": checks,
        "passed": all(checks.values()),
        "interpretation": (
            "Machine render sanity gate only. It cannot judge geometric coherence, route "
            "quality, Go2 camera equivalence, physical collision, or A0-A7 validity."
        ),
    }


def mean_pairwise_rgb_l1(images: list[np.ndarray]) -> float:
    if len(images) < 2:
        raise ValueError("at least two rendered views are required")
    normalized = [np.asarray(image, dtype=np.float64) / 255.0 for image in images]
    values = [
        float(np.mean(np.abs(normalized[left] - normalized[right])))
        for left in range(len(normalized))
        for right in range(left + 1, len(normalized))
    ]
    return float(np.mean(values))
