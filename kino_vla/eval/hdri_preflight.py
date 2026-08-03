"""Radiometric and seam preflight for frozen equirectangular HDR backgrounds."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "kinofail.hdri-preflight.v1"


def evaluate_hdri_array(
    bgr: np.ndarray,
    *,
    expected_width: int,
    expected_height: int,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("HDRI must be an HxWx3 array")
    height, width, channels = bgr.shape
    rgb = np.asarray(bgr[..., ::-1], dtype=np.float64)
    finite = np.isfinite(rgb)
    finite_fraction = float(np.mean(finite))
    safe = np.where(finite, rgb, 0.0)
    luminance = 0.2126 * safe[..., 0] + 0.7152 * safe[..., 1] + 0.0722 * safe[..., 2]
    mean_luminance = float(np.mean(luminance))
    seam_mae = float(np.mean(np.abs(safe[:, 0, :] - safe[:, -1, :])))
    seam_relative_to_mean = seam_mae / max(mean_luminance, 1.0e-12)
    p01, p50, p99 = [float(value) for value in np.percentile(luminance, [1, 50, 99])]
    dynamic_range_p99_p01 = p99 / max(p01, 1.0e-12)
    metrics = {
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "finite_fraction": finite_fraction,
        "negative_fraction": float(np.mean(safe < 0.0)),
        "mean_luminance": mean_luminance,
        "luminance_std": float(np.std(luminance)),
        "luminance_p01": p01,
        "luminance_p50": p50,
        "luminance_p99": p99,
        "dynamic_range_p99_p01": dynamic_range_p99_p01,
        "horizontal_seam_mae": seam_mae,
        "horizontal_seam_relative_to_mean": seam_relative_to_mean,
    }
    checks = {
        "expected_dimensions": width == expected_width and height == expected_height,
        "rgb_channels": channels == 3,
        "all_values_finite": finite_fraction == 1.0,
        "radiance_nonnegative": metrics["negative_fraction"]
        <= float(thresholds["maximum_negative_fraction"]),
        "nondegenerate_luminance": metrics["luminance_std"]
        >= float(thresholds["minimum_luminance_std"]),
        "radiometric_dynamic_range": dynamic_range_p99_p01
        >= float(thresholds["minimum_dynamic_range_p99_p01"]),
        "horizontal_seam": seam_relative_to_mean
        <= float(thresholds["maximum_seam_relative_to_mean"]),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "metrics": metrics,
        "thresholds": dict(thresholds),
        "checks": checks,
        "passed": all(checks.values()),
        "interpretation": (
            "This verifies a nondegenerate radiometric panorama and wrap seam only. It does "
            "not validate route geometry, physical scale, Go2 sensing, or operator behavior."
        ),
    }
