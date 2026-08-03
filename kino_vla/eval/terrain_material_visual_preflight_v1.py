"""Static base-colour screening before expensive RTX material validation."""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "kinofail.terrain-material-visual-preflight.v1"
MIN_P01_P99_LUMINANCE_RANGE = 0.15
MIN_QUANTIZED_COLOR_COUNT_5BIT = 64
MIN_PATCH_STD_P10 = 0.025
MIN_PATCH_COLOR_COUNT_P10 = 64.0


def evaluate_basecolor_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Screen texture informativeness; RTX phase-aware admission remains authoritative."""

    checks = {
        "global_luminance_range": float(metrics["p01_p99_luminance_range"])
        >= MIN_P01_P99_LUMINANCE_RANGE,
        "global_color_count": int(metrics["quantized_color_count_5bit"])
        >= MIN_QUANTIZED_COLOR_COUNT_5BIT,
        "local_patch_contrast": float(metrics["patch_std_p10"]) >= MIN_PATCH_STD_P10,
        "local_patch_color_count": float(metrics["patch_color_count_5bit_p10"])
        >= MIN_PATCH_COLOR_COUNT_P10,
    }
    return {
        "checks": checks,
        "eligible_for_rtx_development": all(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }
