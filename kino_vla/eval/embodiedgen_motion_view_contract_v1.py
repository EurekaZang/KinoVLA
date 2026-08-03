"""Operator-blind visual contract for a short Go2-front route sweep."""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "kinofail.embodiedgen-go2-front-motion-view.v1"
PROGRESS_M = (0.0, 0.25, 0.5)
MIN_STD_LUMINANCE = 0.035
MAX_BLACK_FRACTION = 0.90
MIN_QUANTIZED_COLORS_5BIT = 64
# Unitree Go2 flat-task standing height used before physics is opened.  This is a
# geometric proxy input, not an image-admission threshold.  The articulated QA
# remains authoritative for the measured base pose.
NOMINAL_STANDING_BASE_HEIGHT_M = 0.41


def evaluate_motion_views(records: list[dict[str, Any]]) -> dict[str, bool]:
    """Reuse the articulated Go2 QA's existing non-degenerate-frame thresholds."""

    ordered_progress = [float(record.get("progress_m", -1.0)) for record in records]
    return {
        "three_frozen_progress_views": ordered_progress == list(PROGRESS_M),
        "all_motion_views_non_degenerate": len(records) == len(PROGRESS_M)
        and all(
            float(record["metrics"]["std_luminance"]) >= MIN_STD_LUMINANCE
            and float(record["metrics"]["black_fraction"]) < MAX_BLACK_FRACTION
            and int(record["metrics"]["quantized_color_count_5bit"])
            >= MIN_QUANTIZED_COLORS_5BIT
            for record in records
        ),
    }
