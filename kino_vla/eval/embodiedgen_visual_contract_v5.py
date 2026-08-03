"""Phase-aware scene-admission checks for EmbodiedGen route-surface views."""

from __future__ import annotations

import math
from typing import Any


SCHEMA_VERSION = "kinofail.embodiedgen-phase-aware-scene-views.v5"
NEAR_CLIP_M = 0.01
FAR_CLIP_M = 1000.0


def evaluate_phase_aware_scene_views(
    records: list[dict[str, Any]],
    canonical_pairwise_l1: dict[str, float],
    clipping_range_m: list[float],
) -> dict[str, bool]:
    """Use context gates for canonical views and sensor gates for pitch stress."""
    canonical = [record for record in records if record["kind"] != "go2_body_pitch_stress"]
    stress = [record for record in records if record["kind"] == "go2_body_pitch_stress"]
    return {
        "seven_views_captured": len(records) == 7,
        "four_canonical_context_views": len(canonical) == 4,
        "three_body_pitch_sensor_views": len(stress) == 3,
        "canonical_context_informative": all(
            record["metrics"]["std_luminance"] >= 0.035
            and record["metrics"]["p99_luminance"]
            - record["metrics"]["p01_luminance"]
            >= 0.12
            and record["metrics"]["black_fraction"] < 0.90
            and record["metrics"]["white_fraction"] < 0.90
            and record["metrics"]["quantized_color_count_5bit"] >= 64
            for record in canonical
        ),
        "body_pitch_frames_sensor_valid": all(
            record["metrics"]["mean_luminance"] >= 0.005
            and record["metrics"]["std_luminance"] >= 0.005
            and record["metrics"]["black_fraction"] < 0.995
            and record["metrics"]["white_fraction"] < 0.995
            and record["metrics"]["quantized_color_count_5bit"] >= 8
            for record in stress
        ),
        "canonical_viewpoints_not_stale": bool(canonical_pairwise_l1)
        and min(canonical_pairwise_l1.values()) >= 0.01,
        "go2_height_view_present": any(
            record["kind"] == "go2_front_height_proxy" for record in records
        ),
        "near_clip_calibrated_for_close_ground": len(clipping_range_m) == 2
        and math.isclose(float(clipping_range_m[0]), NEAR_CLIP_M, abs_tol=1.0e-9)
        and math.isclose(float(clipping_range_m[1]), FAR_CLIP_M, abs_tol=1.0e-6),
    }
