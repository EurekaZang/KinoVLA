from __future__ import annotations

from kino_vla.eval.terrain_material_visual_preflight_v1 import (
    MIN_P01_P99_LUMINANCE_RANGE,
    MIN_PATCH_COLOR_COUNT_P10,
    MIN_PATCH_STD_P10,
    MIN_QUANTIZED_COLOR_COUNT_5BIT,
    evaluate_basecolor_metrics,
)


def _metrics() -> dict:
    return {
        "p01_p99_luminance_range": MIN_P01_P99_LUMINANCE_RANGE,
        "quantized_color_count_5bit": MIN_QUANTIZED_COLOR_COUNT_5BIT,
        "patch_std_p10": MIN_PATCH_STD_P10,
        "patch_color_count_5bit_p10": MIN_PATCH_COLOR_COUNT_P10,
    }


def test_threshold_boundary_is_eligible_for_rtx_development() -> None:
    result = evaluate_basecolor_metrics(_metrics())
    assert result["eligible_for_rtx_development"], result


def test_low_local_contrast_is_rejected_even_with_many_global_colors() -> None:
    metrics = _metrics()
    metrics["patch_std_p10"] = MIN_PATCH_STD_P10 - 1.0e-6
    result = evaluate_basecolor_metrics(metrics)
    assert not result["eligible_for_rtx_development"]
    assert result["failed_checks"] == ["local_patch_contrast"]


def test_low_global_range_and_color_count_report_both_failures() -> None:
    metrics = _metrics()
    metrics["p01_p99_luminance_range"] = 0.0
    metrics["quantized_color_count_5bit"] = 1
    result = evaluate_basecolor_metrics(metrics)
    assert not result["eligible_for_rtx_development"]
    assert result["failed_checks"] == ["global_luminance_range", "global_color_count"]
