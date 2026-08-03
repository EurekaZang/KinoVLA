from __future__ import annotations

import numpy as np

from kino_vla.eval.visual_shell_novel_view import (
    evaluate_rendered_view,
    mean_pairwise_rgb_l1,
)


def _thresholds() -> dict[str, float | int]:
    return {
        "minimum_foreground_fraction": 0.7,
        "minimum_luminance_std": 10.0,
        "minimum_5bit_colors": 8,
        "maximum_black_fraction": 0.35,
        "maximum_white_fraction": 0.35,
        "minimum_mean_pairwise_rgb_l1": 0.02,
    }


def test_rendered_view_gate_passes_complex_foreground() -> None:
    values = np.arange(64, dtype=np.uint8).reshape(8, 8)
    rgb = np.stack((values * 3, values * 2, values), axis=-1)
    mask = np.ones((8, 8), dtype=bool)

    result = evaluate_rendered_view(
        rgb,
        mask,
        expected_width=8,
        expected_height=8,
        thresholds=_thresholds(),
    )

    assert result["passed"] is True


def test_rendered_view_gate_rejects_empty_white_render() -> None:
    rgb = np.full((8, 8, 3), 255, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)

    result = evaluate_rendered_view(
        rgb,
        mask,
        expected_width=8,
        expected_height=8,
        thresholds=_thresholds(),
    )

    assert result["passed"] is False
    assert result["checks"]["foreground_coverage"] is False


def test_pairwise_l1_detects_distinct_views() -> None:
    dark = np.zeros((4, 4, 3), dtype=np.uint8)
    bright = np.full((4, 4, 3), 255, dtype=np.uint8)

    assert mean_pairwise_rgb_l1([dark, bright]) == 1.0
