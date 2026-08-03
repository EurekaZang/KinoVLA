from __future__ import annotations

import numpy as np

from kino_vla.eval.hdri_preflight import evaluate_hdri_array


THRESHOLDS = {
    "maximum_negative_fraction": 0.0,
    "minimum_luminance_std": 0.01,
    "minimum_dynamic_range_p99_p01": 2.0,
    "maximum_seam_relative_to_mean": 0.2,
}


def test_valid_wrapped_hdri_passes() -> None:
    x = np.linspace(0.1, 2.0, 16, dtype=np.float32)
    luminance = np.tile(x[:, None], (1, 32))
    bgr = np.repeat(luminance[..., None], 3, axis=2)
    bgr[:, -1] = bgr[:, 0]
    result = evaluate_hdri_array(
        bgr, expected_width=32, expected_height=16, thresholds=THRESHOLDS
    )
    assert result["passed"] is True


def test_nonfinite_or_open_seam_hdri_fails() -> None:
    bgr = np.ones((16, 32, 3), dtype=np.float32)
    bgr[:, -1] = 10.0
    bgr[0, 0, 0] = np.nan
    result = evaluate_hdri_array(
        bgr, expected_width=32, expected_height=16, thresholds=THRESHOLDS
    )
    assert result["checks"]["all_values_finite"] is False
    assert result["checks"]["horizontal_seam"] is False
    assert result["passed"] is False
