from __future__ import annotations

import numpy as np
import pytest

from kino_vla.sim.depth_pipeline import DepthFaultRegion, apply_timestamped_depth_faults


def _capture() -> dict[str, object]:
    return {
        "rgb": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
        "depth": np.array([[1.0, 2.0, 3.0, np.inf], [4.0, 5.0, 6.0, 7.0]]),
        "seg": np.array([[0, 7, 7, 7], [0, 0, 7, 0]]),
        "id_to_labels": {"0": {"class": "BACKGROUND"}, "7": {"class": "o7_patch"}},
    }


def test_depth_fault_is_region_local_timestamped_and_preserves_rgb() -> None:
    source = _capture()
    result = apply_timestamped_depth_faults(
        source,
        [DepthFaultRegion("o7_patch", 0.18)],
        capture_time_s=1.25,
        frame_id=9,
        read_time_s=1.29,
    )
    expected = np.array([[1.0, 2.18, 3.18, np.inf], [4.0, 5.0, 6.18, 7.0]])
    np.testing.assert_allclose(result["depth"], expected)
    np.testing.assert_array_equal(result["rgb"], source["rgb"])
    telemetry = result["depth_fault_telemetry"]
    assert telemetry["affected_pixels"] == 3
    assert telemetry["faults"][0]["mean_applied_bias_m"] == pytest.approx(0.18)
    assert telemetry["camera_timestamp_s"] == telemetry["depth_source_timestamp_s"] == 1.25
    assert telemetry["effective_depth_age_s"] == pytest.approx(0.04)
    assert telemetry["frame_id"] == 9
    assert telemetry["input_rgb_sha256"] == telemetry["output_rgb_sha256"]


def test_zero_bias_keeps_depth_hash_identical() -> None:
    result = apply_timestamped_depth_faults(
        _capture(),
        [DepthFaultRegion("o7_patch", 0.0)],
        capture_time_s=0.0,
        frame_id=0,
    )
    telemetry = result["depth_fault_telemetry"]
    assert telemetry["raw_depth_sha256"] == telemetry["output_depth_sha256"]
