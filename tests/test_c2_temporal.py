from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kino_vla.eval.c2_temporal import (
    FEATURE_COUNT,
    PROPRIO_SIGNAL_INDICES,
    SUMMARY_BLOCKS,
    geometry_aligned_proprio_summary,
    project_proprio,
)
from kino_vla.eval.c2_temporal_v5 import (
    geometry_aligned_invariant_summary,
    invariant_signals,
)
from kino_vla.eval.realistic_multimodal import proprio_summary


def test_project_proprio_keeps_only_scene_invariant_signals() -> None:
    values = np.arange(
        2 * FEATURE_COUNT * SUMMARY_BLOCKS,
        dtype=np.float32,
    ).reshape(2, -1)
    projected = project_proprio(values)
    expected_indices = [
        block * FEATURE_COUNT + signal
        for block in range(SUMMARY_BLOCKS)
        for signal in PROPRIO_SIGNAL_INDICES
    ]
    assert projected.shape == (2, 130)
    assert np.array_equal(projected, values[:, expected_indices])


def test_geometry_alignment_uses_fixed_observable_event_window(
    tmp_path: Path,
) -> None:
    timestamps = np.arange(51, dtype=np.float64) * 0.02
    features = np.arange(
        len(timestamps) * FEATURE_COUNT,
        dtype=np.float32,
    ).reshape(len(timestamps), FEATURE_COUNT)
    np.savez_compressed(
        tmp_path / "proprio.npz",
        features=features,
        timestamps_s=timestamps,
    )
    telemetry = [
        {
            "timestamp_s": float(timestamp),
            "position_xy_m": [-1.0, 0.0]
            if index < 30
            else [0.5, 0.0],
        }
        for index, timestamp in enumerate(timestamps)
    ]
    (tmp_path / "privileged.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in telemetry),
        encoding="utf-8",
    )
    manifest = {
        "geometry_readback": {
            "operator_region": {
                "cx": 1.0,
                "cy": 0.0,
                "hx": 0.1,
                "hy": 0.1,
            }
        },
        "artifacts": {
            "proprio": {
                "path": "proprio.npz",
                "features_key": "features",
                "timestamps_key": "timestamps_s",
            },
            "telemetry": {"path": "privileged.jsonl"},
        },
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    summary, audit = geometry_aligned_proprio_summary(tmp_path)

    assert np.array_equal(summary, proprio_summary(features[21:42]))
    assert audit["encounter_time_s"] == timestamps[30]
    assert audit["decision_time_s"] == timestamps[30] + 0.2
    assert audit["window_start_s"] == timestamps[21]
    assert audit["window_end_s"] == timestamps[41]
    assert audit["end_skew_s"] <= 0.021


def test_v5_invariant_signals_remove_axis_and_heading_sign() -> None:
    first = np.zeros((21, FEATURE_COUNT), dtype=np.float32)
    second = np.zeros_like(first)
    ramp = np.linspace(0.1, 2.1, 21, dtype=np.float32)
    first[:, 3:6] = np.column_stack(
        [ramp, 2.0 * ramp, -3.0 * ramp]
    )
    second[:, 3:6] = np.column_stack(
        [-3.0 * ramp, -ramp, 2.0 * ramp]
    )
    first[:, 6:9] = np.column_stack(
        [2.0 * ramp, -ramp, ramp]
    )
    second[:, 6:9] = np.column_stack(
        [-ramp, -2.0 * ramp, -ramp]
    )
    first[:, 12:14] = np.column_stack([ramp, -2.0 * ramp])
    second[:, 12:14] = np.column_stack([-2.0 * ramp, -ramp])
    first[:, 14:] = second[:, 14:] = np.column_stack(
        [ramp, ramp + 1, ramp + 2, ramp + 3, ramp + 4]
    )

    assert np.allclose(
        invariant_signals(first), invariant_signals(second)
    )


def test_v5_geometry_alignment_reads_post_interaction_window(
    tmp_path: Path,
) -> None:
    timestamps = np.arange(61, dtype=np.float64) * 0.02
    features = np.arange(
        len(timestamps) * FEATURE_COUNT,
        dtype=np.float32,
    ).reshape(len(timestamps), FEATURE_COUNT)
    np.savez_compressed(
        tmp_path / "proprio.npz",
        features=features,
        timestamps_s=timestamps,
    )
    telemetry = [
        {
            "timestamp_s": float(timestamp),
            "position_xy_m": [-1.0, 0.0]
            if index < 30
            else [0.56, 0.0],
        }
        for index, timestamp in enumerate(timestamps)
    ]
    (tmp_path / "privileged.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in telemetry),
        encoding="utf-8",
    )
    manifest = {
        "geometry_readback": {
            "operator_region": {
                "cx": 1.0,
                "cy": 0.0,
                "hx": 0.1,
                "hy": 0.1,
            }
        },
        "artifacts": {
            "proprio": {
                "path": "proprio.npz",
                "features_key": "features",
                "timestamps_key": "timestamps_s",
            },
            "telemetry": {"path": "privileged.jsonl"},
        },
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    summary, audit = geometry_aligned_invariant_summary(tmp_path)

    expected = proprio_summary(invariant_signals(features[26:47]))
    assert np.array_equal(summary, expected)
    assert summary.shape == (80,)
    assert audit["encounter_time_s"] == timestamps[30]
    assert audit["decision_time_s"] == timestamps[30] + 0.3
    assert audit["window_start_s"] == timestamps[26]
    assert audit["window_end_s"] == timestamps[46]
    assert audit["end_skew_s"] <= 0.021
