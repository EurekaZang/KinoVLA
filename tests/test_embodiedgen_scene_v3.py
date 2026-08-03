from __future__ import annotations

import pytest

from kino_vla.sim.embodiedgen_scene_v3 import (
    evaluate_pitch_aware_metrics,
    pitch_aware_camera_views,
)


def test_pitch_aware_views_cover_three_downward_attitudes() -> None:
    route = [[-1.1, 0.0], [-0.55, 0.0], [0.0, 0.0], [0.7, 0.0], [1.35, 0.0]]
    views = pitch_aware_camera_views(route, 0.0)
    assert len(views) == 7
    stress = [view for view in views if view["kind"] == "go2_body_pitch_stress"]
    assert [view["nominal_pitch_down_deg"] for view in stress] == pytest.approx(
        [45.0, 70.0, 80.0], abs=0.2
    )
    assert all(view["eye_xyz_m"][2] == pytest.approx(0.42) for view in stress)


def test_pitch_aware_gate_rejects_one_degenerate_downward_view() -> None:
    route = [[-1.1, 0.0], [-0.55, 0.0], [0.0, 0.0], [0.7, 0.0], [1.35, 0.0]]
    views = pitch_aware_camera_views(route, 0.0)
    records = []
    for view in views:
        records.append(
            {
                **view,
                "metrics": {
                    "std_luminance": 0.10,
                    "p01_luminance": 0.05,
                    "p99_luminance": 0.80,
                    "black_fraction": 0.01,
                    "white_fraction": 0.01,
                    "quantized_color_count_5bit": 200,
                },
            }
        )
    records[-1]["metrics"]["std_luminance"] = 0.01
    checks = evaluate_pitch_aware_metrics(records, {"canonical_pair": 0.1})
    assert checks["three_body_pitch_stress_views"]
    assert not checks["non_degenerate_luminance"]
