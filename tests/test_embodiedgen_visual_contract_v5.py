from __future__ import annotations

from kino_vla.eval.embodiedgen_visual_contract_v5 import (
    evaluate_phase_aware_scene_views,
)


def _record(name: str, kind: str, *, context: bool) -> dict:
    metrics = {
        "mean_luminance": 0.50,
        "std_luminance": 0.08 if context else 0.019,
        "p01_luminance": 0.10,
        "p99_luminance": 0.90,
        "black_fraction": 0.0,
        "white_fraction": 0.05,
        "quantized_color_count_5bit": 128,
    }
    return {"name": name, "kind": kind, "metrics": metrics}


def _records() -> list[dict]:
    return [
        _record("entry", "canonical_review", context=True),
        _record("middle", "canonical_review", context=True),
        _record("reverse", "canonical_review", context=True),
        _record("go2", "go2_front_height_proxy", context=True),
        _record("pitch45", "go2_body_pitch_stress", context=False),
        _record("pitch70", "go2_body_pitch_stress", context=False),
        _record("pitch80", "go2_body_pitch_stress", context=False),
    ]


def test_phase_aware_scene_views_accept_informative_context_and_valid_sensor_frames() -> None:
    checks = evaluate_phase_aware_scene_views(
        _records(), {"a__b": 0.1}, [0.01, 1000.0]
    )
    assert all(checks.values())


def test_phase_aware_scene_views_reject_black_pitch_frame() -> None:
    records = _records()
    records[-1]["metrics"].update(
        mean_luminance=0.0,
        std_luminance=0.0,
        black_fraction=1.0,
        quantized_color_count_5bit=1,
    )
    checks = evaluate_phase_aware_scene_views(records, {"a__b": 0.1}, [0.01, 1000.0])
    assert checks["body_pitch_frames_sensor_valid"] is False


def test_phase_aware_scene_views_reject_default_near_clip() -> None:
    checks = evaluate_phase_aware_scene_views(
        _records(), {"a__b": 0.1}, [1.0, 1000000.0]
    )
    assert checks["near_clip_calibrated_for_close_ground"] is False
