from kino_vla.eval.embodiedgen_motion_view_contract_v1 import (
    NOMINAL_STANDING_BASE_HEIGHT_M,
    PROGRESS_M,
    evaluate_motion_views,
)


def test_proxy_base_height_is_physical_go2_scale():
    assert 0.35 <= NOMINAL_STANDING_BASE_HEIGHT_M <= 0.50


def _records(std: float = 0.2, colors: int = 200):
    return [
        {
            "progress_m": progress,
            "metrics": {
                "std_luminance": std,
                "black_fraction": 0.0,
                "quantized_color_count_5bit": colors,
            },
        }
        for progress in PROGRESS_M
    ]


def test_motion_views_accept_existing_go2_thresholds():
    assert all(evaluate_motion_views(_records()).values())


def test_motion_views_reject_low_colour_wall():
    checks = evaluate_motion_views(_records(std=0.05, colors=54))
    assert checks["three_frozen_progress_views"]
    assert not checks["all_motion_views_non_degenerate"]


def test_motion_views_reject_non_frozen_progress_order():
    records = _records()
    records[-1]["progress_m"] = 0.6
    assert not evaluate_motion_views(records)["three_frozen_progress_views"]
