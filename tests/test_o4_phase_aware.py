from __future__ import annotations

from kino_vla.eval.o4_phase_aware import adjudicate_phase_aware_visuals


def _frame(step: int, *, mean: float = 0.1, std: float = 0.1, black: float = 0.1, colors: int = 100):
    return {
        "step": step,
        "metrics": {
            "mean_luminance": mean,
            "std_luminance": std,
            "black_fraction": black,
            "quantized_color_count_5bit": colors,
        },
    }


def test_real_low_texture_terminal_frame_is_sensor_valid() -> None:
    nominal = [_frame(step) for step in (-1, 0, 10, 20, 30, 40, 50)]
    anomaly = [_frame(step) for step in (-1, 0, 10, 20, 30, 40)]
    anomaly.extend([_frame(50, mean=0.03, std=0.01, black=0.7, colors=14)])
    result = adjudicate_phase_aware_visuals(
        nominal,
        anomaly,
        first_attachment_step=42,
        anomaly_fell=True,
        first_fall_step=50,
    )
    assert result["passed"]


def test_pure_black_terminal_frame_is_rejected() -> None:
    nominal = [_frame(step) for step in (-1, 0, 10)]
    anomaly = [_frame(-1), _frame(0), _frame(10, mean=0.0, std=0.0, black=1.0, colors=1)]
    result = adjudicate_phase_aware_visuals(
        nominal,
        anomaly,
        first_attachment_step=0,
        anomaly_fell=True,
        first_fall_step=10,
    )
    assert not result["passed"]
    assert not result["checks"]["all_captured_frames_sensor_valid"]
