from __future__ import annotations

import numpy as np

from scripts.isaac_collect_embodiedgen_o4_pair import (
    _command_phase_at,
    _paired_consequence_metrics,
    _registered_terminal_outcome,
)


def _row(
    step: int,
    progress: float,
    *,
    attached: bool = False,
    tilt: float = 0.02,
    height: float = 0.40,
) -> dict[str, object]:
    return {
        "step": step,
        "time_s": step * 0.5,
        "phase": "forward",
        "progress_m": progress,
        "tilt_rad": tilt,
        "base_height_m": height,
        "adhesion": {
            "feet": [{"event": "attached" if attached else "none"}],
        },
    }


def test_paired_metrics_start_at_first_attachment_and_report_two_channels() -> None:
    nominal = [_row(step, step * 0.25) for step in range(6)]
    anomaly = [
        _row(0, 0.00),
        _row(1, 0.25),
        _row(2, 0.50, attached=True),
        _row(3, 0.68, tilt=0.10, height=0.37),
        _row(4, 0.84, tilt=0.18, height=0.34),
        _row(5, 1.00, tilt=0.14, height=0.35),
    ]

    metrics = _paired_consequence_metrics(nominal, anomaly)

    assert metrics["first_attachment_step"] == 2
    assert metrics["window_duration_s"] == 1.5
    assert abs(metrics["progress_gain_lag_m"] - 0.25) < 1.0e-9
    assert abs(metrics["relative_forward_speed_suppression"] - (1.0 / 3.0)) < 1.0e-9
    assert abs(metrics["max_tilt_increase_rad"] - 0.16) < 1.0e-9
    assert abs(metrics["min_base_height_drop_m"] - 0.06) < 1.0e-9


def test_paired_metrics_ignore_pre_attachment_progress_difference() -> None:
    nominal = [_row(step, step * 0.25) for step in range(5)]
    anomaly = [
        _row(0, -0.10),
        _row(1, 0.15),
        _row(2, 0.40, attached=True),
        _row(3, 0.65),
        _row(4, 0.90),
    ]

    metrics = _paired_consequence_metrics(nominal, anomaly)

    assert abs(metrics["progress_gain_lag_m"]) < 1.0e-9
    assert abs(metrics["relative_forward_speed_suppression"]) < 1.0e-9


def test_paired_metrics_report_missing_attachment_without_crashing() -> None:
    nominal = [_row(step, step * 0.25) for step in range(5)]
    anomaly = [_row(step, step * 0.25) for step in range(5)]

    metrics = _paired_consequence_metrics(nominal, anomaly)

    assert metrics["available"] is False
    assert metrics["failure_reason"] == "no_attachment_event"
    assert metrics["first_attachment_step"] is None


def test_release_recovery_retreat_schedule_is_explicit() -> None:
    kwargs = {
        "forward_s": 2.3,
        "peel_pulse_s": 0.1,
        "recovery_s": 1.0,
        "reverse_s": 3.0,
    }

    forward, forward_phase = _command_phase_at(2.29, **kwargs)
    peel, peel_phase = _command_phase_at(2.30, **kwargs)
    recovery, recovery_phase = _command_phase_at(2.40, **kwargs)
    retreat, retreat_phase = _command_phase_at(3.40, **kwargs)
    stop, stop_phase = _command_phase_at(6.40, **kwargs)

    assert forward_phase == "forward" and np.allclose(forward, [0.5, 0.0, 0.0])
    assert peel_phase == "peel_release" and np.allclose(peel, [-0.3, 0.0, 0.0])
    assert recovery_phase == "post_peel_recovery" and np.allclose(recovery, 0.0)
    assert retreat_phase == "reverse_retreat" and np.allclose(retreat, [-0.3, 0.0, 0.0])
    assert stop_phase == "stop" and np.allclose(stop, 0.0)


def _summary(
    *,
    steps: int,
    fell: bool,
    fall_step: int | None = None,
    fall_phase: str | None = None,
    peeled_step: int | None = None,
    terminal_frame: bool = False,
) -> dict[str, object]:
    events = []
    if peeled_step is not None:
        events.append({"step": peeled_step, "phase": "peel_release", "event": "peeled"})
    return {
        "steps": steps,
        "fell": fell,
        "first_fall_step": fall_step,
        "first_fall_phase": fall_phase,
        "event_trace": events,
        "terminal_frame_at_fall": terminal_frame,
    }


def test_registered_terminal_outcome_accepts_complete_recovery() -> None:
    result = _registered_terminal_outcome(
        _summary(steps=320, fell=False),
        _summary(steps=320, fell=False),
        {"available": True, "window_last_step": 100},
        total_steps=320,
    )

    assert result["registered"] is True
    assert result["outcome_class"] == "recoverable_fixed_horizon"


def test_registered_terminal_outcome_accepts_ordered_o4_fall() -> None:
    result = _registered_terminal_outcome(
        _summary(steps=320, fell=False),
        _summary(
            steps=128,
            fell=True,
            fall_step=127,
            fall_phase="post_peel_recovery",
            peeled_step=107,
            terminal_frame=True,
        ),
        {"available": True, "window_last_step": 106},
        total_steps=320,
    )

    assert result["registered"] is True
    assert result["outcome_class"] == "o4_induced_terminal_fall"


def test_registered_terminal_outcome_rejects_fall_before_peel_or_without_frame() -> None:
    nominal = _summary(steps=320, fell=False)
    before_peel = _registered_terminal_outcome(
        nominal,
        _summary(
            steps=106,
            fell=True,
            fall_step=105,
            fall_phase="forward",
            peeled_step=None,
            terminal_frame=True,
        ),
        {"available": True, "window_last_step": 104},
        total_steps=320,
    )
    uncaptured = _registered_terminal_outcome(
        nominal,
        _summary(
            steps=128,
            fell=True,
            fall_step=127,
            fall_phase="post_peel_recovery",
            peeled_step=107,
            terminal_frame=False,
        ),
        {"available": True, "window_last_step": 106},
        total_steps=320,
    )

    assert before_peel["registered"] is False
    assert uncaptured["registered"] is False
