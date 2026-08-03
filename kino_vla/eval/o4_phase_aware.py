"""Phase-aware visual validity for O4 counterfactual sequences.

Scene admission retains its strict multi-view appearance-complexity gate.  During
an O4 episode, however, a real body-fixed camera can legitimately end up close to
a dark or homogeneous floor as the robot pitches or falls.  This module separates
informative pre-effect context from post-attachment sensor integrity so physical
failure is not censored merely because it changes camera content.
"""

from __future__ import annotations

from typing import Any


CONTEXT_MIN_STD_LUMINANCE = 0.025
CONTEXT_MAX_BLACK_FRACTION = 0.90
CONTEXT_MIN_QUANTIZED_COLORS = 32
SENSOR_MIN_MEAN_LUMINANCE = 0.005
SENSOR_MIN_STD_LUMINANCE = 0.005
SENSOR_MAX_BLACK_FRACTION = 0.995
SENSOR_MIN_QUANTIZED_COLORS = 8


def context_informative(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics["std_luminance"]) >= CONTEXT_MIN_STD_LUMINANCE
        and float(metrics["black_fraction"]) < CONTEXT_MAX_BLACK_FRACTION
        and int(metrics["quantized_color_count_5bit"]) >= CONTEXT_MIN_QUANTIZED_COLORS
    )


def sensor_valid(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics["mean_luminance"]) >= SENSOR_MIN_MEAN_LUMINANCE
        and float(metrics["std_luminance"]) >= SENSOR_MIN_STD_LUMINANCE
        and float(metrics["black_fraction"]) < SENSOR_MAX_BLACK_FRACTION
        and int(metrics["quantized_color_count_5bit"]) >= SENSOR_MIN_QUANTIZED_COLORS
    )


def adjudicate_phase_aware_visuals(
    nominal_frames: list[dict[str, Any]],
    anomaly_frames: list[dict[str, Any]],
    *,
    first_attachment_step: int,
    anomaly_fell: bool,
    first_fall_step: int | None,
) -> dict[str, Any]:
    if not nominal_frames or not anomaly_frames:
        raise ValueError("both counterfactual takes require at least one RGB frame")
    if first_attachment_step < 0:
        raise ValueError("first attachment step must be non-negative")
    pre_nominal = [frame for frame in nominal_frames if int(frame["step"]) <= first_attachment_step]
    pre_anomaly = [frame for frame in anomaly_frames if int(frame["step"]) <= first_attachment_step]
    terminal = anomaly_frames[-1]
    checks = {
        "pre_effect_context_present_in_both_takes": bool(pre_nominal) and bool(pre_anomaly),
        "pre_effect_context_informative": all(
            context_informative(frame["metrics"]) for frame in pre_nominal + pre_anomaly
        ),
        "all_captured_frames_sensor_valid": all(
            sensor_valid(frame["metrics"]) for frame in nominal_frames + anomaly_frames
        ),
        "registered_terminal_frame_sensor_valid": (
            not anomaly_fell
            or (
                first_fall_step is not None
                and int(terminal["step"]) == int(first_fall_step)
                and sensor_valid(terminal["metrics"])
            )
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "counts": {
            "nominal_frames": len(nominal_frames),
            "anomaly_frames": len(anomaly_frames),
            "nominal_pre_effect_frames": len(pre_nominal),
            "anomaly_pre_effect_frames": len(pre_anomaly),
            "nominal_sensor_valid_frames": sum(
                sensor_valid(frame["metrics"]) for frame in nominal_frames
            ),
            "anomaly_sensor_valid_frames": sum(
                sensor_valid(frame["metrics"]) for frame in anomaly_frames
            ),
        },
        "thresholds": {
            "context_min_std_luminance": CONTEXT_MIN_STD_LUMINANCE,
            "context_max_black_fraction": CONTEXT_MAX_BLACK_FRACTION,
            "context_min_quantized_colors_5bit": CONTEXT_MIN_QUANTIZED_COLORS,
            "sensor_min_mean_luminance": SENSOR_MIN_MEAN_LUMINANCE,
            "sensor_min_std_luminance": SENSOR_MIN_STD_LUMINANCE,
            "sensor_max_black_fraction": SENSOR_MAX_BLACK_FRACTION,
            "sensor_min_quantized_colors_5bit": SENSOR_MIN_QUANTIZED_COLORS,
        },
    }
