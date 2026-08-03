"""Frozen observable temporal contract for realistic C2 v4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from kino_vla.eval.realistic_multimodal import proprio_summary


FEATURE_COUNT = 19
SUMMARY_BLOCKS = 10
PROPRIO_SIGNAL_INDICES = (
    3,
    4,
    5,
    6,
    7,
    8,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
)
FOOTPRINT_MARGIN_M = 0.40
POST_ENCOUNTER_DELAY_S = 0.20
WINDOW_S = 0.50
WINDOW_SAMPLES = 21
MAX_END_SKEW_S = 0.021


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def project_proprio(values: np.ndarray) -> np.ndarray:
    """Remove static gravity and absolute pose from 10×19 summaries."""

    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != FEATURE_COUNT * SUMMARY_BLOCKS:
        raise ValueError(f"unexpected proprio summary shape: {array.shape}")
    indices = [
        block * FEATURE_COUNT + signal
        for block in range(SUMMARY_BLOCKS)
        for signal in PROPRIO_SIGNAL_INDICES
    ]
    return np.asarray(array[:, indices], dtype=np.float32)


def _distance_to_region(
    x: float,
    y: float,
    region: dict[str, Any],
) -> float:
    dx = max(abs(x - float(region["cx"])) - float(region["hx"]), 0.0)
    dy = max(abs(y - float(region["cy"])) - float(region["hy"]), 0.0)
    return float(np.hypot(dx, dy))


def geometry_aligned_proprio_summary(
    episode_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the fixed v4 window without using outcome or semantic labels."""

    manifest = _json(episode_dir / "manifest.json")
    region = manifest["geometry_readback"]["operator_region"]
    telemetry_path = (
        episode_dir / manifest["artifacts"]["telemetry"]["path"]
    )
    telemetry = _jsonl(telemetry_path)
    encounter_times = [
        float(row["timestamp_s"])
        for row in telemetry
        if _distance_to_region(
            float(row["position_xy_m"][0]),
            float(row["position_xy_m"][1]),
            region,
        )
        <= FOOTPRINT_MARGIN_M
    ]
    if not encounter_times:
        raise ValueError(f"no geometry encounter: {episode_dir}")
    encounter_time_s = encounter_times[0]
    decision_time_s = encounter_time_s + POST_ENCOUNTER_DELAY_S
    artifact = manifest["artifacts"]["proprio"]
    with np.load(
        episode_dir / artifact["path"], allow_pickle=False
    ) as archive:
        features = np.asarray(
            archive[artifact["features_key"]], dtype=np.float32
        )
        timestamps = np.asarray(
            archive[artifact["timestamps_key"]], dtype=np.float64
        )
    candidates = np.flatnonzero(
        (timestamps > decision_time_s - WINDOW_S - 1.0e-9)
        & (timestamps <= decision_time_s + MAX_END_SKEW_S)
    )
    if len(candidates) < WINDOW_SAMPLES:
        raise ValueError(f"short aligned window: {episode_dir}")
    selected = candidates[-WINDOW_SAMPLES:]
    end_skew_s = abs(float(timestamps[selected[-1]]) - decision_time_s)
    if end_skew_s > MAX_END_SKEW_S:
        raise ValueError(f"aligned-window end skew: {episode_dir}")
    summary = proprio_summary(features[selected])
    return summary, {
        "encounter_time_s": encounter_time_s,
        "decision_time_s": decision_time_s,
        "window_start_s": float(timestamps[selected[0]]),
        "window_end_s": float(timestamps[selected[-1]]),
        "end_skew_s": end_skew_s,
    }
