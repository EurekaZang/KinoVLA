"""T3 post-interaction temporal contract for realistic C2 v5."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from kino_vla.eval.c2_temporal import _distance_to_region
from kino_vla.eval.realistic_multimodal import proprio_summary


FEATURE_COUNT = 19
FOOTPRINT_MARGIN_M = 0.35
POST_ENCOUNTER_DELAY_S = 0.30
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


def invariant_signals(window: np.ndarray) -> np.ndarray:
    """Map 19 channels to eight rotation-invariant physical signals."""

    value = np.asarray(window, dtype=np.float32)
    if (
        value.ndim != 2
        or value.shape[0] != WINDOW_SAMPLES
        or value.shape[1] != FEATURE_COUNT
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"unexpected proprio window: {value.shape}")
    return np.column_stack(
        [
            np.linalg.norm(value[:, 3:6], axis=1),
            np.linalg.norm(value[:, 6:9], axis=1),
            np.linalg.norm(value[:, 12:14], axis=1),
            value[:, 14],
            value[:, 15],
            value[:, 16],
            value[:, 17],
            value[:, 18],
        ]
    ).astype(np.float32)


def invariant_summary(window: np.ndarray) -> np.ndarray:
    return proprio_summary(invariant_signals(window))


def geometry_aligned_invariant_summary(
    episode_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Summarize a fixed post-interaction window without outcome alignment."""

    manifest = _json(episode_dir / "manifest.json")
    region = manifest["geometry_readback"]["operator_region"]
    telemetry = _jsonl(
        episode_dir
        / manifest["artifacts"]["telemetry"]["path"]
    )
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
        raise ValueError(f"no footprint encounter: {episode_dir}")
    encounter_time_s = encounter_times[0]
    decision_time_s = encounter_time_s + POST_ENCOUNTER_DELAY_S
    artifact = manifest["artifacts"]["proprio"]
    with np.load(
        episode_dir / artifact["path"], allow_pickle=False
    ) as archive:
        features = np.asarray(
            archive[artifact["features_key"]],
            dtype=np.float32,
        )
        timestamps = np.asarray(
            archive[artifact["timestamps_key"]],
            dtype=np.float64,
        )
    candidates = np.flatnonzero(
        timestamps <= decision_time_s + MAX_END_SKEW_S
    )
    if len(candidates) < WINDOW_SAMPLES:
        raise ValueError(f"short post-interaction window: {episode_dir}")
    selected = candidates[-WINDOW_SAMPLES:]
    end_skew_s = abs(
        float(timestamps[selected[-1]]) - decision_time_s
    )
    if end_skew_s > MAX_END_SKEW_S:
        raise ValueError(f"post-interaction end skew: {episode_dir}")
    return invariant_summary(features[selected]), {
        "encounter_time_s": encounter_time_s,
        "decision_time_s": decision_time_s,
        "window_start_s": float(timestamps[selected[0]]),
        "window_end_s": float(timestamps[selected[-1]]),
        "end_skew_s": end_skew_s,
    }
