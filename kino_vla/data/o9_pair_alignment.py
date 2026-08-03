"""Matched-horizon physical audit for direct high-centering pairs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kino_vla.data.o9_semantics import evaluate_o9_high_centering


MINIMUM_DECISION_TIME_S = 0.42
MAXIMUM_TIMESTAMP_SKEW_S = 0.021
MAXIMUM_NOMINAL_ROUTE_DEVIATION_M = 0.55


def read_privileged_rows(manifest_path: Path) -> list[dict[str, Any]]:
    path = manifest_path.parent / "privileged.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise ValueError(f"empty privileged telemetry: {path}")
    timestamps = [float(row["timestamp_s"]) for row in rows]
    if not all(math.isfinite(value) for value in timestamps):
        raise ValueError(f"nonfinite privileged timestamp: {path}")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f"nonmonotonic privileged timestamps: {path}")
    return rows


def first_direct_event_time(rows: list[dict[str, Any]]) -> float:
    for row in rows:
        telemetry = row.get("operator")
        if isinstance(telemetry, Mapping) and evaluate_o9_high_centering(dict(telemetry))["passed"]:
            return float(row["timestamp_s"])
    raise ValueError("no strict direct O9 event in anomaly telemetry")


def audit_matched_horizon(
    anomaly_manifest_path: Path,
    nominal_manifest_path: Path,
) -> dict[str, Any]:
    anomaly_rows = read_privileged_rows(anomaly_manifest_path)
    nominal_rows = read_privileged_rows(nominal_manifest_path)
    event_time = first_direct_event_time(anomaly_rows)
    decision_time = max(MINIMUM_DECISION_TIME_S, event_time)
    nominal_prefix = [
        row for row in nominal_rows if float(row["timestamp_s"]) <= decision_time + 1.0e-9
    ]
    anomaly_prefix = [
        row for row in anomaly_rows if float(row["timestamp_s"]) <= decision_time + 1.0e-9
    ]
    nominal_last = float(nominal_prefix[-1]["timestamp_s"]) if nominal_prefix else float("-inf")
    anomaly_last = float(anomaly_prefix[-1]["timestamp_s"]) if anomaly_prefix else float("-inf")
    nominal_deviation = max(
        (abs(float(row["route_lateral_offset_m"])) for row in nominal_prefix),
        default=float("inf"),
    )
    checks = {
        "strict_direct_event_exists": True,
        "anomaly_covers_decision_time": decision_time - anomaly_last <= MAXIMUM_TIMESTAMP_SKEW_S,
        "nominal_covers_decision_time": decision_time - nominal_last <= MAXIMUM_TIMESTAMP_SKEW_S,
        "nominal_not_fallen_by_decision_time": bool(nominal_prefix)
        and not any(bool(row.get("fallen")) for row in nominal_prefix),
        "nominal_within_route_by_decision_time": nominal_deviation
        <= MAXIMUM_NOMINAL_ROUTE_DEVIATION_M,
    }
    return {
        "schema_version": "kinofail.o9-matched-diagnostic-horizon.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "event_time_s": event_time,
        "decision_time_s": decision_time,
        "minimum_decision_time_s": MINIMUM_DECISION_TIME_S,
        "maximum_timestamp_skew_s": MAXIMUM_TIMESTAMP_SKEW_S,
        "nominal_last_prefix_timestamp_s": nominal_last,
        "anomaly_last_prefix_timestamp_s": anomaly_last,
        "nominal_prefix_samples": len(nominal_prefix),
        "anomaly_prefix_samples": len(anomaly_prefix),
        "nominal_max_route_deviation_before_decision_m": nominal_deviation,
        "nominal_max_tilt_before_decision_rad": max(
            (float(row["tilt_rad"]) for row in nominal_prefix), default=float("inf")
        ),
        "nominal_min_base_height_before_decision_m": min(
            (float(row["base_height_m"]) for row in nominal_prefix), default=float("-inf")
        ),
        "post_decision_outcome_not_used_for_attribution_admission": True,
    }
