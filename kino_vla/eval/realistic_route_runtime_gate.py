"""Fail-closed runtime gates shared by realistic route-lane collectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def audit_route_lane_runtime(
    rows: Sequence[Mapping[str, Any]],
    *,
    lane: str,
    maximum_route_deviation_m: float,
    minimum_nominal_region_samples: int,
) -> dict[str, Any]:
    """Gate full-lane tracking while allowing a causally early anomaly failure."""
    if lane not in {"nominal", "anomaly"}:
        raise ValueError(f"unsupported lane: {lane}")
    if maximum_route_deviation_m <= 0.0:
        raise ValueError("maximum_route_deviation_m must be positive")
    if minimum_nominal_region_samples < 1:
        raise ValueError("minimum_nominal_region_samples must be positive")
    region_rows = [row for row in rows if bool(row.get("in_operator_region"))]
    load_bearing_region_rows = [
        row
        for row in rows
        if row.get("operator_region_foot_contact", {}).get(
            "any_load_bearing_in_region"
        )
        is True
    ]
    maximum_absolute_lateral_offset = max(
        (abs(float(row["route_lateral_offset_m"])) for row in rows),
        default=float("inf"),
    )
    checks = {
        "operator_region_reached": bool(region_rows),
        "full_lane_tracking_within_admitted_budget": maximum_absolute_lateral_offset
        <= maximum_route_deviation_m + 1.0e-9,
        "lane_appropriate_operator_region_coverage": (
            len(region_rows) >= minimum_nominal_region_samples
            if lane == "nominal"
            else bool(load_bearing_region_rows)
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": {
            "maximum_absolute_route_lateral_offset_m": maximum_absolute_lateral_offset,
            "operator_region_sample_count": len(region_rows),
            "load_bearing_operator_region_sample_count": len(load_bearing_region_rows),
        },
        "contract": {
            "lane": lane,
            "maximum_route_deviation_m": maximum_route_deviation_m,
            "minimum_nominal_region_samples": minimum_nominal_region_samples,
            "anomaly_coverage_rule": "at_least_one_load_bearing_region_contact",
        },
    }
