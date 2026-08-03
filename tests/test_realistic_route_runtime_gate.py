from __future__ import annotations

from kino_vla.eval.realistic_route_runtime_gate import audit_route_lane_runtime


def _row(*, lateral: float, inside: bool, load: bool) -> dict:
    return {
        "route_lateral_offset_m": lateral,
        "in_operator_region": inside,
        "operator_region_foot_contact": {"any_load_bearing_in_region": load},
    }


def test_nominal_requires_sustained_region_coverage() -> None:
    rows = [_row(lateral=0.1, inside=True, load=True) for _ in range(39)]
    result = audit_route_lane_runtime(
        rows,
        lane="nominal",
        maximum_route_deviation_m=0.3,
        minimum_nominal_region_samples=40,
    )
    assert result["passed"] is False
    rows.append(_row(lateral=0.1, inside=True, load=True))
    assert audit_route_lane_runtime(
        rows,
        lane="nominal",
        maximum_route_deviation_m=0.3,
        minimum_nominal_region_samples=40,
    )["passed"] is True


def test_anomaly_allows_early_failure_after_measured_contact() -> None:
    rows = [_row(lateral=0.1, inside=False, load=False)] * 5 + [
        _row(lateral=0.12, inside=True, load=True)
    ]
    result = audit_route_lane_runtime(
        rows,
        lane="anomaly",
        maximum_route_deviation_m=0.3,
        minimum_nominal_region_samples=40,
    )
    assert result["passed"] is True
    assert result["measurements"]["operator_region_sample_count"] == 1


def test_anomaly_must_have_load_bearing_region_contact() -> None:
    result = audit_route_lane_runtime(
        [_row(lateral=0.1, inside=True, load=False)],
        lane="anomaly",
        maximum_route_deviation_m=0.3,
        minimum_nominal_region_samples=40,
    )
    assert result["passed"] is False


def test_tracking_budget_applies_to_both_lanes() -> None:
    result = audit_route_lane_runtime(
        [_row(lateral=0.31, inside=True, load=True)],
        lane="anomaly",
        maximum_route_deviation_m=0.3,
        minimum_nominal_region_samples=40,
    )
    assert result["checks"]["full_lane_tracking_within_admitted_budget"] is False
