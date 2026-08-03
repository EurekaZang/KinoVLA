"""Fail-closed semantic admission for O9 high-centering evidence.

High-centering is a chassis-support failure, not merely proximity to a ridge or
contact by the head/legs.  This module deliberately consumes only recorded
telemetry so the same contract can be applied during capture and in later
artifact audits.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


DEFAULT_THRESHOLDS = {
    "minimum_base_contact_force_n": 20.0,
    "minimum_consecutive_belly_contact_steps": 10,
    "minimum_belly_contact_duty_cycle": 0.10,
    "maximum_support_ratio": 0.75,
    "maximum_longitudinal_center_offset_m": 0.05,
}


def _body_force(
    values: Mapping[str, Any],
    *,
    names: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
) -> float:
    allowed = set(names) if names is not None else None
    excluded = set(exclude)
    candidates = [
        float(force)
        for name, force in values.items()
        if name not in excluded and (allowed is None or name in allowed)
    ]
    return max(candidates, default=0.0)


def evaluate_o9_high_centering(
    telemetry: Mapping[str, Any],
    *,
    robot_position_xy_m: Sequence[float] | None = None,
    route_axis: str = "x",
    thresholds: Mapping[str, float | int] | None = None,
) -> dict[str, Any]:
    """Evaluate whether recorded telemetry proves chassis-on-ridge beaching.

    A passing region must show sustained base contact and partial foot unloading.
    Head- or limb-only collision is explicitly rejected.  When a robot position
    is supplied, the chassis center must also overlap the ridge along the route
    axis; this prevents a pre-contact frame from passing on accumulated metadata.
    """

    limits = {**DEFAULT_THRESHOLDS, **dict(thresholds or {})}
    regions = list(telemetry.get("regions", []))
    region_results: list[dict[str, Any]] = []
    for index, region in enumerate(regions):
        body_forces = region.get("max_nonfoot_contact_by_body_n", {})
        if not isinstance(body_forces, Mapping):
            body_forces = {}
        base_force_n = max(
            float(region.get("max_belly_contact_force_n", 0.0)),
            float(body_forces.get("base", 0.0)),
        )
        head_force_n = _body_force(
            body_forces,
            names=("Head_lower", "Head_upper"),
        )
        nonbase_force_n = _body_force(body_forces, exclude=("base",))
        head_or_limb_only = (
            nonbase_force_n
            >= float(limits["minimum_base_contact_force_n"])
            and base_force_n
            < float(limits["minimum_base_contact_force_n"])
        )

        geometry_under_chassis = True
        longitudinal_offset_m: float | None = None
        if robot_position_xy_m is not None:
            rect = region.get("region", {})
            axis = 0 if route_axis == "x" else 1
            center_key = "cx" if axis == 0 else "cy"
            half_key = "hx" if axis == 0 else "hy"
            longitudinal_offset_m = abs(
                float(robot_position_xy_m[axis]) - float(rect.get(center_key, 0.0))
            )
            geometry_under_chassis = longitudinal_offset_m <= (
                float(rect.get(half_key, 0.0))
                + float(limits["maximum_longitudinal_center_offset_m"])
            )

        checks = {
            "telemetry_enabled": telemetry.get("enabled") is True,
            "base_contact_force_sufficient": (
                base_force_n >= float(limits["minimum_base_contact_force_n"])
            ),
            "belly_contact_sustained": (
                int(region.get("max_consecutive_belly_contact_steps", 0))
                >= int(limits["minimum_consecutive_belly_contact_steps"])
            ),
            "belly_contact_not_single_impulse": (
                float(region.get("belly_contact_duty_cycle", 0.0))
                >= float(limits["minimum_belly_contact_duty_cycle"])
            ),
            "partial_foot_unloading_measured": (
                float(region.get("min_measured_support", 1.0))
                <= float(limits["maximum_support_ratio"])
            ),
            "not_head_or_limb_only_collision": not head_or_limb_only,
            "ridge_under_chassis_center": geometry_under_chassis,
        }
        region_results.append(
            {
                "region_index": index,
                "passed": all(checks.values()),
                "checks": checks,
                "measurements": {
                    "base_contact_force_n": base_force_n,
                    "head_contact_force_n": head_force_n,
                    "maximum_nonbase_contact_force_n": nonbase_force_n,
                    "max_consecutive_belly_contact_steps": int(
                        region.get("max_consecutive_belly_contact_steps", 0)
                    ),
                    "belly_contact_duty_cycle": float(
                        region.get("belly_contact_duty_cycle", 0.0)
                    ),
                    "min_measured_support": float(
                        region.get("min_measured_support", 1.0)
                    ),
                    "longitudinal_center_offset_m": longitudinal_offset_m,
                },
            }
        )

    return {
        "schema_version": "kinofail.o9-semantic-admission.v1",
        "definition": (
            "sustained Go2 base/belly load on ridge with partial foot unloading; "
            "head- or limb-only collision is not high-centering"
        ),
        "thresholds": limits,
        "region_count": len(regions),
        "regions": region_results,
        "passed": bool(region_results) and any(row["passed"] for row in region_results),
    }
