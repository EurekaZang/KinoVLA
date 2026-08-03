from __future__ import annotations

import math

import numpy as np
import pytest

from kino_vla.sim.realistic_route_protocol import StraightRouteFrame


def _compiled(direction=(1.0, 0.0), left=(0.0, 1.0)) -> dict:
    return {
        "route": {"waypoints_xy_m": [[0.2, -0.3], [4.2, -0.3]]},
        "physics_contract": {
            "route_geometry": {
                "direction_xy": list(direction),
                "left_xy": list(left),
                "route_length_m": 4.0,
                "surface_width_m": 1.4,
            }
        },
    }


def test_route_frame_projects_and_reconstructs_metric_coordinates() -> None:
    frame = StraightRouteFrame.from_compiled_audit(_compiled())
    point = frame.point(1.7, -0.12)
    assert point == pytest.approx([1.9, -0.42])
    assert frame.project(point) == pytest.approx((1.7, -0.12))


def test_controller_corrects_cross_track_and_heading_in_body_frame() -> None:
    frame = StraightRouteFrame.from_compiled_audit(_compiled())
    command = frame.controller_command(
        position_xy_m=np.asarray([0.2, -0.10]),
        heading_rad=0.10,
        forward_speed_mps=0.32,
        target_lateral_offset_m=0.0,
    )
    assert command[0] > 0.0
    assert command[1] < 0.0
    assert command[2] == pytest.approx(-0.20)


def test_rotated_route_command_uses_route_heading() -> None:
    compiled = _compiled(direction=(0.0, 1.0), left=(-1.0, 0.0))
    compiled["route"]["waypoints_xy_m"] = [[0.2, -0.3], [0.2, 3.7]]
    frame = StraightRouteFrame.from_compiled_audit(compiled)
    assert frame.heading_rad == pytest.approx(math.pi / 2.0)
    command = frame.controller_command(
        position_xy_m=np.asarray([0.2, -0.3]),
        heading_rad=math.pi / 2.0,
        forward_speed_mps=0.32,
        target_lateral_offset_m=0.0,
    )
    assert command == pytest.approx([0.32, 0.0, 0.0], abs=1.0e-12)


def test_invalid_nonunit_route_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="unit length"):
        StraightRouteFrame.from_compiled_audit(_compiled(direction=(2.0, 0.0)))
