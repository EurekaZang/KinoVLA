from __future__ import annotations

import pytest

from kino_vla.sim.embodiedgen_scene import Box2D
from kino_vla.sim.embodiedgen_scene_v2 import (
    CORRIDOR_LENGTH_M,
    ROBUST_CENTERLINE_CLEARANCE_M,
    find_robust_experiment_corridor,
)


def test_robust_corridor_reserves_tracking_envelope() -> None:
    floor = Box2D("floor", -3.0, 3.0, -2.0, 2.0)
    obstacle = Box2D("table", -0.4, 0.4, 0.7, 1.4)
    result = find_robust_experiment_corridor(floor, (obstacle,))
    assert (
        result["measured_centerline_clearance_m"]
        >= ROBUST_CENTERLINE_CLEARANCE_M - 1.0e-6
    )
    assert len(result["waypoints_xy_m"]) == 5
    assert result["collector_start_waypoint_index"] == 2
    points = result["waypoints_xy_m"]
    length = sum(
        ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        for a, b in zip(points, points[1:])
    )
    assert length == pytest.approx(CORRIDOR_LENGTH_M)


def test_robust_corridor_rejects_room_without_straight_run() -> None:
    floor = Box2D("floor", 0.0, 2.0, 0.0, 2.0)
    with pytest.raises(ValueError, match="no straight experiment corridor"):
        find_robust_experiment_corridor(floor, ())
