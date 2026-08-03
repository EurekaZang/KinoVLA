from __future__ import annotations

import pytest
import numpy as np

from kino_vla.sim.embodiedgen_scene import Box2D, _adhesion_region, plan_go2_route
from kino_vla.sim.isaac_policy_backend import body_command_to_world_xy


def test_route_planner_handles_staggered_furniture_with_go2_clearance() -> None:
    floor = Box2D("floor", -1.6, 1.6, -4.6, 4.6)
    obstacles = (
        Box2D("cabinet", -0.5, 0.25, -3.45, -2.70),
        Box2D("desk", 0.95, 1.55, -0.90, 0.96),
    )
    route = plan_go2_route(floor, obstacles)
    assert route["long_axis"] == "y"
    assert route["route_length_m"] > 8.0
    assert route["measured_min_clearance_m"] >= (
        route["required_clearance_m"] - route["grid_resolution_m"]
    )
    assert len(route["waypoints_xy_m"]) >= 3
    assert max(
        np.linalg.norm(np.asarray(end) - np.asarray(start))
        for start, end in zip(route["waypoints_xy_m"], route["waypoints_xy_m"][1:])
    ) <= route["maximum_waypoint_spacing_m"] + 1.0e-6


def test_route_planner_rejects_room_narrower_than_robot_envelope() -> None:
    with pytest.raises(ValueError, match="smaller than"):
        plan_go2_route(Box2D("floor", -0.3, 0.3, -2.0, 2.0), ())


def test_route_planner_rejects_fully_blocked_cross_section() -> None:
    floor = Box2D("floor", -1.5, 1.5, -3.0, 3.0)
    barrier = (Box2D("barrier", -1.5, 1.5, -0.2, 0.2),)
    with pytest.raises(ValueError, match="no Go2-width route"):
        plan_go2_route(floor, barrier)


def test_route_planner_uses_longest_internal_component_without_relaxing_clearance() -> None:
    floor = Box2D("floor", -4.0, 4.0, -4.0, 4.0)
    # The inflated barrier disconnects the two long-axis boundaries, but each half still has a
    # publication-usable internal trajectory wider than the exact same Go2 envelope.
    barrier = (Box2D("barrier", -4.0, 4.0, -0.2, 0.2),)

    route = plan_go2_route(floor, barrier)

    assert route["route_selection"] == "largest_free_component_geodesic_diameter"
    assert route["route_length_m"] >= route["minimum_route_length_m"]
    assert route["measured_min_clearance_m"] >= (
        route["required_clearance_m"] - route["grid_resolution_m"]
    )


def test_body_command_is_rotated_to_world_for_nonzero_scene_heading() -> None:
    assert np.allclose(body_command_to_world_xy(np.array([0.4, 0.0]), np.pi / 2), [0.0, 0.4])
    assert np.allclose(body_command_to_world_xy(np.array([-0.3, 0.0]), np.pi / 2), [0.0, -0.3])


def test_adhesion_region_is_relative_to_collector_start_not_whole_route_length() -> None:
    route = {
        "long_axis": "y",
        "waypoints_xy_m": [[0.0, float(y)] for y in range(8)],
    }

    operator = _adhesion_region(route, 0.0)
    vertices = np.asarray(operator["vertices_xy_m"])

    assert operator["collector_start_waypoint_index"] == 4
    assert operator["entry_offset_from_collector_start_m"] == 0.38
    assert abs(vertices[:, 1].min() - 4.38) < 1.0e-6
    assert operator["placement_contract"] == "route_relative_from_collector_start_v2"
