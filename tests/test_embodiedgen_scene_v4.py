from __future__ import annotations

import math

import pytest

from kino_vla.sim.embodiedgen_scene_v4 import (
    route_envelope_light_positions,
    route_surface_geometry,
)


def test_route_surface_extends_straight_corridor_without_changing_height() -> None:
    route = [[0.0, 2.0], [0.5, 2.0], [1.0, 2.0], [1.7, 2.0], [2.45, 2.0]]
    value = route_surface_geometry(route, 0.1)

    assert value["direction_xy"] == [1.0, 0.0]
    assert value["surface_length_m"] == pytest.approx(3.35)
    assert value["surface_width_m"] == pytest.approx(1.2)
    assert value["maximum_waypoint_line_deviation_m"] == pytest.approx(0.0)
    xs = [row[0] for row in value["vertices_xyz_m"]]
    ys = [row[1] for row in value["vertices_xyz_m"]]
    zs = [row[2] for row in value["vertices_xyz_m"]]
    assert min(xs) == pytest.approx(-0.45)
    assert max(xs) == pytest.approx(2.90)
    assert min(ys) == pytest.approx(1.4)
    assert max(ys) == pytest.approx(2.6)
    assert zs == pytest.approx([0.103] * 4)


def test_route_envelope_has_two_endpoint_extensions() -> None:
    route = [[-1.0, 1.0], [-1.0, 1.5], [-1.0, 2.0], [-1.0, 2.7], [-1.0, 3.45]]
    positions = route_envelope_light_positions(route, 2.5)

    assert len(positions) == 7
    assert positions[0] == pytest.approx([-1.0, 0.55, 2.5])
    assert positions[-1] == pytest.approx([-1.0, 3.90, 2.5])
    assert all(math.isclose(row[2], 2.5) for row in positions)


def test_route_surface_rejects_non_straight_route() -> None:
    with pytest.raises(ValueError, match="straight corridor-v2"):
        route_surface_geometry([[0.0, 0.0], [1.0, 0.2], [2.0, 0.0]], 0.0)
