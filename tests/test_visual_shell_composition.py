from __future__ import annotations

import numpy as np
import pytest

from kino_vla.sim.visual_shell_composition import (
    coordinate_matrix,
    route_surface_geometry,
    transform_vertices,
)


def test_coordinate_transform_is_right_handed_and_matches_go2_axes() -> None:
    contract = {
        "matrix": [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "determinant": 1.0,
    }
    matrix = coordinate_matrix(contract)
    source_axes = np.eye(3)
    transformed = transform_vertices(source_axes, matrix)
    assert np.allclose(transformed[0], [0.0, 1.0, 0.0])
    assert np.allclose(transformed[1], [0.0, 0.0, 1.0])
    assert np.allclose(transformed[2], [1.0, 0.0, 0.0])
    assert np.isclose(np.linalg.det(matrix), 1.0)


def test_coordinate_transform_rejects_reflection() -> None:
    with pytest.raises(ValueError, match="preserve handedness"):
        coordinate_matrix(
            {
                "matrix": [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "determinant": -1.0,
            }
        )


def test_route_surface_contains_straight_route() -> None:
    route = {
        "waypoints_xy_m": [[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]],
        "surface_width_m": 1.4,
        "endpoint_margin_m": 0.75,
        "floor_thickness_m": 0.08,
    }
    geometry = route_surface_geometry(route)
    assert geometry["direction_xy"] == [1.0, 0.0]
    assert geometry["surface_length_m"] == pytest.approx(5.5)
    assert geometry["surface_width_m"] == pytest.approx(1.4)
    assert np.asarray(geometry["corners_xy_m"])[:, 1].ptp() == pytest.approx(1.4)


def test_route_surface_rejects_bent_development_route() -> None:
    with pytest.raises(ValueError, match="must be straight"):
        route_surface_geometry(
            {
                "waypoints_xy_m": [[0.0, 0.0], [1.0, 0.2], [2.0, 0.0]],
                "surface_width_m": 1.4,
                "endpoint_margin_m": 0.75,
                "floor_thickness_m": 0.08,
            }
        )
