from __future__ import annotations

import pytest

from kino_vla.sim.embodiedgen_terrain_route_v2 import dense_route_grid


def _inputs() -> tuple[dict, dict]:
    route = {"waypoints_xy_m": [[-0.4, -0.8], [0.2, -0.8], [2.05, -0.8]]}
    geometry = {
        "vertices_xyz_m": [
            [-0.85, -1.4, 0.003],
            [2.5, -1.4, 0.003],
            [2.5, -0.2, 0.003],
            [-0.85, -0.2, 0.003],
        ],
        "direction_xy": [1.0, 0.0],
        "left_xy": [0.0, 1.0],
        "route_length_m": 2.45,
        "surface_length_m": 3.35,
        "surface_width_m": 1.2,
        "endpoint_margin_m": 0.45,
        "uv_repeat": [1.675, 0.6],
    }
    return route, geometry


def test_dense_grid_is_fine_enough_and_covers_surface() -> None:
    route, geometry = _inputs()
    grid = dense_route_grid(route=route, geometry=geometry, max_spacing_m=0.04)
    assert len(grid["vertices"]) == grid["n_long"] * grid["n_lat"]
    assert len(grid["faces"]) == (grid["n_long"] - 1) * (grid["n_lat"] - 1)
    assert len(grid["vertices"]) >= 1000
    assert grid["actual_longitudinal_spacing_m"] <= 0.04
    assert grid["actual_lateral_spacing_m"] <= 0.04
    assert grid["vertices"][0] == pytest.approx((-0.85, -1.4, 0.003))
    assert grid["vertices"][-1] == pytest.approx((2.5, -0.2, 0.003))


def test_dense_grid_rejects_coarse_operator_surface() -> None:
    route, geometry = _inputs()
    with pytest.raises(ValueError, match="max_spacing_m"):
        dense_route_grid(route=route, geometry=geometry, max_spacing_m=0.2)
