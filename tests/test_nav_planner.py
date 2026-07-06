"""Deployed grid route planner over the costmap (decoupled nav, #44) — pure-geometry unit tests."""

from __future__ import annotations

import numpy as np

from kino_vla.vla.nav_planner import obstacle_mask, occupancy_count, plan_route

ORIGIN = np.array([0.0, 0.0])
RES = 1.0


def _grid(ny: int = 11, nx: int = 21) -> np.ndarray:
    return np.zeros((ny, nx), dtype=np.float64)


def _fill(grid: np.ndarray, x0: float, x1: float, y0: float, y1: float, cost: float = 1.0) -> None:
    """Mark the world rect [x0,x1]x[y0,y1] (cell centres at i+0.5, j+0.5 with ORIGIN/RES)."""
    for j in range(grid.shape[0]):
        for i in range(grid.shape[1]):
            if x0 <= i + 0.5 <= x1 and y0 <= j + 0.5 <= y1:
                grid[j, i] = cost


def _route(grid: np.ndarray, start, goal, **kw) -> list[np.ndarray]:
    return plan_route(grid, ORIGIN, RES, np.asarray(start), np.asarray(goal), **kw)


def _segments_clear(grid: np.ndarray, start, wps, threshold: float = 0.5) -> bool:
    """Every segment of start->wp1->...->goal misses occupied cells (sampled at 0.25-cell steps)."""
    pts = [np.asarray(start, dtype=np.float64)] + [np.asarray(w, dtype=np.float64) for w in wps]
    ny, nx = grid.shape
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        n = max(2, int(np.linalg.norm(b - a) / (0.25 * RES)))
        for k in range(n + 1):
            p = a + (b - a) * (k / n)
            i, j = int(p[0] - ORIGIN[0]), int(p[1] - ORIGIN[1])
            if 0 <= i < nx and 0 <= j < ny and grid[j, i] > threshold:
                return False
    return True


def test_clear_grid_goes_straight() -> None:
    wps = _route(_grid(), (0.5, 5.5), (20.5, 5.5))
    assert len(wps) == 1 and np.allclose(wps[0], [20.5, 5.5])


def test_routes_around_a_central_block() -> None:
    grid = _grid()
    _fill(grid, 9.0, 12.0, 0.0, 8.0)  # a wall blocking the straight line, gap only near the top
    wps = _route(grid, (0.5, 5.5), (20.5, 5.5), inflation_m=0.0)
    assert np.allclose(wps[-1], [20.5, 5.5]), "the true goal is the final commitment"
    assert _segments_clear(grid, (0.5, 5.5), wps), "the committed route misses the marked cells"
    assert any(w[1] > 8.0 for w in wps), "it detoured up over the wall (used the gap)"


def test_keeps_clearance_margin_when_inflated() -> None:
    grid = _grid()
    _fill(grid, 9.0, 11.0, 0.0, 7.0)
    wps = _route(grid, (0.5, 5.5), (20.5, 5.5), inflation_m=1.0)
    # every routed waypoint (bar the true goal) stays >= ~1 cell clear of the marked region
    for w in wps[:-1]:
        assert not (8.0 <= w[0] <= 12.0 and w[1] <= 8.0), f"waypoint {w} hugs the inflated wall"
    assert _segments_clear(grid, (0.5, 5.5), wps)


def test_goal_off_grid_still_ends_at_goal() -> None:
    grid = _grid()
    _fill(grid, 9.0, 12.0, 0.0, 8.0)
    goal = np.array([25.0, 5.5])  # past the nx=21 grid edge
    wps = _route(grid, (0.5, 5.5), goal, inflation_m=0.0)
    assert np.allclose(wps[-1], goal), "off-grid goal appended as the final waypoint"
    assert _segments_clear(grid, (0.5, 5.5), wps[:-1] + [wps[-2]] if len(wps) > 1 else wps)


def test_start_on_hazard_recovers_and_routes_out() -> None:
    grid = _grid()
    _fill(grid, 4.0, 12.0, 3.0, 8.0)
    wps = _route(grid, (5.5, 5.5), (20.5, 5.5), inflation_m=0.0)  # start INSIDE the marked block
    assert np.allclose(wps[-1], [20.5, 5.5])
    assert len(wps) >= 2, "it had to route out of and around the block"


def test_boxed_in_goal_falls_back_not_through() -> None:
    grid = _grid()
    # wall the goal off completely (a full frame around it) ⇒ no path ⇒ fall back to [goal]
    _fill(grid, 15.0, 16.0, 0.0, 11.0)  # vertical wall spanning the whole height before the goal
    wps = _route(grid, (0.5, 5.5), (20.5, 5.5), inflation_m=0.0)
    assert len(wps) == 1 and np.allclose(wps[0], [20.5, 5.5]), "never fabricates a route through"


def test_occupancy_count_is_the_commit_trigger() -> None:
    grid = _grid()
    assert occupancy_count(grid, 0.5) == 0
    _fill(grid, 9.0, 12.0, 4.0, 7.0)  # cell centres 9.5/10.5/11.5 x 4.5/5.5/6.5 ⇒ a 3x3 block
    assert occupancy_count(grid, 0.5) == 9
    assert occupancy_count(grid, 1.5) == 0, "threshold respected"


def test_obstacle_mask_is_marked_plus_clearance() -> None:
    """The visualization mask = the EXACT set the planner routes around: marked cells (cost>thr)
    dilated by the clearance. A marked cell is occupied; a free cell beyond the inflation is not."""
    grid = _grid()
    _fill(grid, 9.0, 11.0, 4.0, 6.0)  # cells (9,10)x(4,5) marked
    occ = obstacle_mask(grid, threshold=0.5, inflation_m=1.0, resolution_m=RES)  # inflate 1 cell
    assert occ[5, 10], "a marked cell is occupied"
    assert occ[5, 8] and occ[6, 10], "1-cell clearance halo around the mark is occupied"
    assert not occ[5, 5], "a cell well clear of the mark + halo is free"
    assert not obstacle_mask(grid, 1.5, 1.0, RES).any(), "threshold respected (nothing above 1.5)"


def test_deterministic() -> None:
    grid = _grid()
    _fill(grid, 9.0, 12.0, 0.0, 8.0)
    a = _route(grid, (0.5, 5.5), (20.5, 5.5))
    b = _route(grid, (0.5, 5.5), (20.5, 5.5))
    assert len(a) == len(b) and all(np.allclose(x, y) for x, y in zip(a, b, strict=True))
