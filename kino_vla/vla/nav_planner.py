"""Deployed geometric route planner over the persistent traversability costmap (decoupled nav).

The decoupled navigation architecture (CLAUDE.md #44 resolution, the open-set/commitment split):
the VLA does OPEN-SET ATTRIBUTION only — on contact it decides "this region is untraversable, type
= <open-vocabulary>" ONCE and writes that into the §7 costmap (sticky, world-frame, propagated).
This module is the other half — a classic grid planner that consumes that persistent costmap and
COMMITS a complete route around every marked hazard. Memory + commitment live here (the planner has
a global view and holds a fixed path); the VLA never re-decides the route per frame.

This cures the per-frame-reactive OSCILLATION the VLA-as-nominal-nav approach hit (#43/#44): a
per-frame veto can REJECT a waypoint that lands on the hazard but cannot COMMIT a multi-frame
detour, so the reactive policy ping-pongs at the patch edge (turn -> re-see patch -> turn back ->
drive in). A planner with cross-frame memory routes around the whole region and holds the plan.

Pure numpy / CPU (no torch, no Isaac): exercised in CI and inside the closed loop alike. A* over the
costmap occupancy (cost > threshold, inflated by a robot-clearance margin), 8-connected with an
octile heuristic and NO diagonal corner-cutting, then line-of-sight string-pulling down to a minimal
committed waypoint list the pursuit controller drives (turn-in-place when misaligned, then advance).
"""

from __future__ import annotations

import heapq
import math

import numpy as np

_NBRS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
_SQRT2 = math.sqrt(2.0)


def occupancy_count(cost_grid: np.ndarray, threshold: float) -> int:
    """Number of marked-untraversable cells — the commit-and-hold re-plan trigger.

    Costmap marks are sticky and propagation only adds, so this count is monotonic; a change means a
    NEW hazard region was attributed and the committed route must be recomputed. Between changes the
    planner holds its plan (no per-frame churn — the cure for the oscillation)."""
    return int((np.asarray(cost_grid) > float(threshold)).sum())


def obstacle_mask(
    cost_grid: np.ndarray, threshold: float, inflation_m: float, resolution_m: float
) -> np.ndarray:
    """The inflated occupancy the planner routes around: cells with cost > ``threshold`` dilated by
    the robot-clearance margin. This is EXACTLY the set :func:`plan_route` treats as blocked —
    exposed so a visualization can show what the DEPLOYED planner actually avoids (the marked
    costmap + a clearance halo), not the retired ``nav_hazards`` disc abstraction (#45)."""
    occ = np.asarray(cost_grid) > float(threshold)
    return _inflate(occ, int(math.ceil(float(inflation_m) / float(resolution_m))))


def _inflate(occ: np.ndarray, r_cells: int) -> np.ndarray:
    """Dilate the occupancy by ``r_cells`` (Chebyshev square) — the robot-clearance margin so a
    routed waypoint and the pursuit drift while skirting stay clear of the marked region."""
    if r_cells <= 0:
        return occ
    out = occ.copy()
    ny, nx = occ.shape
    for j, i in zip(*np.nonzero(occ), strict=True):
        j0, j1 = max(0, j - r_cells), min(ny, j + r_cells + 1)
        i0, i1 = max(0, i - r_cells), min(nx, i + r_cells + 1)
        out[j0:j1, i0:i1] = True
    return out


def _line_clear(occ: np.ndarray, c0: tuple[int, int], c1: tuple[int, int]) -> bool:
    """Is the straight cell-space segment c0->c1 free of occupied cells? (string-pull LOS test)."""
    (i0, j0), (i1, j1) = c0, c1
    n = max(abs(i1 - i0), abs(j1 - j0))
    if n == 0:
        return not bool(occ[j0, i0])
    for k in range(n + 1):
        t = k / n
        i = int(round(i0 + (i1 - i0) * t))
        j = int(round(j0 + (j1 - j0) * t))
        if occ[j, i]:
            return False
    return True


def _nearest_free(occ: np.ndarray, cell: tuple[int, int]) -> tuple[int, int] | None:
    """The closest free cell to ``cell`` (BFS rings) — frees an endpoint that fell on the hazard."""
    ny, nx = occ.shape
    i0, j0 = cell
    if not occ[j0, i0]:
        return cell
    for r in range(1, max(nx, ny) + 1):
        best: tuple[int, int] | None = None
        best_d = math.inf
        for j in range(max(0, j0 - r), min(ny, j0 + r + 1)):
            for i in range(max(0, i0 - r), min(nx, i0 + r + 1)):
                if not occ[j, i]:
                    d = (i - i0) ** 2 + (j - j0) ** 2
                    if d < best_d:
                        best_d, best = d, (i, j)
        if best is not None:
            return best
    return None


def _astar(
    occ: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    """8-connected A* (octile heuristic, no diagonal corner-cutting); [] if no path."""
    ny, nx = occ.shape
    if occ[start[1], start[0]] or occ[goal[1], goal[0]]:
        return []

    def h(c: tuple[int, int]) -> float:
        dx, dy = abs(c[0] - goal[0]), abs(c[1] - goal[1])
        return (dx + dy) + (_SQRT2 - 2.0) * min(dx, dy)

    g: dict[tuple[int, int], float] = {start: 0.0}
    came: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    openh: list[tuple[float, float, int, tuple[int, int]]] = [(h(start), 0.0, 0, start)]
    cnt = 0
    while openh:
        _, gc, _, cur = heapq.heappop(openh)
        if cur == goal:
            path: list[tuple[int, int]] = []
            node: tuple[int, int] | None = cur
            while node is not None:
                path.append(node)
                node = came[node]
            return path[::-1]
        if gc > g.get(cur, math.inf):
            continue
        ci, cj = cur
        for di, dj in _NBRS:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < nx and 0 <= nj < ny) or occ[nj, ni]:
                continue
            if di != 0 and dj != 0 and (occ[cj, ni] or occ[nj, ci]):
                continue  # do not squeeze diagonally between two orthogonal obstacles
            ng = gc + (_SQRT2 if di and dj else 1.0)
            nc = (ni, nj)
            if ng < g.get(nc, math.inf):
                g[nc] = ng
                came[nc] = cur
                cnt += 1
                heapq.heappush(openh, (ng + h(nc), ng, cnt, nc))
    return []


def _string_pull(occ: np.ndarray, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Drop interior cells that the line of sight can skip — a minimal turn-point waypoint list."""
    if len(cells) <= 2:
        return cells
    out = [cells[0]]
    anchor = 0
    for i in range(1, len(cells) - 1):
        if not _line_clear(occ, cells[anchor], cells[i + 1]):
            out.append(cells[i])
            anchor = i
    out.append(cells[-1])
    return out


def plan_route(
    cost_grid: np.ndarray,
    origin_xy: np.ndarray,
    resolution_m: float,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    *,
    threshold: float = 0.5,
    inflation_m: float = 0.5,
) -> list[np.ndarray]:
    """A committed world-frame waypoint route from ``start_xy`` to ``goal_xy`` around the costmap's
    marked-untraversable cells (cost > ``threshold``), inflated by ``inflation_m`` of clearance.

    Returns ``[hop1, …, goal]`` (the start is excluded; the pursuit controller is already at it);
    ``[goal]`` when the straight line is already clear. The true goal is always the final waypoint
    (it may lie just past, or off, the grid — the off-grid tail is assumed clear beyond the mapped
    region). If the goal is boxed in (no path), falls back to ``[goal]`` and lets the monitor / CBF
    shield guard — the planner never fabricates a route through a hazard."""
    grid = np.asarray(cost_grid, dtype=np.float64)
    ny, nx = grid.shape
    origin = np.asarray(origin_xy, dtype=np.float64)
    res = float(resolution_m)
    start_w = np.asarray(start_xy, dtype=np.float64)
    goal_w = np.asarray(goal_xy, dtype=np.float64)

    occ = obstacle_mask(grid, threshold, inflation_m, res)

    def to_cell(xy: np.ndarray) -> tuple[int, int]:
        rel = (np.asarray(xy, dtype=np.float64) - origin) / res
        return (
            int(np.clip(math.floor(rel[0]), 0, nx - 1)),
            int(np.clip(math.floor(rel[1]), 0, ny - 1)),
        )

    def to_world(cell: tuple[int, int]) -> np.ndarray:
        i, j = cell
        return origin + np.array([(i + 0.5) * res, (j + 0.5) * res])

    start_c = to_cell(start_w)
    goal_c = to_cell(goal_w)
    if not occ.any() or _line_clear(occ, start_c, goal_c):
        return [goal_w.copy()]

    free_start = _nearest_free(occ, start_c) or start_c
    target_c = _nearest_free(occ, goal_c) or goal_c
    cells = _astar(occ, free_start, target_c)
    if not cells:
        return [goal_w.copy()]  # boxed in ⇒ defer to the monitor / shield, never route through

    wps = [to_world(c) for c in _string_pull(occ, cells)[1:]]
    if not wps or float(np.linalg.norm(wps[-1] - goal_w)) > 0.5 * res:
        wps.append(goal_w.copy())  # the true goal is always the final commitment
    return wps
