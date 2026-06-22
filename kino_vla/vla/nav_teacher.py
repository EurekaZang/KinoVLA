"""Geometric route-around teacher for the nav SFT data (distil the routing into the VLA).

The deployed planner forbids any geometric routing — ALL high-level nav must come from the VLA
(user directive, [[vla-driven-nav]]). To make the VLA OWN that decision robustly (instead of a
brittle zero-shot prompt), we fine-tune it on geometric labels: for a sampled robot pose this
module computes the correct next nav action — a ``Replan_Waypoint`` pixel that skirts the patch
toward the goal, or a ``Turn`` when the clear way forward is outside the camera's view — and the
generator renders the matching real RTX frame. At deploy the VLA reproduces this from pixels alone;
no geometry runs in the loop.

The route is a visibility-graph shortest path over {robot, the four inflated-patch corners, goal};
the first hop is the immediate target. A look-ahead point one ``lookahead`` along that hop is
projected into the real camera: in frame ⇒ a waypoint pixel (the VLA steers to it), out of the
horizontal field ⇒ a Turn toward the hop (bring clear ground into view, then pick a waypoint).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from kino_vla.utils.geometry import Rect, wrap_angle


def _seg_crosses_rect(p: np.ndarray, q: np.ndarray, rect: Rect) -> bool:
    """True if the segment p→q passes through the axis-aligned rect interior (Liang–Barsky clip)."""
    x0, y0 = float(p[0]), float(p[1])
    dx, dy = float(q[0]) - x0, float(q[1]) - y0
    xmin, xmax = rect.cx - rect.hx, rect.cx + rect.hx
    ymin, ymax = rect.cy - rect.hy, rect.cy + rect.hy
    t0, t1 = 0.0, 1.0
    for num, den in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
        if abs(num) < 1e-12:
            if den < 0.0:  # parallel to this slab AND outside it ⇒ never inside the rect
                return False
        else:
            r = den / num
            if num < 0.0:
                if r > t1:
                    return False
                t0 = max(t0, r)
            else:
                if r < t0:
                    return False
                t1 = min(t1, r)
    return t0 < t1  # the clipped sub-segment has positive length ⇒ it crosses the interior


def next_waypoint(
    start: np.ndarray, goal: np.ndarray, patch: Rect, *, margin: float = 0.5
) -> np.ndarray:
    """The first hop of the shortest robot→goal route that skirts the inflated patch.

    Visibility graph over {start, the 4 corners of patch+margin, goal}; an edge is free when its
    segment misses the patch+margin/2 rect (so corner→corner edges, tangent to the inflated patch,
    stay free). Dijkstra; returns the second node on the path (the immediate target), or the goal
    if the straight line is already clear."""
    inf = Rect(patch.cx, patch.cy, patch.hx + margin, patch.hy + margin)
    corners = [
        np.array([inf.cx - inf.hx, inf.cy - inf.hy]),
        np.array([inf.cx + inf.hx, inf.cy - inf.hy]),
        np.array([inf.cx + inf.hx, inf.cy + inf.hy]),
        np.array([inf.cx - inf.hx, inf.cy + inf.hy]),
    ]
    nodes = [np.asarray(start, dtype=np.float64), *corners, np.asarray(goal, dtype=np.float64)]
    test = Rect(patch.cx, patch.cy, patch.hx + 0.5 * margin, patch.hy + 0.5 * margin)
    n = len(nodes)

    def free(i: int, j: int) -> bool:
        return not _seg_crosses_rect(nodes[i], nodes[j], test)

    dist = [math.inf] * n
    prev = [-1] * n
    visited = [False] * n
    dist[0] = 0.0
    for _ in range(n):
        u = min((i for i in range(n) if not visited[i]), key=lambda i: dist[i], default=-1)
        if u == -1 or dist[u] == math.inf:
            break
        visited[u] = True
        for v in range(n):
            if v == u or visited[v] or not free(u, v):
                continue
            d = dist[u] + float(np.linalg.norm(nodes[u] - nodes[v]))
            if d < dist[v]:
                dist[v] = d
                prev[v] = u
    goal_i = n - 1
    if dist[goal_i] == math.inf:  # boxed in (shouldn't happen with margin) ⇒ aim at the goal
        return np.asarray(goal, dtype=np.float64)
    path: list[int] = []
    c = goal_i
    while c != -1:
        path.append(c)
        c = prev[c]
    path.reverse()
    return nodes[path[1]] if len(path) >= 2 else np.asarray(goal, dtype=np.float64)


def next_nav_label(
    pose_xy: np.ndarray,
    heading: float,
    patch: Rect,
    goal: np.ndarray,
    project: Callable[[np.ndarray], np.ndarray | None],
    *,
    margin: float = 0.5,
    lookahead: float = 1.6,
) -> dict:
    """Geometric nav label for one pose. ``project(world_xy)`` → (u_frac, v_frac)∈[0,1] in the real
    camera frame, or None if out of frame. Returns a dict: ``{"kind": "waypoint", "point_px":[u,v],
    "wp": …}`` (0..1000 pixel) when the look-ahead lands in view, else ``{"kind": "turn",
    "yaw_deg": d, "wp": …}`` (turn toward the hop to bring clear ground into view)."""
    pose_xy = np.asarray(pose_xy, dtype=np.float64)
    wp = next_waypoint(pose_xy, goal, patch, margin=margin)
    direction = wp - pose_xy
    dist = float(np.linalg.norm(direction))
    if dist < 1e-6:
        wp = np.asarray(goal, dtype=np.float64)
        direction = wp - pose_xy
        dist = float(np.linalg.norm(direction)) or 1.0
    look = pose_xy + min(lookahead, dist) * (direction / dist)
    px = project(look)
    if px is not None:
        return {
            "kind": "waypoint",
            "point_px": [int(round(float(px[0]) * 1000)), int(round(float(px[1]) * 1000))],
            "wp": [float(wp[0]), float(wp[1])],
        }
    bearing = math.atan2(float(direction[1]), float(direction[0]))
    yaw = math.degrees(wrap_angle(bearing - float(heading)))
    yaw = max(-90.0, min(90.0, yaw))
    return {"kind": "turn", "yaw_deg": round(yaw, 1), "wp": [float(wp[0]), float(wp[1])]}
