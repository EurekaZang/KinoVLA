"""Kino-Fail-owned physics and route compilation for EmbodiedGen visual rooms.

EmbodiedGen is admitted only as an appearance/layout source.  This module derives conservative
axis-aligned collision proxies, plans a Go2-width route, and authors independent USD layers whose
provenance can be audited without treating generated collision or mass as ground truth.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from kino_vla.sim.embodiedgen_asset import (
    SOURCE_KIND,
    audit_embodiedgen_room_manifest,
    compile_embodiedgen_visual_wrapper,
)

SCHEMA_VERSION = "kinofail.embodiedgen-compiled-scene.v1"
MAX_ABS_INDOOR_COORDINATE_M = 100.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Box2D:
    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def expanded(self, amount: float) -> "Box2D":
        return Box2D(
            self.name,
            self.xmin - amount,
            self.xmax + amount,
            self.ymin - amount,
            self.ymax + amount,
        )

    def contains(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


@dataclass(frozen=True)
class CollisionProxy:
    name: str
    source_prim: str
    center_xyz_m: tuple[float, float, float]
    size_xyz_m: tuple[float, float, float]
    semantic: str


def _point_box_distance(x: float, y: float, box: Box2D) -> float:
    dx = max(box.xmin - x, 0.0, x - box.xmax)
    dy = max(box.ymin - y, 0.0, y - box.ymax)
    return math.hypot(dx, dy)


def _sample_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    spacing: float,
) -> Iterable[tuple[float, float]]:
    length = math.dist(start, end)
    steps = max(1, math.ceil(length / spacing))
    for index in range(steps + 1):
        alpha = index / steps
        yield (
            start[0] + alpha * (end[0] - start[0]),
            start[1] + alpha * (end[1] - start[1]),
        )


def _route_clear(
    start: tuple[float, float],
    end: tuple[float, float],
    free_bounds: Box2D,
    inflated_obstacles: tuple[Box2D, ...],
    spacing: float,
) -> bool:
    for x, y in _sample_segment(start, end, spacing):
        if not free_bounds.contains(x, y):
            return False
        if any(box.contains(x, y) for box in inflated_obstacles):
            return False
    return True


def plan_go2_route(
    floor_bounds: Box2D,
    obstacles: tuple[Box2D, ...],
    *,
    robot_radius_m: float = 0.34,
    safety_margin_m: float = 0.10,
    grid_resolution_m: float = 0.08,
    minimum_route_length_m: float = 4.0,
    maximum_waypoint_spacing_m: float = 0.75,
) -> dict[str, Any]:
    """Plan a long-axis route with explicit swept-disc clearance.

    The obstacle boxes are raw visual-layout bounds.  Planning occurs after Minkowski expansion by
    ``robot_radius + safety_margin``; the returned clearance is recomputed against the raw boxes.
    """
    if (
        robot_radius_m <= 0
        or safety_margin_m < 0
        or grid_resolution_m <= 0
        or minimum_route_length_m <= 0
        or maximum_waypoint_spacing_m <= 0
    ):
        raise ValueError("route dimensions must be positive")
    clearance = robot_radius_m + safety_margin_m
    free = Box2D(
        "room_interior",
        floor_bounds.xmin + clearance,
        floor_bounds.xmax - clearance,
        floor_bounds.ymin + clearance,
        floor_bounds.ymax - clearance,
    )
    if free.xmin >= free.xmax or free.ymin >= free.ymax:
        raise ValueError("room is smaller than the required Go2 clearance envelope")
    inflated = tuple(box.expanded(clearance) for box in obstacles)
    span_x = floor_bounds.xmax - floor_bounds.xmin
    span_y = floor_bounds.ymax - floor_bounds.ymin
    long_axis = "y" if span_y >= span_x else "x"

    nx = max(2, math.floor((free.xmax - free.xmin) / grid_resolution_m) + 1)
    ny = max(2, math.floor((free.ymax - free.ymin) / grid_resolution_m) + 1)
    xs = [free.xmin + i * (free.xmax - free.xmin) / (nx - 1) for i in range(nx)]
    ys = [free.ymin + j * (free.ymax - free.ymin) / (ny - 1) for j in range(ny)]

    def is_free(node: tuple[int, int]) -> bool:
        x, y = xs[node[0]], ys[node[1]]
        return not any(box.contains(x, y) for box in inflated)

    def endpoint(candidates: Iterable[tuple[int, int]]) -> tuple[int, int]:
        available = [node for node in candidates if is_free(node)]
        if not available:
            raise ValueError("no collision-free route endpoint")

        def score(node: tuple[int, int]) -> tuple[float, float]:
            x, y = xs[node[0]], ys[node[1]]
            wall_gap = min(x - free.xmin, free.xmax - x, y - free.ymin, free.ymax - y)
            obstacle_gap = min(
                (_point_box_distance(x, y, box) for box in inflated), default=1.0e6
            )
            short_offset = abs(x - (free.xmin + free.xmax) / 2) if long_axis == "y" else abs(
                y - (free.ymin + free.ymax) / 2
            )
            return (min(wall_gap, obstacle_gap), -short_offset)

        return max(available, key=score)

    if long_axis == "y":
        start = endpoint((i, 0) for i in range(nx))
        goal = endpoint((i, ny - 1) for i in range(nx))
    else:
        start = endpoint((0, j) for j in range(ny))
        goal = endpoint((nx - 1, j) for j in range(ny))

    neighbours = (
        (-1, -1, math.sqrt(2.0)),
        (-1, 0, 1.0),
        (-1, 1, math.sqrt(2.0)),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, math.sqrt(2.0)),
        (1, 0, 1.0),
        (1, 1, math.sqrt(2.0)),
    )

    def valid_neighbours(node: tuple[int, int]) -> Iterable[tuple[tuple[int, int], float]]:
        for di, dj, step_cost in neighbours:
            nxt = (node[0] + di, node[1] + dj)
            if not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny) or not is_free(nxt):
                continue
            # Prevent a diagonal move from cutting an inflated obstacle corner.
            if di and dj and (
                not is_free((node[0] + di, node[1]))
                or not is_free((node[0], node[1] + dj))
            ):
                continue
            yield nxt, step_cost

    frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost_so_far = {start: 0.0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for nxt, step_cost in valid_neighbours(current):
            x, y = xs[nxt[0]], ys[nxt[1]]
            short_center = (free.xmin + free.xmax) / 2 if long_axis == "y" else (
                free.ymin + free.ymax
            ) / 2
            short_value = x if long_axis == "y" else y
            centre_penalty = 0.015 * abs(short_value - short_center)
            new_cost = cost_so_far[current] + step_cost + centre_penalty
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                heuristic = math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                heapq.heappush(frontier, (new_cost + heuristic, nxt))
                came_from[nxt] = current

    route_selection = "long_axis_boundary_to_boundary"
    if goal not in came_from:
        # Real rooms can have furniture along a boundary while still containing a long, useful
        # free-space trajectory.  Fall back to an approximate geodesic diameter of the largest
        # connected Go2-clear component; never reduce the robot envelope or obstacle inflation.
        all_free = {(i, j) for i in range(nx) for j in range(ny) if is_free((i, j))}
        components: list[set[tuple[int, int]]] = []
        unseen = set(all_free)
        while unseen:
            root = next(iter(unseen))
            unseen.remove(root)
            component = {root}
            stack = [root]
            while stack:
                current = stack.pop()
                for nxt, _ in valid_neighbours(current):
                    if nxt in unseen:
                        unseen.remove(nxt)
                        component.add(nxt)
                        stack.append(nxt)
            components.append(component)
        if not components:
            raise ValueError("no Go2-width free-space component in generated room")
        component = max(components, key=len)

        def farthest(
            source: tuple[int, int],
        ) -> tuple[tuple[int, int], dict[tuple[int, int], tuple[int, int] | None], float]:
            distances = {source: 0.0}
            predecessors: dict[tuple[int, int], tuple[int, int] | None] = {source: None}
            queue: list[tuple[float, tuple[int, int]]] = [(0.0, source)]
            while queue:
                distance, current = heapq.heappop(queue)
                if distance != distances[current]:
                    continue
                for nxt, step_cost in valid_neighbours(current):
                    if nxt not in component:
                        continue
                    new_distance = distance + step_cost * grid_resolution_m
                    if new_distance < distances.get(nxt, math.inf):
                        distances[nxt] = new_distance
                        predecessors[nxt] = current
                        heapq.heappush(queue, (new_distance, nxt))
            destination = max(distances, key=distances.get)
            return destination, predecessors, distances[destination]

        first, _, _ = farthest(next(iter(component)))
        second, fallback_predecessors, geodesic_length = farthest(first)
        if geodesic_length < minimum_route_length_m:
            raise ValueError(
                "no Go2-width route meets the minimum usable length: "
                f"longest_component_path={geodesic_length:.3f}m "
                f"required={minimum_route_length_m:.3f}m"
            )
        start, goal = first, second
        came_from = fallback_predecessors
        route_selection = "largest_free_component_geodesic_diameter"

    nodes: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = goal
    while cursor is not None:
        nodes.append(cursor)
        cursor = came_from[cursor]
    nodes.reverse()
    dense = [(xs[i], ys[j]) for i, j in nodes]

    # Greedy line-of-sight reduction retains the same swept-disc safety predicate.
    simplified = [dense[0]]
    anchor = 0
    while anchor < len(dense) - 1:
        candidate = len(dense) - 1
        while candidate > anchor + 1 and not _route_clear(
            dense[anchor],
            dense[candidate],
            free,
            inflated,
            grid_resolution_m / 2,
        ):
            candidate -= 1
        simplified.append(dense[candidate])
        anchor = candidate

    samples = [
        point
        for a, b in zip(simplified, simplified[1:])
        for point in _sample_segment(a, b, grid_resolution_m / 4)
    ]
    min_obstacle_clearance = min(
        (_point_box_distance(x, y, box) for x, y in samples for box in obstacles),
        default=1.0e6,
    )
    min_wall_clearance = min(
        min(
            x - floor_bounds.xmin,
            floor_bounds.xmax - x,
            y - floor_bounds.ymin,
            floor_bounds.ymax - y,
        )
        for x, y in samples
    )
    measured_clearance = min(min_obstacle_clearance, min_wall_clearance)
    if measured_clearance + 1.0e-6 < clearance - grid_resolution_m:
        raise RuntimeError("route clearance verification disagrees with grid planner")
    length = sum(math.dist(a, b) for a, b in zip(simplified, simplified[1:]))
    if length < minimum_route_length_m:
        raise ValueError(
            "collision-free route became shorter than the minimum after simplification: "
            f"{length:.3f}m < {minimum_route_length_m:.3f}m"
        )

    # Keep downstream start selection tied to arc length rather than to an arbitrary number of
    # line-of-sight corners.  Every retained corner remains present, and long straight segments
    # receive uniformly spaced intermediate waypoints.
    waypoints = [simplified[0]]
    for segment_start, segment_end in zip(simplified, simplified[1:]):
        segment_length = math.dist(segment_start, segment_end)
        subdivisions = max(1, math.ceil(segment_length / maximum_waypoint_spacing_m))
        for index in range(1, subdivisions + 1):
            alpha = index / subdivisions
            waypoints.append(
                (
                    segment_start[0] + alpha * (segment_end[0] - segment_start[0]),
                    segment_start[1] + alpha * (segment_end[1] - segment_start[1]),
                )
            )
    return {
        "long_axis": long_axis,
        "route_selection": route_selection,
        "waypoints_xy_m": [[round(x, 6), round(y, 6)] for x, y in waypoints],
        "route_length_m": length,
        "minimum_route_length_m": minimum_route_length_m,
        "maximum_waypoint_spacing_m": maximum_waypoint_spacing_m,
        "robot_radius_m": robot_radius_m,
        "safety_margin_m": safety_margin_m,
        "required_clearance_m": clearance,
        "measured_min_clearance_m": measured_clearance,
        "grid_resolution_m": grid_resolution_m,
        "dense_node_count": len(dense),
    }


def _adhesion_region(route: dict[str, Any], floor_z: float) -> dict[str, Any]:
    waypoints = [tuple(value) for value in route["waypoints_xy_m"]]
    start_index = len(waypoints) // 2
    start = np.asarray(waypoints[start_index], dtype=np.float64)
    half_long, half_short = 0.48, 0.34
    entry_offset_m = 0.38
    centre_offset_m = entry_offset_m + half_long
    remaining = centre_offset_m
    centre: np.ndarray | None = None
    direction: np.ndarray | None = None
    for segment_start_value, segment_end_value in zip(
        waypoints[start_index:], waypoints[start_index + 1 :]
    ):
        segment_start = np.asarray(segment_start_value, dtype=np.float64)
        segment_end = np.asarray(segment_end_value, dtype=np.float64)
        delta = segment_end - segment_start
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-9:
            continue
        if remaining <= length:
            direction = delta / length
            centre = segment_start + remaining * direction
            break
        remaining -= length
    if centre is None or direction is None:
        raise ValueError(
            "route after the collector start is too short for the O4 exposure contract: "
            f"need {centre_offset_m:.3f}m to region centre"
        )
    left = np.array([-direction[1], direction[0]], dtype=np.float64)
    vertices_array = (
        centre - half_long * direction - half_short * left,
        centre - half_long * direction + half_short * left,
        centre + half_long * direction + half_short * left,
        centre + half_long * direction - half_short * left,
    )
    vertices = tuple((float(point[0]), float(point[1])) for point in vertices_array)
    return {
        "operator_id": "O4",
        "operator_name": "foot_adhesion",
        "surface_z_m": floor_z,
        "vertices_xy_m": [[round(x, 6), round(y, 6)] for x, y in vertices],
        "area_m2": 4.0 * half_long * half_short,
        "collector_start_waypoint_index": start_index,
        "collector_start_xy_m": [round(float(value), 6) for value in start],
        "entry_offset_from_collector_start_m": entry_offset_m,
        "centre_offset_from_collector_start_m": centre_offset_m,
        "route_tangent_xy": [round(float(value), 8) for value in direction],
        "placement_contract": "route_relative_from_collector_start_v2",
        "half_extent_long_m": half_long,
        "half_extent_short_m": half_short,
        "visual_marker_authoritative": False,
        "physics_owned_by": "Kino-Fail",
    }


def compile_embodiedgen_kinofail_scene(
    source_manifest: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compile independent Kino collision/route/operator layers around a frozen visual source."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

    source_manifest = Path(source_manifest).resolve()
    source_audit = audit_embodiedgen_room_manifest(source_manifest)
    if not source_audit["passed"]:
        raise ValueError(f"source manifest failed: {source_audit}")
    source_usd = Path(str(source_audit["main_usd"]))
    stage = Usd.Stage.Open(str(source_usd))
    if stage is None or not stage.GetDefaultPrim():
        raise ValueError("EmbodiedGen source USD cannot be opened")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError("EmbodiedGen scene is not Z-up")
    if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("EmbodiedGen scene is not authored in metres")

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    mesh_rows: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        world_transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        world_translation = world_transform.ExtractTranslation()
        if any(
            not math.isfinite(float(world_translation[index]))
            or abs(float(world_translation[index])) > MAX_ABS_INDOOR_COORDINATE_M
            for index in range(3)
        ):
            raise ValueError(
                "EmbodiedGen source has an implausible world transform: "
                f"{prim.GetPath()} translation={tuple(float(v) for v in world_translation)}"
            )
        mesh = UsdGeom.Mesh(prim)
        bound = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        low, high = bound.GetMin(), bound.GetMax()
        coordinates = [float(low[i]) for i in range(3)] + [float(high[i]) for i in range(3)]
        if any(
            not math.isfinite(value) or abs(value) > MAX_ABS_INDOOR_COORDINATE_M
            for value in coordinates
        ):
            raise ValueError(
                "EmbodiedGen source has implausible metric bounds: "
                f"{prim.GetPath()} low={tuple(float(low[i]) for i in range(3))} "
                f"high={tuple(float(high[i]) for i in range(3))}"
            )
        mesh_rows.append(
            {
                "path": str(prim.GetPath()),
                "low": [float(low[i]) for i in range(3)],
                "high": [float(high[i]) for i in range(3)],
                "points": len(mesh.GetPointsAttr().Get() or []),
                "faces": len(mesh.GetFaceVertexCountsAttr().Get() or []),
            }
        )
    floor_candidates = [
        row
        for row in mesh_rows
        if "floor" in row["path"].lower()
        and row["high"][2] - row["low"][2] < 0.25
    ]
    if not floor_candidates:
        raise ValueError("no metric floor mesh found in EmbodiedGen room")
    floor = max(
        floor_candidates,
        key=lambda row: (row["high"][0] - row["low"][0])
        * (row["high"][1] - row["low"][1]),
    )
    floor_z = float(floor["high"][2])
    floor_bounds = Box2D(
        "navigable_floor",
        float(floor["low"][0]),
        float(floor["high"][0]),
        float(floor["low"][1]),
        float(floor["high"][1]),
    )
    structural_tokens = ("wall", "floor", "ceiling", "exterior", "window")
    furniture_rows = [
        row
        for row in mesh_rows
        if not any(token in row["path"].lower() for token in structural_tokens)
        and row["low"][2] <= floor_z + 0.45
        and row["high"][2] >= floor_z + 0.12
    ]
    furniture_obstacles = tuple(
        Box2D(
            row["path"],
            row["low"][0],
            row["high"][0],
            row["low"][1],
            row["high"][1],
        )
        for row in furniture_rows
    )

    # RoomGen exports a combined visual wall mesh, not trustworthy source collision.  Derive
    # conservative Kino-owned boxes from its vertical faces so routes and physics cannot pass
    # through visible internal partitions.  Perimeter faces are skipped because four explicit
    # boundary proxies are authored below.  Duplicate triangle faces collapse to one rounded box.
    perimeter_tolerance_m = 0.16
    wall_proxy_thickness_m = 0.10
    internal_wall_by_key: dict[tuple[float, ...], dict[str, Any]] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or "wall" not in str(prim.GetPath()).lower():
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world_points = [transform.Transform(Gf.Vec3d(*point)) for point in points]
        offset = 0
        for face_index, count in enumerate(counts):
            face = [world_points[int(indices[offset + index])] for index in range(int(count))]
            offset += int(count)
            low = [min(float(point[axis]) for point in face) for axis in range(3)]
            high = [max(float(point[axis]) for point in face) for axis in range(3)]
            if (
                high[2] - low[2] < 0.60
                or low[2] > floor_z + 0.25
                or high[2] < floor_z + 0.60
                or max(high[0] - low[0], high[1] - low[1]) < 0.12
            ):
                continue
            on_perimeter = (
                high[0] <= floor_bounds.xmin + perimeter_tolerance_m
                or low[0] >= floor_bounds.xmax - perimeter_tolerance_m
                or high[1] <= floor_bounds.ymin + perimeter_tolerance_m
                or low[1] >= floor_bounds.ymax - perimeter_tolerance_m
            )
            if on_perimeter:
                continue
            xmin, xmax = low[0], high[0]
            ymin, ymax = low[1], high[1]
            if xmax - xmin < wall_proxy_thickness_m:
                centre = (xmin + xmax) / 2
                xmin, xmax = (
                    centre - wall_proxy_thickness_m / 2,
                    centre + wall_proxy_thickness_m / 2,
                )
            if ymax - ymin < wall_proxy_thickness_m:
                centre = (ymin + ymax) / 2
                ymin, ymax = (
                    centre - wall_proxy_thickness_m / 2,
                    centre + wall_proxy_thickness_m / 2,
                )
            key = tuple(round(value, 3) for value in (xmin, xmax, ymin, ymax, low[2], high[2]))
            internal_wall_by_key.setdefault(
                key,
                {
                    "path": str(prim.GetPath()),
                    "face_index": face_index,
                    "low": [xmin, ymin, low[2]],
                    "high": [xmax, ymax, high[2]],
                },
            )
    internal_wall_rows = list(internal_wall_by_key.values())
    internal_wall_obstacles = tuple(
        Box2D(
            f"{row['path']}#face{row['face_index']}",
            row["low"][0],
            row["high"][0],
            row["low"][1],
            row["high"][1],
        )
        for row in internal_wall_rows
    )
    obstacles = furniture_obstacles + internal_wall_obstacles
    route = plan_go2_route(floor_bounds, obstacles)
    operator = _adhesion_region(route, floor_z)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    visual = compile_embodiedgen_visual_wrapper(source_manifest, output_dir)

    proxies: list[CollisionProxy] = []
    floor_thickness = 0.08
    proxies.append(
        CollisionProxy(
            "Floor",
            floor["path"],
            (
                (floor_bounds.xmin + floor_bounds.xmax) / 2,
                (floor_bounds.ymin + floor_bounds.ymax) / 2,
                floor_z - floor_thickness / 2,
            ),
            (
                floor_bounds.xmax - floor_bounds.xmin,
                floor_bounds.ymax - floor_bounds.ymin,
                floor_thickness,
            ),
            "floor",
        )
    )
    room_height = max(row["high"][2] for row in mesh_rows) - floor_z
    wall_height = min(max(room_height, 2.4), 4.0)
    wall_t = 0.10
    cx = (floor_bounds.xmin + floor_bounds.xmax) / 2
    cy = (floor_bounds.ymin + floor_bounds.ymax) / 2
    sx = floor_bounds.xmax - floor_bounds.xmin
    sy = floor_bounds.ymax - floor_bounds.ymin
    for name, centre, size in (
        ("WallXMin", (floor_bounds.xmin - wall_t / 2, cy, floor_z + wall_height / 2), (wall_t, sy + 2 * wall_t, wall_height)),
        ("WallXMax", (floor_bounds.xmax + wall_t / 2, cy, floor_z + wall_height / 2), (wall_t, sy + 2 * wall_t, wall_height)),
        ("WallYMin", (cx, floor_bounds.ymin - wall_t / 2, floor_z + wall_height / 2), (sx, wall_t, wall_height)),
        ("WallYMax", (cx, floor_bounds.ymax + wall_t / 2, floor_z + wall_height / 2), (sx, wall_t, wall_height)),
    ):
        proxies.append(CollisionProxy(name, floor["path"], centre, size, "boundary_wall"))
    for index, row in enumerate(internal_wall_rows):
        low, high = row["low"], row["high"]
        proxies.append(
            CollisionProxy(
                f"InternalWall_{index:03d}",
                f"{row['path']}#face{row['face_index']}",
                tuple((low[axis] + high[axis]) / 2 for axis in range(3)),
                tuple(max(high[axis] - low[axis], 0.03) for axis in range(3)),
                "internal_wall_face_aabb",
            )
        )
    for index, row in enumerate(furniture_rows):
        low, high = row["low"], row["high"]
        proxies.append(
            CollisionProxy(
                f"Furniture_{index:02d}",
                row["path"],
                tuple((low[i] + high[i]) / 2 for i in range(3)),
                tuple(max(high[i] - low[i], 0.03) for i in range(3)),
                "furniture_conservative_aabb",
            )
        )

    collision_path = output_dir / "collision.usda"
    collision_stage = Usd.Stage.CreateNew(str(collision_path))
    collision_root = UsdGeom.Xform.Define(collision_stage, "/KinoScene")
    collision_stage.SetDefaultPrim(collision_root.GetPrim())
    collision_group = UsdGeom.Xform.Define(collision_stage, "/KinoScene/Collision")
    collision_group.GetPrim().CreateAttribute(
        "kino:sourceCollisionAuthoritative", Sdf.ValueTypeNames.Bool
    ).Set(False)
    for proxy in proxies:
        cube = UsdGeom.Cube.Define(collision_stage, f"/KinoScene/Collision/{proxy.name}")
        cube.CreateSizeAttr(1.0)
        UsdGeom.Imageable(cube.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        xform = UsdGeom.XformCommonAPI(cube)
        xform.SetTranslate(Gf.Vec3d(*proxy.center_xyz_m))
        xform.SetScale(Gf.Vec3f(*proxy.size_xyz_m))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        cube.GetPrim().CreateAttribute("kino:sourcePrim", Sdf.ValueTypeNames.String).Set(
            proxy.source_prim
        )
        cube.GetPrim().CreateAttribute("kino:semantic", Sdf.ValueTypeNames.String).Set(
            proxy.semantic
        )
    collision_stage.GetRootLayer().Save()

    lighting_path = output_dir / "lighting.usda"
    lighting_stage = Usd.Stage.CreateNew(str(lighting_path))
    lighting_root = UsdGeom.Xform.Define(lighting_stage, "/KinoScene")
    lighting_stage.SetDefaultPrim(lighting_root.GetPrim())
    lighting_scope = UsdGeom.Scope.Define(lighting_stage, "/KinoScene/Lighting")
    lighting_scope.GetPrim().CreateAttribute("kino:ownedBy", Sdf.ValueTypeNames.String).Set(
        "Kino-Fail"
    )
    dome = UsdLux.DomeLight.Define(lighting_stage, "/KinoScene/Lighting/NeutralDome")
    dome.CreateIntensityAttr(250.0)
    dome.CreateExposureAttr(0.0)
    panel_y = (
        floor_bounds.ymin + 0.18 * (floor_bounds.ymax - floor_bounds.ymin),
        (floor_bounds.ymin + floor_bounds.ymax) / 2,
        floor_bounds.ymin + 0.82 * (floor_bounds.ymax - floor_bounds.ymin),
    )
    panel_z = floor_z + min(max(wall_height - 0.25, 2.3), 2.75)
    for index, y in enumerate(panel_y):
        panel = UsdLux.RectLight.Define(
            lighting_stage, f"/KinoScene/Lighting/CeilingPanel_{index}"
        )
        panel.CreateIntensityAttr(450.0)
        panel.CreateExposureAttr(7.0)
        panel.CreateColorAttr(Gf.Vec3f(1.0, 0.90, 0.80))
        panel.CreateWidthAttr(min(1.0, 0.45 * sx))
        panel.CreateHeightAttr(0.35)
        UsdGeom.XformCommonAPI(panel).SetTranslate(Gf.Vec3d(cx, y, panel_z))
    lighting_stage.GetRootLayer().Save()

    route_path = output_dir / "route_operator.usda"
    route_stage = Usd.Stage.CreateNew(str(route_path))
    route_root = UsdGeom.Xform.Define(route_stage, "/KinoScene")
    route_stage.SetDefaultPrim(route_root.GetPrim())
    route_prim = UsdGeom.Xform.Define(route_stage, "/KinoScene/Route").GetPrim()
    flat_waypoints = [value for point in route["waypoints_xy_m"] for value in point]
    route_prim.CreateAttribute("kino:waypointsXY", Sdf.ValueTypeNames.DoubleArray).Set(
        flat_waypoints
    )
    route_prim.CreateAttribute("kino:requiredClearanceM", Sdf.ValueTypeNames.Double).Set(
        route["required_clearance_m"]
    )
    operator_prim = UsdGeom.Xform.Define(route_stage, "/KinoScene/Operators/O4").GetPrim()
    operator_prim.CreateAttribute("kino:operator", Sdf.ValueTypeNames.String).Set("foot_adhesion")
    operator_prim.CreateAttribute("kino:regionXY", Sdf.ValueTypeNames.DoubleArray).Set(
        [value for point in operator["vertices_xy_m"] for value in point]
    )
    operator_prim.CreateAttribute("kino:surfaceZM", Sdf.ValueTypeNames.Double).Set(floor_z)
    operator_prim.CreateAttribute("kino:visualMarkerAuthoritative", Sdf.ValueTypeNames.Bool).Set(
        False
    )
    route_stage.GetRootLayer().Save()

    episode_path = output_dir / "episode.usda"
    episode_layer = Sdf.Layer.CreateNew(str(episode_path))
    episode_layer.subLayerPaths = [
        "visual.usda",
        "collision.usda",
        "lighting.usda",
        "route_operator.usda",
    ]
    episode_layer.defaultPrim = "KinoScene"
    episode_layer.Save()

    files = [
        Path(visual["visual_usd"]),
        collision_path,
        lighting_path,
        route_path,
        episode_path,
    ]
    issues: list[str] = []
    if len(mesh_rows) < 5:
        issues.append("insufficient_visual_meshes")
    if sum(row["faces"] for row in mesh_rows) < 1000:
        issues.append("insufficient_visual_faces")
    if route["measured_min_clearance_m"] < route["required_clearance_m"] - route[
        "grid_resolution_m"
    ]:
        issues.append("route_clearance_failed")
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "scene_id": json.loads(source_manifest.read_text())["scene_id"],
        "source_kind": SOURCE_KIND,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "visual_metrics": {
            "mesh_count": len(mesh_rows),
            "point_count": sum(row["points"] for row in mesh_rows),
            "face_count": sum(row["faces"] for row in mesh_rows),
            "floor_source_prim": floor["path"],
            "floor_bounds_xy_m": asdict(floor_bounds),
            "floor_z_m": floor_z,
        },
        "physics_contract": {
            "owned_by": "Kino-Fail",
            "embodiedgen_collision_used": False,
            "proxy_count": len(proxies),
            "internal_wall_proxy_count": len(internal_wall_rows),
            "furniture_proxy_count": len(furniture_rows),
            "proxies": [asdict(proxy) for proxy in proxies],
        },
        "lighting_contract": {
            "owned_by": "Kino-Fail",
            "source_lights_retained": True,
            "neutral_dome_intensity": 250.0,
            "ceiling_rect_lights": {
                "count": 3,
                "intensity": 450.0,
                "exposure": 7.0,
                "positions_xyz_m": [[cx, y, panel_z] for y in panel_y],
            },
            "formal_randomization_fields": [
                "color_temperature",
                "intensity",
                "exposure",
                "enabled_panel_subset",
            ],
        },
        "route": route,
        "operator": operator,
        "files": {path.name: _sha256(path) for path in files},
        "passed": not issues,
        "issues": issues,
        "admission_state": (
            "geometry_route_passed_pending_rtx_front_camera_physics_qa"
            if not issues
            else "geometry_route_failed"
        ),
    }
    audit_path = output_dir / "compiled_scene_audit.json"
    audit_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return {**result, "audit_path": str(audit_path), "episode_usd": str(episode_path)}
