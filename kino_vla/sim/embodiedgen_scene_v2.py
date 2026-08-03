"""Robust experiment-corridor refinement for admitted EmbodiedGen scenes.

Version 1 deliberately remains immutable because its code hash is part of a sealed
confirmation batch.  This module consumes a passed v1 compilation, preserves its
Kino-owned collision layer, and authors a versioned route/lighting episode with an
explicit allowance for closed-loop Go2 tracking error.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from kino_vla.sim.embodiedgen_scene import Box2D, _adhesion_region


SCHEMA_VERSION = "kinofail.embodiedgen-compiled-scene.v2"
ROBOT_RADIUS_M = 0.34
PHYSICAL_SAFETY_MARGIN_M = 0.10
# Frozen only after development traces exposed 0.252 m and 0.277 m lateral
# excursions in the rejected DiningRoom scene.  The 0.30 m bound is a rounded,
# conservative envelope; it is not estimated from any future formal scene.
MAX_TRACKING_ERROR_M = 0.30
ROBUST_CENTERLINE_CLEARANCE_M = (
    ROBOT_RADIUS_M + PHYSICAL_SAFETY_MARGIN_M + MAX_TRACKING_ERROR_M
)
CORRIDOR_REVERSE_EXTENT_M = 1.10
CORRIDOR_FORWARD_EXTENT_M = 1.35
CORRIDOR_LENGTH_M = CORRIDOR_REVERSE_EXTENT_M + CORRIDOR_FORWARD_EXTENT_M


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _point_box_distance(x: float, y: float, box: Box2D) -> float:
    dx = max(box.xmin - x, 0.0, x - box.xmax)
    dy = max(box.ymin - y, 0.0, y - box.ymax)
    return math.hypot(dx, dy)


def _subtract_intervals(
    lower: float,
    upper: float,
    forbidden: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    clipped = sorted(
        (max(lower, start), min(upper, end))
        for start, end in forbidden
        if end > lower and start < upper
    )
    merged: list[list[float]] = []
    for start, end in clipped:
        if start >= end:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    free: list[tuple[float, float]] = []
    cursor = lower
    for start, end in merged:
        if start > cursor:
            free.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < upper:
        free.append((cursor, upper))
    return free


def _sample_values(lower: float, upper: float, spacing: float) -> list[float]:
    count = max(1, math.floor((upper - lower) / spacing))
    values = [lower + index * (upper - lower) / count for index in range(count + 1)]
    values.append((lower + upper) / 2.0)
    return sorted(set(round(value, 9) for value in values))


def _corridor_clearance(
    points: list[tuple[float, float]],
    floor: Box2D,
    obstacles: tuple[Box2D, ...],
) -> float:
    values = []
    for x, y in points:
        wall_clearance = min(
            x - floor.xmin,
            floor.xmax - x,
            y - floor.ymin,
            floor.ymax - y,
        )
        obstacle_clearance = min(
            (_point_box_distance(x, y, box) for box in obstacles),
            default=1.0e6,
        )
        values.append(min(wall_clearance, obstacle_clearance))
    return min(values)


def find_robust_experiment_corridor(
    floor: Box2D,
    obstacles: tuple[Box2D, ...],
    *,
    robust_clearance_m: float = ROBUST_CENTERLINE_CLEARANCE_M,
    corridor_length_m: float = CORRIDOR_LENGTH_M,
    lane_spacing_m: float = 0.08,
) -> dict[str, Any]:
    """Find a straight corridor safe under a bounded lateral tracking error.

    Obstacles and walls are inflated by ``robust_clearance_m``. Candidate lanes are
    evaluated on both room axes. The fixed-length experiment segment is centred in
    each admissible gap, then ranked by independently recomputed raw clearance.
    """
    if robust_clearance_m <= 0.0 or corridor_length_m <= 0.0 or lane_spacing_m <= 0.0:
        raise ValueError("corridor dimensions must be positive")
    free = Box2D(
        "robust_room_interior",
        floor.xmin + robust_clearance_m,
        floor.xmax - robust_clearance_m,
        floor.ymin + robust_clearance_m,
        floor.ymax - robust_clearance_m,
    )
    if free.xmin >= free.xmax or free.ymin >= free.ymax:
        raise ValueError("room has no interior after robust Go2 envelope inflation")
    inflated = tuple(box.expanded(robust_clearance_m) for box in obstacles)
    candidates: list[dict[str, Any]] = []

    for axis in ("x", "y"):
        lane_lower, lane_upper = (
            (free.ymin, free.ymax) if axis == "x" else (free.xmin, free.xmax)
        )
        run_lower, run_upper = (
            (free.xmin, free.xmax) if axis == "x" else (free.ymin, free.ymax)
        )
        for lane in _sample_values(lane_lower, lane_upper, lane_spacing_m):
            forbidden = []
            for box in inflated:
                intersects_lane = (
                    box.ymin <= lane <= box.ymax
                    if axis == "x"
                    else box.xmin <= lane <= box.xmax
                )
                if intersects_lane:
                    forbidden.append(
                        (box.xmin, box.xmax) if axis == "x" else (box.ymin, box.ymax)
                    )
            for gap_start, gap_end in _subtract_intervals(
                run_lower, run_upper, forbidden
            ):
                gap_length = gap_end - gap_start
                if gap_length + 1.0e-9 < corridor_length_m:
                    continue
                segment_start = (gap_start + gap_end - corridor_length_m) / 2.0
                segment_end = segment_start + corridor_length_m
                sample_count = max(2, math.ceil(corridor_length_m / 0.04))
                if axis == "x":
                    points = [
                        (
                            segment_start
                            + index * (segment_end - segment_start) / sample_count,
                            lane,
                        )
                        for index in range(sample_count + 1)
                    ]
                    room_centre_offset = abs(lane - (floor.ymin + floor.ymax) / 2.0)
                else:
                    points = [
                        (
                            lane,
                            segment_start
                            + index * (segment_end - segment_start) / sample_count,
                        )
                        for index in range(sample_count + 1)
                    ]
                    room_centre_offset = abs(lane - (floor.xmin + floor.xmax) / 2.0)
                measured_clearance = _corridor_clearance(points, floor, obstacles)
                if measured_clearance + 1.0e-6 < robust_clearance_m:
                    continue
                candidates.append(
                    {
                        "axis": axis,
                        "lane_coordinate_m": lane,
                        "gap_start_m": gap_start,
                        "gap_end_m": gap_end,
                        "gap_length_m": gap_length,
                        "segment_start_m": segment_start,
                        "segment_end_m": segment_end,
                        "measured_centerline_clearance_m": measured_clearance,
                        "room_centre_offset_m": room_centre_offset,
                    }
                )
    if not candidates:
        raise ValueError(
            "no straight experiment corridor satisfies the robust Go2 envelope: "
            f"clearance={robust_clearance_m:.3f}m length={corridor_length_m:.3f}m"
        )
    selected = max(
        candidates,
        key=lambda row: (
            row["measured_centerline_clearance_m"],
            row["gap_length_m"],
            -row["room_centre_offset_m"],
            row["axis"] == "x",
        ),
    )
    axis = selected["axis"]
    back = selected["segment_start_m"]
    start_scalar = back + CORRIDOR_REVERSE_EXTENT_M
    scalars = [back, start_scalar - 0.55, start_scalar, start_scalar + 0.70, selected["segment_end_m"]]
    if axis == "x":
        waypoints = [[value, selected["lane_coordinate_m"]] for value in scalars]
        direction = [1.0, 0.0]
    else:
        waypoints = [[selected["lane_coordinate_m"], value] for value in scalars]
        direction = [0.0, 1.0]
    selected.update(
        {
            "waypoints_xy_m": [
                [round(float(x), 6), round(float(y), 6)] for x, y in waypoints
            ],
            "collector_start_waypoint_index": 2,
            "collector_start_xy_m": [round(float(v), 6) for v in waypoints[2]],
            "direction_xy": direction,
            "candidate_count": len(candidates),
        }
    )
    return selected


def _obstacles_from_base(base: dict[str, Any]) -> tuple[Box2D, ...]:
    rows = []
    for proxy in base["physics_contract"]["proxies"]:
        if proxy["semantic"] not in {
            "furniture_conservative_aabb",
            "internal_wall_face_aabb",
        }:
            continue
        cx, cy, _ = proxy["center_xyz_m"]
        sx, sy, _ = proxy["size_xyz_m"]
        rows.append(
            Box2D(
                proxy["name"],
                float(cx) - float(sx) / 2.0,
                float(cx) + float(sx) / 2.0,
                float(cy) - float(sy) / 2.0,
                float(cy) + float(sy) / 2.0,
            )
        )
    return tuple(rows)


def refine_embodiedgen_kinofail_scene_v2(
    base_compiled_audit: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Author a robust-corridor episode while retaining the frozen v1 collision source."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

    base_path = Path(base_compiled_audit).resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("passed") is not True:
        raise ValueError("base compiled-scene audit has not passed")
    base_dir = base_path.parent
    for name, expected in base["files"].items():
        path = base_dir / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"stale base compiled-scene file: {path}")
    source_manifest = Path(base["source_manifest"]).resolve()
    if _sha256(source_manifest) != base["source_manifest_sha256"]:
        raise ValueError("source manifest no longer matches base compilation")

    floor_dict = base["visual_metrics"]["floor_bounds_xy_m"]
    floor = Box2D(
        "navigable_floor",
        float(floor_dict["xmin"]),
        float(floor_dict["xmax"]),
        float(floor_dict["ymin"]),
        float(floor_dict["ymax"]),
    )
    floor_z = float(base["visual_metrics"]["floor_z_m"])
    obstacles = _obstacles_from_base(base)
    corridor = find_robust_experiment_corridor(floor, obstacles)
    waypoints = corridor["waypoints_xy_m"]
    route = {
        "long_axis": corridor["axis"],
        "route_selection": "robust_straight_experiment_corridor_v2",
        "waypoints_xy_m": waypoints,
        "route_length_m": CORRIDOR_LENGTH_M,
        "minimum_route_length_m": CORRIDOR_LENGTH_M,
        "maximum_waypoint_spacing_m": 0.70,
        "robot_radius_m": ROBOT_RADIUS_M,
        "safety_margin_m": PHYSICAL_SAFETY_MARGIN_M,
        "required_clearance_m": ROBOT_RADIUS_M + PHYSICAL_SAFETY_MARGIN_M,
        "measured_min_clearance_m": corridor["measured_centerline_clearance_m"],
        "grid_resolution_m": 0.08,
        "dense_node_count": 0,
        "maximum_admissible_tracking_error_m": MAX_TRACKING_ERROR_M,
        "tracking_error_budget_basis": {
            "development_only_scene": "indoor_diningroom_02",
            "observed_rejected_v1_max_deviation_m": 0.2522817205920219,
            "observed_rejected_v2_calibration_max_deviation_m": 0.276611407851696,
            "rounded_frozen_budget_m": MAX_TRACKING_ERROR_M,
            "formal_scene_used_to_set_budget": False,
        },
        "robust_centerline_clearance_m": ROBUST_CENTERLINE_CLEARANCE_M,
        "collector_start_waypoint_index": 2,
        "corridor_candidate_count": corridor["candidate_count"],
        "corridor_gap_length_m": corridor["gap_length_m"],
        "corridor_lane_coordinate_m": corridor["lane_coordinate_m"],
    }
    operator = _adhesion_region(route, floor_z)

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lighting_path = output / "lighting_v2.usda"
    lighting_stage = Usd.Stage.CreateNew(str(lighting_path))
    lighting_root = UsdGeom.Xform.Define(lighting_stage, "/KinoScene")
    lighting_stage.SetDefaultPrim(lighting_root.GetPrim())
    scope = UsdGeom.Scope.Define(lighting_stage, "/KinoScene/Lighting")
    scope.GetPrim().CreateAttribute("kino:ownedBy", Sdf.ValueTypeNames.String).Set("Kino-Fail")
    dome = UsdLux.DomeLight.Define(lighting_stage, "/KinoScene/Lighting/RouteAwareDome")
    dome.CreateIntensityAttr(350.0)
    dome.CreateExposureAttr(0.0)
    room_height = max(
        float(proxy["center_xyz_m"][2]) + float(proxy["size_xyz_m"][2]) / 2.0
        for proxy in base["physics_contract"]["proxies"]
    ) - floor_z
    panel_z = floor_z + min(max(room_height - 0.25, 2.3), 2.75)
    panel_positions = []
    for index, waypoint_index in enumerate((0, 1, 2, 3, 4)):
        x, y = waypoints[waypoint_index]
        panel_positions.append([x, y, panel_z])
        panel = UsdLux.RectLight.Define(
            lighting_stage, f"/KinoScene/Lighting/CorridorPanel_{index}"
        )
        panel.CreateIntensityAttr(450.0)
        panel.CreateExposureAttr(7.0)
        panel.CreateColorAttr(Gf.Vec3f(1.0, 0.92, 0.84))
        panel.CreateWidthAttr(0.85)
        panel.CreateHeightAttr(0.45)
        UsdGeom.XformCommonAPI(panel).SetTranslate(Gf.Vec3d(x, y, panel_z))
    lighting_stage.GetRootLayer().Save()

    route_path = output / "route_operator_v2.usda"
    route_stage = Usd.Stage.CreateNew(str(route_path))
    route_root = UsdGeom.Xform.Define(route_stage, "/KinoScene")
    route_stage.SetDefaultPrim(route_root.GetPrim())
    route_prim = UsdGeom.Xform.Define(route_stage, "/KinoScene/Route").GetPrim()
    route_prim.CreateAttribute("kino:waypointsXY", Sdf.ValueTypeNames.DoubleArray).Set(
        [value for point in waypoints for value in point]
    )
    route_prim.CreateAttribute("kino:requiredClearanceM", Sdf.ValueTypeNames.Double).Set(
        ROBUST_CENTERLINE_CLEARANCE_M
    )
    route_prim.CreateAttribute("kino:maxTrackingErrorM", Sdf.ValueTypeNames.Double).Set(
        MAX_TRACKING_ERROR_M
    )
    operator_prim = UsdGeom.Xform.Define(route_stage, "/KinoScene/Operators/O4").GetPrim()
    operator_prim.CreateAttribute("kino:operator", Sdf.ValueTypeNames.String).Set("foot_adhesion")
    operator_prim.CreateAttribute("kino:regionXY", Sdf.ValueTypeNames.DoubleArray).Set(
        [value for point in operator["vertices_xy_m"] for value in point]
    )
    operator_prim.CreateAttribute("kino:surfaceZM", Sdf.ValueTypeNames.Double).Set(floor_z)
    operator_prim.CreateAttribute("kino:visualMarkerAuthoritative", Sdf.ValueTypeNames.Bool).Set(False)
    route_stage.GetRootLayer().Save()

    episode_path = output / "episode_v2.usda"
    episode_layer = Sdf.Layer.CreateNew(str(episode_path))
    episode_layer.subLayerPaths = [
        os.path.relpath(base_dir / "visual.usda", output),
        os.path.relpath(base_dir / "collision.usda", output),
        lighting_path.name,
        route_path.name,
    ]
    episode_layer.defaultPrim = "KinoScene"
    episode_layer.Save()

    lighting_contract = {
        "owned_by": "Kino-Fail",
        "source_lights_retained": True,
        "neutral_dome_intensity": 350.0,
        "route_aware_rect_lights": {
            "count": len(panel_positions),
            "intensity": 450.0,
            "exposure": 7.0,
            "positions_xyz_m": panel_positions,
        },
        "formal_randomization_fields": [
            "color_temperature",
            "intensity",
            "exposure",
            "enabled_panel_subset",
        ],
    }
    files = {
        "visual.usda": _sha256(base_dir / "visual.usda"),
        "collision.usda": _sha256(base_dir / "collision.usda"),
        lighting_path.name: _sha256(lighting_path),
        route_path.name: _sha256(route_path),
        episode_path.name: _sha256(episode_path),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "scene_id": base["scene_id"],
        "source_kind": base["source_kind"],
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "base_compiled_audit": str(base_path),
        "base_compiled_audit_sha256": _sha256(base_path),
        "visual_metrics": base["visual_metrics"],
        "physics_contract": base["physics_contract"],
        "lighting_contract": lighting_contract,
        "route": route,
        "operator": operator,
        "corridor_search": corridor,
        "files": files,
        "passed": True,
        "issues": [],
        "admission_state": "robust_corridor_v2_passed_pending_rtx_front_camera_physics_qa",
    }
    audit_path = output / "compiled_scene_audit.json"
    audit_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {**result, "audit_path": str(audit_path), "episode_usd": str(episode_path)}
