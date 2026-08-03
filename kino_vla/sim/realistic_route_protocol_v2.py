"""Cross-domain route binding for realistic Kino-Fail terrain operators.

Version 1 is intentionally frozen for the forest G01--G07 evidence.  This
module adds an explicit adapter for EmbodiedGen v4 route-surface scenes without
changing that historical implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from kino_vla.utils.geometry import world_to_body, wrap_angle


FOREST_SCHEMA_PREFIX = "kinofail.forest-hybrid-compiled-scene."
EMBODIEDGEN_V4_SCHEMA = "kinofail.embodiedgen-compiled-scene.v4-development"
EMBODIEDGEN_TERRAIN_V1_SCHEMA = (
    "kinofail.embodiedgen-terrain-route-compiled-scene.v1-development"
)
EMBODIEDGEN_TERRAIN_V2_SCHEMA = (
    "kinofail.embodiedgen-terrain-route-compiled-scene.v2-development"
)


@dataclass(frozen=True)
class SceneRouteBinding:
    """Runtime paths and metric frame bound to one compiled scene audit."""

    source_kind: str
    scene_id: str
    episode_filename: str
    floor_prim_suffix: str
    route_surface_prim_suffix: str
    route_surface_collision_authored: bool
    collector_start_waypoint_index: int
    collector_start_progress_m: float
    frame: "RouteFrameV2"

    def floor_prim_path(self, scene_prim: str) -> str:
        return f"{scene_prim}/{self.floor_prim_suffix}"

    def route_surface_prim_path(self, scene_prim: str) -> str:
        return f"{scene_prim}/{self.route_surface_prim_suffix}"


@dataclass(frozen=True)
class RouteFrameV2:
    """Straight route frame shared by forest and EmbodiedGen scene packages."""

    origin_xy_m: tuple[float, float]
    direction_xy: tuple[float, float]
    left_xy: tuple[float, float]
    heading_rad: float
    route_length_m: float
    surface_width_m: float

    @classmethod
    def from_compiled_audit(cls, compiled: Mapping[str, Any]) -> "RouteFrameV2":
        binding = scene_route_binding(compiled)
        return binding.frame

    @property
    def origin(self) -> np.ndarray:
        return np.asarray(self.origin_xy_m, dtype=np.float64)

    @property
    def direction(self) -> np.ndarray:
        return np.asarray(self.direction_xy, dtype=np.float64)

    @property
    def left(self) -> np.ndarray:
        return np.asarray(self.left_xy, dtype=np.float64)

    def point(self, progress_m: float, lateral_offset_m: float = 0.0) -> np.ndarray:
        return self.origin + float(progress_m) * self.direction + float(lateral_offset_m) * self.left

    def project(self, position_xy_m: np.ndarray) -> tuple[float, float]:
        delta = np.asarray(position_xy_m, dtype=np.float64) - self.origin
        return float(delta @ self.direction), float(delta @ self.left)

    def controller_command(
        self,
        *,
        position_xy_m: np.ndarray,
        heading_rad: float,
        forward_speed_mps: float,
        target_lateral_offset_m: float,
        cross_track_gain: float = 1.0,
        heading_gain: float = 2.0,
        lateral_limit_mps: float = 0.16,
        yaw_rate_limit_radps: float = 0.60,
    ) -> np.ndarray:
        _, lateral = self.project(position_xy_m)
        lateral_speed = float(
            np.clip(
                -float(cross_track_gain) * (lateral - float(target_lateral_offset_m)),
                -float(lateral_limit_mps),
                float(lateral_limit_mps),
            )
        )
        desired_world = float(forward_speed_mps) * self.direction + lateral_speed * self.left
        body_xy = world_to_body(desired_world, float(heading_rad))
        yaw_rate = float(
            np.clip(
                float(heading_gain) * wrap_angle(self.heading_rad - float(heading_rad)),
                -float(yaw_rate_limit_radps),
                float(yaw_rate_limit_radps),
            )
        )
        return np.asarray([body_xy[0], body_xy[1], yaw_rate], dtype=np.float64)


def is_supported_compiled_scene(compiled: Mapping[str, Any]) -> bool:
    schema = str(compiled.get("schema_version", ""))
    return schema.startswith(FOREST_SCHEMA_PREFIX) or schema in {
        EMBODIEDGEN_V4_SCHEMA,
        EMBODIEDGEN_TERRAIN_V1_SCHEMA,
        EMBODIEDGEN_TERRAIN_V2_SCHEMA,
    }


def _frame_from_geometry(
    *, route: Mapping[str, Any], geometry: Mapping[str, Any]
) -> RouteFrameV2:
    points = np.asarray(route.get("waypoints_xy_m", []), dtype=np.float64)
    direction = np.asarray(geometry.get("direction_xy", []), dtype=np.float64)
    left = np.asarray(geometry.get("left_xy", []), dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2:
        raise ValueError("compiled audit has no valid straight-route waypoints")
    if direction.shape != (2,) or left.shape != (2,):
        raise ValueError("compiled audit has no valid route frame")
    if not np.isfinite(points).all() or not np.isfinite(direction).all() or not np.isfinite(left).all():
        raise ValueError("compiled route frame must be finite")
    if abs(float(np.linalg.norm(direction)) - 1.0) > 1.0e-6:
        raise ValueError("compiled route direction must be unit length")
    if abs(float(np.linalg.norm(left)) - 1.0) > 1.0e-6:
        raise ValueError("compiled route left axis must be unit length")
    if abs(float(direction @ left)) > 1.0e-6:
        raise ValueError("compiled route axes must be orthogonal")
    route_length = float(geometry["route_length_m"])
    width = float(geometry["surface_width_m"])
    projected = (points - points[0]) @ left
    if float(np.max(np.abs(projected))) > 1.0e-5:
        raise ValueError("terrain operator adapter requires a straight route")
    endpoint_progress = float((points[-1] - points[0]) @ direction)
    if abs(endpoint_progress - route_length) > 1.0e-5:
        raise ValueError("route waypoints and metric geometry disagree on length")
    if route_length <= 0.0 or width <= 0.0:
        raise ValueError("route length and surface width must be positive")
    return RouteFrameV2(
        origin_xy_m=(float(points[0, 0]), float(points[0, 1])),
        direction_xy=(float(direction[0]), float(direction[1])),
        left_xy=(float(left[0]), float(left[1])),
        heading_rad=float(math.atan2(direction[1], direction[0])),
        route_length_m=route_length,
        surface_width_m=width,
    )


def scene_route_binding(compiled: Mapping[str, Any]) -> SceneRouteBinding:
    """Fail closed when a scene cannot support route-local O1--O3 physics."""
    if compiled.get("passed") is not True:
        raise ValueError("compiled scene must pass before route binding")
    schema = str(compiled.get("schema_version", ""))
    route = compiled.get("route", {})
    if schema.startswith(FOREST_SCHEMA_PREFIX):
        geometry = compiled.get("physics_contract", {}).get("route_geometry", {})
        frame = _frame_from_geometry(route=route, geometry=geometry)
        episode = "episode.usda"
        surface_suffix = "Appearance/RouteSurface"
        source_kind = "forest_hybrid"
        collision_authored = False
        collector_start_index = 0
        collector_start_progress = 0.0
    elif schema in {
        EMBODIEDGEN_V4_SCHEMA,
        EMBODIEDGEN_TERRAIN_V1_SCHEMA,
        EMBODIEDGEN_TERRAIN_V2_SCHEMA,
    }:
        appearance = compiled.get("appearance_contract", {})
        if appearance.get("schema") != "kinofail.render-only-route-surface.v1":
            raise ValueError("EmbodiedGen scene lacks the frozen render-only route surface")
        if appearance.get("visual_intervention_only") is not True:
            raise ValueError("EmbodiedGen route surface is not visual-only")
        if appearance.get("collision_authored") is not False:
            raise ValueError("EmbodiedGen route surface must not author collision")
        geometry = appearance.get("geometry", {})
        full_frame = _frame_from_geometry(route=route, geometry=geometry)
        points = np.asarray(route.get("waypoints_xy_m", []), dtype=np.float64)
        collector_start_index = int(route.get("collector_start_waypoint_index", -1))
        if not 0 <= collector_start_index < len(points):
            raise ValueError("EmbodiedGen route lacks a valid collector start waypoint")
        collector_start_progress, collector_start_lateral = full_frame.project(
            points[collector_start_index]
        )
        if abs(collector_start_lateral) > 1.0e-5:
            raise ValueError("EmbodiedGen collector start is not on the route centerline")
        surface_end_progress = float(geometry["route_length_m"]) + float(
            geometry["endpoint_margin_m"]
        )
        available_length = surface_end_progress - collector_start_progress
        if available_length <= 0.0:
            raise ValueError("EmbodiedGen collector start leaves no operator route length")
        frame = RouteFrameV2(
            origin_xy_m=(
                float(points[collector_start_index, 0]),
                float(points[collector_start_index, 1]),
            ),
            direction_xy=full_frame.direction_xy,
            left_xy=full_frame.left_xy,
            heading_rad=full_frame.heading_rad,
            route_length_m=available_length,
            surface_width_m=full_frame.surface_width_m,
        )
        episode = {
            EMBODIEDGEN_V4_SCHEMA: "episode_v4.usda",
            EMBODIEDGEN_TERRAIN_V1_SCHEMA: "episode_terrain_v1.usda",
            EMBODIEDGEN_TERRAIN_V2_SCHEMA: "episode_terrain_v2.usda",
        }[schema]
        surface_suffix = "AppearanceV4/RenderOnlyRouteSurface"
        source_kind = {
            EMBODIEDGEN_V4_SCHEMA: "embodiedgen_v4",
            EMBODIEDGEN_TERRAIN_V1_SCHEMA: "embodiedgen_terrain_v1",
            EMBODIEDGEN_TERRAIN_V2_SCHEMA: "embodiedgen_terrain_v2",
        }[schema]
        collision_authored = False
    else:
        raise ValueError(f"unsupported compiled scene schema: {schema}")
    if episode not in compiled.get("files", {}):
        raise ValueError(f"compiled audit does not bind {episode}")
    return SceneRouteBinding(
        source_kind=source_kind,
        scene_id=str(compiled.get("scene_id", "")),
        episode_filename=episode,
        floor_prim_suffix="Collision/Floor",
        route_surface_prim_suffix=surface_suffix,
        route_surface_collision_authored=collision_authored,
        collector_start_waypoint_index=collector_start_index,
        collector_start_progress_m=collector_start_progress,
        frame=frame,
    )


def resolve_bound_episode(compiled_path: Path, compiled: Mapping[str, Any]) -> Path:
    """Resolve the episode sublayer bound by the compiled audit."""
    binding = scene_route_binding(compiled)
    episode = compiled_path.resolve().parent / binding.episode_filename
    if not episode.is_file():
        raise FileNotFoundError(episode)
    return episode
