"""Frozen straight-route frames for realistic Kino-Fail collection.

The realistic terrain collector originally assumed a world-X route, a world-Y
centerline at zero, and a single hard-coded operator location.  This module
keeps route geometry and controller state explicit so physically distinct
scene realizations cannot silently collapse back to the same trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from kino_vla.utils.geometry import world_to_body, wrap_angle


@dataclass(frozen=True)
class StraightRouteFrame:
    """Metric frame recovered from a frozen compiled-scene audit."""

    origin_xy_m: tuple[float, float]
    direction_xy: tuple[float, float]
    left_xy: tuple[float, float]
    heading_rad: float
    route_length_m: float
    surface_width_m: float

    @classmethod
    def from_compiled_audit(cls, compiled: Mapping[str, Any]) -> "StraightRouteFrame":
        geometry = compiled.get("physics_contract", {}).get("route_geometry", {})
        route = compiled.get("route", {})
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
        return cls(
            origin_xy_m=(float(points[0, 0]), float(points[0, 1])),
            direction_xy=(float(direction[0]), float(direction[1])),
            left_xy=(float(left[0]), float(left[1])),
            heading_rad=float(math.atan2(direction[1], direction[0])),
            route_length_m=float(geometry["route_length_m"]),
            surface_width_m=float(geometry["surface_width_m"]),
        )

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
        """Return a body-frame velocity command for this route."""
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

