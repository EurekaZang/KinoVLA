"""Damped route controller for long realistic Kino-Fail terrain traversals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kino_vla.sim.realistic_route_protocol_v2 import RouteFrameV2
from kino_vla.utils.geometry import body_to_world, world_to_body, wrap_angle


SCHEMA_VERSION = "kinofail.damped-straight-route-controller.v3"


@dataclass(frozen=True)
class DampedRouteControllerV3:
    """World-frame lateral PD tracking with body-frame velocity commands."""

    cross_track_gain_per_s: float = 1.0
    lateral_velocity_damping: float = 1.0
    heading_gain_per_s: float = 2.0
    lateral_limit_mps: float = 0.16
    yaw_rate_limit_radps: float = 0.60

    def __post_init__(self) -> None:
        if self.cross_track_gain_per_s <= 0.0:
            raise ValueError("cross_track_gain_per_s must be positive")
        if self.lateral_velocity_damping < 0.0:
            raise ValueError("lateral_velocity_damping must be nonnegative")
        if self.heading_gain_per_s <= 0.0:
            raise ValueError("heading_gain_per_s must be positive")
        if self.lateral_limit_mps <= 0.0 or self.yaw_rate_limit_radps <= 0.0:
            raise ValueError("command limits must be positive")

    def command(
        self,
        frame: RouteFrameV2,
        *,
        position_xy_m: np.ndarray,
        heading_rad: float,
        velocity_body_xy_mps: np.ndarray,
        forward_speed_mps: float,
        target_lateral_offset_m: float,
    ) -> tuple[np.ndarray, dict[str, float]]:
        _, lateral = frame.project(position_xy_m)
        velocity_body = np.asarray(velocity_body_xy_mps, dtype=np.float64)
        if velocity_body.shape != (2,):
            raise ValueError("velocity_body_xy_mps must have shape (2,)")
        velocity_world = body_to_world(velocity_body, float(heading_rad))
        measured_lateral_velocity = float(velocity_world @ frame.left)
        error = lateral - float(target_lateral_offset_m)
        raw_lateral_command = -(
            self.cross_track_gain_per_s * error
            + self.lateral_velocity_damping * measured_lateral_velocity
        )
        lateral_command = float(
            np.clip(raw_lateral_command, -self.lateral_limit_mps, self.lateral_limit_mps)
        )
        desired_world = (
            float(forward_speed_mps) * frame.direction + lateral_command * frame.left
        )
        body_xy = world_to_body(desired_world, float(heading_rad))
        heading_error = wrap_angle(frame.heading_rad - float(heading_rad))
        yaw_rate = float(
            np.clip(
                self.heading_gain_per_s * heading_error,
                -self.yaw_rate_limit_radps,
                self.yaw_rate_limit_radps,
            )
        )
        return np.asarray([body_xy[0], body_xy[1], yaw_rate], dtype=np.float64), {
            "route_lateral_offset_m": lateral,
            "target_lateral_offset_m": float(target_lateral_offset_m),
            "lateral_error_m": error,
            "measured_lateral_velocity_mps": measured_lateral_velocity,
            "raw_lateral_command_mps": raw_lateral_command,
            "clipped_lateral_command_mps": lateral_command,
            "heading_error_rad": heading_error,
        }

    def contract(self) -> dict[str, float | str]:
        return {
            "schema_version": SCHEMA_VERSION,
            "cross_track_gain_per_s": self.cross_track_gain_per_s,
            "lateral_velocity_damping": self.lateral_velocity_damping,
            "heading_gain_per_s": self.heading_gain_per_s,
            "lateral_limit_mps": self.lateral_limit_mps,
            "yaw_rate_limit_radps": self.yaw_rate_limit_radps,
        }
