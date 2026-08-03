"""Settled lookahead route controller for realistic Kino-Fail collection.

The controller has one operator-independent initialization phase.  It first
acquires the requested lane offset without advancing toward the intervention,
then traverses using heading/lookahead feedback instead of sustained lateral
strafing.  The phase never resets after traversal starts, so an anomaly cannot
make the collector pause and silently rescue the robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from kino_vla.sim.realistic_route_protocol_v2 import RouteFrameV2
from kino_vla.utils.geometry import body_to_world, world_to_body, wrap_angle


SCHEMA_VERSION = "kinofail.settled-lookahead-route-controller.v4"


@dataclass
class SettledLookaheadRouteControllerV4:
    """Acquire a common lane, then steer along it with pure heading feedback."""

    alignment_gain_per_s: float = 0.8
    alignment_lateral_limit_mps: float = 0.12
    alignment_tolerance_m: float = 0.035
    alignment_dwell_steps: int = 10
    lookahead_m: float = 0.45
    heading_gain_per_s: float = 2.0
    yaw_rate_limit_radps: float = 0.60
    _phase: str = field(default="align", init=False, repr=False)
    _alignment_dwell: int = field(default=0, init=False, repr=False)
    _transition_step: int | None = field(default=None, init=False, repr=False)
    _command_step: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.alignment_gain_per_s <= 0.0:
            raise ValueError("alignment_gain_per_s must be positive")
        if self.alignment_lateral_limit_mps <= 0.0:
            raise ValueError("alignment_lateral_limit_mps must be positive")
        if self.alignment_tolerance_m <= 0.0:
            raise ValueError("alignment_tolerance_m must be positive")
        if self.alignment_dwell_steps < 1:
            raise ValueError("alignment_dwell_steps must be positive")
        if self.lookahead_m <= 0.0:
            raise ValueError("lookahead_m must be positive")
        if self.heading_gain_per_s <= 0.0 or self.yaw_rate_limit_radps <= 0.0:
            raise ValueError("heading gains and limits must be positive")

    def command(
        self,
        frame: RouteFrameV2,
        *,
        position_xy_m: np.ndarray,
        heading_rad: float,
        velocity_body_xy_mps: np.ndarray,
        forward_speed_mps: float,
        target_lateral_offset_m: float,
    ) -> tuple[np.ndarray, dict[str, float | int | str | None]]:
        if forward_speed_mps <= 0.0:
            raise ValueError("forward_speed_mps must be positive")
        velocity_body = np.asarray(velocity_body_xy_mps, dtype=np.float64)
        if velocity_body.shape != (2,):
            raise ValueError("velocity_body_xy_mps must have shape (2,)")

        self._command_step += 1
        progress, lateral = frame.project(position_xy_m)
        error = lateral - float(target_lateral_offset_m)
        velocity_world = body_to_world(velocity_body, float(heading_rad))
        measured_lateral_velocity = float(velocity_world @ frame.left)

        if self._phase == "align":
            if abs(error) <= self.alignment_tolerance_m:
                self._alignment_dwell += 1
            else:
                self._alignment_dwell = 0
            if self._alignment_dwell >= self.alignment_dwell_steps:
                self._phase = "traverse"
                self._transition_step = self._command_step

        if self._phase == "align":
            raw_lateral = -self.alignment_gain_per_s * error
            lateral_command = float(
                np.clip(
                    raw_lateral,
                    -self.alignment_lateral_limit_mps,
                    self.alignment_lateral_limit_mps,
                )
            )
            desired_world = lateral_command * frame.left
            body_xy = world_to_body(desired_world, float(heading_rad))
            desired_heading = frame.heading_rad
            correction_angle = 0.0
        else:
            raw_lateral = 0.0
            lateral_command = 0.0
            correction_angle = math.atan2(-error, self.lookahead_m)
            desired_heading = wrap_angle(frame.heading_rad + correction_angle)
            # Traverse longitudinally in the robot frame.  Cross-track error is
            # corrected by steering, not by a second lateral velocity loop.
            body_xy = np.asarray([float(forward_speed_mps), 0.0], dtype=np.float64)

        heading_error = wrap_angle(desired_heading - float(heading_rad))
        yaw_rate = float(
            np.clip(
                self.heading_gain_per_s * heading_error,
                -self.yaw_rate_limit_radps,
                self.yaw_rate_limit_radps,
            )
        )
        command = np.asarray([body_xy[0], body_xy[1], yaw_rate], dtype=np.float64)
        return command, {
            "controller_phase": self._phase,
            "controller_step": self._command_step,
            "alignment_dwell_steps_observed": self._alignment_dwell,
            "alignment_transition_step": self._transition_step,
            "route_progress_m": progress,
            "route_lateral_offset_m": lateral,
            "target_lateral_offset_m": float(target_lateral_offset_m),
            "lateral_error_m": error,
            "measured_lateral_velocity_mps": measured_lateral_velocity,
            "raw_alignment_lateral_command_mps": raw_lateral,
            "clipped_alignment_lateral_command_mps": lateral_command,
            "lookahead_correction_angle_rad": correction_angle,
            "desired_heading_rad": desired_heading,
            "heading_error_rad": heading_error,
            "yaw_rate_command_radps": yaw_rate,
        }

    def contract(self) -> dict[str, float | int | str]:
        return {
            "schema_version": SCHEMA_VERSION,
            "alignment_gain_per_s": self.alignment_gain_per_s,
            "alignment_lateral_limit_mps": self.alignment_lateral_limit_mps,
            "alignment_tolerance_m": self.alignment_tolerance_m,
            "alignment_dwell_steps": self.alignment_dwell_steps,
            "lookahead_m": self.lookahead_m,
            "heading_gain_per_s": self.heading_gain_per_s,
            "yaw_rate_limit_radps": self.yaw_rate_limit_radps,
            "phase_policy": "one_way_align_then_traverse_no_anomaly_reacquisition",
            "traverse_cross_track_actuation": "heading_only_no_lateral_strafe",
        }
