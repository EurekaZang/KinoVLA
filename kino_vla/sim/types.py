"""Shared sim-layer types: observation snapshot and friction regions.

The observation is the Sport-Client-level view a real Go2 controller would have
(odometry, IMU-derived attitude, commanded vs. measured velocity, contact slip
estimate). Operators on Axis IV (embodiment degradation, e.g. O11 Obs-Bias)
corrupt this *measured* view; the backend's internal state stays the privileged
truth (spec §8.2 P2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kino_vla.utils.geometry import Rect


@dataclass(frozen=True)
class Obs:
    """One control-step observation snapshot (body frame where noted)."""

    t: float
    pos: np.ndarray  # (2,) world xy [m] (odometry)
    heading: float  # yaw [rad]
    vel_body: np.ndarray  # (2,) measured body-frame (vx, vy) [m/s]
    yaw_rate: float  # [rad/s]
    cmd_prev: np.ndarray  # (3,) last applied command (vx, vy, wz)
    slip_ratio: float  # contact slip estimate in [0, 1)
    base_height: float  # [m]
    tilt: float  # angle from vertical [rad]
    fallen: bool


# Observation fields O11 may bias — the IMU/odometry-derived channels (spec §8.2:
# "inject IMU bias/drift into the observation stream").
BIASABLE_OBS_FIELDS: tuple[str, ...] = ("vel_body", "yaw_rate", "heading", "tilt")


@dataclass(frozen=True)
class FrictionRegion:
    """A rectangular ground region with overridden contact material (operator O1)."""

    rect: Rect
    mu_s: float  # static friction
    mu_d: float  # dynamic friction
    restitution: float = 0.0
