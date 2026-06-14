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
    slip_ratio: float  # contact slip estimate in [0, 1) — friction shortfall (O1/O3)
    base_height: float  # [m]
    tilt: float  # angle from vertical [rad]
    fallen: bool
    # M2: actuator-effort saturation estimate in [0, 1) — torque shortfall (O5/O10).
    # The third Kino-Monitor channel; 0 when the demanded tangential accel is within
    # the effort budget. Defaulted so M1 call sites and tests stay valid.
    effort_ratio: float = 0.0
    # M2: fraction of nominal foot support in (0, 1]; <1 when feet lose contact
    # (O9 high-centering). 1.0 nominal. Defaulted for back-compat.
    support_ratio: float = 1.0


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


@dataclass(frozen=True)
class CollapseRegion:
    """A region whose friction collapses once dwelled-on past a threshold (operator O3).

    Models thin ice / a trigger-and-swap collider: intact (``mu_intact``) until the
    robot accumulates ``trigger_dwell_s`` of contact inside it, then it drops to
    ``mu_collapsed`` for the rest of the episode (region-level topology hazard).
    """

    rect: Rect
    mu_intact: float
    mu_collapsed: float
    trigger_dwell_s: float


@dataclass(frozen=True)
class BlockingRegion:
    """An impassable collider with no effect until touched (operator O8: invisible wall).

    Forward motion into the region is hard-stopped at its boundary; nothing is
    rendered, so only proprioception (a tracking-error spike) reveals it.
    """

    rect: Rect


@dataclass(frozen=True)
class SupportLossRegion:
    """A region where feet lose ground support (operator O9: high-centering / beaching).

    Inside it, the fraction of foot support drops to ``residual_support`` in (0, 1]:
    the legs cannot transmit tangential force, so control authority is scaled down
    even though friction is nominal (the belly bears the load).
    """

    rect: Rect
    residual_support: float
