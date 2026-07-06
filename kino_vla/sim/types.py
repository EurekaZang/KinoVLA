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
class ResistanceRegion:
    """A region applying a displacement-dependent tangential resistance (operators O2/O4).

    Models the shared mechanical signature of compliant sinking (O2) and elastic
    adhesion/tether (O4): a force opposing motion that grows linearly with the path
    length ``s`` travelled inside the region plus a viscous term:

        ``F_resist(s, v) = stiffness * s + damping * |v|``

    The two operators differ in *kind* (privileged truth) and *appearance* (mud vs
    adhesive board) but, by construction (matched ``stiffness``/``damping``), produce an
    **identical tangential-resistance-vs-displacement curve** — the spec §8.1 P4 ambiguity
    pair: proprioception alone cannot separate them, so the visual map must. ``sink_depth_m``
    lowers the measured base height (O2 geometric sink, 0 for O4); ``break_force_n`` snaps
    the resistance once exceeded (O4 tether break, ``inf`` for O2).
    """

    rect: Rect
    stiffness_n_per_m: float
    damping_ns_per_m: float
    sink_depth_m: float = 0.0
    break_force_n: float = float("inf")
    slack_length_m: float = 0.0  # O4 tether free length L_0 before the spring engages
    kind: str = "compliance"
    # O4 ADHESIVE GRIP (Bug-1): the hold grows with PENETRATION (forward progress into the patch),
    # resists going deeper at full strength (push-through stalls), but is scaled by peel_factor (<1)
    # when reversing, so the dog escapes by PEELING OUT (backing off), not by pushing through. The
    # grip still snaps at break_force_n (a LOW-break tether tears under forward load, the M5 gate).
    peel_factor: float = 1.0
    # #49 PEEL-PLATEAU force shaping (E1 calibration; tether kind only): cap the spring grip at
    # ``force_cap_n`` and add a constant pre-load ``force_offset_n`` ⇒ ``grip = min(k·(pen−L₀),
    # force_cap_n) + force_offset_n``. Physically a real adhesive peel is a BOUNDED, near-constant
    # force, not an unbounded Hookean spring. This lets O4's forward signal be made a constant drag
    # matching O2 (set force_cap_n=0, force_offset_n=k_c) so the E1 C2ST can reach the strong claim.
    # Defaults (inf, 0) reproduce the unshaped tether EXACTLY — no behaviour change for any caller.
    force_cap_n: float = float("inf")
    force_offset_n: float = 0.0
    # A4.1 TWO-PHASE (delayed-divergence) adhesion (experiments_design.md §4 A4.1). When ``p0_m>0``
    # the grip law is REPLACED by a plateau-then-ramp: ``grip = force_offset_n`` (a constant plateau
    # byte-identical to O2 compliance's k_c drag, for pen ≤ p0_m) then ``force_offset_n +
    # k2_n_per_m·(pen − p0_m)`` beyond. The plateau is where attribution happens (C2ST-
    # indistinguishable from O2; A1.3 re-certifies it); the ramp is the CONSEQUENCE region.
    # ``f_break`` then fires on the RAMP grip: finite ⇒ the tether tears under forward lean
    # (a "catapult" release); ``f_break=inf`` + high ``k2`` ⇒ unbounded hold ("immobilization").
    # Default (0, 0) reproduces the #49 path EXACTLY — no behaviour change for any caller. R7: the
    # plateau magnitude is a free parameter (force_offset_n), so A4 places the pair where the
    # consequence structure exists and A1.3 re-certifies byte-identity at that point.
    p0_m: float = 0.0
    k2_n_per_m: float = 0.0


@dataclass(frozen=True)
class SupportLossRegion:
    """A region where feet lose ground support (operator O9: high-centering / beaching).

    Inside it, the fraction of foot support drops to ``residual_support`` in (0, 1]:
    the legs cannot transmit tangential force, so control authority is scaled down
    even though friction is nominal (the belly bears the load).
    """

    rect: Rect
    residual_support: float
