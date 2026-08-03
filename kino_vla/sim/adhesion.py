"""Foot-local adhesion model used by the realistic Kino-Fail scene pipeline.

The historical O4 implementation intentionally remains available for the frozen A0--A7
artifacts.  This module is the higher-fidelity vertical-slice implementation: every attachment
belongs to a named foot, has an explicit world anchor, and produces a force at that foot rather
than an anonymous wrench on the trunk.

The state transition is kept independent of Isaac Sim so it can be unit-tested and audited.  The
Isaac backend is responsible only for reading foot/contact state and applying the returned force.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class IrregularRegion:
    """Simple non-self-intersecting floor polygon in the episode/start frame."""

    vertices_xy: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.vertices_xy) < 3:
            raise ValueError("an adhesion footprint needs at least three vertices")
        if abs(self.signed_area_m2) < 1.0e-6:
            raise ValueError("adhesion footprint has zero area")

    @property
    def signed_area_m2(self) -> float:
        points = self.vertices_xy
        return 0.5 * sum(
            x0 * y1 - x1 * y0
            for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1], strict=True)
        )

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xs = [point[0] for point in self.vertices_xy]
        ys = [point[1] for point in self.vertices_xy]
        return min(xs), max(xs), min(ys), max(ys)

    def contains(self, xy: np.ndarray | tuple[float, float]) -> bool:
        """Return whether ``xy`` is inside the polygon (boundary counts as inside)."""
        x, y = float(xy[0]), float(xy[1])
        points = self.vertices_xy
        inside = False
        for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1], strict=True):
            # Boundary check first; it also handles horizontal edges robustly.
            dx, dy = x1 - x0, y1 - y0
            cross = (x - x0) * dy - (y - y0) * dx
            if abs(cross) <= 1.0e-9:
                dot = (x - x0) * (x - x1) + (y - y0) * (y - y1)
                if dot <= 1.0e-9:
                    return True
            if (y0 > y) != (y1 > y):
                x_intersection = x0 + (y - y0) * dx / dy
                if x <= x_intersection:
                    inside = not inside
        return inside


@dataclass(frozen=True)
class FootAdhesionConfig:
    """Calibratable contact law for an irregular adhesive floor film."""

    region: IrregularRegion
    surface_z_m: float = 0.0
    attach_contact_force_n: float = 5.0
    attach_height_tolerance_m: float = 0.08
    stiffness_xy_n_per_m: float = 260.0
    damping_xy_ns_per_m: float = 5.0
    stiffness_z_n_per_m: float = 180.0
    damping_z_ns_per_m: float = 3.0
    force_cap_n: float = 55.0
    break_force_n: float = 105.0
    peel_release_force_n: float = 16.0
    peel_velocity_threshold_mps: float = 0.03
    progress_axis_xy: tuple[float, float] = (1.0, 0.0)
    max_active_feet: int = 2

    def __post_init__(self) -> None:
        positive = {
            "attach_contact_force_n": self.attach_contact_force_n,
            "attach_height_tolerance_m": self.attach_height_tolerance_m,
            "stiffness_xy_n_per_m": self.stiffness_xy_n_per_m,
            "stiffness_z_n_per_m": self.stiffness_z_n_per_m,
            "force_cap_n": self.force_cap_n,
            "break_force_n": self.break_force_n,
            "peel_release_force_n": self.peel_release_force_n,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.damping_xy_ns_per_m < 0.0 or self.damping_z_ns_per_m < 0.0:
            raise ValueError("adhesion damping must be non-negative")
        if self.force_cap_n >= self.break_force_n:
            raise ValueError(
                "force_cap_n must stay below break_force_n so break remains observable"
            )
        if self.peel_release_force_n >= self.break_force_n:
            raise ValueError("peel release must be easier than forward break")
        if self.max_active_feet < 1:
            raise ValueError("max_active_feet must be at least one")
        if math.hypot(*self.progress_axis_xy) <= 1.0e-9:
            raise ValueError("progress_axis_xy cannot be zero")

    @property
    def unit_progress_axis_xy(self) -> np.ndarray:
        axis = np.asarray(self.progress_axis_xy, dtype=np.float64)
        return axis / np.linalg.norm(axis)


@dataclass(frozen=True)
class FootAdhesionState:
    """Episode state for one foot; all coordinates are in the episode/start frame."""

    attached: bool = False
    terminal_release: bool = False
    broken: bool = False
    anchor_xyz: tuple[float, float, float] | None = None
    phase: str = "idle"
    last_event: str = "none"
    attachment_count: int = 0
    max_extension_m: float = 0.0
    max_force_n: float = 0.0
    force_world_n: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class FootAdhesionUpdate:
    """One transition result and force to apply to the corresponding foot body."""

    state: FootAdhesionState
    force_world_n: np.ndarray
    raw_force_n: float
    extension_m: float
    event: str


def step_foot_adhesion(
    config: FootAdhesionConfig,
    state: FootAdhesionState,
    foot_pos_xyz: np.ndarray,
    foot_vel_xyz: np.ndarray,
    contact_force_n: float,
    *,
    allow_attach: bool = True,
    peel_requested: bool = False,
) -> FootAdhesionUpdate:
    """Advance one foot-local adhesion state and return the world-frame force.

    Loading in the nominal travel direction uses the high forward break threshold.  The caller
    marks an intentional reversal as a peel attempt; this must not be inferred from instantaneous
    foot velocity because a stance foot moves backward during an ordinary forward gait.  Once the
    explicit reversal is requested, reaching the lower peel-force threshold releases the bond
    without waiting for the tethered foot to reverse first.  A released or broken contact never
    silently reattaches within the same episode.
    """
    pos = np.asarray(foot_pos_xyz, dtype=np.float64)
    vel = np.asarray(foot_vel_xyz, dtype=np.float64)
    if pos.shape != (3,) or vel.shape != (3,):
        raise ValueError("foot position and velocity must both have shape (3,)")
    zero = np.zeros(3, dtype=np.float64)

    if state.terminal_release:
        phase = "broken" if state.broken else "released"
        next_state = replace(
            state,
            attached=False,
            phase=phase,
            last_event="none",
            force_world_n=(0.0, 0.0, 0.0),
        )
        return FootAdhesionUpdate(next_state, zero, 0.0, 0.0, "none")

    if not state.attached:
        on_surface = abs(float(pos[2]) - config.surface_z_m) <= config.attach_height_tolerance_m
        can_attach = (
            allow_attach
            and config.region.contains(pos[:2])
            and on_surface
            and float(contact_force_n) >= config.attach_contact_force_n
        )
        if not can_attach:
            next_state = replace(
                state,
                phase="eligible" if config.region.contains(pos[:2]) else "idle",
                last_event="none",
                force_world_n=(0.0, 0.0, 0.0),
            )
            return FootAdhesionUpdate(next_state, zero, 0.0, 0.0, "none")
        anchor = (float(pos[0]), float(pos[1]), float(config.surface_z_m))
        next_state = replace(
            state,
            attached=True,
            anchor_xyz=anchor,
            phase="attached",
            last_event="attached",
            attachment_count=state.attachment_count + 1,
            force_world_n=(0.0, 0.0, 0.0),
        )
        return FootAdhesionUpdate(next_state, zero, 0.0, 0.0, "attached")

    if state.anchor_xyz is None:
        raise RuntimeError("attached foot adhesion state has no anchor")
    extension = pos - np.asarray(state.anchor_xyz, dtype=np.float64)
    extension_m = float(np.linalg.norm(extension))
    raw_force = np.array(
        [
            -config.stiffness_xy_n_per_m * extension[0] - config.damping_xy_ns_per_m * vel[0],
            -config.stiffness_xy_n_per_m * extension[1] - config.damping_xy_ns_per_m * vel[1],
            -config.stiffness_z_n_per_m * extension[2] - config.damping_z_ns_per_m * vel[2],
        ],
        dtype=np.float64,
    )
    raw_force_n = float(np.linalg.norm(raw_force))
    # ``peel_requested`` already encodes an intentional command reversal in the backend.  Requiring
    # negative measured foot speed here creates a deadlock: a loaded adhesive bond is precisely
    # what can prevent that foot from reversing.  Foot velocity remains part of the force damping
    # term, but it is not a second authorization gate for release.
    peeling = bool(peel_requested)

    max_extension = max(state.max_extension_m, extension_m)
    max_force = max(state.max_force_n, raw_force_n)
    if peeling and raw_force_n >= config.peel_release_force_n:
        next_state = replace(
            state,
            attached=False,
            terminal_release=True,
            phase="released",
            last_event="peeled",
            max_extension_m=max_extension,
            max_force_n=max_force,
            force_world_n=(0.0, 0.0, 0.0),
        )
        return FootAdhesionUpdate(next_state, zero, raw_force_n, extension_m, "peeled")
    if raw_force_n >= config.break_force_n:
        next_state = replace(
            state,
            attached=False,
            terminal_release=True,
            broken=True,
            phase="broken",
            last_event="broken",
            max_extension_m=max_extension,
            max_force_n=max_force,
            force_world_n=(0.0, 0.0, 0.0),
        )
        return FootAdhesionUpdate(next_state, zero, raw_force_n, extension_m, "broken")

    applied = raw_force
    if raw_force_n > config.force_cap_n:
        applied = raw_force * (config.force_cap_n / raw_force_n)
    applied_tuple = tuple(float(value) for value in applied)
    next_state = replace(
        state,
        phase="peeling" if peeling else "loaded",
        last_event="none",
        max_extension_m=max_extension,
        max_force_n=max_force,
        force_world_n=applied_tuple,
    )
    return FootAdhesionUpdate(next_state, applied, raw_force_n, extension_m, "none")
