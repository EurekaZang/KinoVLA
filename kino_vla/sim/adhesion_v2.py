"""Contact-aware surface adhesion, distinct from persistent tether entanglement.

The v1 world-anchor model is retained for the ``tether_entanglement`` subtype.  This module
implements ``adhesive_contact``: a stance foot bonds to a sticky surface, resists tangential
motion, peels when it unloads/lifts, and may bond again on a later stance while still inside the
surface.  The distinction prevents a single stance contact from becoming an unphysical permanent
world tether across the swing phase.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from kino_vla.sim.adhesion import IrregularRegion


@dataclass(frozen=True)
class SurfaceAdhesionConfig:
    """Per-foot contact law for a peelable adhesive floor surface."""

    region: IrregularRegion
    surface_z_m: float = 0.0
    attach_contact_force_n: float = 5.0
    detach_contact_force_n: float = 2.0
    attach_height_tolerance_m: float = 0.04
    tangential_stiffness_n_per_m: float = 120.0
    tangential_damping_ns_per_m: float = 2.0
    normal_stiffness_n_per_m: float = 80.0
    normal_damping_ns_per_m: float = 1.0
    tangential_force_cap_n: float = 12.0
    normal_force_cap_n: float = 8.0
    peel_height_m: float = 0.018
    unload_steps_to_peel: int = 2
    reattach_cooldown_steps: int = 2
    max_active_feet: int = 2
    progress_axis_xy: tuple[float, float] = (1.0, 0.0)

    def __post_init__(self) -> None:
        positive = {
            "attach_contact_force_n": self.attach_contact_force_n,
            "attach_height_tolerance_m": self.attach_height_tolerance_m,
            "tangential_stiffness_n_per_m": self.tangential_stiffness_n_per_m,
            "normal_stiffness_n_per_m": self.normal_stiffness_n_per_m,
            "tangential_force_cap_n": self.tangential_force_cap_n,
            "normal_force_cap_n": self.normal_force_cap_n,
            "peel_height_m": self.peel_height_m,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.detach_contact_force_n < 0.0:
            raise ValueError("detach_contact_force_n must be nonnegative")
        if self.detach_contact_force_n >= self.attach_contact_force_n:
            raise ValueError("detach contact threshold must be below attach threshold")
        if self.tangential_damping_ns_per_m < 0.0 or self.normal_damping_ns_per_m < 0.0:
            raise ValueError("adhesion damping must be nonnegative")
        if self.unload_steps_to_peel < 1:
            raise ValueError("unload_steps_to_peel must be at least one")
        if self.reattach_cooldown_steps < 0:
            raise ValueError("reattach_cooldown_steps must be nonnegative")
        if self.max_active_feet < 1:
            raise ValueError("max_active_feet must be at least one")
        if math.hypot(*self.progress_axis_xy) <= 1.0e-9:
            raise ValueError("progress_axis_xy cannot be zero")

    @property
    def unit_progress_axis_xy(self) -> np.ndarray:
        axis = np.asarray(self.progress_axis_xy, dtype=np.float64)
        return axis / np.linalg.norm(axis)


@dataclass(frozen=True)
class SurfaceAdhesionState:
    attached: bool = False
    anchor_xyz: tuple[float, float, float] | None = None
    unload_steps: int = 0
    cooldown_steps: int = 0
    attachment_count: int = 0
    peel_count: int = 0
    phase: str = "idle"
    last_event: str = "none"
    max_tangential_extension_m: float = 0.0
    max_tangential_force_n: float = 0.0
    max_normal_force_n: float = 0.0
    tangential_work_j: float = 0.0
    force_world_n: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SurfaceAdhesionUpdate:
    state: SurfaceAdhesionState
    force_world_n: np.ndarray
    raw_tangential_force_n: float
    raw_normal_force_n: float
    tangential_extension_m: float
    event: str


def _capped(vector: np.ndarray, cap: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= cap else vector * (cap / max(norm, 1.0e-12))


def step_surface_adhesion(
    config: SurfaceAdhesionConfig,
    state: SurfaceAdhesionState,
    foot_pos_xyz: np.ndarray,
    foot_vel_xyz: np.ndarray,
    contact_force_n: float,
    *,
    dt_s: float,
    allow_attach: bool = True,
    force_release: bool = False,
) -> SurfaceAdhesionUpdate:
    """Advance one contact-aware adhesive bond.

    A bond peels when a recovery command explicitly requests release or when the foot has been
    unloaded for the configured dwell and lifted above the peel height.  Unlike the tether model,
    peel is not terminal for the whole episode: after a short cooldown the foot may attach on a
    later load-bearing stance inside the region.
    """

    pos = np.asarray(foot_pos_xyz, dtype=np.float64)
    vel = np.asarray(foot_vel_xyz, dtype=np.float64)
    if pos.shape != (3,) or vel.shape != (3,):
        raise ValueError("foot position and velocity must have shape (3,)")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    zero = np.zeros(3, dtype=np.float64)

    if not state.attached:
        cooldown = max(0, state.cooldown_steps - 1)
        inside = config.region.contains(pos[:2])
        on_surface = abs(float(pos[2]) - config.surface_z_m) <= config.attach_height_tolerance_m
        can_attach = (
            allow_attach
            and cooldown == 0
            and inside
            and on_surface
            and float(contact_force_n) >= config.attach_contact_force_n
        )
        if not can_attach:
            next_state = replace(
                state,
                cooldown_steps=cooldown,
                phase="cooldown" if cooldown else "eligible" if inside else "idle",
                last_event="none",
                force_world_n=(0.0, 0.0, 0.0),
            )
            return SurfaceAdhesionUpdate(next_state, zero, 0.0, 0.0, 0.0, "none")
        next_state = replace(
            state,
            attached=True,
            anchor_xyz=(float(pos[0]), float(pos[1]), float(config.surface_z_m)),
            unload_steps=0,
            phase="attached",
            last_event="attached",
            attachment_count=state.attachment_count + 1,
            force_world_n=(0.0, 0.0, 0.0),
        )
        return SurfaceAdhesionUpdate(next_state, zero, 0.0, 0.0, 0.0, "attached")

    if state.anchor_xyz is None:
        raise RuntimeError("attached surface adhesion state has no anchor")
    anchor = np.asarray(state.anchor_xyz, dtype=np.float64)
    extension = pos - anchor
    tangential_extension = extension[:2]
    tangential_extension_m = float(np.linalg.norm(tangential_extension))
    raw_xy = -(
        config.tangential_stiffness_n_per_m * tangential_extension
        + config.tangential_damping_ns_per_m * vel[:2]
    )
    # Adhesive normal force resists lifting only; it must never push a foot into the floor.
    lift_m = max(0.0, float(extension[2]))
    lift_speed_mps = max(0.0, float(vel[2]))
    raw_z_magnitude = (
        config.normal_stiffness_n_per_m * lift_m
        + config.normal_damping_ns_per_m * lift_speed_mps
    )
    unloaded = float(contact_force_n) < config.detach_contact_force_n
    unload_steps = state.unload_steps + 1 if unloaded else 0
    lifted_for_peel = lift_m >= config.peel_height_m
    should_peel = force_release or (
        unload_steps >= config.unload_steps_to_peel and lifted_for_peel
    )
    if should_peel:
        next_state = replace(
            state,
            attached=False,
            anchor_xyz=None,
            unload_steps=0,
            cooldown_steps=config.reattach_cooldown_steps,
            peel_count=state.peel_count + 1,
            phase="peeled",
            last_event="peeled",
            max_tangential_extension_m=max(
                state.max_tangential_extension_m, tangential_extension_m
            ),
            max_tangential_force_n=max(
                state.max_tangential_force_n, float(np.linalg.norm(raw_xy))
            ),
            max_normal_force_n=max(state.max_normal_force_n, raw_z_magnitude),
            force_world_n=(0.0, 0.0, 0.0),
        )
        return SurfaceAdhesionUpdate(
            next_state,
            zero,
            float(np.linalg.norm(raw_xy)),
            raw_z_magnitude,
            tangential_extension_m,
            "peeled",
        )

    applied_xy = _capped(raw_xy, config.tangential_force_cap_n)
    applied_z = -min(raw_z_magnitude, config.normal_force_cap_n)
    force = np.asarray([applied_xy[0], applied_xy[1], applied_z], dtype=np.float64)
    tangential_power = max(0.0, -float(applied_xy @ vel[:2]))
    next_state = replace(
        state,
        unload_steps=unload_steps,
        phase="unloading" if unloaded else "loaded",
        last_event="none",
        max_tangential_extension_m=max(
            state.max_tangential_extension_m, tangential_extension_m
        ),
        max_tangential_force_n=max(
            state.max_tangential_force_n, float(np.linalg.norm(raw_xy))
        ),
        max_normal_force_n=max(state.max_normal_force_n, raw_z_magnitude),
        tangential_work_j=state.tangential_work_j + tangential_power * float(dt_s),
        force_world_n=tuple(float(value) for value in force),
    )
    return SurfaceAdhesionUpdate(
        next_state,
        force,
        float(np.linalg.norm(raw_xy)),
        raw_z_magnitude,
        tangential_extension_m,
        "none",
    )
