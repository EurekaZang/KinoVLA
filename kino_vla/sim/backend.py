"""Locomotion backend protocol — the walking skeleton's view of the robot.

Two implementations exist:
- ``kino_vla.sim.surrogate.SurrogateBackend``: CPU point-robot model, keeps the
  demo and CI green on machines without a GPU.
- ``kino_vla.sim.isaac_policy_backend.IsaacPolicyBackend``: Isaac Lab Go2 physically
  simulated and walked by the trained RSL-RL velocity policy (M2); GPU-only.

Commands are Sport-Client-style body-frame velocities ``(vx, vy, wz)``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from kino_vla.sim.terramechanics import FootTerramechanicsConfig
from kino_vla.sim.types import (
    BlockingRegion,
    CollapseRegion,
    FrictionRegion,
    Obs,
    ResistanceRegion,
    SupportLossRegion,
)


@runtime_checkable
class LocomotionBackend(Protocol):
    """Structural interface required by the episode loop and the failure operators."""

    dt: float  # control period [s]
    mass_kg: float

    def reset(self, seed: int) -> Obs:
        """Reset world and robot to the start pose; reseed all backend randomness."""
        ...

    def step(self, cmd_vel: np.ndarray) -> Obs:
        """Apply a body-frame (vx, vy, wz) command for one control period."""
        ...

    def add_friction_regions(self, regions: list[FrictionRegion]) -> None:
        """Append per-region ground material overrides (operator O1, spec §8.2 Axis I).

        Appending (not replacing) lets multiple O1 instances stack (Suite-Comp);
        ``reset`` clears the list. Overlaps resolve first-added-wins.
        """
        ...

    def friction_at(self, pos: np.ndarray) -> float:
        """Effective dynamic friction at a world xy position (θ-application gate, QA 5.2a)."""
        ...

    def apply_push(self, impulse_xy_ns: np.ndarray, yaw_impulse_nms: float) -> None:
        """Apply a world-frame impulse to the base (operator O6, spec §8.2 Axis II)."""
        ...

    def start_push_pulse(
        self,
        impulse_xy_ns: np.ndarray,
        yaw_impulse_nms: float,
        duration_s: float,
        application_point_body_m: np.ndarray,
    ) -> None:
        """Apply O6 as a finite PhysX force pulse at a body-frame point."""
        ...

    # ---- M2 operator-effect hooks (O3/O5/O8/O9/O10, spec §8.2) -------------------
    # Each is additive and cleared by ``reset`` (mirrors ``add_friction_regions``),
    # so operators stack for Suite-Comp. Backends without a given mechanism may treat
    # it as a no-op, but both shipped backends implement all five.

    def add_collapse_regions(self, regions: list[CollapseRegion]) -> None:
        """Append load-triggered support-topology regions (operator O3, Axis I)."""
        ...

    def add_blocking_regions(self, regions: list[BlockingRegion]) -> None:
        """Append impassable invisible colliders (operator O8, Axis III)."""
        ...

    def add_support_loss_regions(
        self,
        regions: list[SupportLossRegion],
        height_m: float | None = None,
        geometry_kind: str = "box_ridge",
    ) -> None:
        """Append foot-support-loss regions (operator O9, Axis III)."""
        ...

    def add_resistance_regions(self, regions: list[ResistanceRegion]) -> None:
        """Append tangential-resistance regions (operators O2 compliance / O4 tether, M5)."""
        ...

    def add_foot_compliance(self, config: FootTerramechanicsConfig) -> None:
        """Enable the realistic O2 per-foot sinkage/shear path for one episode."""
        ...

    def add_payload(
        self,
        mass_kg: float,
        com_offset_m: np.ndarray,
        size_m: np.ndarray | None = None,
    ) -> None:
        """Rigidly attach a cuboid payload with CoM and dimensions (operator O5, Axis II)."""
        ...

    def set_effort_scale(self, scale: float) -> None:
        """Scale the actuator-effort budget in (0, 1] (operator O10, Axis IV)."""
        ...

    # ---- M4 privileged-distillation target (spec §4) ----------------------------

    def privileged_physics(self) -> dict[str, float]:
        """God's-eye physics truth at the current step — the Kino-Tokens regression
        target (``mu``, ``payload_kg``, ``effort_scale``, ``support_ratio``; spec §4)."""
        ...
