"""Locomotion backend protocol — the walking skeleton's view of the robot.

Two implementations exist at M1:
- ``kino_vla.sim.surrogate.SurrogateBackend``: CPU point-robot model, keeps the
  demo and CI green on machines without a GPU (CLAUDE.md Section 6 issue #4).
- ``kino_vla.sim.isaac_backend.IsaacKinematicBackend``: Isaac Lab Go2 with a real
  PhysX material patch; GPU-only, verified manually on the RTX 5090 machine.

Commands are Sport-Client-style body-frame velocities ``(vx, vy, wz)``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from kino_vla.sim.types import FrictionRegion, Obs


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
