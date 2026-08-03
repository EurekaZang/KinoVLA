"""O5 Payload — torso-fixed overload with a center-of-mass offset (spec §8.2, Axis II).

θ = (m, r_offset). A rigid body fixed to the base raises the mass the actuators must
drive and, when offset, adds a persistent disturbance the controller fights — both
push the effort budget toward saturation (the "隐形超载死锁 / invisible overload
deadlock" [B] case: torque saturates, the policy must shed load or request help).
Forms the **constructive ambiguity pair with O10**: identical torque-saturation
symptom, opposite cause (external load vs. internal decay) and opposite recovery.

M2 ships the static-overload case. The pendulum-slosh [A] and sudden-detach [A]
variants (spec's l_pend, t_detach) are deferred to a later milestone; θ here is the
overload sub-vector (m, r_offset).
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from kino_vla.sim.backend import LocomotionBackend
from kino_vla.sim.operators.base import FailureOperator


class Payload(FailureOperator):
    """Attach a visible rigid cuboid with explicit mass, CoM and dimensions."""

    name: ClassVar[str] = "O5_payload"
    axis: ClassVar[str] = "II_external_wrench"

    def __init__(
        self,
        mass_kg: float,
        com_offset_m: np.ndarray | tuple[float, float] | tuple[float, float, float],
        *,
        size_m: np.ndarray | tuple[float, float, float] = (0.30, 0.20, 0.16),
        default_com_height_m: float = 0.14,
    ) -> None:
        if mass_kg < 0.0:
            raise ValueError(f"payload mass must be non-negative, got {mass_kg}")
        self._mass = float(mass_kg)
        offset = np.asarray(com_offset_m, dtype=np.float64).copy()
        if offset.shape == (2,):
            offset = np.concatenate([offset, [float(default_com_height_m)]])
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError(f"com_offset must have shape (2,) or (3,), got {offset.shape}")
        size = np.asarray(size_m, dtype=np.float64).copy()
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0.0):
            raise ValueError("size_m must be a finite positive 3-vector")
        self._com_offset = offset
        self._size = size

    def on_reset(self, backend: LocomotionBackend) -> None:
        backend.add_payload(self._mass, self._com_offset, self._size)

    def get_privileged_state(self) -> dict[str, float]:
        return {
            "mass_kg": self._mass,
            "com_offset_x": float(self._com_offset[0]),
            "com_offset_y": float(self._com_offset[1]),
            "com_offset_z": float(self._com_offset[2]),
            "size_x": float(self._size[0]),
            "size_y": float(self._size[1]),
            "size_z": float(self._size[2]),
        }
