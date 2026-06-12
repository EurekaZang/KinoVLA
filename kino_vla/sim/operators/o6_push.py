"""O6 Push — external impulse (spec §8.2, Axis II: external wrench & attachment).

θ = (J, direction, application point, time). The classic push-recovery operator:
spec marks it [A] and the dedicated validation operator for the Reflex layer and
the §6.9 T_safe calibration. At M1 the application point is the base link.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from kino_vla.sim.backend import LocomotionBackend
from kino_vla.sim.operators.base import FailureOperator


class Push(FailureOperator):
    """Apply one world-frame impulse to the base at a scheduled sim time."""

    name: ClassVar[str] = "O6_push"
    axis: ClassVar[str] = "II_external_wrench"

    def __init__(
        self,
        impulse_xy_ns: np.ndarray,
        t_push_s: float,
        yaw_impulse_nms: float = 0.0,
    ) -> None:
        self._impulse = np.asarray(impulse_xy_ns, dtype=np.float64).copy()
        if self._impulse.shape != (2,):
            raise ValueError(f"impulse must have shape (2,), got {self._impulse.shape}")
        self._t_push = float(t_push_s)
        self._yaw_impulse = float(yaw_impulse_nms)
        self._fired = False

    @property
    def fired(self) -> bool:
        return self._fired

    def on_reset(self, backend: LocomotionBackend) -> None:
        self._fired = False

    def on_step(self, backend: LocomotionBackend, t: float) -> None:
        if not self._fired and t >= self._t_push:
            backend.apply_push(self._impulse, self._yaw_impulse)
            self._fired = True

    def get_privileged_state(self) -> dict[str, float]:
        return {
            "impulse_norm_ns": float(np.linalg.norm(self._impulse)),
            "impulse_dir_rad": math.atan2(float(self._impulse[1]), float(self._impulse[0])),
            "yaw_impulse_nms": self._yaw_impulse,
            "t_push_s": self._t_push,
        }
