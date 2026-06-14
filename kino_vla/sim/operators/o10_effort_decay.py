"""O10 Effort-Decay — actuator effort-limit decay (spec §8.2, Axis IV: embodiment degradation).

θ = (decay_rate, floor, t_start). Schedules the actuator effort budget τ_max(t) down
over time (thermal/derating limp): zero extra sim cost, just an actuator-config edit.
Forms the **second constructive ambiguity pair with O5** — both saturate torque, but
the cause is internal (decay) vs. external (overload), so the recovery flips (limp
gait + speed cap vs. shed load / request help). Spec marks it an A/B boundary along
the decay-rate axis (Suite-Bound): mild decay the low level can compensate [A], severe
decay needs a semantic gait switch [B].
"""

from __future__ import annotations

from typing import ClassVar

from kino_vla.sim.backend import LocomotionBackend
from kino_vla.sim.operators.base import FailureOperator


class EffortDecay(FailureOperator):
    """Linearly decay the actuator-effort budget after ``t_start`` down to ``floor``."""

    name: ClassVar[str] = "O10_effort_decay"
    axis: ClassVar[str] = "IV_embodiment_degradation"

    def __init__(self, decay_rate_per_s: float, floor: float = 0.2, t_start_s: float = 0.0) -> None:
        if not 0.0 <= floor <= 1.0:
            raise ValueError(f"floor must be in [0, 1], got {floor}")
        if decay_rate_per_s < 0.0:
            raise ValueError(f"decay_rate must be non-negative, got {decay_rate_per_s}")
        self._decay_rate = float(decay_rate_per_s)
        self._floor = float(floor)
        self._t_start = float(t_start_s)

    def scale_at(self, t: float) -> float:
        """Effort budget scale in [floor, 1] at sim time ``t`` (deterministic schedule)."""
        if t < self._t_start:
            return 1.0
        return max(self._floor, 1.0 - self._decay_rate * (t - self._t_start))

    def on_reset(self, backend: LocomotionBackend) -> None:
        backend.set_effort_scale(1.0)

    def on_step(self, backend: LocomotionBackend, t: float) -> None:
        backend.set_effort_scale(self.scale_at(t))

    def get_privileged_state(self) -> dict[str, float]:
        return {
            "decay_rate_per_s": self._decay_rate,
            "floor": self._floor,
            "t_start_s": self._t_start,
        }
