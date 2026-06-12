"""O11 Obs-Bias — IMU bias/drift injection (spec §8.2, Axis IV: embodiment degradation).

θ = per-channel additive bias plus optional drift rate. Spec instance: a fake
slope-compensation signal (tilt bias) [A]. The corruption is applied to the
*measured* observation the monitor/planner consume; backend truth is untouched,
so privileged supervision (spec §4) still sees the real state.
"""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import numpy as np

from kino_vla.sim.operators.base import FailureOperator
from kino_vla.sim.types import BIASABLE_OBS_FIELDS, Obs


class ObsBias(FailureOperator):
    """Add constant bias (and linear drift) to selected IMU/odometry observation fields."""

    name: ClassVar[str] = "O11_obs_bias"
    axis: ClassVar[str] = "IV_embodiment_degradation"

    def __init__(
        self,
        bias: dict[str, float | np.ndarray],
        drift_per_s: dict[str, float | np.ndarray] | None = None,
    ) -> None:
        drift_per_s = drift_per_s or {}
        for field in list(bias) + list(drift_per_s):
            if field not in BIASABLE_OBS_FIELDS:
                raise ValueError(f"O11 cannot bias field {field!r}; allowed: {BIASABLE_OBS_FIELDS}")
        self._bias = {k: np.asarray(v, dtype=np.float64) for k, v in bias.items()}
        self._drift = {k: np.asarray(v, dtype=np.float64) for k, v in drift_per_s.items()}

    def transform_obs(self, obs: Obs) -> Obs:
        updates: dict[str, float | np.ndarray] = {}
        fields = set(self._bias) | set(self._drift)
        for field in fields:
            value = getattr(obs, field)
            offset = self._bias.get(field, 0.0) + self._drift.get(field, 0.0) * obs.t
            if isinstance(value, np.ndarray):
                updates[field] = value + offset
            else:
                updates[field] = float(value + offset)
        return replace(obs, **updates)  # type: ignore[arg-type]

    def get_privileged_state(self) -> dict[str, float]:
        state: dict[str, float] = {}
        for prefix, table in (("bias", self._bias), ("drift_per_s", self._drift)):
            for field, value in table.items():
                flat = np.atleast_1d(value)
                if flat.size == 1:
                    state[f"{prefix}.{field}"] = float(flat[0])
                else:
                    for i, item in enumerate(flat):
                        state[f"{prefix}.{field}.{i}"] = float(item)
        return state
