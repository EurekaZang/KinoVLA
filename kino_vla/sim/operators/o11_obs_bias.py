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

from kino_vla.sim.backend import LocomotionBackend
from kino_vla.sim.operators.base import FailureOperator
from kino_vla.sim.proprio_pipeline import ProprioFaultConfig, ProprioStatePipeline
from kino_vla.sim.types import BIASABLE_OBS_FIELDS, Obs


class ObsBias(FailureOperator):
    """Add constant bias (and linear drift) to selected IMU/odometry observation fields."""

    name: ClassVar[str] = "O11_obs_bias"
    axis: ClassVar[str] = "IV_embodiment_degradation"

    def __init__(
        self,
        bias: dict[str, float | np.ndarray],
        drift_per_s: dict[str, float | np.ndarray] | None = None,
        *,
        random_walk_per_sqrt_s: dict[str, float] | None = None,
        latency_s: float = 0.0,
        dropout_probability: float = 0.0,
        seed: int = 0,
    ) -> None:
        drift_per_s = drift_per_s or {}
        random_walk_per_sqrt_s = random_walk_per_sqrt_s or {}
        for field in list(bias) + list(drift_per_s):
            if field not in BIASABLE_OBS_FIELDS:
                raise ValueError(f"O11 cannot bias field {field!r}; allowed: {BIASABLE_OBS_FIELDS}")
        if set(random_walk_per_sqrt_s) - {"tilt"}:
            raise ValueError("realistic O11 currently supports random walk only on raw IMU tilt")
        self._bias = {k: np.asarray(v, dtype=np.float64) for k, v in bias.items()}
        self._drift = {k: np.asarray(v, dtype=np.float64) for k, v in drift_per_s.items()}
        self._random_walk = {
            key: float(value) for key, value in random_walk_per_sqrt_s.items()
        }
        self._latency_s = float(latency_s)
        self._dropout_probability = float(dropout_probability)
        self._seed = int(seed)
        self._raw_source = None
        self._pipeline = ProprioStatePipeline(self._pipeline_config(), seed=self._seed)

    @staticmethod
    def _scalar(table: dict[str, np.ndarray], key: str) -> float:
        value = np.asarray(table.get(key, 0.0), dtype=np.float64)
        if value.size != 1:
            raise ValueError(f"O11 field {key!r} must be scalar")
        return float(value.reshape(-1)[0])

    @staticmethod
    def _vector2(table: dict[str, np.ndarray], key: str) -> tuple[float, float]:
        value = np.asarray(table.get(key, np.zeros(2)), dtype=np.float64)
        if value.shape != (2,):
            raise ValueError(f"O11 field {key!r} must be a 2-vector")
        return float(value[0]), float(value[1])

    def _pipeline_config(self) -> ProprioFaultConfig:
        return ProprioFaultConfig(
            tilt_bias_rad=self._scalar(self._bias, "tilt"),
            tilt_drift_radps=self._scalar(self._drift, "tilt"),
            tilt_random_walk_rad_sqrt_s=float(self._random_walk.get("tilt", 0.0)),
            gyro_z_bias_radps=self._scalar(self._bias, "yaw_rate"),
            gyro_z_drift_radps2=self._scalar(self._drift, "yaw_rate"),
            odom_vel_bias_mps=self._vector2(self._bias, "vel_body"),
            odom_vel_drift_mps2=self._vector2(self._drift, "vel_body"),
            odom_heading_bias_rad=self._scalar(self._bias, "heading"),
            odom_heading_drift_radps=self._scalar(self._drift, "heading"),
            odom_latency_s=self._latency_s,
            odom_dropout_probability=self._dropout_probability,
        )

    def on_reset(self, backend: LocomotionBackend) -> None:
        """Bind Isaac's timestamped raw sensor source; surrogate runs retain legacy behavior."""
        source = getattr(backend, "raw_proprio_packet", None)
        self._raw_source = source if callable(source) else None
        self._pipeline.reset(self._seed)

    def transform_obs(self, obs: Obs) -> Obs:
        if self._raw_source is not None:
            return self._pipeline.transform(self._raw_source(), obs)
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
        state["random_walk_per_sqrt_s.tilt"] = float(self._random_walk.get("tilt", 0.0))
        state["latency_s"] = self._latency_s
        state["dropout_probability"] = self._dropout_probability
        state["seed"] = float(self._seed)
        return state

    def sensor_telemetry(self) -> dict[str, object]:
        telemetry = self._pipeline.telemetry()
        telemetry["raw_pipeline_active"] = self._raw_source is not None
        return telemetry
