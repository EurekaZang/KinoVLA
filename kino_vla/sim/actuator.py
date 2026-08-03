"""Auditable joint-level actuator telemetry for O5 overload and O10 derating."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActuatorSample:
    torque_utilization: np.ndarray
    mechanical_power_w: np.ndarray
    utilization_mean: float
    utilization_p90: float
    utilization_max: float
    binding_joint_fraction: float
    total_abs_mechanical_power_w: float


def actuator_sample(
    applied_torque_nm: np.ndarray,
    joint_velocity_radps: np.ndarray,
    effort_limit_nm: np.ndarray | float,
    *,
    binding_threshold: float = 0.95,
) -> ActuatorSample:
    """Compute dimensioned power and torque-margin diagnostics for one control sample."""
    torque = np.asarray(applied_torque_nm, dtype=np.float64).reshape(-1)
    velocity = np.asarray(joint_velocity_radps, dtype=np.float64).reshape(-1)
    limits = np.broadcast_to(np.asarray(effort_limit_nm, dtype=np.float64), torque.shape)
    if torque.shape != velocity.shape or torque.size == 0:
        raise ValueError("torque and velocity must be non-empty vectors with matching shape")
    if not (
        np.isfinite(torque).all()
        and np.isfinite(velocity).all()
        and np.isfinite(limits).all()
    ):
        raise ValueError("actuator inputs must be finite")
    if np.any(limits <= 0.0) or not 0.0 < binding_threshold <= 1.0:
        raise ValueError("effort limits must be positive and binding_threshold in (0, 1]")
    utilization = np.abs(torque) / limits
    power = np.abs(torque * velocity)
    return ActuatorSample(
        torque_utilization=utilization,
        mechanical_power_w=power,
        utilization_mean=float(utilization.mean()),
        utilization_p90=float(np.quantile(utilization, 0.90)),
        utilization_max=float(utilization.max()),
        binding_joint_fraction=float(np.mean(utilization >= binding_threshold)),
        total_abs_mechanical_power_w=float(power.sum()),
    )
