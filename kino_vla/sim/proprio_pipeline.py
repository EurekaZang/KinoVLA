"""Timestamped IMU/odometry corruption followed by one shared state-estimation path."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from kino_vla.sim.types import Obs
from kino_vla.utils.geometry import wrap_angle


@dataclass(frozen=True)
class RawProprioPacket:
    """Ideal sensor packet sampled from Isaac before any O11 corruption."""

    t: float
    imu_projected_gravity_b: np.ndarray
    imu_ang_vel_b_radps: np.ndarray
    imu_lin_acc_b_mps2: np.ndarray
    odom_pos_xy_m: np.ndarray
    odom_heading_rad: float
    odom_vel_body_mps: np.ndarray


@dataclass(frozen=True)
class ProprioFaultConfig:
    """Raw-channel fault parameters; zero values define the identity pipeline."""

    tilt_bias_rad: float = 0.0
    tilt_drift_radps: float = 0.0
    tilt_random_walk_rad_sqrt_s: float = 0.0
    gyro_z_bias_radps: float = 0.0
    gyro_z_drift_radps2: float = 0.0
    odom_vel_bias_mps: tuple[float, float] = (0.0, 0.0)
    odom_vel_drift_mps2: tuple[float, float] = (0.0, 0.0)
    odom_heading_bias_rad: float = 0.0
    odom_heading_drift_radps: float = 0.0
    odom_latency_s: float = 0.0
    odom_dropout_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.tilt_random_walk_rad_sqrt_s < 0.0:
            raise ValueError("tilt random-walk scale must be non-negative")
        if self.odom_latency_s < 0.0:
            raise ValueError("odometry latency must be non-negative")
        if not 0.0 <= self.odom_dropout_probability <= 1.0:
            raise ValueError("odometry dropout probability must be in [0, 1]")


def _tilt_from_projected_gravity(projected_gravity_b: np.ndarray) -> float:
    gravity = np.asarray(projected_gravity_b, dtype=np.float64)
    norm = float(np.linalg.norm(gravity))
    if gravity.shape != (3,) or norm <= 1.0e-9:
        raise ValueError("projected gravity must be a finite nonzero 3-vector")
    return float(np.arccos(np.clip(-gravity[2] / norm, -1.0, 1.0)))


def _rotate_gravity_about_body_y(gravity: np.ndarray, angle_rad: float) -> np.ndarray:
    """Apply a raw attitude error before tilt is recovered by the estimator."""
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    x, y, z = (float(value) for value in gravity)
    return np.array([c * x + s * z, y, -s * x + c * z], dtype=np.float64)


class ProprioStatePipeline:
    """Corrupt raw channels, enforce timestamps, then estimate the shared ``Obs`` state."""

    def __init__(self, config: ProprioFaultConfig, *, seed: int = 0) -> None:
        self.config = config
        self.reset(seed)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._odom_buffer: deque[RawProprioPacket] = deque()
        self._last_odom: RawProprioPacket | None = None
        self._last_t: float | None = None
        self._tilt_random_walk_rad = 0.0
        self._last_telemetry: dict[str, object] = {
            "initialized": False,
            "seed": int(seed),
            "imu_source": "IsaacLab.Imu(projected_gravity_b,ang_vel_b,lin_acc_b)",
            "odometry_source": "Isaac articulation root state",
        }

    def _delayed_odometry(self, packet: RawProprioPacket) -> tuple[RawProprioPacket, bool]:
        self._odom_buffer.append(packet)
        target_t = packet.t - self.config.odom_latency_s
        candidates = [sample for sample in self._odom_buffer if sample.t <= target_t + 1.0e-12]
        delayed = candidates[-1] if candidates else self._odom_buffer[0]
        while len(self._odom_buffer) > 2 and self._odom_buffer[1].t <= target_t:
            self._odom_buffer.popleft()
        dropped = bool(self._rng.random() < self.config.odom_dropout_probability)
        if dropped and self._last_odom is not None:
            delayed = self._last_odom
        else:
            self._last_odom = delayed
        return delayed, dropped

    def transform(self, packet: RawProprioPacket, template: Obs) -> Obs:
        dt = 0.0 if self._last_t is None else max(0.0, packet.t - self._last_t)
        self._last_t = packet.t
        if dt > 0.0 and self.config.tilt_random_walk_rad_sqrt_s > 0.0:
            self._tilt_random_walk_rad += (
                self.config.tilt_random_walk_rad_sqrt_s
                * np.sqrt(dt)
                * float(self._rng.standard_normal())
            )
        tilt_error = (
            self.config.tilt_bias_rad
            + self.config.tilt_drift_radps * packet.t
            + self._tilt_random_walk_rad
        )
        corrupted_gravity = _rotate_gravity_about_body_y(
            np.asarray(packet.imu_projected_gravity_b, dtype=np.float64), tilt_error
        )
        delayed, dropped = self._delayed_odometry(packet)
        odom_velocity = (
            np.asarray(delayed.odom_vel_body_mps, dtype=np.float64)
            + np.asarray(self.config.odom_vel_bias_mps, dtype=np.float64)
            + np.asarray(self.config.odom_vel_drift_mps2, dtype=np.float64) * packet.t
        )
        heading = wrap_angle(
            delayed.odom_heading_rad
            + self.config.odom_heading_bias_rad
            + self.config.odom_heading_drift_radps * packet.t
        )
        yaw_rate = (
            float(packet.imu_ang_vel_b_radps[2])
            + self.config.gyro_z_bias_radps
            + self.config.gyro_z_drift_radps2 * packet.t
        )
        measured = replace(
            template,
            pos=np.asarray(delayed.odom_pos_xy_m, dtype=np.float64).copy(),
            heading=float(heading),
            vel_body=odom_velocity.copy(),
            yaw_rate=float(yaw_rate),
            tilt=_tilt_from_projected_gravity(corrupted_gravity),
        )
        self._last_telemetry = {
            "initialized": True,
            "packet_time_s": float(packet.t),
            "odom_source_time_s": float(delayed.t),
            "effective_odom_age_s": float(packet.t - delayed.t),
            "requested_odom_latency_s": float(self.config.odom_latency_s),
            "odom_dropped_this_step": dropped,
            "tilt_bias_command_rad": float(self.config.tilt_bias_rad),
            "tilt_drift_component_rad": float(self.config.tilt_drift_radps * packet.t),
            "tilt_random_walk_state_rad": float(self._tilt_random_walk_rad),
            "total_raw_tilt_error_rad": float(tilt_error),
            "ideal_tilt_rad": _tilt_from_projected_gravity(
                packet.imu_projected_gravity_b
            ),
            "estimated_tilt_rad": float(measured.tilt),
            "imu_source": "IsaacLab.Imu(projected_gravity_b,ang_vel_b,lin_acc_b)",
            "odometry_source": "Isaac articulation root state",
            "estimator": "kinofail.timestamped_proprio_state_pipeline.v1",
        }
        return measured

    def telemetry(self) -> dict[str, object]:
        return dict(self._last_telemetry)
