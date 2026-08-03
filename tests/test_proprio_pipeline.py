from __future__ import annotations

import numpy as np
import pytest

from kino_vla.sim.proprio_pipeline import (
    ProprioFaultConfig,
    ProprioStatePipeline,
    RawProprioPacket,
)
from kino_vla.sim.types import Obs


def _obs(t: float) -> Obs:
    return Obs(
        t=t,
        pos=np.array([t, 0.0]),
        heading=0.1 * t,
        vel_body=np.array([t, -t]),
        yaw_rate=0.2 * t,
        cmd_prev=np.zeros(3),
        slip_ratio=0.0,
        base_height=0.3,
        tilt=0.0,
        fallen=False,
    )


def _packet(t: float) -> RawProprioPacket:
    return RawProprioPacket(
        t=t,
        imu_projected_gravity_b=np.array([0.0, 0.0, -1.0]),
        imu_ang_vel_b_radps=np.array([0.0, 0.0, 0.2 * t]),
        imu_lin_acc_b_mps2=np.zeros(3),
        odom_pos_xy_m=np.array([t, 0.0]),
        odom_heading_rad=0.1 * t,
        odom_vel_body_mps=np.array([t, -t]),
    )


def test_raw_tilt_fault_precedes_estimator_and_latency_uses_timestamps() -> None:
    pipeline = ProprioStatePipeline(
        ProprioFaultConfig(tilt_bias_rad=0.18, odom_latency_s=0.10), seed=7
    )
    measured = None
    for index in range(11):
        t = index * 0.02
        measured = pipeline.transform(_packet(t), _obs(t))
    assert measured is not None
    assert measured.tilt == pytest.approx(0.18, abs=1.0e-7)
    np.testing.assert_allclose(measured.pos, [0.10, 0.0], atol=1.0e-9)
    np.testing.assert_allclose(measured.vel_body, [0.10, -0.10], atol=1.0e-9)
    telemetry = pipeline.telemetry()
    assert telemetry["effective_odom_age_s"] == pytest.approx(0.10)
    assert telemetry["odom_source_time_s"] == pytest.approx(0.10)
    assert telemetry["estimator"] == "kinofail.timestamped_proprio_state_pipeline.v1"


def test_random_walk_is_seeded_and_reset_reproducible() -> None:
    config = ProprioFaultConfig(tilt_random_walk_rad_sqrt_s=0.02)
    first = ProprioStatePipeline(config, seed=42)
    second = ProprioStatePipeline(config, seed=42)
    a, b = [], []
    for index in range(1, 30):
        t = index * 0.02
        a.append(first.transform(_packet(t), _obs(t)).tilt)
        b.append(second.transform(_packet(t), _obs(t)).tilt)
    np.testing.assert_allclose(a, b, atol=0.0, rtol=0.0)
    assert np.std(a) > 0.0
    first.reset(42)
    replay = [
        first.transform(_packet(index * 0.02), _obs(index * 0.02)).tilt
        for index in range(1, 30)
    ]
    np.testing.assert_allclose(a, replay, atol=0.0, rtol=0.0)


def test_full_odometry_dropout_holds_last_delivered_packet() -> None:
    pipeline = ProprioStatePipeline(
        ProprioFaultConfig(odom_dropout_probability=1.0), seed=3
    )
    first = pipeline.transform(_packet(0.0), _obs(0.0))
    second = pipeline.transform(_packet(0.2), _obs(0.2))
    np.testing.assert_array_equal(first.pos, second.pos)
    np.testing.assert_array_equal(first.vel_body, second.vel_body)
    assert pipeline.telemetry()["odom_dropped_this_step"]
