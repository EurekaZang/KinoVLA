"""Unit tests for the shared friction-limited traction model."""

from __future__ import annotations

import numpy as np
import pytest

from kino_vla.sim.traction import traction_step

PARAMS = {"tau_track_s": 0.25, "gait_demand_per_speed": 3.0, "gravity": 9.81}


def test_no_slip_within_budget():
    result = traction_step(np.zeros(2), np.array([0.2, 0.0]), mu=0.8, **PARAMS)
    np.testing.assert_allclose(result.accel_body, [0.8, 0.0])
    assert result.slip_ratio == 0.0
    assert result.authority == 1.0


def test_saturated_accel_capped_at_mu_g():
    # From rest the gait term vanishes: achieved |a| must equal exactly mu * g.
    result = traction_step(np.zeros(2), np.array([1.0, 0.0]), mu=0.1, **PARAMS)
    assert np.linalg.norm(result.accel_body) == pytest.approx(0.1 * 9.81)
    assert 0.0 < result.slip_ratio < 1.0
    assert result.authority == pytest.approx(1.0 - result.slip_ratio)


def test_gait_demand_causes_slip_at_constant_speed():
    # Cruising (v == v_des) on ice still slips: the gait term alone exceeds mu*g.
    vel = np.array([0.8, 0.0])
    result = traction_step(vel, vel, mu=0.1, **PARAMS)
    assert result.slip_ratio > 0.4
    # Same cruise on nominal ground is slip-free.
    assert traction_step(vel, vel, mu=0.8, **PARAMS).slip_ratio == 0.0


def test_slip_increases_with_speed():
    slips = [
        traction_step(np.array([v, 0.0]), np.array([v, 0.0]), mu=0.1, **PARAMS).slip_ratio
        for v in (0.2, 0.5, 0.9)
    ]
    assert slips == sorted(slips)


def test_zero_demand_zero_slip():
    result = traction_step(np.zeros(2), np.zeros(2), mu=0.0, **PARAMS)
    assert result.slip_ratio == 0.0
    np.testing.assert_allclose(result.accel_body, 0.0)
