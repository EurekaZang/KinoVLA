from __future__ import annotations

import numpy as np
import pytest

from kino_vla.sim.actuator import actuator_sample


def test_actuator_sample_reports_joint_margin_and_power() -> None:
    sample = actuator_sample(
        np.array([5.0, -9.5, 10.0]),
        np.array([2.0, -3.0, 0.5]),
        10.0,
    )
    assert sample.torque_utilization == pytest.approx([0.5, 0.95, 1.0])
    assert sample.mechanical_power_w == pytest.approx([10.0, 28.5, 5.0])
    assert sample.utilization_max == pytest.approx(1.0)
    assert sample.binding_joint_fraction == pytest.approx(2.0 / 3.0)
    assert sample.total_abs_mechanical_power_w == pytest.approx(43.5)


def test_lower_limit_increases_utilization_for_same_torque() -> None:
    nominal = actuator_sample(np.array([2.0, 4.0]), np.ones(2), 10.0)
    derated = actuator_sample(np.array([2.0, 4.0]), np.ones(2), 5.0)
    assert derated.utilization_p90 > nominal.utilization_p90


def test_actuator_sample_rejects_zero_limit() -> None:
    with pytest.raises(ValueError, match="effort limits"):
        actuator_sample(np.ones(2), np.ones(2), 0.0)
