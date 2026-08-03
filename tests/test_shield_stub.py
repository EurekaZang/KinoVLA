"""Pass-through shield stub: identity filtering, no interventions, defensive copy."""

from __future__ import annotations

import numpy as np

from kino_vla.shield.passthrough import PassThroughShield
from tests._monitor_stub import make_obs


def test_passthrough_identity():
    shield = PassThroughShield()
    cmd = np.array([0.5, 0.1, -0.3])
    decision = shield.filter(cmd, make_obs(1.0))
    np.testing.assert_array_equal(decision.cmd, cmd)
    assert decision.intervened is False
    assert decision.codes == ()


def test_passthrough_returns_copy():
    shield = PassThroughShield()
    cmd = np.array([0.5, 0.0, 0.0])
    decision = shield.filter(cmd, make_obs(1.0))
    cmd[0] = 99.0
    assert decision.cmd[0] == 0.5
