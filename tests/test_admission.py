"""Mode-switch admission gates (spec §6.7) — the M3 admission exit criterion.

A discrete mode switch σ→σ' is admitted iff the current state already lies inside the
*new* mode's contracted safe set: h_j^{σ'}(x) ≥ ε_switch ∀j. Otherwise it is rejected
with a structured reason code for the next reflection round. This is what makes the
piecewise-switched closed loop inductively forward-invariant.
"""

from __future__ import annotations

import numpy as np

from kino_vla.shield.cbf_shield import CbfShield
from kino_vla.sim.types import Obs
from kino_vla.utils.config import load_config

CFG = "shield/cbf_v0.yaml"


def _obs(vx: float = 0.0, vy: float = 0.0) -> Obs:
    return Obs(
        t=0.0,
        pos=np.zeros(2),
        heading=0.0,
        vel_body=np.array([vx, vy]),
        yaw_rate=0.0,
        cmd_prev=np.zeros(3),
        slip_ratio=0.0,
        base_height=0.31,
        tilt=0.0,
        fallen=False,
    )


def test_admit_when_state_deep_inside_target_set():
    sh = CbfShield(load_config(CFG))
    res = sh.admit_mode_switch("crawl", _obs(0.2, 0.0))
    assert res.approved
    assert res.margin >= 0.0
    assert res.code.startswith("ADMIT")


def test_reject_gait_switch_when_barrier_below_epsilon():
    # The exit criterion: at high forward speed the capture point sits outside the
    # narrow high_step safe set, so the switch must be rejected (h^{σ'} < ε_switch).
    sh = CbfShield(load_config(CFG))
    res = sh.admit_mode_switch("high_step", _obs(1.5, 0.0))
    assert not res.approved
    assert res.margin < 0.0
    assert res.code.startswith("REJECT")
    assert "h_min" in res.code


def test_epsilon_switch_boundary_is_enforced():
    # Just outside the target set ⇒ reject; well inside ⇒ admit. Sweep forward speed
    # and confirm a single monotone flip (admitted below, rejected above).
    sh = CbfShield(load_config(CFG))
    eps = float(load_config(CFG).epsilon_switch)
    approvals = [
        sh.admit_mode_switch("high_step", _obs(v, 0.0)).approved for v in np.linspace(0.0, 2.0, 41)
    ]
    # Monotone: once rejected (too fast), never admitted again at higher speed.
    first_reject = approvals.index(False) if False in approvals else len(approvals)
    assert all(approvals[:first_reject])
    assert not any(approvals[first_reject:])
    assert eps > 0.0


def test_reject_unknown_mode():
    sh = CbfShield(load_config(CFG))
    res = sh.admit_mode_switch("moonwalk", _obs(0.0, 0.0))
    assert not res.approved
    assert "unknown" in res.code


def test_admission_margin_matches_barrier():
    # margin == min_j h_j^{σ'}(x) − ε_switch, with the sign deciding approval.
    sh = CbfShield(load_config(CFG))
    eps = float(load_config(CFG).epsilon_switch)
    mode = sh.modes_table()["crawl"]
    obs = _obs(0.5, 0.1)
    omega = mode.omega(load_config(CFG).gravity)
    xi = np.array([0.5, 0.1]) / omega
    h_min = float(mode.barriers(xi).min())
    res = sh.admit_mode_switch("crawl", obs)
    assert abs(res.margin - (h_min - eps)) < 1e-9
