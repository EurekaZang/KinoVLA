"""M2 Reflex gates: survival-time extension and push-recovery (spec §6.9 / O6).

Exit criteria: the Reflex stance measurably extends survival under sustained pushes
(T_safe groundwork) and the push-recovery protocol passes over an impulse range.
"""

from __future__ import annotations

import numpy as np

from kino_vla.monitor.reflex import (
    Reflex,
    run_push_recovery_sweep,
    run_survival_episode,
)
from kino_vla.monitor.rule_monitor import MonitorEvent
from kino_vla.sim.surrogate import SurrogateBackend
from kino_vla.utils.config import load_config

SIM_CFG = load_config("sim/surrogate.yaml")
REFLEX_CFG = load_config("recovery/reflex_v0.yaml")


def test_reflex_extends_survival_time():
    # Same push schedule, same seed: the only difference is the Reflex stance.
    for seed in range(3):
        no_reflex = run_survival_episode(SIM_CFG, REFLEX_CFG, seed=seed, use_reflex=False)
        with_reflex = run_survival_episode(SIM_CFG, REFLEX_CFG, seed=seed, use_reflex=True)
        assert no_reflex.fell, "the nominal stance must eventually topple under repeated pushes"
        assert with_reflex.survived_s > no_reflex.survived_s
        # Substantial, not marginal: at least double the survival time.
        assert with_reflex.survived_s >= 2.0 * no_reflex.survived_s


def test_push_recovery_protocol_passes_range():
    # Operator O6: the robot recovers single impulses up to a spec'd range, and the
    # Reflex stance never recovers *less* than the nominal stance.
    no_reflex = run_push_recovery_sweep(SIM_CFG, REFLEX_CFG, seed=0, use_reflex=False)
    with_reflex = run_push_recovery_sweep(SIM_CFG, REFLEX_CFG, seed=0, use_reflex=True)
    # A standing Go2 (~15 kg) must absorb at least a 15 Ns impulse (Δv ≈ 1 m/s).
    assert no_reflex.max_recovered_ns >= 15.0
    assert with_reflex.max_recovered_ns >= no_reflex.max_recovered_ns


def test_reflex_engages_and_releases():
    backend = SurrogateBackend(SIM_CFG, np.zeros(2), 0.0)
    backend.reset(0)
    reflex = Reflex(REFLEX_CFG, backend)
    assert not reflex.engaged
    event = MonitorEvent(
        t=1.0, pos=np.zeros(2), channel="slip_ratio", value=0.9, threshold=0.4, summary="x"
    )
    reflex.on_event(event)
    # Engaged during the hold window; the command is a damping stand (zero velocity).
    cmd = reflex.step(_obs_at(1.2))
    assert reflex.engaged
    np.testing.assert_array_equal(cmd, np.zeros(3))
    # Released after the hold; control yielded to the planner (None command).
    released = reflex.step(_obs_at(1.0 + float(REFLEX_CFG.hold_s) + 0.1))
    assert not reflex.engaged
    assert released is None


def _obs_at(t):
    from kino_vla.sim.types import Obs

    return Obs(
        t=t,
        pos=np.zeros(2),
        heading=0.0,
        vel_body=np.zeros(2),
        yaw_rate=0.0,
        cmd_prev=np.zeros(3),
        slip_ratio=0.0,
        base_height=0.31,
        tilt=0.0,
        fallen=False,
    )


def test_survival_is_deterministic():
    a = run_survival_episode(SIM_CFG, REFLEX_CFG, seed=1, use_reflex=True)
    b = run_survival_episode(SIM_CFG, REFLEX_CFG, seed=1, use_reflex=True)
    assert a.survived_s == b.survived_s
    assert a.pushes_applied == b.pushes_applied
