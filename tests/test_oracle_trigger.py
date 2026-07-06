"""A0.2 oracle-trigger unit tests (offline; no Isaac). The primary trigger must fire ONCE, at the
privileged onset, deterministically and agent-independently — the fair-instant guarantee A2–A6 rely
on. θ never enters it (R8-clean)."""

from __future__ import annotations

import numpy as np

from kino_vla.eval.oracle_trigger import OracleTrigger
from kino_vla.sim.types import Obs
from kino_vla.utils.geometry import Rect

DT = 0.02
RECT = Rect(cx=3.0, cy=0.0, hx=1.0, hy=1.0)  # hazard patch x∈[2,4]


def _obs(t: float, x: float, y: float = 0.0) -> Obs:
    return Obs(
        t=t, pos=np.array([x, y]), heading=0.0, vel_body=np.array([0.6, 0.0]), yaw_rate=0.0,
        cmd_prev=np.zeros(3), slip_ratio=0.0, base_height=0.35, tilt=0.0, fallen=False,
    )


def _walk(trigger: OracleTrigger, *, y: float = 0.0, x0: float = 0.0, v: float = 0.6, n: int = 400):
    """Drive a straight +x walk through the trigger; return (fire_step, fire_event)."""
    fired_at, event = None, None
    for k in range(n):
        t = k * DT
        ev = trigger.step(_obs(t, x0 + v * t, y))
        if ev is not None and fired_at is None:
            fired_at, event = k, ev
    return fired_at, event


def test_fires_exactly_once_at_region_entry_plus_arm() -> None:
    trig = OracleTrigger(rect=RECT, dt=DT, arm_delay_s=0.2, operator_name="O4_tether")
    fired_at, ev = _walk(trig)
    assert fired_at is not None and ev is not None
    assert len(trig.events) == 1, "oracle must fire exactly once"
    # entry to x=2 is at t=2/0.6≈3.33 s; +0.2 s arm ⇒ fires while still inside [2,4]
    assert 2.0 <= float(ev.pos[0]) <= 4.0
    entry_t = 2.0 / 0.6
    assert ev.t >= entry_t + 0.2 - 1e-6


def test_deterministic_and_agent_independent() -> None:
    """Same obs sequence ⇒ identical fire — reads only (pos, t), nothing agent-specific."""
    a = OracleTrigger(rect=RECT, dt=DT, arm_delay_s=0.2)
    b = OracleTrigger(rect=RECT, dt=DT, arm_delay_s=0.2)
    fa, ea = _walk(a)
    fb, eb = _walk(b)
    assert fa == fb
    assert ea is not None and eb is not None
    assert ea.t == eb.t and np.array_equal(ea.pos, eb.pos)


def test_no_fire_if_never_enters_region() -> None:
    trig = OracleTrigger(rect=RECT, dt=DT, arm_delay_s=0.2)
    fired_at, _ = _walk(trig, y=10.0)  # a lane far from the patch
    assert fired_at is None and trig.events == []


def test_reset_rearms() -> None:
    trig = OracleTrigger(rect=RECT, dt=DT, arm_delay_s=0.2)
    _walk(trig)
    assert trig._fired
    trig.reset()
    assert not trig._fired and trig.events == []
    fired_at, _ = _walk(trig)
    assert fired_at is not None


def test_onset_time_gates_time_operators() -> None:
    """With an onset time, the trigger must not fire before it even if already in the region."""
    trig = OracleTrigger(rect=RECT, onset_time_s=5.0, dt=DT, arm_delay_s=0.0, operator_name="O10")
    fired_at, ev = _walk(trig)
    assert ev is not None and ev.t >= 5.0


def test_arm_delay_requires_persistence() -> None:
    """A transient dip into the region shorter than the arm delay must NOT fire (spike ≠ onset)."""
    trig = OracleTrigger(rect=RECT, dt=DT, arm_delay_s=0.5)
    # brush the edge (x∈[2,4]) for only ~0.1 s then leave, never persisting 0.5 s inside
    events = []
    xs = [1.9, 2.05, 2.1, 1.9, 1.8, 1.7]  # in for 2 steps (~0.04 s) < arm
    for k, x in enumerate(xs):
        ev = trig.step(_obs(k * DT, x))
        if ev:
            events.append(ev)
    assert events == []


def test_for_scenario_factory() -> None:
    class _Region:
        rect = RECT

    class _Scn:
        operator_name = "O4_tether"
        scene_region = _Region()

    trig = OracleTrigger.for_scenario(_Scn(), dt=DT)
    fired_at, _ = _walk(trig)
    assert fired_at is not None

    class _ScnGlobal:
        operator_name = "O10_effort_decay"
        scene_region = _Region()

    trig2 = OracleTrigger.for_scenario(_ScnGlobal(), dt=DT)
    # O10 is a time-onset operator ⇒ the factory also gates on its 2.0 s onset
    _, ev = _walk(trig2)
    assert ev is not None and ev.t >= 2.0
