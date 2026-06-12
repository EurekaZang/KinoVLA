"""Rule-based Kino-Monitor v0: thresholding, debounce, arm delay, cooldown."""

from __future__ import annotations

import numpy as np

from kino_vla.monitor.rule_monitor import RuleMonitor
from kino_vla.sim.types import Obs
from kino_vla.utils.config import load_config

CFG = load_config("monitor/rule_v0.yaml")
DT = 0.02


def make_obs(t, slip=0.0, cmd=(0.0, 0.0, 0.0), vel=(0.0, 0.0)) -> Obs:
    return Obs(
        t=t,
        pos=np.array([t, 0.0]),
        heading=0.0,
        vel_body=np.asarray(vel, dtype=np.float64),
        yaw_rate=0.0,
        cmd_prev=np.asarray(cmd, dtype=np.float64),
        slip_ratio=slip,
        base_height=0.31,
        tilt=0.0,
        fallen=False,
    )


def feed(monitor, signal):
    """Feed (t, obs) pairs; return list of fired events."""
    events = []
    for obs in signal:
        event = monitor.step(obs)
        if event is not None:
            events.append(event)
    return events


def steps(t0, n, **kw):
    return [make_obs(t0 + k * DT, **kw) for k in range(n)]


def test_quiet_stream_never_fires():
    monitor = RuleMonitor(CFG, dt=DT)
    quiet = steps(0.0, 500, slip=0.1, cmd=(0.5, 0.0, 0.0), vel=(0.45, 0.0))
    assert feed(monitor, quiet) == []


def test_slip_fires_after_debounce():
    monitor = RuleMonitor(CFG, dt=DT)
    t0 = float(CFG.arm_delay_s) + 0.1
    events = feed(monitor, steps(t0, 50, slip=0.9))
    assert len(events) == 1
    event = events[0]
    assert event.channel == "slip_ratio"
    assert event.value > event.threshold
    # EMA warm-up plus debounce: fires within a small window after onset.
    fire_step = round((event.t - t0) / DT)
    assert int(CFG.debounce_steps) <= fire_step <= int(CFG.debounce_steps) + 10


def test_arm_delay_suppresses_startup_transient():
    monitor = RuleMonitor(CFG, dt=DT)
    n_before_arm = int(float(CFG.arm_delay_s) / DT) - 1
    assert feed(monitor, steps(0.0, n_before_arm, slip=0.9)) == []


def test_cooldown_then_refire():
    monitor = RuleMonitor(CFG, dt=DT)
    t0 = float(CFG.arm_delay_s) + 0.1
    n = int((2.5 * float(CFG.cooldown_s)) / DT)
    events = feed(monitor, steps(t0, n, slip=0.9))
    assert len(events) >= 2
    gap = events[1].t - events[0].t
    assert gap >= float(CFG.cooldown_s)


def test_tracking_error_channel():
    monitor = RuleMonitor(CFG, dt=DT)
    t0 = float(CFG.arm_delay_s) + 0.1
    events = feed(monitor, steps(t0, 50, cmd=(0.8, 0.0, 0.0), vel=(0.1, 0.0)))
    assert events and events[0].channel == "tracking_error"


def test_summary_stub_format():
    monitor = RuleMonitor(CFG, dt=DT)
    t0 = float(CFG.arm_delay_s) + 0.1
    events = feed(monitor, steps(t0, 50, slip=0.9))
    summary = events[0].summary
    assert "slip_ratio" in summary and "traction" in summary and "t=" in summary
