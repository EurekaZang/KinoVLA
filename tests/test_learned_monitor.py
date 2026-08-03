"""LearnedMonitor online firing logic: arm, debounce, det-attr agreement gate, cooldown, EMA.

Uses a fake model (scripted ``predict``) so the wrapper logic is tested deterministically without
torch or a trained checkpoint. The real detector's TPR/FPR is validated on the Go2 (outputs/
monitor_learned/RESULTS.md): TPR 1.0 on all 11 operators, FPR 0 over 80 negative lanes.
"""

from __future__ import annotations

import numpy as np

from kino_vla.monitor.hazard_lab import N_CLASSES
from kino_vla.monitor.learned_monitor import LearnedMonitor
from kino_vla.sim.types import Obs

DT = 0.02


class _Cfg:
    window_len = 5


class FakeModel:
    """Returns a scripted (hazard_prob, attribution) regardless of the window content."""

    def __init__(self, prob: float, attr_cls: int):
        self.cfg = _Cfg()
        self.prob = float(prob)
        self.attr_cls = int(attr_cls)

    def predict(self, _windows):
        attr = np.zeros((1, N_CLASSES))
        attr[0, self.attr_cls] = 1.0
        return np.array([self.prob]), attr


def _obs(t: float) -> Obs:
    return Obs(
        t=t, pos=np.array([t, 0.0]), heading=0.0, vel_body=np.array([0.5, 0.0]), yaw_rate=0.0,
        cmd_prev=np.array([0.6, 0.0, 0.0]), slip_ratio=0.0, base_height=0.32, tilt=0.05,
        fallen=False, effort_ratio=0.0, support_ratio=1.0,
    )


def _run(model, **kw):
    mon = LearnedMonitor(model, dt=DT, ema_alpha=1.0, arm_s=0.1, cooldown_s=6.0, **kw)
    fires = [mon.step(_obs(k * DT)) for k in range(60)]
    return mon, [f for f in fires if f is not None]


def test_fires_when_both_heads_agree_hazard():
    """High detector prob + a non-normal attribution argmax fires after the debounce."""
    _mon, fires = _run(FakeModel(prob=0.95, attr_cls=5), threshold=0.6, debounce_steps=3)
    assert fires, "should fire when det>=threshold and attribution is a hazard class"
    assert fires[0].channel == "O5"  # class 5 -> O5
    assert fires[0].value >= 0.6


def test_agreement_gate_suppresses_normal_attribution():
    """A confident detector prob whose attribution argmax is 'normal' (class 0) does NOT fire."""
    _mon, fires = _run(FakeModel(prob=0.95, attr_cls=0), threshold=0.6, debounce_steps=3)
    assert not fires, "agreement gate must suppress det/attr disagreement (attr=normal)"


def test_below_threshold_never_fires():
    _mon, fires = _run(FakeModel(prob=0.3, attr_cls=4), threshold=0.6, debounce_steps=3)
    assert not fires


def test_arm_delay_blocks_early_fire():
    """Nothing fires before arm_s even with a strong, agreeing signal."""
    mon = LearnedMonitor(FakeModel(0.95, 5), dt=DT, threshold=0.6, debounce_steps=2,
                         arm_s=0.5, ema_alpha=1.0)
    early = [mon.step(_obs(k * DT)) for k in range(20)]  # 0..0.38s, all < arm 0.5s
    assert all(e is None for e in early)


def test_debounce_requires_consecutive():
    """A single over-threshold window is not enough; debounce_steps consecutive are required."""
    mon = LearnedMonitor(FakeModel(0.95, 5), dt=DT, threshold=0.6, debounce_steps=4,
                         arm_s=0.0, ema_alpha=1.0)
    seq = [mon.step(_obs(k * DT)) for k in range(3)]  # only 3 windows over threshold
    assert all(s is None for s in seq[:3])
