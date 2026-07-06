"""Threshold monitor TEST DOUBLE — scaffolding for unit tests of components other than the monitor
(the walking-skeleton demo, DPO rollouts, the coupler gate, the data-collection probe).

This is test-only: the DEPLOYED Kino-Monitor is the trained learning-based
:class:`~kino_vla.monitor.learned_monitor.LearnedMonitor` (real-Go2, TPR 1.0 / FPR 0, outputs/
monitor_learned/RESULTS.md). The learned detector is Isaac-trained and does not transfer to the
surrogate point-robot, so these surrogate CPU tests inject this deterministic double, which fires on
a raw proprioceptive channel using the legacy ``configs/monitor/rule_v0.yaml`` calibration
(EMA + debounce + arm + cooldown) so the tuned demo/data behaviour is reproduced exactly.
"""

from __future__ import annotations

import numpy as np

from kino_vla.monitor.event import MonitorEvent
from kino_vla.sim.types import Obs
from kino_vla.utils.config import load_config

_CHANNELS = ("slip_ratio", "tracking_error", "effort_ratio", "tilt")
_HINTS = {
    "slip_ratio": "suspected traction loss",
    "tracking_error": "velocity command tracking degraded",
    "effort_ratio": "suspected actuator-effort saturation",
    "tilt": "loss of balance / imminent fall",
}


class StubMonitor:
    """EMA + debounce + arm + cooldown threshold monitor (legacy rule_v0 calibration)."""

    def __init__(self, cfg=None):
        self._cfg = cfg if cfg is not None else load_config("monitor/rule_v0.yaml")
        self._arm = float(self._cfg.arm_delay_s)
        self._tilt_arm = float(self._cfg.get("tilt_arm_delay_s", self._cfg.arm_delay_s))
        self.reset()

    def _threshold(self, ch: str) -> float:
        t = self._cfg.thresholds.get(ch)
        return float(t) if t is not None else float("inf")

    def reset(self) -> None:
        self._ema = dict.fromkeys(_CHANNELS, 0.0)
        self._above = dict.fromkeys(_CHANNELS, 0)
        self._cooldown_until = 0.0
        self.events: list[MonitorEvent] = []

    @property
    def anomaly_score(self) -> float:
        return max(self._ema[c] / self._threshold(c) for c in _CHANNELS)

    def step(self, obs: Obs) -> MonitorEvent | None:
        alpha = float(self._cfg.ema_alpha)
        raw = {
            "slip_ratio": obs.slip_ratio,
            "tracking_error": float(np.linalg.norm(obs.cmd_prev[:2] - obs.vel_body)),
            "effort_ratio": obs.effort_ratio,
            "tilt": obs.tilt,
        }
        for ch, v in raw.items():
            self._ema[ch] += alpha * (v - self._ema[ch])
        if obs.t < self._cooldown_until:
            for ch in self._above:
                self._above[ch] = 0
            return None
        for ch in _CHANNELS:
            arm = self._tilt_arm if ch == "tilt" else self._arm
            if obs.t < arm:
                self._above[ch] = 0
                continue
            thr = self._threshold(ch)
            self._above[ch] = self._above[ch] + 1 if self._ema[ch] > thr else 0
            if self._above[ch] >= int(self._cfg.debounce_steps):
                ev = MonitorEvent(
                    t=obs.t, pos=obs.pos.copy(), channel=ch, value=self._ema[ch], threshold=thr,
                    summary=(
                        f"[stub] t={obs.t:.2f}s {ch}={self._ema[ch]:.2f}>{thr:.2f}; {_HINTS[ch]}"
                    ),
                )
                self.events.append(ev)
                self._cooldown_until = obs.t + float(self._cfg.cooldown_s)
                self._above = dict.fromkeys(_CHANNELS, 0)
                return ev
        return None


def make_obs(t, slip=0.0, cmd=(0.0, 0.0, 0.0), vel=(0.0, 0.0)) -> Obs:
    """Minimal Obs builder shared by component tests (was tests/test_monitor.py before the
    rule monitor was removed)."""
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
