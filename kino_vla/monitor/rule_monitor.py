"""Rule-based Kino-Monitor v0 — walking-skeleton anomaly detector (spec §3).

Thresholds two proprioceptive channels (M1 scope): contact slip ratio and
velocity-command tracking error. Each channel is EMA-smoothed and must stay above
its threshold for a debounce window before firing; a cooldown suppresses repeat
events while a recovery is already in flight.

The ``MonitorEvent.summary`` string is the walking skeleton's
[STUB: text anomaly summary] — at M4/M7 the Kino-Tokens extractor and the VLA
planner replace it with grounded attribution. The monitor consumes the *measured*
observation stream, so Axis-IV operators (O11) corrupt exactly what it sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kino_vla.sim.types import Obs
from kino_vla.utils.config import Config


@dataclass(frozen=True)
class MonitorEvent:
    """One anomaly detection: which channel fired, where, and a text summary stub."""

    t: float
    pos: np.ndarray  # (2,) world xy at fire time
    channel: str  # "slip_ratio" | "tracking_error"
    value: float  # smoothed channel value at fire time
    threshold: float
    summary: str


_CHANNEL_HINTS = {
    "slip_ratio": "suspected traction loss",
    "tracking_error": "velocity command tracking degraded",
}


class RuleMonitor:
    """Stateful per-episode anomaly monitor; call ``step`` once per control step."""

    def __init__(self, cfg: Config, dt: float) -> None:
        self._cfg = cfg
        self._dt = dt
        self.reset()

    def reset(self) -> None:
        self._ema = {"slip_ratio": 0.0, "tracking_error": 0.0}
        self._above = {"slip_ratio": 0, "tracking_error": 0}
        self._cooldown_until = 0.0
        self.events: list[MonitorEvent] = []

    def step(self, obs: Obs) -> MonitorEvent | None:
        alpha = float(self._cfg.ema_alpha)
        raw = {
            "slip_ratio": obs.slip_ratio,
            "tracking_error": float(np.linalg.norm(obs.cmd_prev[:2] - obs.vel_body)),
        }
        for channel, value in raw.items():
            self._ema[channel] += alpha * (value - self._ema[channel])

        if obs.t < float(self._cfg.arm_delay_s) or obs.t < self._cooldown_until:
            for channel in self._above:
                self._above[channel] = 0
            return None

        for channel in ("slip_ratio", "tracking_error"):
            threshold = float(self._cfg.thresholds.get(channel))
            if self._ema[channel] > threshold:
                self._above[channel] += 1
            else:
                self._above[channel] = 0
            if self._above[channel] >= int(self._cfg.debounce_steps):
                event = self._make_event(obs, channel, self._ema[channel], threshold)
                self.events.append(event)
                self._cooldown_until = obs.t + float(self._cfg.cooldown_s)
                self._above = {key: 0 for key in self._above}
                return event
        return None

    def _make_event(self, obs: Obs, channel: str, value: float, threshold: float) -> MonitorEvent:
        summary = (
            f"[anomaly] t={obs.t:.2f}s pos=({obs.pos[0]:.2f},{obs.pos[1]:.2f}): "
            f"{channel}={value:.2f} > thr={threshold:.2f}; {_CHANNEL_HINTS[channel]}"
        )
        return MonitorEvent(
            t=obs.t,
            pos=obs.pos.copy(),
            channel=channel,
            value=value,
            threshold=threshold,
            summary=summary,
        )
