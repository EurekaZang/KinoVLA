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
    "effort_ratio": "suspected actuator-effort saturation",
    "tilt": "loss of balance / imminent fall",
}

# The proprioceptive channels the monitor thresholds, in priority order (spec §3).
# ``tilt`` (added 2026-06-17) is the imminent-fall channel: a body tilt blow-up is an
# UNAMBIGUOUS anomaly (not a push-off artifact), so it arms early (``tilt_arm_delay_s``,
# default = the main arm delay) — catching topples that occur before the slip/tracking/
# effort channels arm (e.g. O5 payload toppling at ~1.7 s, before the 2.5 s Isaac arm).
# It is inert on any config that omits a ``tilt`` threshold (treated as +inf) and on the
# flat surrogate (tilt ≡ 0), so it adds detection without touching the existing channels.
_CHANNELS = ("slip_ratio", "tracking_error", "effort_ratio", "tilt")


class RuleMonitor:
    """Stateful per-episode anomaly monitor; call ``step`` once per control step."""

    def __init__(self, cfg: Config, dt: float) -> None:
        self._cfg = cfg
        self._dt = dt
        self._arm = float(cfg.arm_delay_s)
        # The tilt channel arms early (a real tilt is not a startup transient); falls back to
        # the main arm delay if the config does not set tilt_arm_delay_s.
        self._tilt_arm = float(cfg.get("tilt_arm_delay_s", cfg.arm_delay_s))
        self.reset()

    def _threshold(self, channel: str) -> float:
        """Fire threshold for a channel; +inf (never fires) when the config omits it."""
        t = self._cfg.thresholds.get(channel)
        return float(t) if t is not None else float("inf")

    def _arm_delay(self, channel: str) -> float:
        return self._tilt_arm if channel == "tilt" else self._arm

    def reset(self) -> None:
        self._ema = {channel: 0.0 for channel in _CHANNELS}
        self._above = {channel: 0 for channel in _CHANNELS}
        self._cooldown_until = 0.0
        self.events: list[MonitorEvent] = []

    @property
    def channel_emas(self) -> dict[str, float]:
        """Current EMA-smoothed value of each channel (telemetry/recording access)."""
        return dict(self._ema)

    @property
    def anomaly_score(self) -> float:
        """Threshold-normalized anomaly score: ``max_c EMA_c / threshold_c``.

        A scale-free detector statistic (1.0 is the firing operating point) used to
        sweep the Monitor ROC across rollouts (spec §12 metric); independent of the
        debounce/cooldown firing logic, which only sets *when* an event is emitted.
        """
        return max(self._ema[channel] / self._threshold(channel) for channel in _CHANNELS)

    def step(self, obs: Obs) -> MonitorEvent | None:
        alpha = float(self._cfg.ema_alpha)
        raw = {
            "slip_ratio": obs.slip_ratio,
            "tracking_error": float(np.linalg.norm(obs.cmd_prev[:2] - obs.vel_body)),
            "effort_ratio": obs.effort_ratio,
            "tilt": obs.tilt,
        }
        for channel, value in raw.items():
            self._ema[channel] += alpha * (value - self._ema[channel])

        if obs.t < self._cooldown_until:
            for channel in self._above:
                self._above[channel] = 0
            return None

        # Each channel is gated by its own arm delay (tilt arms early; the rest skip the
        # push-off transient). Channels are checked in _CHANNELS priority order.
        for channel in _CHANNELS:
            if obs.t < self._arm_delay(channel):
                self._above[channel] = 0
                continue
            threshold = self._threshold(channel)
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
