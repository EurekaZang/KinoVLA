"""The Kino-Monitor event contract — what the monitor emits when it fires (spec §3).

A neutral home for ``MonitorEvent`` so every consumer (the loop, the planner, the recovery FSM,
the snapshot recorder) depends on the event type, not on any particular monitor implementation.
The deployed monitor is the learning-based :class:`kino_vla.monitor.learned_monitor.LearnedMonitor`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from kino_vla.sim.types import Obs


@dataclass(frozen=True)
class MonitorEvent:
    """One anomaly detection: which cause fired, where, and a text summary.

    ``channel`` is the inferred physical cause (the learned monitor's attribution argmax, e.g.
    ``"O5"`` / ``"low_friction"``-style), ``value`` the firing statistic (the hazard probability),
    ``threshold`` the firing threshold.
    """

    t: float
    pos: np.ndarray  # (2,) world xy at fire time
    channel: str
    value: float
    threshold: float
    summary: str


class Monitor(Protocol):
    """The Kino-Monitor interface every consumer (loop, coupler, planner) depends on — implemented
    by :class:`kino_vla.monitor.learned_monitor.LearnedMonitor`."""

    events: list[MonitorEvent]

    @property
    def anomaly_score(self) -> float: ...

    def reset(self) -> None: ...

    def step(self, obs: Obs) -> MonitorEvent | None: ...
