"""A0.2 — Primary trigger protocol: a privileged, agent-independent, deterministic hazard-onset
event (Paper-A §2 A0.2, rule R3/R4).

Attribution is evaluated GIVEN an event; the detector operating point is a separate problem (R3).
So the primary experiments (A2–A6, A4 label-swap matrix) trigger from privileged sim state, not from
the learned detector — every agent then snapshots the IDENTICAL (t, pos), making the attribution
comparison fair (R4: ground truths independent of the evaluated agents). :class:`OracleTrigger`
implements the :class:`kino_vla.monitor.event.Monitor` protocol, so it drops into
``run_closed_loop`` / the collectors exactly where the deployed ``LearnedMonitor`` goes.

It fires ONCE, when the privileged onset condition first holds and has persisted for a fixed arm
delay: the robot is inside the tagged hazard region AND (for time-onset operators) the onset time
has passed. θ NEVER enters it — only region geometry and a fixed time/step delay — so it is
R8-clean and cannot leak the answer. The hardened realistic detector (``monitor_abaware``) is the
SECONDARY trigger (A0.2-secondary / A4.5); this oracle stays primary everywhere.
"""

from __future__ import annotations

import numpy as np

from kino_vla.monitor.event import MonitorEvent
from kino_vla.sim.types import Obs
from kino_vla.utils.geometry import Rect

# Default arm delay: hold the onset condition this long before firing, so the operator's proprio
# signature (slip / resistance / effort spike) is established in the snapshot window rather than the
# region-entry transient. 0.2 s ≈ 10 control steps at 50 Hz (mirrors the collection gate margins).
DEFAULT_ARM_DELAY_S = 0.2
# Operators whose fault is a TIME onset (not a region): the trigger also waits for the onset time.
_TIME_ONSET_S: dict[str, float] = {
    "O5_payload": 2.0,
    "O10_effort_decay": 2.0,
    "O6_push": 2.0,
}
# Boundary operators: an impassable wall (O8) blocks the robot at the region's NEAR FACE (~a body
# radius before it), so it never gets INSIDE the rect. The onset is "reached the obstacle" — fire on
# an approach-margin-expanded rect so the blocked pose (tracking-error spike) is captured.
_BOUNDARY_OPS: frozenset[str] = frozenset({"O8_invisible_collider"})
_BOUNDARY_MARGIN_M = 0.5


class OracleTrigger:
    """Privileged, deterministic, agent-independent hazard-onset trigger (A0.2 primary).

    Construct with a hazard ``rect`` and/or an ``onset_time_s``; the trigger fires the first control
    step at which BOTH hold (rect containment if given, onset-time elapsed if given) AND the arm
    delay has passed since that condition first became true. Fires exactly once per episode; call
    :meth:`reset` between episodes. ``channel`` labels the emitted event (default the neutral
    ``"oracle"``; the true operator is carried in ``summary`` for logging, never as the answer).
    """

    def __init__(
        self,
        *,
        rect: Rect | None = None,
        onset_time_s: float | None = None,
        dt: float,
        arm_delay_s: float = DEFAULT_ARM_DELAY_S,
        channel: str = "oracle",
        operator_name: str = "",
    ) -> None:
        if rect is None and onset_time_s is None:
            raise ValueError("OracleTrigger needs a rect and/or onset_time_s (a privileged onset)")
        self.events: list[MonitorEvent] = []
        self._rect = rect
        self._onset_time = None if onset_time_s is None else float(onset_time_s)
        self._dt = float(dt)
        self._arm = float(arm_delay_s)
        self._channel = channel
        self._operator_name = operator_name
        self._fired = False
        self._cond_t: float | None = None  # first sim-time the privileged condition held

    @property
    def anomaly_score(self) -> float:
        return 1.0 if self._fired else 0.0

    def reset(self) -> None:
        self.events = []
        self._fired = False
        self._cond_t = None

    def step(self, obs: Obs) -> MonitorEvent | None:
        if self._fired:
            return None
        cond = True
        if self._rect is not None:
            cond = cond and bool(self._rect.contains(np.asarray(obs.pos)))
        if self._onset_time is not None:
            cond = cond and (float(obs.t) >= self._onset_time)
        if cond:
            if self._cond_t is None:
                self._cond_t = float(obs.t)
        else:
            self._cond_t = None  # left before arming ⇒ re-require persistence (a spike ≠ onset)
        if self._cond_t is not None and float(obs.t) + 1e-9 >= self._cond_t + self._arm:
            ev = MonitorEvent(
                t=float(obs.t),
                pos=np.asarray(obs.pos).copy(),
                channel=self._channel,
                value=1.0,
                threshold=0.5,
                summary=f"oracle onset (privileged): {self._operator_name}",
            )
            self.events.append(ev)
            self._fired = True
            return ev
        return None

    # ------------------------------------------------------------------- factories
    @classmethod
    def for_scenario(
        cls, scenario: object, *, dt: float, arm_delay_s: float = DEFAULT_ARM_DELAY_S
    ) -> OracleTrigger:
        """Build the oracle trigger for a rollout :class:`~kino_vla.vla.rollout.Scenario`.

        Per the spec's two-form onset (region entry OR operator activation time): a REGION operator
        (friction/resistance patch) triggers on entry into ``scene_region.rect``; a GLOBAL
        time-onset operator (payload / effort / push — the failure is everywhere, robot need not
        reach a patch) triggers on ``activation time + Δ``, position-free; a BOUNDARY operator
        (O8 wall) triggers on an approach-margin-expanded rect (it is blocked at the near face). All
        from sim geometry + fixed constants, never θ."""
        op_name = str(getattr(scenario, "operator_name", ""))
        onset = _TIME_ONSET_S.get(op_name)
        if onset is not None:  # global time-onset operator ⇒ NO rect gate (failure is everywhere)
            return cls(onset_time_s=onset, dt=dt, arm_delay_s=arm_delay_s, operator_name=op_name)
        region = getattr(scenario, "scene_region", None)
        rect = getattr(region, "rect", None)
        if rect is not None and op_name in _BOUNDARY_OPS:  # expand so the blocked pose is in-rect
            rect = Rect(cx=rect.cx, cy=rect.cy, hx=rect.hx + _BOUNDARY_MARGIN_M,
                        hy=rect.hy + _BOUNDARY_MARGIN_M)
        return cls(rect=rect, dt=dt, arm_delay_s=arm_delay_s, operator_name=op_name)
