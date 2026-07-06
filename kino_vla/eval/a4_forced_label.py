"""A4.2 — the forced-label executor + outcome recorder (Paper-A §4 A4.2, serves C3).

Forces a scripted canonical recovery (a LABEL) onto a physical scene and measures the outcome,
holding everything else fixed: oracle trigger (A0.2-primary) marks the decision moment identically
for every label, the deterministic backend (A0.1 deep_reset) makes the episode reproducible, the
base locomotion policy is frozen, and a fixed safety envelope (the backend's velocity/yaw caps +
fall-abort) replaces the CBF shield (experiments_design.md §0.3 cuts the CBF chapter). The label,
not any agent, decides the action ⇒ ``M(s, ℓ)`` is a pure CAUSAL object.

The executor implements the loop's :class:`~kino_vla.loop.RecoveryPolicy` contract (``on_event`` /
``step``): it cruises until the oracle fires, then enacts the forced label's scripted controller —
``continue`` (no intervention), ``backstep_detour`` (escape then route around), ``high_step`` /
``crawl`` (gait switches), ``slow_low`` (speed + posture cap), ``hold_request`` (safe stop),
``detour_replan`` (mark blocked + route around). :class:`OutcomeRecorder` is an ``on_step`` tap that
reduces the EXTENDED readouts the loop's result does not carry — peak |ω|, immobilization (v < ε for
> 5 s in-hazard), catapult (a tether-tear energy-release motion spike), left_hazard, forward
progress — and computes the pre-registered success verdict (R5). Works on the surrogate (import-
smoke / unit-test scaffold) and the real Go2 (the deliverable).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kino_vla.eval.a4_consequence import Outcome
from kino_vla.utils.geometry import Rect


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class ForcedLabelPolicy:
    """Drive one forced recovery label deterministically (no agent decision). Implements the loop's
    RecoveryPolicy (on_event/step); sets ``posture_height``/``posture_stiffness`` which the loop
    forwards to the backend's body-posture controller each step."""

    label: str
    rect: Rect
    goal_xy: np.ndarray
    cruise: float = 0.6
    dt: float = 0.02
    back_speed: float = 0.30
    back_dist: float = 0.45
    detour_clear: float = 0.60  # y-margin beyond rect.hy for the detour offset
    k_yaw: float = 2.5  # proportional yaw gain for waypoint pursuit
    max_yaw: float = 1.6  # rad/s yaw-rate cap (within the backend's own cap)
    high_z: float = 0.30  # high-step gait body height
    low_z: float = 0.22  # slow+low-posture body height
    crawl_z: float = 0.20  # crawl-gait body height
    wp_tol: float = 0.30  # waypoint-reached tolerance [m]

    def __post_init__(self) -> None:
        self.goal_xy = np.asarray(self.goal_xy, dtype=np.float64)
        self.fired = False
        self.event_pos: np.ndarray | None = None
        self.t_post = 0.0
        self.subphase = "cruise"
        self.waypoints: list[np.ndarray] = []
        self._wp_idx = 0
        self._back_origin: np.ndarray | None = None
        # the loop forwards these to backend.set_posture each step (None ⇒ nominal trot)
        self.posture_height: float | None = None
        self.posture_stiffness: float = 1.0

    # --------------------------------------------------------------- RecoveryPolicy
    def on_event(self, event) -> None:  # noqa: ANN001 (MonitorEvent)
        if self.fired:
            return
        self.fired = True
        self.event_pos = np.asarray(event.pos, dtype=np.float64).copy()
        self.t_post = 0.0
        self._begin_label()

    def step(self, obs) -> np.ndarray:  # noqa: ANN001 (Obs)
        if self.fired:
            self.t_post += self.dt
        if not self.fired or self.label == "continue":
            # `continue` = no semantic intervention: keep cruising through (the nominal T5 answer,
            # OR the push-through consequence on a hazard the robot was going to enter anyway).
            return np.array([self.cruise, 0.0, 0.0])
        if self.label == "hold_request":
            return np.zeros(3)  # safe stop (the loop's fall-abort + the halt verdict handle it)
        if self.label in ("high_step", "slow_low", "crawl"):
            # `high_step` = a forward POWER-THROUGH cruise (the mud-canonical): the sim has no
            # distinct high-step gait, and RAISING the body cuts push-through traction
            # (it stalled in mud in the A4 smoke), so high_step is a forward cruise at full
            # speed. slow_low / crawl cap speed (a real distinct low-effort response, #41).
            spd = (
                self.cruise
                if self.label == "high_step"
                else (0.30 if self.label == "slow_low" else 0.40)
            )
            return np.array([spd, 0.0, 0.0])
        if self.label == "backstep_detour":
            return self._step_backstep(obs)
        if self.label == "detour_replan":
            return self._step_detour(obs)
        return np.array([self.cruise, 0.0, 0.0])

    # --------------------------------------------------------------- controllers
    def _begin_label(self) -> None:
        # No forced label changes body posture: the sim's posture channel (#41) introduces push-
        # through artifacts (a raised body stalls in mud; a lowered body stalls on ice), so the
        # forced labels are pure SPEED/DIRECTION modulations and the consequence structure is driven
        # by OPERATOR physics (adhesion ramp, wall, overload, decay) — the honest C3 story.
        if self.label in ("continue", "high_step", "slow_low", "crawl"):
            return
        if self.label == "backstep_detour":
            self.subphase = "back"
            self._back_origin = self.event_pos.copy()
        elif self.label == "detour_replan":
            self.subphase = "detour"
            self._setup_detour_waypoints(reverse=False)

    def _detour_y(self) -> float:
        # clear the rect on the +y side (lane_y + rect.hy + margin); canonical for the matrix
        return self.rect.cy + self.rect.hy + self.detour_clear

    def _setup_detour_waypoints(self, *, reverse: bool) -> None:
        y_clear = self._detour_y()
        x_near = float(self.event_pos[0])
        # veer to y_clear before the rect's far edge, cross past it, then come back to the goal
        self.waypoints = [
            np.array([x_near, y_clear]),
            np.array([float(self.goal_xy[0]), y_clear]),
            np.asarray(self.goal_xy, dtype=np.float64).copy(),
        ]
        self._wp_idx = 0
        if reverse:  # backstep_detour: waypoints set after the back phase completes
            self.subphase = "detour"

    def _step_backstep(self, obs) -> np.ndarray:  # noqa: ANN001
        if self.subphase == "back":
            backed = float(np.linalg.norm(obs.pos - self._back_origin))
            if backed >= self.back_dist:
                self._setup_detour_waypoints(reverse=True)
                return self._step_detour(obs)
            return np.array([-self.back_speed, 0.0, 0.0])  # peel out (the adhesion escape)
        return self._step_detour(obs)

    def _step_detour(self, obs) -> np.ndarray:  # noqa: ANN001
        if self._wp_idx >= len(self.waypoints):
            return np.array([self.cruise, 0.0, 0.0])  # all waypoints cleared ⇒ straight to goal
        wp = self.waypoints[self._wp_idx]
        if float(np.linalg.norm(wp - obs.pos)) < self.wp_tol:
            self._wp_idx += 1
            return self._step_detour(obs)
        to = wp - obs.pos
        ang = float(np.arctan2(to[1], to[0]))
        err = _wrap_pi(ang - float(obs.heading))
        wz = float(np.clip(self.k_yaw * err, -self.max_yaw, self.max_yaw))
        fwd = max(0.0, float(np.cos(err)))  # drive forward only when roughly facing the waypoint
        return np.array([self.cruise * fwd, 0.0, wz])


@dataclass
class OutcomeRecorder:
    """An ``on_step`` tap accumulating the extended A4 readouts + privileged tether telemetry
    and computes the pre-registered success verdict (R5) at episode end."""

    rect: Rect
    goal_xy: np.ndarray
    success_criterion: str  # one of reach / safe_halt / escape / continue (A0.5)
    backend: object  # for privileged_physics() (tether break marker)
    label: str
    spatial: bool = True  # region op (immobilized = stuck IN the rect) vs global (after event)
    cruise: float = 0.6
    immob_speed: float = 0.05  # v below this counts as "not making progress"
    immob_time: float = 5.0  # ...for this many consecutive seconds ⇒ immobilized
    cat_omega: float = 4.0  # peak |ω| above this ⇒ a lurch (the catapult motion signature)
    cat_lunge: float = 1.2  # max |v| / cruise above this after the event ⇒ a forward lunge

    def __post_init__(self) -> None:
        self.peak_omega = 0.0
        self.max_speed = 0.0
        self.max_speed_post_event = 0.0  # peak |v| AFTER the event (catapult lunge)
        self.final_speed = 0.0  # |v| on the LAST step (halt detection: settled vs wobbling)
        self.max_fwd_after_event = 0.0
        self.max_x = -1e9  # peak forward x (catapult overshoot detection)
        self.event_pos: np.ndarray | None = None
        self._immob_run = 0.0  # consecutive seconds of low-speed in-hazard
        self.immobilized = False
        self.left_hazard = False
        self.tether_broke = False  # privileged mechanism marker (the report's explanation)
        self.ever_in_hazard = False
        self.steps = 0

    def __call__(self, obs, event, cmd, decision) -> None:  # noqa: ANN001 (the on_step signature)
        self.steps += 1
        if event is not None and self.event_pos is None:  # the loop passes the fire-step event here
            self.event_pos = np.asarray(event.pos, dtype=np.float64).copy()
        omega = float(abs(obs.yaw_rate))
        if omega > self.peak_omega:
            self.peak_omega = omega
        spd = float(np.linalg.norm(obs.vel_body))
        if spd > self.max_speed:
            self.max_speed = spd
        self.final_speed = spd
        if float(obs.pos[0]) > self.max_x:
            self.max_x = float(obs.pos[0])
        in_haz = bool(self.rect.contains(obs.pos))
        if in_haz:
            self.ever_in_hazard = True
        else:
            if self.ever_in_hazard:
                self.left_hazard = True
        # immobilization: sustained low speed while the hazard is active — IN the rect for a region
        # op (spatial), or anywhere after the event for a global op (temporal). A DELIBERATE hold is
        # not "immobilized" (the stop is intended, not a trap). Observable, R8-clean.
        post_event = self.event_pos is not None
        stuck = self.label != "hold_request" and (
            (in_haz if self.spatial else post_event) and spd < self.immob_speed
        )
        if stuck:
            self._immob_run += getattr(obs, "dt", None) or 0.02
            if self._immob_run >= self.immob_time:
                self.immobilized = True
        else:
            self._immob_run = 0.0
        # post-event speed (halt detection + catapult lunge): the pre-event cruise must NOT count
        if self.event_pos is not None:
            if spd > self.max_fwd_after_event:
                self.max_fwd_after_event = spd
            if spd > self.max_speed_post_event:
                self.max_speed_post_event = spd
        # privileged tether-tear marker (the mechanism; never agent-visible)
        try:
            priv = self.backend.privileged_physics()
            if float(priv.get("tether_broken", 0.0)) > 0.0:
                self.tether_broke = True
        except Exception:  # noqa: BLE001 (backend may not expose privileged physics)
            pass

    # --------------------------------------------------------------- verdict
    def build_outcome(
        self,
        *,
        scenario: str,
        seed: int,
        reached: bool,
        fell: bool,
        final_dist_m: float,
        sim_time_s: float,
    ) -> Outcome:
        n_interv = 0 if self.label == "continue" else 1
        overshoot = (
            self.max_x > float(self.goal_xy[0]) + 1.5
        )  # blew past the goal (a catapult lunge)
        catapult = bool(
            self.tether_broke
            and (
                self.peak_omega > self.cat_omega
                or self.max_fwd_after_event > self.cat_lunge * self.cruise
                or overshoot
            )
        )
        forward_progress = reached or (
            self.event_pos is not None
            and final_dist_m < (float(np.linalg.norm(self.goal_xy - self.event_pos)))
        )
        # the pre-registered R5 verdict
        if self.success_criterion == "reach":
            success = reached and not fell
        elif self.success_criterion == "safe_halt":
            # a deliberate Hold: settled to near-zero speed at the end (a stop, not a cruise).
            halted = self.label == "hold_request" and self.final_speed < 0.2
            success = (reached or halted) and not fell
        elif self.success_criterion == "escape":
            success = (
                self.left_hazard
                and forward_progress
                and not fell
                and not catapult
                and not self.immobilized
            )
        else:  # continue (nominal T5): reached with NO unnecessary semantic intervention
            success = reached and not fell and n_interv == 0
        return Outcome(
            scenario=scenario,
            label=self.label,
            seed=seed,
            reached=reached,
            fell=fell,
            immobilized=self.immobilized,
            catapult=catapult,
            final_dist_m=round(final_dist_m, 3),
            sim_time_s=round(sim_time_s, 2),
            peak_omega=round(self.peak_omega, 3),
            n_semantic_interventions=n_interv,
            left_hazard=self.left_hazard,
            success=bool(success),
        )
