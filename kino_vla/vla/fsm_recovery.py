"""Scripted FSM recovery — walking-skeleton stand-in for the VLA Recovery Planner.

This is the [STUB: scripted FSM recovery (Backstep + replan)] of CLAUDE.md §1:
on a monitor event it executes the §5 primitives ``Backstep`` then
``Replan_Waypoint`` (a box detour around an avoid circle marked at the event
position — a degenerate ``Update_Topology``). It is replaced by the trained VLA
planner at M7 and promoted to the tuned rule-FSM baseline B2 at M8 (spec §12).

Command shaping: outputs are slew-rate-limited Sport-Client velocities so the
low-level tracking error the monitor watches stays small during nominal maneuvers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from kino_vla.monitor.rule_monitor import MonitorEvent
from kino_vla.sim.types import Obs
from kino_vla.utils.config import Config
from kino_vla.utils.geometry import rot90, segment_hits_circle, unit, wrap_angle


class Phase(Enum):
    NOMINAL = "nominal"
    BACKSTEP = "backstep"
    DETOUR = "detour"


@dataclass
class AvoidCircle:
    """Region marked untraversable around a failure site (stub topology memory)."""

    center: np.ndarray  # (2,)
    radius: float


def plan_detour(
    start: np.ndarray,
    goal: np.ndarray,
    circles: list[AvoidCircle],
    clearance_m: float,
    max_legs: int = 6,
) -> list[np.ndarray]:
    """Waypoints from ``start`` to ``goal`` skirting avoid circles with a box detour.

    For the first circle blocking the direct leg to the goal, insert two corner
    waypoints offset perpendicular to the path at radius + clearance, then recurse
    on the remaining path. Pure geometry; deterministic.
    """
    waypoints: list[np.ndarray] = []
    cur = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    for _ in range(max_legs):
        blocking = next(
            (
                c
                for c in circles
                if segment_hits_circle(cur, goal, c.center, c.radius + clearance_m)
            ),
            None,
        )
        if blocking is None:
            waypoints.append(goal)
            return waypoints
        reach = blocking.radius + clearance_m
        u = unit(goal - cur)
        perp = rot90(u)
        cross = float(u[0] * (blocking.center[1] - cur[1]) - u[1] * (blocking.center[0] - cur[0]))
        side = -1.0 if cross > 0.0 else 1.0
        corner1 = blocking.center + side * perp * reach
        corner2 = corner1 + u * reach
        waypoints.extend([corner1, corner2])
        cur = corner2
    waypoints.append(goal)
    return waypoints


class FsmRecovery:
    """Backstep-then-replan recovery policy over a waypoint-pursuit controller."""

    def __init__(self, cfg: Config, goal_xy: np.ndarray, dt: float) -> None:
        self._cfg = cfg
        self._goal = np.asarray(goal_xy, dtype=np.float64).copy()
        self._dt = dt
        self.phase = Phase.NOMINAL
        self.waypoints: list[np.ndarray] = [self._goal.copy()]
        self.avoid_circles: list[AvoidCircle] = []
        self.transitions: list[str] = []
        self.backstep_count = 0
        self._backstep_until = 0.0
        self._grace_until = 0.0
        self._cmd_prev = np.zeros(3)

    def on_event(self, event: MonitorEvent) -> bool:
        """Handle a monitor event; returns False when ignored (already recovering)."""
        if self.phase is Phase.BACKSTEP or event.t < self._grace_until:
            return False
        merged = False
        for circle in self.avoid_circles:
            if float(np.linalg.norm(circle.center - event.pos)) < circle.radius:
                circle.radius *= float(self._cfg.avoid.growth)
                merged = True
                break
        if not merged:
            self.avoid_circles.append(
                AvoidCircle(center=event.pos.copy(), radius=float(self._cfg.avoid.radius_m))
            )
        self.phase = Phase.BACKSTEP
        self.backstep_count += 1
        self._backstep_until = event.t + float(self._cfg.backstep.duration_s)
        self.transitions.append(
            f"t={event.t:.2f}s {event.channel} -> Backstep"
            f"{' (avoid region grown)' if merged else ' (avoid region marked)'}"
        )
        return True

    def step(self, obs: Obs) -> np.ndarray:
        if self.phase is Phase.BACKSTEP:
            if obs.t >= self._backstep_until:
                self.waypoints = plan_detour(
                    obs.pos,
                    self._goal,
                    self.avoid_circles,
                    float(self._cfg.avoid.clearance_m),
                )
                self.phase = Phase.DETOUR
                self._grace_until = obs.t + float(self._cfg.event_grace_s)
                self.transitions.append(
                    f"t={obs.t:.2f}s Replan_Waypoint ({len(self.waypoints)} waypoints)"
                )
                raw = self._pursuit(obs)
            else:
                raw = np.array([-float(self._cfg.backstep.speed_mps), 0.0, 0.0])
        else:
            raw = self._pursuit(obs)
        return self._slew(raw)

    def _pursuit(self, obs: Obs) -> np.ndarray:
        """Unicycle waypoint pursuit: turn toward the waypoint, drive when aligned."""
        while self.waypoints:
            to_wp = self.waypoints[0] - obs.pos
            if float(np.linalg.norm(to_wp)) > float(self._cfg.waypoint_tol_m):
                break
            self.waypoints.pop(0)
        if not self.waypoints:
            return np.zeros(3)
        to_wp = self.waypoints[0] - obs.pos
        dist = float(np.linalg.norm(to_wp))
        heading_err = wrap_angle(math.atan2(to_wp[1], to_wp[0]) - obs.heading)
        wz = float(self._cfg.heading_gain) * heading_err
        speed = float(self._cfg.cruise_speed_mps) * min(
            1.0, dist / float(self._cfg.approach_slowdown_m)
        )
        # Turn in place when grossly misaligned, else the curved pursuit path can
        # bow off the planned segment (e.g. back into the avoid region).
        if abs(heading_err) > float(self._cfg.align_angle_rad):
            vx = 0.0
        else:
            vx = speed * max(0.0, math.cos(heading_err))
        return np.array([vx, 0.0, wz])

    def _slew(self, raw: np.ndarray) -> np.ndarray:
        max_dv = float(self._cfg.cmd_slew_mps2) * self._dt
        max_dw = float(self._cfg.yaw_slew_radps2) * self._dt
        cmd = self._cmd_prev.copy()
        cmd[:2] += np.clip(raw[:2] - cmd[:2], -max_dv, max_dv)
        cmd[2] += float(np.clip(raw[2] - cmd[2], -max_dw, max_dw))
        self._cmd_prev = cmd
        return cmd.copy()
