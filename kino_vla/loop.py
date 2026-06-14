"""Closed-loop episode runner — the walking skeleton (CLAUDE.md §1).

Pipeline per control step:
    backend obs -> operator obs-corruption (Axis IV) -> Kino-Monitor ->
    recovery policy -> Safety Shield -> backend command

At M1 every stage downstream of the monitor is a stub (scripted FSM, pass-through
shield); M2–M7 each replace exactly one stage. The loop runs single-rate at the
backend's control frequency; the dual-rate (1 kHz monitor / planner-rate) split
arrives with M3 (spec §6.9 latency budget).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from kino_vla.monitor.rule_monitor import MonitorEvent, RuleMonitor
from kino_vla.shield.passthrough import ShieldDecision
from kino_vla.sim.backend import LocomotionBackend
from kino_vla.sim.operators.base import OperatorStack
from kino_vla.sim.types import Obs
from kino_vla.utils.seeding import trajectory_hash


class RecoveryPolicy(Protocol):
    """Anything that maps observations to commands and reacts to monitor events."""

    def on_event(self, event: MonitorEvent) -> bool: ...

    def step(self, obs: Obs) -> np.ndarray: ...


class Shield(Protocol):
    """Command filter contract (pass-through at M1, CBF-QP from M3)."""

    def filter(self, cmd: np.ndarray, obs: Obs) -> ShieldDecision: ...


@dataclass(frozen=True)
class EpisodeResult:
    """Everything the demo assertions and the QA determinism gates need."""

    monitor_fired: bool
    events: list[MonitorEvent]
    fell: bool
    goal_reached: bool
    final_dist_m: float
    sim_time_s: float
    wall_time_s: float
    n_steps: int
    shield_interventions: int
    traj_hash: str


def run_episode(
    backend: LocomotionBackend,
    operators: OperatorStack,
    monitor: RuleMonitor,
    policy: RecoveryPolicy,
    shield: Shield,
    *,
    seed: int,
    goal_xy: np.ndarray,
    goal_tol_m: float,
    max_time_s: float,
    on_step: Callable[[Obs, MonitorEvent | None, np.ndarray, ShieldDecision], None] | None = None,
) -> EpisodeResult:
    """Run one seeded episode to goal, fall, or timeout.

    ``on_step`` (optional) is called once per control step after the shield decision
    and before the backend steps, with ``(obs_measured, event, cmd, decision)`` — a
    non-invasive hook for telemetry recording (scripts/record_demo.py). Default None
    keeps the demo/CI path untouched.
    """
    wall_start = time.perf_counter()
    goal_xy = np.asarray(goal_xy, dtype=np.float64)
    obs = backend.reset(seed)
    operators.on_reset(backend)
    monitor.reset()
    if hasattr(shield, "reset"):
        shield.reset()  # per-episode isolation of CBF intervention/latency stats

    events: list[MonitorEvent] = []
    interventions = 0
    positions: list[np.ndarray] = [obs.pos.copy()]
    velocities: list[np.ndarray] = [obs.vel_body.copy()]
    goal_reached = False
    max_steps = int(round(max_time_s / backend.dt))

    for _ in range(max_steps):
        obs_measured = operators.transform_obs(obs)
        event = monitor.step(obs_measured)
        if event is not None:
            events.append(event)
            policy.on_event(event)
        cmd = policy.step(obs_measured)
        decision = shield.filter(cmd, obs_measured)
        interventions += int(decision.intervened)
        # Spec §6.8 fallback coupling: when the shield escalates to the brace stance
        # or halts (QP infeasible at the nominal mode), engage the physical Reflex
        # stance so the robot's real support/capture limits match the mode the shield
        # adjudicated against. No-op for the pass-through stub (empty codes) and for
        # backends without a Reflex hook.
        if hasattr(backend, "set_reflex"):
            backend.set_reflex(
                any(c.startswith(("INFEASIBLE_FALLBACK", "HALT")) for c in decision.codes)
            )
        if on_step is not None:
            on_step(obs_measured, event, cmd, decision)
        operators.on_step(backend, obs.t)
        obs = backend.step(decision.cmd)
        positions.append(obs.pos.copy())
        velocities.append(obs.vel_body.copy())
        if obs.fallen:
            break
        if float(np.linalg.norm(obs.pos - goal_xy)) < goal_tol_m:
            goal_reached = True
            break

    return EpisodeResult(
        monitor_fired=len(events) > 0,
        events=events,
        fell=obs.fallen,
        goal_reached=goal_reached,
        final_dist_m=float(np.linalg.norm(obs.pos - goal_xy)),
        sim_time_s=obs.t,
        wall_time_s=time.perf_counter() - wall_start,
        n_steps=len(positions) - 1,
        shield_interventions=interventions,
        traj_hash=trajectory_hash(np.asarray(positions), np.asarray(velocities)),
    )
