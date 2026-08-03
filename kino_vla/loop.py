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

from kino_vla.monitor.event import Monitor, MonitorEvent
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


class Coupler(Protocol):
    """Online perception→safety coupler (M4): updates the shield's μ̂ each step.

    Called after the monitor and before the shield so the tightened friction
    constraint applies on the same step. Default ``None`` keeps the torch-free
    CI/demo path untouched (the shield runs on nominal μ).
    """

    def step(self, obs: Obs, monitor: Monitor) -> object: ...


class NavMap(Protocol):
    """Semantic traversability map (M5): persistent topological memory fed to the planner.

    Each step the map observes the visible scene (visual prior); on a monitor event the
    failure site is overwritten untraversable and propagated to visually-homogeneous
    neighbours, and the resulting avoid hazards are handed to the planner (spec §7).
    Default ``None`` keeps the demo path untouched.
    """

    def observe(self, pose_xy: np.ndarray, heading: float) -> int: ...

    def mark_failure(
        self, world_xy: np.ndarray, embedding: object = None, radius_m: float | None = None
    ) -> dict: ...

    def nav_hazards(self) -> list[tuple[np.ndarray, float]]: ...


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
    monitor: Monitor,
    policy: RecoveryPolicy,
    shield: Shield,
    *,
    seed: int,
    goal_xy: np.ndarray,
    goal_tol_m: float,
    max_time_s: float,
    on_step: Callable[[Obs, MonitorEvent | None, np.ndarray, ShieldDecision], None] | None = None,
    coupler: Coupler | None = None,
    nav_map: NavMap | None = None,
    perceive_every: int = 1,
    deep_reset: bool = False,
) -> EpisodeResult:
    """Run one seeded episode to goal, fall, or timeout.

    ``on_step`` (optional) is called once per control step after the shield decision
    and before the backend steps, with ``(obs_measured, event, cmd, decision)`` — a
    non-invasive hook for telemetry recording (scripts/record_demo.py). Default None
    keeps the demo/CI path untouched.

    ``perceive_every`` throttles the live semantic-map ``nav_map.observe`` (the RTX render +
    CLIP segmentation, the per-step cost) to every Nth step. Default 1 = unchanged (every step) so
    every existing caller/gate/demo is byte-identical; the DAgger collector sets it >1 to render
    the perception camera at ~2 Hz instead of 50 Hz (≈25x fewer renders) — quality-neutral, since
    the costmap only needs to be fresh at the ~1 Hz reflections and the VLA reads its own RGB via
    the recorder, while hazard MARKING is monitor-fire-driven (independent of observe)."""
    wall_start = time.perf_counter()
    goal_xy = np.asarray(goal_xy, dtype=np.float64)
    # A0.1 determinism: deep_reset scrubs the operator-ORDER PhysX residue so a reused-app closed
    # loop is order-independent (default False ⇒ the walking skeleton / all existing callers are
    # byte-identical; only the A0 determinism harness + A4/A6 collection opt in).
    if deep_reset and hasattr(backend, "deep_reset"):
        obs = backend.deep_reset(seed)
    else:
        obs = backend.reset(seed)
    operators.on_reset(backend)
    monitor.reset()
    if hasattr(shield, "reset"):
        shield.reset()  # per-episode isolation of CBF intervention/latency stats
    if coupler is not None and hasattr(coupler, "reset"):
        coupler.reset()  # per-episode window/μ̂ state

    events: list[MonitorEvent] = []
    interventions = 0
    positions: list[np.ndarray] = [obs.pos.copy()]
    velocities: list[np.ndarray] = [obs.vel_body.copy()]
    goal_reached = False
    max_steps = int(round(max_time_s / backend.dt))
    perceive_every = max(1, int(perceive_every))

    for _step in range(max_steps):
        obs_measured = operators.transform_obs(obs)
        # M5 semantic map: paint the visual prior from the current view (persistent in the
        # odometry frame, so a later turn-around cannot erase a marked region, spec §7).
        # Throttled by perceive_every (default 1 = every step ⇒ unchanged for all existing callers).
        if nav_map is not None and _step % perceive_every == 0:
            # Egocentric roll for long-distance / multi-patch courses (#47): keep the costmap window
            # on the robot BEFORE observing/marking. No-op unless the map is configured ``rolling``,
            # so fixed-grid scenarios + the pinned demo are byte-identical.
            if hasattr(nav_map, "recenter"):
                nav_map.recenter(obs_measured.pos)
            nav_map.observe(obs_measured.pos, obs_measured.heading)
        event = monitor.step(obs_measured)
        if event is not None:
            events.append(event)
            policy.on_event(event)
            # Physics writes the map: overwrite the failure site untraversable, propagate to
            # visually-homogeneous cells, and hand the resulting hazards to the planner.
            if nav_map is not None:
                nav_map.mark_failure(event.pos)
                if hasattr(policy, "adopt_map_hazards"):
                    policy.adopt_map_hazards(nav_map.nav_hazards())
        # M4 perception→safety coupling: update the shield's μ̂ from the proprioception
        # window (anomaly-gated) *before* the shield filters, so a detected ice patch
        # tightens the friction cone on this same step (spec §6.5).
        if coupler is not None:
            coupler.step(obs_measured, monitor)
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
                any(
                    c.startswith(("INFEASIBLE_FALLBACK", "HALT", "RECOVER_BRACE"))
                    for c in decision.codes
                )
            )
        # Forward the planner's commanded BODY POSTURE (Switch_Gait/Adjust_Posture/Set_Constraint)
        # to the backend's closed-loop height controller, so each posture/gait primitive produces a
        # real, distinct, dog-executed physical response — not just a speed cap (#41). getattr-
        # guarded so the FsmRecovery stub (no posture) and hook-less backends are unaffected.
        if hasattr(backend, "set_posture"):
            backend.set_posture(
                getattr(policy, "posture_height", None),
                float(getattr(policy, "posture_stiffness", 1.0)),
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
