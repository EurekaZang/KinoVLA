"""Reflex layer — the fast self-stabilization stage of the 1 kHz Monitor+Reflex loop.

Spec §6.9 latency budget: anomaly -> Kino-Monitor (<5 ms) -> **Reflex self-stabilize
(<20 ms)** -> Kino-Token window (500 ms) -> VLA (~1 s) -> CBF -> primitive. The
Reflex's job is to *pin the state inside the safe set and buy time* (the quasi-static
hold time T_safe) until the slow VLA planner responds. It is not the recovery
planner — it only keeps the robot up.

At M2 the Reflex is a damping stand that simultaneously widens and lowers the stance
(``backend.set_reflex(True)``): standing kills the gait's self-generated demand, and a
wider/lower support polygon raises both the topple threshold and the capturable-speed
limit. ``run_survival_episode`` quantifies the resulting T_safe extension under a
repeated-push protocol (operator O6) — the spec §6.9 groundwork and the M2 gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kino_vla.monitor.rule_monitor import MonitorEvent
from kino_vla.sim.surrogate import SurrogateBackend
from kino_vla.sim.types import Obs
from kino_vla.utils.config import Config


class Reflex:
    """Damping/widen/lower protective stance, engaged on a monitor event for a hold window.

    Holds a reference to the backend so it can toggle the physical stance (the spec's
    stance-widen / CoM-lower actions); the command it issues is a damping stand
    (zero velocity). When disengaged it yields control by returning ``None`` so a
    downstream planner can drive.
    """

    def __init__(self, cfg: Config, backend: SurrogateBackend) -> None:
        self._cfg = cfg
        self._backend = backend
        self._hold_s = float(cfg.hold_s)
        self._engaged_until = float("-inf")
        self._engaged = False

    @property
    def engaged(self) -> bool:
        return self._engaged

    def on_event(self, event: MonitorEvent) -> bool:
        """Engage (or extend) the protective stance for ``hold_s`` after the event."""
        self._engaged_until = max(self._engaged_until, event.t + self._hold_s)
        return True

    def step(self, obs: Obs) -> np.ndarray | None:
        engage = obs.t < self._engaged_until
        if engage != self._engaged:
            self._backend.set_reflex(engage)
            self._engaged = engage
        return np.zeros(3) if engage else None


@dataclass(frozen=True)
class SurvivalResult:
    """Time-to-fall under the repeated-push survival protocol."""

    survived_s: float
    fell: bool
    pushes_applied: int


def run_survival_episode(
    sim_cfg: Config,
    reflex_cfg: Config,
    *,
    seed: int,
    use_reflex: bool,
) -> SurvivalResult:
    """Stand a robot under periodic O6 pushes; return how long it stays up.

    With ``use_reflex=True`` a persistent Reflex stance (engaged the whole episode)
    holds the robot; with ``use_reflex=False`` the robot damping-stands in its nominal
    stance with no widen/lower. Same seed, same push schedule, so the only difference
    is the Reflex — the T_safe comparison the M2 gate asserts (spec §6.9).
    """
    surv = reflex_cfg.survival
    backend = SurrogateBackend(sim_cfg, np.zeros(2), 0.0)
    obs = backend.reset(seed)
    if use_reflex:
        backend.set_reflex(True)

    period = float(surv.push_period_s)
    impulse = float(surv.impulse_ns)
    max_time = float(surv.max_time_s)
    direction = float(surv.push_dir_rad)
    push_vec = impulse * np.array([np.cos(direction), np.sin(direction)])

    pushes = 0
    next_push = period
    max_steps = int(round(max_time / backend.dt))
    for _ in range(max_steps):
        if obs.t >= next_push:
            backend.apply_push(push_vec, 0.0)
            pushes += 1
            next_push += period
        obs = backend.step(np.zeros(3))  # damping stand
        if obs.fallen:
            return SurvivalResult(survived_s=obs.t, fell=True, pushes_applied=pushes)
    return SurvivalResult(survived_s=obs.t, fell=False, pushes_applied=pushes)


@dataclass(frozen=True)
class PushRecoveryResult:
    """Largest single impulse the standing robot recovers (operator O6 protocol)."""

    max_recovered_ns: float
    first_fall_ns: float | None


def run_push_recovery_sweep(
    sim_cfg: Config,
    reflex_cfg: Config,
    *,
    seed: int,
    use_reflex: bool,
) -> PushRecoveryResult:
    """Sweep single-push impulse magnitude; report the largest the robot recovers.

    Classic push-recovery (spec O6 [A]): one impulse to a standing robot, settle,
    score recovered if it never falls. ``use_reflex`` engages the protective stance.
    """
    sweep = reflex_cfg.push_recovery
    impulses = np.arange(float(sweep.start_ns), float(sweep.stop_ns) + 1e-9, float(sweep.step_ns))
    direction = float(sweep.push_dir_rad)
    settle_steps = int(round(float(sweep.settle_s) / (1.0 / float(sim_cfg.control_hz))))
    max_recovered = 0.0
    first_fall: float | None = None
    for impulse in impulses:
        backend = SurrogateBackend(sim_cfg, np.zeros(2), 0.0)
        backend.reset(seed)
        if use_reflex:
            backend.set_reflex(True)
        push_vec = float(impulse) * np.array([np.cos(direction), np.sin(direction)])
        backend.apply_push(push_vec, 0.0)
        obs = None
        for _ in range(settle_steps):
            obs = backend.step(np.zeros(3))
            if obs.fallen:
                break
        if obs is not None and obs.fallen:
            if first_fall is None:
                first_fall = float(impulse)
        else:
            max_recovered = float(impulse)
    return PushRecoveryResult(max_recovered_ns=max_recovered, first_fall_ns=first_fall)
