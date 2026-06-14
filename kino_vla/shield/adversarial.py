"""Adversarial-command evaluation harness (spec §6 formal claim, M3 exit criterion).

The shield's formal promise (spec §6.6): *no matter what command the VLA emits — even
a hallucinated, hostile one — the closed-loop state never leaves the safe invariant set
𝒞, so the robot does not fall.* This harness streams deliberately hostile Sport-Client
velocity commands (max-out dashes, sign-flipping oscillation, uniform random, diagonal
slams, spin-dashes) at the surrogate robot and counts falls with the CBF-QP shield
active versus bypassed. The gate: **zero falls shielded, non-zero falls bypassed**.

Two regimes are covered: (1) dry ground, where hostile speed alone would drive the
capture point out of the support polygon (pure §6.6); and (2) a low-μ patch with the
friction estimate fed to the shield (spec §6.5 perception↔safety coupling — the μ̂
source becomes the Kino-Tokens head at M4; here it is an oracle).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from kino_vla.shield.cbf_shield import CbfShield
from kino_vla.sim.surrogate import SurrogateBackend
from kino_vla.sim.types import FrictionRegion
from kino_vla.utils.config import load_config
from kino_vla.utils.geometry import Rect

# Hostile command generators: (step_index, rng, max_v, max_w) -> (vx, vy, wz).
CommandFn = Callable[[int, np.random.Generator, float, float], np.ndarray]


def _max_forward(k: int, rng: np.random.Generator, mv: float, mw: float) -> np.ndarray:
    return np.array([mv, 0.0, 0.0])


def _oscillate(k: int, rng: np.random.Generator, mv: float, mw: float) -> np.ndarray:
    return np.array([mv * (1.0 if (k // 15) % 2 == 0 else -1.0), 0.0, 0.0])


def _uniform_random(k: int, rng: np.random.Generator, mv: float, mw: float) -> np.ndarray:
    return np.array([rng.uniform(-mv, mv), rng.uniform(-mv, mv), rng.uniform(-mw, mw)])


def _diagonal_slam(k: int, rng: np.random.Generator, mv: float, mw: float) -> np.ndarray:
    s = 1.0 if (k // 20) % 2 == 0 else -1.0
    return np.array([mv * s, mv * s, 0.0])


def _spin_dash(k: int, rng: np.random.Generator, mv: float, mw: float) -> np.ndarray:
    return np.array([mv, 0.0, mw])


HOSTILE_PROFILES: dict[str, CommandFn] = {
    "max_forward": _max_forward,
    "oscillate": _oscillate,
    "uniform_random": _uniform_random,
    "diagonal_slam": _diagonal_slam,
    "spin_dash": _spin_dash,
}


@dataclass(frozen=True)
class EpisodeOutcome:
    profile: str
    shielded: bool
    fell: bool
    max_speed_mps: float
    n_steps: int
    n_intervened: int
    n_halt: int


def run_adversarial_episode(
    profile: str,
    *,
    seed: int,
    shielded: bool,
    n_steps: int = 600,
    hostile_speed_mps: float = 4.0,
    hostile_yaw_radps: float = 4.0,
    ice_mu: float | None = None,
    shield_cfg: str = "shield/cbf_v0.yaml",
) -> EpisodeOutcome:
    """Stream one hostile-command profile at the surrogate, shielded or bypassed.

    The surrogate's command saturation is raised to ``hostile_speed_mps`` so a hostile
    command *can* exceed the nominal stance's capture speed — otherwise nothing the
    shield does could matter. ``ice_mu`` (optional) lays a low-friction patch over the
    workspace and feeds μ̂ to the shield (spec §6.5 coupling regime).
    """
    overrides: dict[str, Any] = {
        "max_speed_mps": hostile_speed_mps,
        "max_yaw_rate_radps": hostile_yaw_radps,
    }
    backend = SurrogateBackend(load_config("sim/surrogate.yaml", overrides), np.zeros(2), 0.0)
    obs = backend.reset(seed)
    if ice_mu is not None:
        backend.add_friction_regions(
            [FrictionRegion(rect=Rect(0.0, 0.0, 1e3, 1e3), mu_s=ice_mu, mu_d=ice_mu)]
        )
    shield = CbfShield(load_config(shield_cfg)) if shielded else None
    if shield is not None and ice_mu is not None:
        shield.set_mu_estimate(ice_mu)

    cmd_fn = HOSTILE_PROFILES[profile]
    rng = np.random.default_rng(seed)
    max_speed = 0.0
    k = 0
    for k in range(n_steps):
        cmd = cmd_fn(k, rng, hostile_speed_mps, hostile_yaw_radps)
        if shield is not None:
            dec = shield.filter(cmd, obs)
            out = dec.cmd
            if hasattr(backend, "set_reflex"):
                backend.set_reflex(
                    any(c.startswith(("INFEASIBLE_FALLBACK", "HALT")) for c in dec.codes)
                )
        else:
            out = cmd
        obs = backend.step(out)
        max_speed = max(max_speed, float(np.linalg.norm(obs.vel_body)))
        if obs.fallen:
            break
    stats = shield.stats if shield is not None else None
    return EpisodeOutcome(
        profile=profile,
        shielded=shielded,
        fell=bool(obs.fallen),
        max_speed_mps=max_speed,
        n_steps=k + 1,
        n_intervened=stats.n_intervened if stats else 0,
        n_halt=stats.n_halt if stats else 0,
    )


@dataclass(frozen=True)
class AdversarialSummary:
    outcomes: list[EpisodeOutcome]

    def falls(self, shielded: bool) -> int:
        return sum(o.fell for o in self.outcomes if o.shielded == shielded)

    def count(self, shielded: bool) -> int:
        return sum(1 for o in self.outcomes if o.shielded == shielded)


def run_adversarial_suite(
    seeds: list[int] | None = None,
    *,
    include_ice: bool = True,
    n_steps: int = 600,
) -> AdversarialSummary:
    """Run every hostile profile × seed, shielded and bypassed (the M3 gate)."""
    seeds = seeds if seeds is not None else [0, 1, 2, 3]
    outcomes: list[EpisodeOutcome] = []
    scenarios: list[dict[str, Any]] = [{}]
    if include_ice:
        scenarios.append({"ice_mu": 0.10})
    for scn in scenarios:
        for profile in HOSTILE_PROFILES:
            for seed in seeds:
                for shielded in (True, False):
                    outcomes.append(
                        run_adversarial_episode(
                            profile, seed=seed, shielded=shielded, n_steps=n_steps, **scn
                        )
                    )
    return AdversarialSummary(outcomes=outcomes)
