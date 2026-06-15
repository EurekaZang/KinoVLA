"""Privileged-distillation dataset builder (spec §4, M4 scope).

Generates the teacher-student training pairs for the Kino-Tokens extractor by driving
the surrogate Go2 *straight across* parameterized failure operators (no FSM detour, no
shield — we want the robot to actually experience the degradation) and logging, per
step, the measured proprioception window and the privileged physics truth
(``backend.privileged_physics()``). Single-operator episodes only — Suite-Comp
composition is test-only (spec §8.3) — so the regression supervision stays clean.

Outputs are deterministic given the seed list. Train/held-out are split by *disjoint
seed ranges* so no window leaks across the boundary. ``ood_sweep`` builds
parameter-extrapolation episodes (θ outside the training range) for the spec §9
OOD-monotonicity check.

Pure numpy + the surrogate backend (no torch).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kino_vla.sim.operators import (
    Collapse,
    EffortDecay,
    HighCentering,
    MuField,
    OperatorStack,
    Payload,
)
from kino_vla.sim.operators.base import FailureOperator
from kino_vla.sim.surrogate import SurrogateBackend
from kino_vla.tokens.features import (
    N_FEATURES,
    N_TARGETS,
    obs_to_features,
    physics_to_target,
)
from kino_vla.tokens.semantics import regime_of
from kino_vla.tokens.window import slice_rollout_to_windows, window_length
from kino_vla.utils.config import Config, load_config
from kino_vla.utils.geometry import Rect
from kino_vla.utils.seeding import rng

# Training regimes (single operator each). "normal" drives clean ground; the rest
# sample one operator's θ from its in-distribution range.
TRAIN_REGIMES: tuple[str, ...] = (
    "normal",
    "low_friction",
    "overload",
    "actuator_decay",
    "high_centered",
    "thin_ice",  # O3 collapse — friction drops mid-crossing (time-varying μ target)
)


@dataclass(frozen=True)
class TokenDataset:
    """Windowed distillation pairs: ``(N, T, F)`` inputs, ``(N, n_targets)`` physics
    targets, ``(N,)`` regime ids."""

    windows: np.ndarray
    targets: np.ndarray
    regimes: np.ndarray

    def __len__(self) -> int:
        return int(self.windows.shape[0])


def _path_region(cfg: Config) -> Rect:
    """A hazard rectangle straddling the robot's straight +x path across the field."""
    return Rect(
        cx=float(cfg.drive.hazard_cx),
        cy=0.0,
        hx=float(cfg.drive.hazard_hx),
        hy=float(cfg.drive.hazard_hy),
    )


def _build_operators(regime: str, params: dict[str, float], cfg: Config) -> list[FailureOperator]:
    """Instantiate the single operator (if any) realizing ``regime`` with ``params``."""
    if regime == "normal":
        return []
    if regime == "low_friction":
        mu = params["mu"]
        return [MuField(region=_path_region(cfg), mu_s=mu * 1.2, mu_d=mu)]
    if regime == "overload":
        return [Payload(mass_kg=params["payload_kg"], com_offset_m=(params["com_off"], 0.0))]
    if regime == "actuator_decay":
        return [
            EffortDecay(
                decay_rate_per_s=params["decay_rate"],
                floor=params["effort_scale"],
                t_start_s=float(cfg.drive.decay_t_start_s),
            )
        ]
    if regime == "high_centered":
        return [HighCentering(region=_path_region(cfg), residual_support=params["support_ratio"])]
    if regime == "thin_ice":
        return [
            Collapse(
                region=_path_region(cfg),
                mu_collapsed=params["mu"],
                trigger_dwell_s=float(cfg.drive.collapse_dwell_s),
                mu_intact=float(cfg.drive.collapse_mu_intact),
            )
        ]
    raise ValueError(f"unknown regime {regime!r}")


def _sample_params(regime: str, r: np.random.Generator, ranges: Config) -> dict[str, float]:
    """Sample an operator's θ uniformly from its (in-distribution) training range."""

    def u(key: str) -> float:
        lo, hi = ranges.get(key)
        return float(r.uniform(lo, hi))

    if regime in ("low_friction", "thin_ice"):
        return {"mu": u("mu")}
    if regime == "overload":
        return {"payload_kg": u("payload_kg"), "com_off": u("com_off")}
    if regime == "actuator_decay":
        return {"decay_rate": u("decay_rate"), "effort_scale": u("effort_scale")}
    if regime == "high_centered":
        return {"support_ratio": u("support_ratio")}
    return {}


def run_scripted_rollout(
    backend: SurrogateBackend,
    operators: OperatorStack,
    speed_lo: float,
    speed_hi: float,
    period_s: float,
    duty: float,
    n_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Drive straight +x for ``n_steps`` logging (features, privileged target) per step.

    The forward command is a bang-bang square wave — ``speed_hi`` for the first ``duty``
    fraction of each ``period_s``, then ``speed_lo`` — rather than a constant cruise. This
    is a deliberate system-identification excitation: payload mass (O5) and effort-decay
    (O10) leave a proprioceptive trace *only* when the effort budget binds (``demand >
    effort_budget``), and demand peaks during the sharp high→low deceleration (the gait
    term ``g·v`` is large while the tracking error ``|cmd−v|/τ`` is also large). A flat or
    gently-ramped driver never reaches that demand, so O5/O10 stay unidentifiable and the
    θ-regression head can do no better than the prior mean (the M4 v0 failure mode). The
    same sharp speed changes give a richer slip signal on ice (better μ inversion). Small
    seeded lateral/yaw jitter adds window diversity. Deliberately *not* the FSM, so the
    robot crosses the hazard instead of detouring and the degradation is actually felt.

    Logging stops at the first fallen step so frozen post-fall windows never enter the
    dataset (they would mislabel a static obs with the operator's θ).
    """
    obs = backend.reset(seed)
    operators.on_reset(backend)
    r = rng(seed + 100_000)
    phase0 = float(r.uniform(0.0, 1.0))
    dt = float(backend.dt)
    feats = np.empty((n_steps, N_FEATURES), dtype=np.float64)
    tgts = np.empty((n_steps, N_TARGETS), dtype=np.float64)
    n_valid = 0
    for k in range(n_steps):
        if obs.fallen:
            break
        feats[n_valid] = obs_to_features(obs)
        tgts[n_valid] = physics_to_target(backend.privileged_physics())
        n_valid += 1
        frac = (phase0 + (k * dt) / period_s) % 1.0
        speed = speed_hi if frac < duty else speed_lo  # bang-bang square wave
        cmd = np.array(
            [speed, 0.05 * r.standard_normal(), 0.05 * r.standard_normal()],
            dtype=np.float64,
        )
        operators.on_step(backend, obs.t)
        obs = backend.step(cmd)
    return feats[:n_valid], tgts[:n_valid]


def _episode_windows(
    regime: str,
    params: dict[str, float],
    seed: int,
    cfg: Config,
    win_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One scripted episode → (windows, targets)."""
    backend = SurrogateBackend(
        load_config("sim/surrogate.yaml"),
        np.zeros(2),
        0.0,
    )
    ops = OperatorStack(_build_operators(regime, params, cfg))
    feats, tgts = run_scripted_rollout(
        backend,
        ops,
        speed_lo=float(cfg.drive.speed_lo),
        speed_hi=float(cfg.drive.speed_hi),
        period_s=float(cfg.drive.speed_period_s),
        duty=float(cfg.drive.speed_duty),
        n_steps=int(cfg.drive.n_steps),
        seed=seed,
    )
    return slice_rollout_to_windows(feats, tgts, win_len, stride=int(cfg.window.stride))


def build_dataset(cfg: Config, seeds: list[int]) -> TokenDataset:
    """Build a windowed dataset over ``seeds`` (one regime per seed, round-robin)."""
    win_len = window_length(float(cfg.window.window_ms), float(cfg.window.control_hz))
    thr = cfg.regime_thresholds
    all_w: list[np.ndarray] = []
    all_t: list[np.ndarray] = []
    all_r: list[np.ndarray] = []
    for i, seed in enumerate(seeds):
        regime = TRAIN_REGIMES[i % len(TRAIN_REGIMES)]
        params = _sample_params(regime, rng(seed + 7), cfg.param_ranges)
        wins, tgts = _episode_windows(regime, params, seed, cfg, win_len)
        if wins.shape[0] == 0:
            continue
        labels = np.array(
            [
                regime_of(
                    t,
                    mu_nominal=float(thr.mu_nominal),
                    mu_low=float(thr.mu_low),
                    payload_min_kg=float(thr.payload_min_kg),
                    effort_low=float(thr.effort_low),
                    support_low=float(thr.support_low),
                )
                for t in tgts
            ],
            dtype=np.int64,
        )
        all_w.append(wins)
        all_t.append(tgts)
        all_r.append(labels)
    return TokenDataset(
        windows=np.concatenate(all_w, axis=0),
        targets=np.concatenate(all_t, axis=0),
        regimes=np.concatenate(all_r, axis=0),
    )


def ood_sweep(
    cfg: Config,
    values: list[float],
    seeds: list[int],
    regime: str = "low_friction",
    param_key: str = "mu",
) -> list[tuple[float, np.ndarray]]:
    """Build a parameter-extrapolation sweep (spec §9): for each value of ``param_key``
    (typically *outside* the training range), return ``(value, windows)`` averaged
    over ``seeds``. Used to check the OOD residual rises monotonically off-manifold.
    """
    win_len = window_length(float(cfg.window.window_ms), float(cfg.window.control_hz))
    out: list[tuple[float, np.ndarray]] = []
    for value in values:
        wins_for_value: list[np.ndarray] = []
        for seed in seeds:
            params = _sample_params(regime, rng(seed + 7), cfg.param_ranges)
            params[param_key] = float(value)
            wins, _ = _episode_windows(regime, params, seed, cfg, win_len)
            if wins.shape[0] > 0:
                wins_for_value.append(wins)
        if wins_for_value:
            out.append((float(value), np.concatenate(wins_for_value, axis=0)))
    return out
