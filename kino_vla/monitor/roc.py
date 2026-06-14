"""Kino-Monitor ROC — detection performance over labeled rollouts (spec §12 metric).

Runs a forward-driving robot across the surrogate under two conditions — nominal
(no operator) and a mix of failure operators (O1 ice, O3 collapse, O8 wall, O9
high-centering, O10 effort-decay) — and records each episode's peak threshold-
normalized monitor anomaly score (``RuleMonitor.anomaly_score``). Sweeping a decision
threshold over those per-episode scores yields the ROC: TPR (failures flagged) vs FPR
(nominal false alarms), summarized by AUC. This is the M2 exit-criterion plot;
``scripts/monitor_roc.py`` renders it and writes the metrics to a tracked file.

The four monitor channels make the failure set separable from nominal, so a good
monitor scores AUC ≈ 1; a regression that desensitizes a channel drops it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from kino_vla.monitor.rule_monitor import RuleMonitor
from kino_vla.sim.operators import (
    Collapse,
    EffortDecay,
    FailureOperator,
    HighCentering,
    InvisibleCollider,
    MuField,
    OperatorStack,
)
from kino_vla.sim.surrogate import SurrogateBackend
from kino_vla.utils.config import Config, load_config
from kino_vla.utils.geometry import Rect

# Failure operators sampled for the positive class. Each is placed on the robot's
# straight path (a band around x ∈ [2.5, 3.5]); the lambda takes a per-episode rng
# for parameter jitter so the ROC reflects a θ-distribution, not one point.
_HAZARD = Rect(cx=3.0, cy=0.0, hx=1.0, hy=1.0)


def _failure_factories() -> list[tuple[str, Callable[[np.random.Generator], FailureOperator]]]:
    return [
        ("O1_ice", lambda r: MuField(region=_HAZARD, mu_s=0.12, mu_d=float(r.uniform(0.06, 0.16)))),
        (
            "O3_thin_ice",
            lambda r: Collapse(
                region=_HAZARD, mu_collapsed=float(r.uniform(0.05, 0.12)), trigger_dwell_s=0.3
            ),
        ),
        ("O8_glass", lambda r: InvisibleCollider(region=Rect(cx=3.0, cy=0.0, hx=0.3, hy=2.0))),
        (
            "O9_ridge",
            lambda r: HighCentering(region=_HAZARD, residual_support=float(r.uniform(0.08, 0.2))),
        ),
        (
            "O10_decay",
            lambda r: EffortDecay(
                decay_rate_per_s=float(r.uniform(1.0, 1.8)),
                floor=float(r.uniform(0.12, 0.18)),
                t_start_s=0.5,
            ),
        ),
    ]


@dataclass(frozen=True)
class RocResult:
    """ROC over labeled episodes: per-episode (score, label) plus the swept curve."""

    scores: np.ndarray  # (N,) peak anomaly score per episode
    labels: np.ndarray  # (N,) 1 = failure present, 0 = nominal
    fpr: np.ndarray  # (T,) false-positive rate at each swept threshold
    tpr: np.ndarray  # (T,) true-positive rate
    thresholds: np.ndarray  # (T,)
    auc: float


def _episode_peak_score(
    sim_cfg: Config, monitor_cfg: Config, stack: OperatorStack, seed: int
) -> float:
    """Drive a forward command across the hazard band; return the peak anomaly score."""
    backend = SurrogateBackend(sim_cfg, np.array([0.0, 0.0]), 0.0)
    monitor = RuleMonitor(monitor_cfg, dt=backend.dt)
    obs = backend.reset(seed)
    stack.on_reset(backend)
    monitor.reset()
    peak = 0.0
    arm_delay = float(monitor_cfg.arm_delay_s)
    n_steps = int(round(6.0 / backend.dt))
    for _ in range(n_steps):
        obs_m = stack.transform_obs(obs)
        monitor.step(obs_m)
        # Score only past the arm window: the monitor ignores the startup acceleration
        # transient for firing, so the ROC must too (else nominal episodes false-alarm).
        if obs.t >= arm_delay:
            peak = max(peak, monitor.anomaly_score)
        stack.on_step(backend, obs.t)
        obs = backend.step(np.array([0.7, 0.0, 0.0]))
        if obs.pos[0] > 5.5:
            break
    return peak


def roc_curve(scores: np.ndarray, labels: np.ndarray, n_thresholds: int = 200) -> RocResult:
    """Sweep a decision threshold over ``scores``; return FPR/TPR/AUC (trapezoidal)."""
    lo, hi = float(scores.min()), float(scores.max())
    pad = 0.05 * (hi - lo + 1e-9)
    thresholds = np.linspace(hi + pad, lo - pad, n_thresholds)
    pos = labels == 1
    neg = labels == 0
    n_pos = max(1, int(pos.sum()))
    n_neg = max(1, int(neg.sum()))
    tpr = np.array([float((scores[pos] > t).sum()) / n_pos for t in thresholds])
    fpr = np.array([float((scores[neg] > t).sum()) / n_neg for t in thresholds])
    auc = float(np.trapz(tpr, fpr))
    return RocResult(scores, labels, fpr, tpr, thresholds, auc)


def compute_monitor_roc(
    n_per_class: int = 40,
    sim_cfg: Config | None = None,
    monitor_cfg: Config | None = None,
    base_seed: int = 1000,
) -> RocResult:
    """Run ``n_per_class`` nominal + failure episodes and return the monitor ROC."""
    sim_cfg = sim_cfg or load_config("sim/surrogate.yaml")
    monitor_cfg = monitor_cfg or load_config("monitor/rule_v0.yaml")
    factories = _failure_factories()
    scores: list[float] = []
    labels: list[int] = []
    for i in range(n_per_class):
        # Nominal: empty operator stack on good ground.
        scores.append(_episode_peak_score(sim_cfg, monitor_cfg, OperatorStack([]), base_seed + i))
        labels.append(0)
        # Failure: rotate through the operator set with per-episode θ jitter.
        name, factory = factories[i % len(factories)]
        rng = np.random.default_rng(base_seed + 7919 * (i + 1))
        stack = OperatorStack([factory(rng)])
        scores.append(_episode_peak_score(sim_cfg, monitor_cfg, stack, base_seed + 13 * (i + 1)))
        labels.append(1)
    return roc_curve(np.asarray(scores), np.asarray(labels, dtype=int))
