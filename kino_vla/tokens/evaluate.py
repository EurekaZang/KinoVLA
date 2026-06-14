"""M4 exit-criterion metrics for the Kino-Tokens extractor (spec §4, §9).

Shared by ``scripts/train_extractor.py`` and ``tests/test_extractor.py`` so the
reported numbers and the asserted gates are computed by identical code (QA 5.2:
learned-component metrics logged to a tracked file; a checkpoint is "done" only when
its eval is reproduced). Imports torch transitively via the extractor — call sites
are torch-gated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kino_vla.tokens.dataset import TokenDataset, ood_sweep
from kino_vla.tokens.extractor import Extractor
from kino_vla.tokens.features import TARGET_SCHEMA
from kino_vla.utils.config import Config


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared), like scipy.stats.rankdata — pure numpy."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # Average tied groups.
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.mean(np.arange(i + 1, j + 2))
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (Pearson on ranks)."""
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt(np.sum(ra**2) * np.sum(rb**2)))
    return 0.0 if denom == 0.0 else float(np.sum(ra * rb) / denom)


def regression_mae(extractor: Extractor, eval_ds: TokenDataset) -> dict[str, float]:
    """Per-target held-out mean absolute error in physical units (spec §4 main gate)."""
    theta_hat = extractor.predict(eval_ds.windows).theta_hat
    mae = np.mean(np.abs(theta_hat - eval_ds.targets), axis=0)
    return {name: float(mae[i]) for i, name in enumerate(TARGET_SCHEMA)}


@dataclass(frozen=True)
class OodResult:
    """OOD-monotonicity outcome over the parameter-extrapolation sweep (spec §9)."""

    values: list[float]
    ood_curve: list[float]
    spearman: float
    separation_ratio: float


def ood_monotonicity(extractor: Extractor, cfg: Config) -> OodResult:
    """Check the reconstruction-residual OOD score rises monotonically as a parameter
    extrapolates beyond its training range, and separates from the in-distribution
    baseline (spec §9 / Suite-OOD)."""
    oc = cfg.ood
    param = str(oc.sweep_param)
    regime = str(oc.sweep_regime)
    seeds = list(range(int(oc.n_seeds)))
    sweep = ood_sweep(cfg, list(oc.sweep_values), seeds, regime=regime, param_key=param)
    values = [v for v, _ in sweep]
    curve = [float(np.mean(extractor.predict(w).ood_score)) for _, w in sweep]

    in_dist = ood_sweep(cfg, list(oc.in_dist_values), seeds, regime=regime, param_key=param)
    in_scores = [float(np.mean(extractor.predict(w).ood_score)) for _, w in in_dist]

    # More extreme = parameter further below training min ⇒ use −value as severity.
    rho = spearman(-np.asarray(values), np.asarray(curve))
    sep = float(np.mean(curve) / max(np.mean(in_scores), 1e-12))
    return OodResult(values=values, ood_curve=curve, spearman=rho, separation_ratio=sep)
