"""Small, dependency-free statistics for preregistered operator dose sweeps."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    """Deterministic percentile interval for a mean over independent episode seeds."""

    mean: float
    lower: float
    upper: float
    confidence: float
    n: int
    resamples: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "n": self.n,
            "resamples": self.resamples,
        }


def bootstrap_mean_interval(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapInterval:
    """Return a reproducible non-parametric percentile interval for the sample mean."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("values must be a non-empty finite one-dimensional sample")
    if not 0.0 < confidence < 1.0 or resamples <= 0:
        raise ValueError("confidence must be in (0, 1) and resamples must be positive")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(resamples, len(array)))
    means = array[indices].mean(axis=1)
    tail = 0.5 * (1.0 - confidence)
    return BootstrapInterval(
        mean=float(array.mean()),
        lower=float(np.quantile(means, tail)),
        upper=float(np.quantile(means, 1.0 - tail)),
        confidence=float(confidence),
        n=int(len(array)),
        resamples=int(resamples),
    )


def paired_dose_monotonicity(
    records: Sequence[Mapping[str, object]],
    *,
    dose_order: Sequence[str],
    metric: str,
    direction: str = "increasing",
    tolerance: float = 0.0,
) -> dict[str, object]:
    """Audit ordered doses both after aggregation and within every paired seed.

    Each ``(seed, dose)`` cell must occur exactly once. ``tolerance`` permits bounded numerical
    jitter: increasing order accepts ``next >= previous - tolerance`` and decreasing order uses
    the symmetric rule. Strict ordering is also reported and is never relaxed by the tolerance.
    """
    if direction not in {"increasing", "decreasing"}:
        raise ValueError("direction must be 'increasing' or 'decreasing'")
    if len(dose_order) < 2 or len(set(dose_order)) != len(dose_order):
        raise ValueError("dose_order must contain at least two unique labels")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    table: dict[tuple[int, str], float] = {}
    seeds: set[int] = set()
    for record in records:
        seed = int(record["seed"])
        dose = str(record["dose"])
        value = float(record[metric])
        if dose not in dose_order or not np.isfinite(value):
            raise ValueError("records contain an unknown dose or non-finite metric")
        key = (seed, dose)
        if key in table:
            raise ValueError(f"duplicate paired cell {key}")
        table[key] = value
        seeds.add(seed)
    expected = {(seed, dose) for seed in seeds for dose in dose_order}
    if set(table) != expected:
        missing = sorted(expected - set(table))
        raise ValueError(f"incomplete paired dose table; missing={missing}")

    sign = 1.0 if direction == "increasing" else -1.0
    medians = {
        dose: float(np.median([table[(seed, dose)] for seed in sorted(seeds)]))
        for dose in dose_order
    }

    def _ordered(values: Sequence[float], *, strict: bool) -> bool:
        signed = sign * np.diff(np.asarray(values, dtype=np.float64))
        return bool(np.all(signed > 0.0)) if strict else bool(np.all(signed >= -tolerance))

    per_seed = {}
    for seed in sorted(seeds):
        values = [table[(seed, dose)] for dose in dose_order]
        per_seed[str(seed)] = {
            "values": {dose: value for dose, value in zip(dose_order, values, strict=True)},
            "ordered_with_tolerance": _ordered(values, strict=False),
            "strictly_ordered": _ordered(values, strict=True),
        }
    median_values = [medians[dose] for dose in dose_order]
    return {
        "metric": metric,
        "direction": direction,
        "tolerance": float(tolerance),
        "dose_order": list(dose_order),
        "median_by_dose": medians,
        "median_ordered_with_tolerance": _ordered(median_values, strict=False),
        "median_strictly_ordered": _ordered(median_values, strict=True),
        "paired_seed_order_fraction": float(
            np.mean([row["ordered_with_tolerance"] for row in per_seed.values()])
        ),
        "paired_seed_strict_fraction": float(
            np.mean([row["strictly_ordered"] for row in per_seed.values()])
        ),
        "per_seed": per_seed,
    }
