from __future__ import annotations

import pytest

from kino_vla.eval.dose_response import bootstrap_mean_interval, paired_dose_monotonicity


def test_bootstrap_mean_interval_is_deterministic_and_contains_mean():
    first = bootstrap_mean_interval([1.0, 2.0, 4.0, 8.0], resamples=2_000, seed=7)
    second = bootstrap_mean_interval([1.0, 2.0, 4.0, 8.0], resamples=2_000, seed=7)
    assert first == second
    assert first.lower <= first.mean <= first.upper
    assert first.n == 4


def test_paired_dose_monotonicity_checks_each_seed_not_only_median():
    records = [
        {"seed": 1, "dose": "nominal", "effect": 0.0},
        {"seed": 1, "dose": "mild", "effect": 1.0},
        {"seed": 1, "dose": "severe", "effect": 2.0},
        {"seed": 2, "dose": "nominal", "effect": 0.0},
        {"seed": 2, "dose": "mild", "effect": 2.0},
        {"seed": 2, "dose": "severe", "effect": 1.5},
    ]
    summary = paired_dose_monotonicity(
        records,
        dose_order=["nominal", "mild", "severe"],
        metric="effect",
    )
    assert summary["median_strictly_ordered"]
    assert summary["paired_seed_strict_fraction"] == 0.5


def test_paired_dose_monotonicity_rejects_incomplete_cells():
    with pytest.raises(ValueError, match="incomplete paired dose table"):
        paired_dose_monotonicity(
            [
                {"seed": 1, "dose": "nominal", "effect": 0.0},
                {"seed": 1, "dose": "mild", "effect": 1.0},
                {"seed": 2, "dose": "nominal", "effect": 0.0},
            ],
            dose_order=["nominal", "mild"],
            metric="effect",
        )
