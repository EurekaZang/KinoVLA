"""Fast CPU golden tests for the C2ST statistic (E1) — no Isaac, no torch.

Mirrors the truth-consistency-filter golden tests (QA 5.2): pin the statistic and the
lane-grouped bootstrap/permutation machinery on synthetic data with a known answer, so a
regression in the analysis is caught independently of the (GPU-only) real-stack collection.
All tests use the ``logreg`` discriminator (numpy, deterministic, torch-free).
"""

from __future__ import annotations

import numpy as np

from kino_vla.eval.c2st import (
    C2STResult,
    c2st,
    c2st_holdout,
    roc_auc,
    windows_from_lanes,
)


def _lane_windows(mean: float, n_lanes: int, steps: int, t: int, f: int, seed: int):
    """n_lanes synthetic rollouts ~ N(mean, 1); return (windows, lane_ids)."""
    rng = np.random.default_rng(seed)
    traces = [rng.normal(mean, 1.0, size=(steps, f)) for _ in range(n_lanes)]
    return windows_from_lanes(traces, t, stride=1)


def test_roc_auc_known_values() -> None:
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    # tied scores → chance
    assert roc_auc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_windows_from_lanes_shape() -> None:
    traces = [np.zeros((30, 4)), np.zeros((10, 4)), np.zeros((25, 4))]
    x, lanes = windows_from_lanes(traces, length=25, stride=1)
    # lane0: 30-25+1=6, lane1: too short → 0, lane2: 25-25+1=1  → 7 windows
    assert x.shape == (7, 25, 4)
    assert lanes.tolist() == [0] * 6 + [2]


def test_identical_distributions_are_indistinguishable() -> None:
    # Same generating distribution → discriminator must be at chance.
    xa, la = _lane_windows(0.0, n_lanes=12, steps=60, t=25, f=4, seed=1)
    xb, lb = _lane_windows(0.0, n_lanes=12, steps=60, t=25, f=4, seed=2)
    # explicit train/test split by lane (disjoint halves)
    res: C2STResult = c2st_holdout(
        xa, xb, la, lb, test_frac=0.5, classifier="logreg",
        n_boot=400, n_perm=300, seed=0, importance=True,
    )
    assert res.indistinguishable is True
    assert res.auc_ci[0] <= 0.5 <= res.auc_ci[1]
    assert res.perm_p >= res.alpha
    assert abs(res.auc - 0.5) < 0.15
    # importance keys present, one per channel
    assert set(res.feature_importance) == {f"ch{c}" for c in range(4)}


def test_shifted_distributions_are_distinguishable() -> None:
    # A clear mean shift → the discriminator separates them (the power check).
    xa, la = _lane_windows(0.0, n_lanes=12, steps=60, t=25, f=4, seed=3)
    xb, lb = _lane_windows(2.0, n_lanes=12, steps=60, t=25, f=4, seed=4)
    res = c2st_holdout(
        xa, xb, la, lb, test_frac=0.5, classifier="logreg",
        n_boot=400, n_perm=300, seed=0, importance=False,
    )
    assert res.indistinguishable is False
    assert res.auc > 0.95
    assert res.auc_ci[0] > 0.5
    assert res.perm_p < res.alpha


def test_lane_grouping_widens_ci_vs_naive() -> None:
    # Lane-grouped resampling must be no tighter than per-window (the false-PASS guard):
    # with correlated within-lane windows, treating each window as its own lane gives a
    # narrower CI. We assert the grouped CI is at least as wide.
    xa, la = _lane_windows(0.0, n_lanes=10, steps=80, t=25, f=3, seed=5)
    xb, lb = _lane_windows(0.4, n_lanes=10, steps=80, t=25, f=3, seed=6)
    grouped = c2st_holdout(
        xa, xb, la, lb, test_frac=0.5, classifier="logreg",
        n_boot=600, n_perm=50, seed=0, importance=False,
    )
    # per-window "lanes": every window its own group
    naive = c2st_holdout(
        xa, xb, np.arange(la.shape[0]), np.arange(lb.shape[0]) + 10_000,
        test_frac=0.5, classifier="logreg", n_boot=600, n_perm=50, seed=0, importance=False,
    )
    grouped_w = grouped.auc_ci[1] - grouped.auc_ci[0]
    naive_w = naive.auc_ci[1] - naive.auc_ci[0]
    assert grouped_w >= naive_w - 1e-9


def test_explicit_train_test_api() -> None:
    # The signature the check script uses: disjoint train/test arrays + test lane ids.
    xa_tr, _ = _lane_windows(0.0, n_lanes=8, steps=60, t=20, f=5, seed=7)
    xb_tr, _ = _lane_windows(0.0, n_lanes=8, steps=60, t=20, f=5, seed=8)
    xa_te, la = _lane_windows(0.0, n_lanes=6, steps=60, t=20, f=5, seed=9)
    xb_te, lb = _lane_windows(0.0, n_lanes=6, steps=60, t=20, f=5, seed=10)
    res = c2st(
        xa_tr, xb_tr, xa_te, xb_te, la, lb, classifier="logreg",
        n_boot=300, n_perm=200, seed=0, window_len=20, feature_set="synthetic",
    )
    assert res.indistinguishable is True
    assert res.window_len == 20
    assert res.feature_set == "synthetic"
    assert res.n_per_class[0] == xa_te.shape[0]
