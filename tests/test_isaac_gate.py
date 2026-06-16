"""Torch-free data-path tests for the M4 strict Isaac gate (kino_vla/tokens/isaac_gate.py).

The GPU collection + extractor training are exercised by the `pytest -m sim` gate; here
we lock down the pure-numpy plumbing the gate relies on: the npz round-trip, the support
trailing-average smoothing, and the disjoint train/eval split (no eval window may share a
step with any train window). Keeps the gate's correctness checkable on the CI machine.
"""

from __future__ import annotations

import numpy as np

from kino_vla.tokens.features import N_FEATURES, N_TARGETS, SUPPORT_INDEX
from kino_vla.tokens.isaac_gate import (
    CHANNEL_IDX,
    Rollout,
    build_split,
    filter_observable,
    load_rollouts,
    save_rollouts,
    smooth_support,
)
from kino_vla.utils.config import load_config


def _synthetic_rollouts(seed: int = 0) -> list[Rollout]:
    rng = np.random.default_rng(seed)
    rollouts: list[Rollout] = []
    for ch, level, regime in [
        ("mu", 0.10, 1),
        ("mu", 0.80, 0),
        ("support", 0.09, 4),
        ("effort", 0.35, 3),
        ("payload", 6.0, 2),
    ]:
        n = 200
        feats = rng.standard_normal((n, N_FEATURES))
        tgts = np.zeros((n, N_TARGETS))
        tgts[:, CHANNEL_IDX[ch]] = level
        # square-wave support so the smoother has something to flatten
        tgts[:, SUPPORT_INDEX] = 0.5 + 0.5 * ((np.arange(n) // 2) % 2)
        rollouts.append(Rollout(ch, level, regime, feats, tgts))
    return rollouts


def test_npz_roundtrip(tmp_path):
    rollouts = _synthetic_rollouts()
    path = tmp_path / "r.npz"
    save_rollouts(path, rollouts)
    back = load_rollouts(path)
    assert len(back) == len(rollouts)
    for a, b in zip(rollouts, back, strict=True):
        assert a.channel == b.channel
        assert a.level == b.level
        assert a.regime == b.regime
        assert np.allclose(a.feats, b.feats)
        assert np.allclose(a.tgts, b.tgts)


def test_smooth_support_flattens_gait_noise():
    tgts = np.zeros((40, N_TARGETS))
    tgts[:, SUPPORT_INDEX] = 0.5 + 0.5 * ((np.arange(40) // 2) % 2)  # 0.5/1.0 square wave
    raw_std = tgts[:, SUPPORT_INDEX].std()
    smoothed = smooth_support(tgts, win=10)
    # Smoothing must reduce the gait-phase variance and leave the mean ~unchanged.
    assert smoothed[:, SUPPORT_INDEX].std() < 0.5 * raw_std
    assert abs(smoothed[15:, SUPPORT_INDEX].mean() - 0.75) < 0.1
    # Other channels untouched.
    assert np.array_equal(smoothed[:, :SUPPORT_INDEX], tgts[:, :SUPPORT_INDEX])


def test_split_is_disjoint_and_per_channel():
    rollouts = _synthetic_rollouts()
    win_len, stride = 25, 3
    train, eval_by_ch = build_split(
        rollouts, win_len, stride, train_frac=0.7, gap=win_len, support_smooth=win_len
    )
    # Each excited channel has its own non-empty eval set; the model trains on the union.
    assert set(eval_by_ch) == set(CHANNEL_IDX)
    assert len(train) > 0
    # Per-channel eval targets only take that channel's drive levels (μ pools its 5 lanes),
    # so the MAE is computed on genuine variation, not a constant the model copies.
    levels: dict[str, set[float]] = {}
    for r in rollouts:
        levels.setdefault(r.channel, set()).add(round(r.level, 3))
    for ch, ds in eval_by_ch.items():
        assert len(ds) > 0
        if ch != "support":  # support target is the measured fraction, not the ridge height
            seen = {round(float(v), 3) for v in ds.targets[:, CHANNEL_IDX[ch]]}
            assert seen <= levels[ch], f"{ch}: eval levels {seen} not subset of {levels[ch]}"


def test_split_windows_share_no_step():
    """Train windows ⊂ feats[:n_tr] and eval windows ⊂ feats[n_tr+gap:] — disjoint steps."""
    n, win_len, stride, train_frac, gap = 220, 25, 3, 0.7, 25
    feats = np.arange(n * N_FEATURES, dtype=np.float64).reshape(n, N_FEATURES)
    tgts = np.zeros((n, N_TARGETS))
    roll = [Rollout("mu", 0.1, 1, feats, tgts)]
    train, eval_by_ch = build_split(
        roll, win_len, stride, train_frac=train_frac, gap=gap, support_smooth=win_len
    )
    n_tr = int(train_frac * n)
    # feats[*,0] encodes the global step index (col 0 = step*N_FEATURES).
    max_train_step = train.windows[..., 0].max() / N_FEATURES
    min_eval_step = eval_by_ch["mu"].windows[..., 0].min() / N_FEATURES
    assert max_train_step <= n_tr - 1
    assert min_eval_step >= n_tr + gap
    assert min_eval_step > max_train_step  # the core invariant: no shared timestep


def test_filter_observable_excludes_only_unobservable_bands():
    ig = load_config("tokens/extractor_v0.yaml").isaac_gate
    rolls = [
        Rollout("mu", 0.10, 1, np.zeros((1, N_FEATURES)), np.zeros((1, N_TARGETS))),  # deep ice
        Rollout("mu", 0.30, 1, np.zeros((1, N_FEATURES)), np.zeros((1, N_TARGETS))),  # knee → drop
        Rollout("mu", 0.80, 0, np.zeros((1, N_FEATURES)), np.zeros((1, N_TARGETS))),  # firm
        Rollout("effort", 1.0, 0, np.zeros((1, N_FEATURES)), np.zeros((1, N_TARGETS))),  # healthy
        Rollout("effort", 0.5, 3, np.zeros((1, N_FEATURES)), np.zeros((1, N_TARGETS))),  # mild→drop
        Rollout("effort", 0.2, 3, np.zeros((1, N_FEATURES)), np.zeros((1, N_TARGETS))),  # binds
        Rollout("payload", 8.0, 2, np.zeros((1, N_FEATURES)), np.zeros((1, N_TARGETS))),  # always
        Rollout("support", 0.09, 4, np.zeros((1, N_FEATURES)), np.zeros((1, N_TARGETS))),  # always
    ]
    kept, dropped = filter_observable(rolls, ig)
    assert {(r.channel, r.level) for r in dropped} == {("mu", 0.30), ("effort", 0.5)}
    assert len(kept) == 6
    assert {(r.channel, r.level) for r in kept} >= {("payload", 8.0), ("support", 0.09)}
