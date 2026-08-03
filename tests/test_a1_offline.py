"""A1 offline-logic guards (fast CPU; no Isaac).

Locks in the two load-bearing properties of the A1.1 determinism-controlled certificate:
  * the C2ST VALIDITY property — identical window sets are indistinguishable (AUC≈0.5), a shifted
    set is distinguishable (AUC≈1.0). This is exactly the fresh-app certificate's logic (matched
    O4↔O2 byte-identical → 0.5; power O2↔O1 different → 1.0) and the same-operator validity gate.
  * the byte-identity metric + the Wilson CI helper used across A1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _lanes(base: np.ndarray, n: int, jitter: float, rng) -> list:
    """n per-lane (T,F) traces = base + small per-lane gaussian jitter (a within-class spread)."""
    return [base + jitter * rng.standard_normal(base.shape) for _ in range(n)]


def test_c2st_validity_identical_vs_shifted() -> None:
    from kino_vla.eval.c2st import c2st, windows_from_lanes

    rng = np.random.default_rng(0)
    base = np.cumsum(rng.standard_normal((300, 6)), axis=0)  # a temporally-correlated trace
    # SAME distribution (the matched pair / same-op control): two jittered copies of one base.
    a_tr = _lanes(base, 6, 0.05, rng)
    b_tr = _lanes(base, 6, 0.05, rng)
    a_te = _lanes(base, 4, 0.05, rng)
    b_te = _lanes(base, 4, 0.05, rng)
    xa, _ = windows_from_lanes(a_tr, 50)
    xb, _ = windows_from_lanes(b_tr, 50)
    xta, la = windows_from_lanes(a_te, 50)
    xtb, lb = windows_from_lanes(b_te, 50)
    r_same = c2st(xa, xb, xta, xtb, la, lb, classifier="cnn1d", n_boot=100, n_perm=50,
                  seed=0, window_len=50, importance=False)
    assert r_same.indistinguishable, f"same dist must be indistinguishable, AUC={r_same.auc}"
    assert abs(r_same.auc - 0.5) < 0.2

    # DIFFERENT distribution (the power control): shift class B by a constant offset.
    b2_tr = _lanes(base + 3.0, 6, 0.05, rng)
    b2_te = _lanes(base + 3.0, 4, 0.05, rng)
    xb2, _ = windows_from_lanes(b2_tr, 50)
    xtb2, lb2 = windows_from_lanes(b2_te, 50)
    r_diff = c2st(xa, xb2, xta, xtb2, la, lb2, classifier="cnn1d", n_boot=100, n_perm=50,
                  seed=0, window_len=50, importance=False)
    assert not r_diff.indistinguishable, "genuinely different classes must be distinguishable"
    assert r_diff.auc > 0.9


def test_byte_identity_and_wilson() -> None:
    from a1_2_a1_4_fresh import max_abs_delta
    from a1_5_triangulation import wilson

    a = np.arange(60.0).reshape(10, 6)
    assert max_abs_delta(a, a.copy()) == 0.0  # byte-identity
    assert max_abs_delta(a, a + 0.5) == 0.5

    lo, hi = wilson(0, 15)  # E2/A1 convention: 0/15 → [0, upper]
    assert lo == 0.0 and 0.15 < hi < 0.25
    lo2, hi2 = wilson(15, 15)
    assert hi2 == 1.0 and 0.75 < lo2 < 0.85
