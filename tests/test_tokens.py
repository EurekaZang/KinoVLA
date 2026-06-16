"""Torch-free unit tests for the Kino-Tokens M4 plumbing (spec §4).

Everything here imports only numpy + the surrogate, so it runs in the torch-free CI:
the window logger, the feature/target schema, the privileged-distillation dataset
builder (determinism + disjoint-seed split), the regime semantics, and — the headline
gate — the anomaly-gated μ̂→shield coupler logic (spec §4 #4), exercised against a fake
extractor so no torch is needed. The trained-model gates live in tests/test_extractor.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from kino_vla.sim.types import Obs
from kino_vla.tokens.coupler import MuEstimateCoupler
from kino_vla.tokens.dataset import build_dataset
from kino_vla.tokens.features import (
    FEATURE_SCHEMA,
    MU_INDEX,
    N_FEATURES,
    N_TARGETS,
    Standardizer,
    obs_to_features,
    physics_to_target,
)
from kino_vla.tokens.semantics import N_REGIMES, REGIMES, regime_of
from kino_vla.tokens.window import (
    RollingWindow,
    slice_rollout_to_windows,
    window_length,
)
from kino_vla.utils.config import load_config

CFG = load_config("tokens/extractor_v0.yaml")


def _obs(t: float = 0.0, *, vx: float = 0.5, slip: float = 0.0, effort: float = 0.0) -> Obs:
    return Obs(
        t=t,
        pos=np.zeros(2),
        heading=0.0,
        vel_body=np.array([vx, 0.0]),
        yaw_rate=0.0,
        cmd_prev=np.array([vx, 0.0, 0.0]),
        slip_ratio=slip,
        base_height=0.31,
        tilt=0.0,
        fallen=False,
        effort_ratio=effort,
        support_ratio=1.0,
    )


# --------------------------------------------------------------------- features
def test_feature_and_target_schema_layout():
    feats = obs_to_features(_obs(vx=0.6, slip=0.3, effort=0.2))
    assert feats.shape == (N_FEATURES,)
    assert feats[FEATURE_SCHEMA.index("slip_ratio")] == pytest.approx(0.3)
    assert feats[FEATURE_SCHEMA.index("effort_ratio")] == pytest.approx(0.2)
    # tracking_err = ||cmd_xy - vel_body||; here cmd==vel so it is zero.
    assert feats[FEATURE_SCHEMA.index("tracking_err")] == pytest.approx(0.0)


def test_physics_to_target_order_matches_schema():
    phys = {"mu": 0.2, "payload_kg": 7.0, "effort_scale": 0.5, "support_ratio": 0.4}
    tgt = physics_to_target(phys)
    assert tgt.shape == (N_TARGETS,)
    assert tgt[MU_INDEX] == pytest.approx(0.2)
    assert tgt.tolist() == [0.2, 7.0, 0.5, 0.4]


def test_standardizer_roundtrip_and_std_floor():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3)) * np.array([5.0, 0.0, 2.0]) + np.array([1.0, 9.0, -3.0])
    std = Standardizer.fit(x, eps=1e-3)
    # Zero-variance channel (col 1) is floored, never zero (no divide-by-zero).
    assert std.std[1] == pytest.approx(1e-3)
    z = std.transform(x)
    assert np.allclose(std.inverse(z), x)
    # Serialization roundtrip is exact.
    back = Standardizer.from_dict(std.to_dict())
    assert np.allclose(back.mean, std.mean) and np.allclose(back.std, std.std)


# ----------------------------------------------------------------------- window
def test_window_length_rounding():
    assert window_length(500.0, 50.0) == 25
    assert window_length(500.0, 1000.0) == 500
    assert window_length(0.0, 50.0) == 1  # floored to >= 1


def test_rolling_window_readiness_and_padding():
    win = RollingWindow(length=4)
    assert not win.ready
    # Before ready, shape is still fixed (left-padded with the earliest sample).
    win.push(_obs(vx=0.1))
    assert win.window().shape == (4, N_FEATURES)
    assert not win.ready
    for vx in (0.2, 0.3, 0.4):
        win.push(_obs(vx=vx))
    assert win.ready
    w = win.window()
    assert w.shape == (4, N_FEATURES)
    # Oldest row first: vx ascending 0.1..0.4 down the window.
    vx_col = FEATURE_SCHEMA.index("vx")
    assert list(w[:, vx_col]) == pytest.approx([0.1, 0.2, 0.3, 0.4])
    # Ring eviction: a fifth push drops the oldest.
    win.push(_obs(vx=0.5))
    assert list(win.window()[:, vx_col]) == pytest.approx([0.2, 0.3, 0.4, 0.5])
    win.reset()
    assert not win.ready


def test_slice_rollout_to_windows_shapes_and_targets():
    s, f, td = 20, N_FEATURES, N_TARGETS
    feats = np.arange(s * f, dtype=np.float64).reshape(s, f)
    tgts = np.arange(s * td, dtype=np.float64).reshape(s, td)
    wins, wt = slice_rollout_to_windows(feats, tgts, length=5, stride=3)
    starts = list(range(0, s - 5 + 1, 3))
    assert wins.shape == (len(starts), 5, f)
    assert wt.shape == (len(starts), td)
    # Target is taken at each window's LAST step (in-place supervision).
    assert np.allclose(wt[0], tgts[4])
    assert np.allclose(wt[1], tgts[3 + 4])
    # Too-short rollout returns empty (not an error).
    empty_w, empty_t = slice_rollout_to_windows(feats[:3], tgts[:3], length=5)
    assert empty_w.shape == (0, 5, f) and empty_t.shape == (0, td)


# --------------------------------------------------------------------- semantics
def test_regime_of_classifies_each_regime():
    idx = {name: i for i, name in enumerate(("mu", "payload_kg", "effort_scale", "support_ratio"))}

    def tgt(mu=0.8, payload=0.0, effort=1.0, support=1.0):
        v = np.zeros(4)
        v[idx["mu"]], v[idx["payload_kg"]] = mu, payload
        v[idx["effort_scale"]], v[idx["support_ratio"]] = effort, support
        return v

    thr = CFG.regime_thresholds
    kw = dict(
        mu_nominal=float(thr.mu_nominal),
        mu_low=float(thr.mu_low),
        payload_min_kg=float(thr.payload_min_kg),
        effort_low=float(thr.effort_low),
        support_low=float(thr.support_low),
    )
    assert REGIMES[regime_of(tgt(), **kw)] == "normal"
    assert REGIMES[regime_of(tgt(mu=0.1), **kw)] == "low_friction"
    assert REGIMES[regime_of(tgt(support=0.4), **kw)] == "high_centered"
    assert REGIMES[regime_of(tgt(payload=8.0), **kw)] == "overload"
    assert REGIMES[regime_of(tgt(effort=0.4), **kw)] == "actuator_decay"
    # Salience order: friction loss dominates a co-occurring payload.
    assert REGIMES[regime_of(tgt(mu=0.1, payload=8.0), **kw)] == "low_friction"


# ----------------------------------------------------------------------- dataset
def test_build_dataset_is_deterministic_and_split_disjoint():
    seeds = list(range(12))
    a = build_dataset(CFG, seeds)
    b = build_dataset(CFG, seeds)
    assert np.array_equal(a.windows, b.windows)
    assert np.array_equal(a.targets, b.targets)
    assert np.array_equal(a.regimes, b.regimes)
    assert a.windows.shape[1:] == (window_length(500.0, 50.0), N_FEATURES)
    assert len(a) > 0
    # A disjoint seed range yields different windows (no leakage / no accidental reuse).
    other = build_dataset(CFG, list(range(1_000_000, 1_000_012)))
    assert not np.array_equal(a.windows, other.windows)


# ------------------------------------------------------------------ μ̂ coupler
class _FakeExtractor:
    """Stand-in for the torch Extractor: counts predict() calls, returns fixed μ̂."""

    def __init__(self, mu: float = 0.12) -> None:
        self.calls = 0
        self._mu = mu

    def predict(self, windows: np.ndarray) -> SimpleNamespace:
        self.calls += 1
        theta = np.zeros((1, N_TARGETS))
        theta[0, MU_INDEX] = self._mu
        logits = np.zeros((1, N_REGIMES))
        logits[0, REGIMES.index("low_friction")] = 1.0
        return SimpleNamespace(theta_hat=theta, ood_score=np.array([0.5]), regime_logits=logits)


class _FakeShield:
    def __init__(self) -> None:
        self.mu_history: list[float] = []

    def set_mu_estimate(self, mu: float) -> None:
        self.mu_history.append(float(mu))


class _FakeMonitor:
    def __init__(self) -> None:
        self.anomaly_score = 0.0


def _make_coupler() -> tuple[MuEstimateCoupler, _FakeExtractor, _FakeShield]:
    ex, sh = _FakeExtractor(), _FakeShield()
    cp = MuEstimateCoupler(CFG, ex, sh)
    cp.reset()
    return cp, ex, sh


def test_coupler_gate_closed_skips_extractor_and_holds_nominal():
    cp, ex, sh = _make_coupler()
    mon = _FakeMonitor()  # anomaly 0 < gate_threshold 0.8
    for k in range(40):
        cp.step(_obs(t=k * 0.02), mon)
    assert ex.calls == 0, "steady state must skip the extractor (anomaly-gated, spec §4 #4)"
    # μ̂ stays pinned at nominal and is pushed to the shield every step.
    assert sh.mu_history[-1] == pytest.approx(float(CFG.coupler.nominal_mu))


def test_coupler_gate_open_drives_mu_estimate_down():
    cp, ex, sh = _make_coupler()
    mon = _FakeMonitor()
    # Fill the rolling window while quiet (no extractor calls yet).
    for k in range(int(cp._win.length)):
        cp.step(_obs(t=k * 0.02), mon)
    assert ex.calls == 0
    # Now the monitor fires: the extractor runs and μ̂ slews toward the ice estimate.
    nominal = float(CFG.coupler.nominal_mu)
    mon.anomaly_score = 0.95
    seq = []
    for k in range(int(cp._win.length), int(cp._win.length) + 30):
        rec = cp.step(_obs(t=k * 0.02, slip=0.6), mon)
        seq.append(rec.mu_hat)
    assert ex.calls > 0
    assert seq[-1] < nominal and seq[-1] < 0.25  # slewed toward μ̂=0.12
    pairs = zip(seq, seq[1:], strict=False)
    assert all(b <= a + 1e-12 for a, b in pairs), "μ̂ should decrease monotonically"
    assert sh.mu_history[-1] == pytest.approx(seq[-1])


def test_coupler_hold_keeps_gate_open_then_relaxes():
    cp, ex, sh = _make_coupler()
    mon = _FakeMonitor()
    dt = 0.1
    for k in range(int(cp._win.length)):
        cp.step(_obs(t=k * dt), mon)
    # One aroused step sets the hold window (hold_s seconds of latched arousal).
    mon.anomaly_score = 0.95
    cp.step(_obs(t=int(cp._win.length) * dt, slip=0.6), mon)
    calls_after_fire = ex.calls
    # Monitor quiets, but within hold_s the extractor keeps running (latched).
    mon.anomaly_score = 0.0
    t0 = (int(cp._win.length) + 1) * dt
    held = cp.step(_obs(t=t0, slip=0.6), mon)
    assert ex.calls > calls_after_fire and held.aroused
    # Advance well past the hold horizon: arousal lapses and μ̂ relaxes toward nominal.
    nominal = float(CFG.coupler.nominal_mu)
    last = held.mu_hat
    for k in range(60):
        rec = cp.step(_obs(t=t0 + float(CFG.coupler.hold_s) + 1.0 + k * dt), mon)
    assert not rec.aroused
    assert rec.mu_hat > last  # climbing back up
    assert rec.mu_hat == pytest.approx(nominal, abs=0.05)


def test_coupler_reset_restores_nominal():
    cp, ex, sh = _make_coupler()
    mon = _FakeMonitor()
    for k in range(int(cp._win.length)):
        cp.step(_obs(t=k * 0.02), mon)
    mon.anomaly_score = 0.95
    for k in range(int(cp._win.length), int(cp._win.length) + 10):
        cp.step(_obs(t=k * 0.02, slip=0.6), mon)
    assert sh.mu_history[-1] < float(CFG.coupler.nominal_mu)
    cp.reset()
    # reset re-pins the shield to nominal and clears the window/records.
    assert sh.mu_history[-1] == pytest.approx(float(CFG.coupler.nominal_mu))
    assert cp.records == [] and not cp._win.ready
