"""Torch-gated tests for the Kino-Tokens extractor model (spec §4, §9).

Skipped automatically where torch is absent (the torch-free CI box) — like the sim
gates, the authoritative milestone check is the manual GPU/CPU run of
``scripts/train_extractor.py`` recorded in the Completed Log. These tests cover the
model machinery on a small fast-trained instance (real layer sizes, reduced epochs/
seeds) and add an opt-in ``slow`` test that reproduces the full exit gates from scratch.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

from tests._monitor_stub import StubMonitor

torch = pytest.importorskip("torch")  # noqa: F841 — gate: skip module when torch absent

from kino_vla.tokens.dataset import build_dataset  # noqa: E402
from kino_vla.tokens.extractor import Extractor, ExtractResult  # noqa: E402
from kino_vla.tokens.features import N_TARGETS  # noqa: E402
from kino_vla.utils.config import load_config  # noqa: E402

TOL = load_config("tolerances.yaml")


def _fast_cfg():
    """Real model dimensions (so latency is meaningful) but few epochs/seeds for speed."""
    return load_config(
        "tokens/extractor_v0.yaml",
        {"train.epochs": 12, "train.n_train_seeds": 24, "train.n_eval_seeds": 12},
    )


@pytest.fixture(scope="module")
def trained():
    cfg = _fast_cfg()
    ds = build_dataset(cfg, list(range(int(cfg.train.n_train_seeds))))
    ex = Extractor(cfg, device="cpu")
    hist = ex.fit(ds)
    return cfg, ex, ds, hist


def test_predict_returns_well_shaped_result(trained):
    _, ex, ds, _ = trained
    res = ex.predict(ds.windows[:5])
    assert isinstance(res, ExtractResult)
    assert res.theta_hat.shape == (5, N_TARGETS)
    assert res.mu_hat.shape == (5,)
    assert res.ood_score.shape == (5,)
    assert res.tokens.ndim == 3 and res.tokens.shape[0] == 5
    assert np.all(np.isfinite(res.theta_hat)) and np.all(res.ood_score >= 0)
    # mu_hat is the convenience view of theta_hat[:, MU_INDEX].
    assert np.allclose(res.mu_hat, res.theta_hat[:, 0])
    # A bare (T, F) window is promoted to a batch of one.
    single = ex.predict(ds.windows[0])
    assert single.theta_hat.shape == (1, N_TARGETS)


def test_training_reduces_losses(trained):
    _, _, _, hist = trained
    assert hist["loss"][-1] < hist["loss"][0]
    assert hist["reg"][-1] < hist["reg"][0]  # the main θ-regression supervision improves
    assert hist["recon"][-1] < hist["recon"][0]


def test_save_load_roundtrip_is_exact(trained, tmp_path):
    cfg, ex, ds, _ = trained
    path = tmp_path / "ex"
    ex.save(path)
    reloaded = Extractor.load(cfg, path, device="cpu")
    a = ex.predict(ds.windows[:8])
    b = reloaded.predict(ds.windows[:8])
    assert np.allclose(a.theta_hat, b.theta_hat, atol=1e-6)
    assert np.allclose(a.ood_score, b.ood_score, atol=1e-6)


def test_ood_residual_rises_off_manifold(trained):
    # Spec §9: the reconstruction residual is the OOD signal. Grossly off-manifold
    # windows (scaled + shifted far outside the training support) must score higher.
    _, ex, ds, _ = trained
    in_dist = float(ex.predict(ds.windows[:64]).ood_score.mean())
    wild = float(ex.predict(ds.windows[:64] * 5.0 + 3.0).ood_score.mean())
    assert wild > in_dist


def test_latent_mahalanobis_fallback_fitted(trained):
    _, ex, ds, _ = trained
    d = ex.ood_mahalanobis(ds.windows[:10])
    assert d.shape == (10,) and np.all(np.isfinite(d)) and np.all(d >= 0)


def test_gate_tokens_anomaly_gating():
    tokens = np.ones((1, 4, 8))
    assert np.all(Extractor.gate_tokens(tokens, 0.0) == 0.0)  # steady state ⇒ no wakeup
    assert np.allclose(Extractor.gate_tokens(tokens, 1.0), tokens)
    assert np.allclose(Extractor.gate_tokens(tokens, 2.0), tokens)  # clipped to [0, 1]
    assert np.allclose(Extractor.gate_tokens(tokens, 0.5), 0.5 * tokens)


def test_inference_latency_under_budget(trained):
    # Real layer sizes (the fixture only shrinks epochs/seeds), so this is the deployed
    # model's single-window p99 — spec §6.9 / CLAUDE.md §5.3 extractor budget.
    _, ex, _, _ = trained
    p99 = ex.inference_latency_ms(n=200)
    assert p99 <= float(TOL.latency.inference_ms_p99)


def _load_script(name: str):
    """Import a scripts/<name>.py module by path (scripts/ is not a package)."""
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # register so @dataclass type resolution can find the module
    spec.loader.exec_module(module)
    return module


def test_coupling_tightens_friction_bound(trained, tmp_path):
    """μ̂→shield coupling MECHANISM (spec §6.5): on detected ice μ̂ drops below the
    firm-ground nominal and the friction-cone radius shrinks. Direction-only (robust to
    the small fixture model); the strict tolerance gate is in the slow test below."""
    _, ex, _, _ = trained
    ckpt = tmp_path / "ex"
    ex.save(ckpt)
    demo = _load_script("coupling_demo")
    res = demo.run_coupling_demo(seed=0, ckpt=str(ckpt), monitor=StubMonitor())
    assert res.fired, "monitor must fire on the ice for the gate to open"
    assert res.mu_hat_firm == pytest.approx(0.8, abs=1e-3)  # gate closed off-ice ⇒ nominal
    assert res.mu_hat_ice < res.mu_hat_firm  # extractor pulls μ̂ down on ice
    assert res.r_ice < res.r_firm  # friction cone tightens (the QP turns conservative)


@pytest.mark.slow
def test_full_milestone_gates_pass(tmp_path):
    """Reproduce ALL M4 exit gates from scratch on the real config (QA 5.2: a checkpoint
    is 'done' only when its eval is reproduced) — θ regression, OOD, latency, and the
    strict μ̂→shield coupling tolerances. Opt-in (slow)."""
    train = _load_script("train_extractor")
    cfg = load_config("tokens/extractor_v0.yaml")
    tol = load_config("tolerances.yaml")
    ckpt = tmp_path / "ex"
    metrics = train.train_and_eval(cfg, tol, device="cpu", out=str(ckpt))
    assert metrics["all_pass"], metrics
    # Exit criterion 3: μ̂→shield coupling on the same checkpoint (spec §6.5).
    demo = _load_script("coupling_demo")
    res = demo.run_coupling_demo(seed=0, ckpt=str(ckpt), monitor=StubMonitor())
    assert res.fired
    assert res.mu_hat_ice <= float(tol.coupling.max_mu_hat_on_ice)
    assert res.mu_hat_firm >= float(tol.coupling.min_mu_hat_on_firm)
    assert res.r_ice < res.r_firm
