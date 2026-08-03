"""Kino-Projector forward + grounding head (spec §4 / §11 route B). Torch-gated, CPU."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kino_vla.tokens.features import N_FEATURES, N_TARGETS  # noqa: E402
from kino_vla.vla.projector import KinoProjector  # noqa: E402


def test_forward_shapes():
    proj = KinoProjector(vlm_dim=2560, n_latents=6)
    window = torch.randn(4, 25, N_FEATURES)
    soft, theta = proj(window)
    assert soft.shape == (4, 6, 2560)
    assert theta.shape == (4, N_TARGETS)
    assert proj.n_soft_tokens == 6
    assert torch.isfinite(soft).all() and torch.isfinite(theta).all()


def test_standardizer_applied_and_serialized():
    proj = KinoProjector(vlm_dim=64)
    mean = np.arange(N_FEATURES, dtype=np.float64)
    std = np.ones(N_FEATURES) * 2.0
    proj.set_standardizer(mean, std)
    assert torch.allclose(proj.feat_mean, torch.tensor(mean, dtype=torch.float32))
    assert torch.allclose(proj.feat_std, torch.tensor(std, dtype=torch.float32))
    # buffers survive a state-dict round-trip (checkpoint portability)
    sd = proj.state_dict()
    proj2 = KinoProjector(vlm_dim=64)
    proj2.load_state_dict(sd)
    assert torch.allclose(proj2.feat_mean, proj.feat_mean)


def test_backprop_reaches_projection_and_theta():
    proj = KinoProjector(vlm_dim=32)
    window = torch.randn(2, 25, N_FEATURES)
    soft, theta = proj(window)
    (soft.sum() + theta.sum()).backward()
    grads = [p.grad for p in proj.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)
