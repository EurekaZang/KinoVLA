"""Contract tests for the network-free pixel appearance encoder (kino_vla/map/pixel_appearance).

The encoder is the M5 strict-GPU drop-in for CLIP: an RGB crop → unit
vector whose cosine similarity tracks visual homogeneity. These CPU tests lock the contract
the map's 0.9 propagation bar relies on — same material near-collinear *even under a lighting
gradient*, different materials well-separated — so the Isaac render only has to reproduce it.
"""

from __future__ import annotations

import numpy as np
import pytest

from kino_vla.map.appearance import cosine_similarity
from kino_vla.map.pixel_appearance import (
    PIXEL_MATERIALS,
    PixelAppearanceEncoder,
    solid_color_image,
)

PROP_BAR = 0.9  # configs/map/traversability_v0.yaml propagation_sim_threshold


def _lit(rgb, seed, size=32):
    """A flat colour with a smooth lighting gradient + sensor noise (a render stand-in)."""
    rng = np.random.default_rng(seed)
    img = solid_color_image(rgb, size)
    ramp = np.linspace(-0.08, 0.08, size)[:, None, None]  # vertical shading gradient
    img = img + ramp + 0.02 * rng.standard_normal((size, size, 3))
    return np.clip(img, 0.0, 1.0)


def test_same_material_is_collinear_under_lighting():
    enc = PixelAppearanceEncoder()
    for name, rgb in PIXEL_MATERIALS.items():
        a = enc.embed(_lit(rgb, seed=1))
        b = enc.embed(_lit(rgb, seed=2))
        sim = cosine_similarity(a, b)
        assert sim >= PROP_BAR, f"{name}: same-material cosine {sim:.3f} below {PROP_BAR}"


def test_different_materials_separate():
    enc = PixelAppearanceEncoder()
    names = list(PIXEL_MATERIALS)
    embs = {n: enc.embed(_lit(PIXEL_MATERIALS[n], seed=3)) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            sim = cosine_similarity(embs[a], embs[b])
            assert sim < PROP_BAR - 0.1, f"{a} vs {b}: cross cosine {sim:.3f} too high"


def test_deterministic():
    enc = PixelAppearanceEncoder()
    img = solid_color_image(PIXEL_MATERIALS["ice"])
    assert np.array_equal(enc.embed(img), enc.embed(img))


def test_unit_norm_and_dim():
    from kino_vla.map.appearance import EMBED_DIM

    enc = PixelAppearanceEncoder(bins=4)  # default: no texture → pure colour histogram
    v = enc.embed(solid_color_image(PIXEL_MATERIALS["mud"]))
    assert v.shape == (enc.dim,)
    assert enc.dim == 4**3 == EMBED_DIM  # drops straight into the costmap embedding array
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-9
    # Texture variant appends two gradient features (for non-map uses).
    assert PixelAppearanceEncoder(bins=4, include_texture=True).dim == 4**3 + 2


def test_rejects_bad_shape():
    enc = PixelAppearanceEncoder()
    with pytest.raises(ValueError):
        enc.embed(np.zeros((4, 4)))
