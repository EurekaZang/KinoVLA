"""Procedural terrain stub: seeded determinism and extent clamping."""

from __future__ import annotations

from kino_vla.sim.terrain import generate_flat_with_patch
from kino_vla.utils.config import load_config


def _terrain_cfg(**overrides):
    return load_config("demo/walking_skeleton.yaml", overrides).terrain


def test_same_seed_same_patch():
    cfg = _terrain_cfg()
    a = generate_flat_with_patch(cfg, seed=42)
    b = generate_flat_with_patch(cfg, seed=42)
    assert a == b


def test_different_seed_different_patch():
    cfg = _terrain_cfg()
    a = generate_flat_with_patch(cfg, seed=1)
    b = generate_flat_with_patch(cfg, seed=2)
    assert a.hazard_patch != b.hazard_patch


def test_patch_clamped_inside_extent():
    cfg = _terrain_cfg(**{"terrain.hazard_patch.jitter_m": 100.0})
    for seed in range(5):
        spec = generate_flat_with_patch(cfg, seed=seed)
        patch = spec.hazard_patch
        assert abs(patch.cx) + patch.hx <= spec.extent[0] / 2 + 1e-9
        assert abs(patch.cy) + patch.hy <= spec.extent[1] / 2 + 1e-9
