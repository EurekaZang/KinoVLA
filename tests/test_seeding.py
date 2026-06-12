"""Seeding utilities: reproducibility and trajectory-hash stability."""

import numpy as np

from kino_vla.utils.seeding import rng, seed_everything, trajectory_hash


def test_seed_everything_reproducible():
    seed_everything(123)
    a = np.random.rand(8)
    seed_everything(123)
    b = np.random.rand(8)
    np.testing.assert_array_equal(a, b)


def test_isolated_rng_reproducible():
    assert rng(5).normal(size=4).tolist() == rng(5).normal(size=4).tolist()


def test_trajectory_hash_deterministic():
    arr = rng(0).normal(size=(100, 3))
    assert trajectory_hash(arr) == trajectory_hash(arr.copy())


def test_trajectory_hash_absorbs_subtolerance_noise():
    arr = rng(0).normal(size=(100, 3))
    noisy = arr + 1e-9  # below the default 6-decimal rounding
    assert trajectory_hash(arr) == trajectory_hash(noisy)


def test_trajectory_hash_detects_real_change():
    arr = rng(0).normal(size=(100, 3))
    changed = arr.copy()
    changed[0, 0] += 1e-3
    assert trajectory_hash(arr) != trajectory_hash(changed)


def test_trajectory_hash_shape_sensitive():
    arr = rng(0).normal(size=12)
    assert trajectory_hash(arr) != trajectory_hash(arr.reshape(3, 4))


def test_trajectory_hash_multiple_arrays():
    a, b = rng(1).normal(size=4), rng(2).normal(size=4)
    assert trajectory_hash(a, b) != trajectory_hash(b, a)
