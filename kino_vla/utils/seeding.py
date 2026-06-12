"""Seeded determinism utilities.

Every stochastic component in the project seeds through here so that demo runs,
operator determinism gates (QA 5.2: same seed => same trajectory hash), and training
runs are reproducible from a single integer.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and (when installed) PyTorch RNGs.

    Torch is optional at M0: the dev laptop has no GPU and no torch install; the
    GPU machine seeds CUDA as well. ``PYTHONHASHSEED`` is set for subprocesses —
    it cannot retroactively affect the current interpreter's hash randomization,
    which is why trajectory hashing below never hashes Python objects directly.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rng(seed: int) -> np.random.Generator:
    """Return an isolated NumPy generator (preferred over global state in new code)."""
    return np.random.default_rng(seed)


def trajectory_hash(*arrays: np.ndarray, decimals: int = 6) -> str:
    """Stable hash of one or more float arrays for determinism gates.

    Arrays are rounded to ``decimals`` before hashing to absorb sub-tolerance
    floating-point noise (e.g. from GPU reduction order), then hashed as raw
    little-endian float64 bytes so the result is platform-stable.
    """
    h = hashlib.sha256()
    for arr in arrays:
        canonical = np.ascontiguousarray(
            np.round(np.asarray(arr, dtype=np.float64), decimals=decimals)
        )
        h.update(str(canonical.shape).encode())
        h.update(canonical.astype("<f8").tobytes())
    return h.hexdigest()
