"""Procedural terrain stub (M1): flat ground with one seeded hazard patch.

Real procedural terrain generation arrives with the operator suites; at M1 the
walking skeleton only needs a deterministic, seed-jittered rectangle for the O1
ice patch so the demo is procedural rather than hard-coded (CLAUDE.md M1 scope).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kino_vla.utils.config import Config
from kino_vla.utils.geometry import Rect
from kino_vla.utils.seeding import rng


@dataclass(frozen=True)
class TerrainSpec:
    """Flat-terrain layout for one episode."""

    extent: tuple[float, float]  # total walkable area (x_size, y_size), centered on origin
    hazard_patch: Rect  # region the demo's O1 operator overrides


def generate_flat_with_patch(cfg: Config, seed: int) -> TerrainSpec:
    """Place the hazard patch at the configured center with seeded uniform jitter.

    The patch is clamped to stay inside the terrain extent so a jittered patch can
    never leak outside the walkable area.
    """
    gen = rng(seed)
    extent = (float(cfg.extent[0]), float(cfg.extent[1]))
    half_size = np.asarray(cfg.hazard_patch.half_size, dtype=np.float64)
    jitter = gen.uniform(-cfg.hazard_patch.jitter_m, cfg.hazard_patch.jitter_m, size=2)
    center = np.asarray(cfg.hazard_patch.center, dtype=np.float64) + jitter
    lo = -0.5 * np.asarray(extent) + half_size
    hi = 0.5 * np.asarray(extent) - half_size
    center = np.clip(center, lo, hi)
    patch = Rect(
        cx=float(center[0]), cy=float(center[1]), hx=float(half_size[0]), hy=float(half_size[1])
    )
    return TerrainSpec(extent=extent, hazard_patch=patch)
