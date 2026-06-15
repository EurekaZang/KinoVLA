"""Persistent odometry-frame traversability costmap (spec §7).

The costmap is the system's *topological memory*: a grid keyed in the odometry/world
frame, so it is invariant to robot heading — the cure for v1.2's "turn around and
forget" failure. Two write channels with a strict confidence ordering:

1. **Visual prior (low confidence):** open-vocab segmentation paints an initial
   traversability estimate from appearance. It may be wrong (O7 visual-physics remap).
2. **Physical overwrite (high confidence):** when a kinodynamic failure is attributed
   on the ground, the affected cells are stamped untraversable and that stamp is
   *sticky* — later visual passes can never lower it. The mark then **propagates** to
   visually-homogeneous neighbours via CLIP-feature similarity (step through one thin-ice
   cell ⇒ the whole homogeneous sheet is down-weighted).

This is the "physics writes the map" inversion of VLMaps/value-maps the spec calls out
as an independently-evaluable contribution (§7).
"""

from __future__ import annotations

import numpy as np

from kino_vla.map.appearance import EMBED_DIM, cosine_similarity
from kino_vla.map.types import MapCrop, ObservedRegion
from kino_vla.utils.config import Config
from kino_vla.utils.geometry import Rect


class Costmap:
    """A grid of traversability costs in [0, 1] over a fixed odometry-frame extent."""

    def __init__(self, cfg: Config) -> None:
        self._res = float(cfg.resolution_m)
        self._extent = (float(cfg.extent[0]), float(cfg.extent[1]))
        self._origin = -0.5 * np.asarray(self._extent, dtype=np.float64)  # lower corner (x,y)
        self._nx = int(round(self._extent[0] / self._res))
        self._ny = int(round(self._extent[1] / self._res))
        self._cost = np.zeros((self._ny, self._nx), dtype=np.float64)
        self._physical = np.zeros((self._ny, self._nx), dtype=bool)
        self._observed = np.zeros((self._ny, self._nx), dtype=bool)
        self._embed = np.zeros((self._ny, self._nx, EMBED_DIM), dtype=np.float64)

    # ------------------------------------------------------------ indexing

    def _cell_of(self, world_xy: np.ndarray) -> tuple[int, int] | None:
        """(i, j) column/row of a world point, or None if outside the grid."""
        rel = (np.asarray(world_xy, dtype=np.float64) - self._origin) / self._res
        i, j = int(np.floor(rel[0])), int(np.floor(rel[1]))
        if 0 <= i < self._nx and 0 <= j < self._ny:
            return i, j
        return None

    def _cells_in_rect(self, rect: Rect) -> list[tuple[int, int]]:
        """All (i, j) cells whose centre falls inside ``rect``."""
        cells: list[tuple[int, int]] = []
        i_lo = max(0, int(np.floor((rect.cx - rect.hx - self._origin[0]) / self._res)))
        i_hi = min(self._nx - 1, int(np.ceil((rect.cx + rect.hx - self._origin[0]) / self._res)))
        j_lo = max(0, int(np.floor((rect.cy - rect.hy - self._origin[1]) / self._res)))
        j_hi = min(self._ny - 1, int(np.ceil((rect.cy + rect.hy - self._origin[1]) / self._res)))
        for j in range(j_lo, j_hi + 1):
            for i in range(i_lo, i_hi + 1):
                cx = self._origin[0] + (i + 0.5) * self._res
                cy = self._origin[1] + (j + 0.5) * self._res
                if rect.contains(np.array([cx, cy])):
                    cells.append((i, j))
        return cells

    # ------------------------------------------------------------ write channels

    def integrate_visual(self, regions: list[ObservedRegion]) -> None:
        """Paint the low-confidence visual prior; never lowers a physical mark (sticky)."""
        for region in regions:
            for i, j in self._cells_in_rect(region.footprint):
                self._observed[j, i] = True
                self._embed[j, i] = region.embedding
                if not self._physical[j, i]:
                    self._cost[j, i] = max(self._cost[j, i], region.visual_cost)

    def overwrite_physical(
        self, world_xy: np.ndarray, cost: float, embedding: np.ndarray, radius_m: float
    ) -> int:
        """Stamp a sticky high-cost disc at a failure site; returns #cells stamped."""
        stamped = 0
        r_cells = int(np.ceil(radius_m / self._res))
        center = self._cell_of(world_xy)
        if center is None:
            return 0
        ci, cj = center
        for j in range(max(0, cj - r_cells), min(self._ny, cj + r_cells + 1)):
            for i in range(max(0, ci - r_cells), min(self._nx, ci + r_cells + 1)):
                wx = self._origin[0] + (i + 0.5) * self._res
                wy = self._origin[1] + (j + 0.5) * self._res
                if float(np.hypot(wx - world_xy[0], wy - world_xy[1])) <= radius_m:
                    self._cost[j, i] = max(self._cost[j, i], float(cost))
                    self._physical[j, i] = True
                    self._observed[j, i] = True
                    if float(np.linalg.norm(self._embed[j, i])) == 0.0:
                        self._embed[j, i] = np.asarray(embedding, dtype=np.float64)
                    stamped += 1
        return stamped

    def propagate_similar(self, embedding: np.ndarray, cost: float, sim_threshold: float) -> int:
        """Raise cost on observed cells visually similar to ``embedding`` (CLIP propagation).

        Returns the number of cells newly down-weighted. Models "one broken thin-ice cell
        condemns the whole homogeneous sheet" (spec §7): the physical evidence at one cell
        is generalised to its visual class.
        """
        propagated = 0
        for j in range(self._ny):
            for i in range(self._nx):
                if not self._observed[j, i] or self._physical[j, i]:
                    continue
                if cosine_similarity(self._embed[j, i], embedding) >= sim_threshold:
                    new_cost = max(self._cost[j, i], float(cost))
                    if new_cost > self._cost[j, i]:
                        propagated += 1
                    self._cost[j, i] = new_cost
        return propagated

    # ------------------------------------------------------------ read

    def cost_at(self, world_xy: np.ndarray) -> float:
        cell = self._cell_of(world_xy)
        if cell is None:
            return 0.0
        i, j = cell
        return float(self._cost[j, i])

    def is_physical(self, world_xy: np.ndarray) -> bool:
        cell = self._cell_of(world_xy)
        if cell is None:
            return False
        i, j = cell
        return bool(self._physical[j, i])

    def crop(self, center_xy: np.ndarray, half_extent_m: float) -> MapCrop:
        """Local cost window centred on ``center_xy`` (planner context, spec §7)."""
        center = self._cell_of(center_xy)
        ci, cj = center if center is not None else (self._nx // 2, self._ny // 2)
        r = int(np.ceil(half_extent_m / self._res))
        i_lo, i_hi = max(0, ci - r), min(self._nx, ci + r + 1)
        j_lo, j_hi = max(0, cj - r), min(self._ny, cj + r + 1)
        origin = self._origin + np.array([i_lo * self._res, j_lo * self._res])
        return MapCrop(
            origin_xy=origin,
            resolution_m=self._res,
            cost=self._cost[j_lo:j_hi, i_lo:i_hi].copy(),
            physical=self._physical[j_lo:j_hi, i_lo:i_hi].copy(),
        )

    def untraversable_points(self, threshold: float) -> list[np.ndarray]:
        """World-frame centres of all cells whose cost exceeds ``threshold``."""
        pts: list[np.ndarray] = []
        for j in range(self._ny):
            for i in range(self._nx):
                if self._cost[j, i] > threshold:
                    pts.append(
                        self._origin + np.array([(i + 0.5) * self._res, (j + 0.5) * self._res])
                    )
        return pts

    @property
    def n_physical(self) -> int:
        return int(self._physical.sum())

    @property
    def shape(self) -> tuple[int, int]:
        return self._ny, self._nx
