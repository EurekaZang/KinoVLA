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

    def __init__(self, cfg: Config, embed_dim: int | None = None) -> None:
        self._res = float(cfg.resolution_m)
        self._extent = (float(cfg.extent[0]), float(cfg.extent[1]))
        self._origin = -0.5 * np.asarray(self._extent, dtype=np.float64)  # lower corner (x,y)
        self._nx = int(round(self._extent[0] / self._res))
        self._ny = int(round(self._extent[1] / self._res))
        # The appearance-feature width: 64 for the class/pixel surrogates, 512 for real CLIP. The
        # map's similarity machinery is encoder-agnostic; only the stored feature width changes.
        self._embed_dim = int(
            embed_dim if embed_dim is not None else cfg.get("embed_dim", EMBED_DIM)
        )
        self._cost = np.zeros((self._ny, self._nx), dtype=np.float64)
        self._physical = np.zeros((self._ny, self._nx), dtype=bool)
        self._observed = np.zeros((self._ny, self._nx), dtype=bool)
        self._embed = np.zeros((self._ny, self._nx, self._embed_dim), dtype=np.float64)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def resolution_m(self) -> float:
        return self._res

    @property
    def origin_xy(self) -> np.ndarray:
        """World-frame coordinate of the grid's lower (i=0, j=0) corner."""
        return self._origin.copy()

    @property
    def cost_grid(self) -> np.ndarray:
        """A copy of the (ny, nx) cost grid — read by the deployed geometric planner (decoupled
        nav, #44): A* routes around every cell above the hazard threshold."""
        return self._cost.copy()

    @property
    def physical_grid(self) -> np.ndarray:
        """A copy of the (ny, nx) bool mask of cells stamped by a physical-failure overwrite (the
        contact 'physics writes the map' / Update_Topology cells, as opposed to the lower-confidence
        visual prior or CLIP-propagated cells) — for the decoupled-nav visualization (#45)."""
        return self._physical.copy()

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

    # ------------------------------------------------------------ rolling (egocentric) window

    def recenter(self, center_xy: np.ndarray, *, margin_m: float) -> bool:
        """Roll the fixed-size grid so it stays centred on ``center_xy`` — an egocentric costmap for
        long-distance / multi-patch courses (#47). Marks move WITH the world (an integer-cell shift,
        no resampling, so ``cost_at(world)`` is invariant for on-window points); cells that scroll
        off the trailing edge are cleared (they left the world window — far behind, irrelevant to
        forward nav). Rolls only when ``center_xy`` is within ``margin_m`` of an edge (hysteresis ⇒
        infrequent), then recentres to the middle. Returns True iff it rolled."""
        rel = (np.asarray(center_xy, dtype=np.float64) - self._origin) / self._res
        m = int(np.ceil(float(margin_m) / self._res))
        near_edge = (
            rel[0] < m or rel[0] > self._nx - m or rel[1] < m or rel[1] > self._ny - m
        )
        if not near_edge:
            return False
        shift_i = int(round(self._nx / 2.0 - rel[0]))
        shift_j = int(round(self._ny / 2.0 - rel[1]))
        if shift_i == 0 and shift_j == 0:
            return False
        self._roll(shift_i, shift_j)
        self._origin = self._origin - np.array([shift_i, shift_j], dtype=np.float64) * self._res
        return True

    def _roll(self, shift_i: int, shift_j: int) -> None:
        """Shift every layer by (shift_i cols, shift_j rows); clear the wrapped-in (newly exposed)
        cells — np.roll wraps, and those cells are now unobserved ground, not stale data."""
        self._cost = np.roll(self._cost, (shift_j, shift_i), axis=(0, 1))
        self._physical = np.roll(self._physical, (shift_j, shift_i), axis=(0, 1))
        self._observed = np.roll(self._observed, (shift_j, shift_i), axis=(0, 1))
        self._embed = np.roll(self._embed, (shift_j, shift_i), axis=(0, 1))
        for arr in (self._cost, self._physical, self._observed, self._embed):
            if shift_i > 0:
                arr[:, :shift_i] = 0
            elif shift_i < 0:
                arr[:, shift_i:] = 0
            if shift_j > 0:
                arr[:shift_j, :] = 0
            elif shift_j < 0:
                arr[shift_j:, :] = 0

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

    def embedding_at(self, world_xy: np.ndarray) -> np.ndarray | None:
        """The appearance feature *actually perceived* at a cell, or None if never observed.

        Propagation must compare like with like: whatever encoder painted the map (class-hash
        surrogate or real pixel feature) is what a failure attribution should generalise with.
        """
        cell = self._cell_of(world_xy)
        if cell is None:
            return None
        i, j = cell
        embed = self._embed[j, i]
        return embed.copy() if float(np.linalg.norm(embed)) > 0.0 else None

    def dominant_feature_in(self, rect: Rect) -> np.ndarray | None:
        """The mean L2-normalised OBSERVED appearance feature over the cells in ``rect`` (None if
        none observed). Robust source of a region's CLIP feature when the exact query cell was not
        directly imaged (the forward-down camera rarely sees the cell under the robot)."""
        feats = [
            self._embed[j, i]
            for i, j in self._cells_in_rect(rect)
            if self._observed[j, i] and float(np.linalg.norm(self._embed[j, i])) > 0.0
        ]
        if not feats:
            return None
        mean = np.mean(feats, axis=0)
        n = float(np.linalg.norm(mean))
        return mean / n if n > 0.0 else None

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
