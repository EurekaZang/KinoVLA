"""Semantic-map data types (spec §7): scene regions, observations, and map crops.

A ``SemanticRegion`` is the perception-side description of a ground patch — its visual
appearance (an embedding) and its world footprint — *decoupled* from its physics. O7
(Visual-Physics Remap) exploits exactly this decoupling: a region can look like
``"solid_ground"`` while physically being an ice/cliff hazard, and can carry a depth
corruption that displaces where the RGB-D back-projection believes it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kino_vla.map.appearance import appearance_embedding, cosine_similarity
from kino_vla.utils.geometry import Rect


@dataclass(frozen=True)
class SemanticRegion:
    """A visually-segmented ground region: appearance + world footprint (perception side).

    ``appearance_class`` names the material; ``embedding`` is its CLIP-surrogate feature
    (defaulted from the class name). ``depth_bias_m`` (O7) is a systematic range error
    injected into the back-projection — a positive bias makes the region appear *farther*
    than it is, so its visual footprint lands beyond the true contact patch (modelling
    over-exposure / IR-absorption depth failure, spec §8.2 O7).
    """

    rect: Rect
    appearance_class: str
    embedding: np.ndarray = field(default=None)  # type: ignore[assignment]
    depth_bias_m: float = 0.0
    visual_cost: float = 0.0  # prior traversability cost from appearance, in [0, 1]

    def __post_init__(self) -> None:
        if self.embedding is None:
            object.__setattr__(self, "embedding", appearance_embedding(self.appearance_class))

    @property
    def center(self) -> np.ndarray:
        return np.array([self.rect.cx, self.rect.cy], dtype=np.float64)

    def similarity(self, other_embedding: np.ndarray) -> float:
        return cosine_similarity(self.embedding, other_embedding)


@dataclass(frozen=True)
class ObservedRegion:
    """A region as the segmenter back-projected it this frame (post depth corruption)."""

    footprint: Rect  # odometry-frame footprint after depth back-projection
    embedding: np.ndarray
    appearance_class: str
    visual_cost: float


@dataclass(frozen=True)
class MapCrop:
    """A local window of the costmap served to the planner (spec §7 "map crop")."""

    origin_xy: np.ndarray  # world xy of cell (0, 0) lower corner
    resolution_m: float
    cost: np.ndarray  # (ny, nx) traversability cost in [0, 1]
    physical: np.ndarray  # (ny, nx) bool: cell was overwritten by a physical failure

    def untraversable_world_points(self, threshold: float) -> list[np.ndarray]:
        """World-frame centres of cells whose cost exceeds ``threshold``."""
        pts: list[np.ndarray] = []
        ny, nx = self.cost.shape
        for j in range(ny):
            for i in range(nx):
                if self.cost[j, i] > threshold:
                    pts.append(
                        self.origin_xy
                        + np.array([(i + 0.5) * self.resolution_m, (j + 0.5) * self.resolution_m])
                    )
        return pts
