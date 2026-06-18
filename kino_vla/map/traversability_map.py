"""Semantic traversability map subsystem (spec §7): perception + persistent memory.

Wires the surrogate segmenter to the persistent costmap and exposes the two online
operations of spec §7 plus the planner hand-off:

- ``observe(pose)``       — segment the visible scene and paint the visual prior.
- ``mark_failure(xy)``    — a kinodynamic failure was attributed on the ground: stamp
                            the site untraversable (sticky) and **propagate** the mark
                            to visually-homogeneous neighbours via CLIP similarity.
- ``crop_for_planner``    — the local map window handed to the recovery planner.
- ``nav_hazards``         — untraversable cells clustered into avoid discs the FSM /
                            VLA planner can route around (the map's effect on behaviour).

The scene (a list of ``SemanticRegion``) is supplied by whoever assembles the world —
in the demo, the failure operators contribute their own visual signatures via
``FailureOperator.scene_region`` (so O2 brown mud and O4 yellow adhesive look different
even though they feel identical), and the segmenter only reveals them within view.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from kino_vla.map.costmap import Costmap
from kino_vla.map.segmentation import SurrogateSegmenter
from kino_vla.map.types import MapCrop, ObservedRegion, SemanticRegion
from kino_vla.utils.config import Config


class Segmenter(Protocol):
    """Perception front-end: scene → back-projected observations from the current pose."""

    def segment(
        self, pose_xy: np.ndarray, heading: float, scene: list[SemanticRegion]
    ) -> list[ObservedRegion]: ...


class TraversabilityMap:
    """Online semantic traversability map maintained in the odometry frame."""

    def __init__(
        self,
        cfg: Config,
        scene: list[SemanticRegion] | None = None,
        segmenter: Segmenter | None = None,
    ) -> None:
        self._cfg = cfg
        self._costmap = Costmap(cfg.costmap)
        # Default: the cheap centre-in-cone surrogate (keeps the live demo loop fast). The
        # real RGB-D pinhole back-projection (rgbd.RgbdSegmenter) is injectable here and is a
        # drop-in for the same .segment contract (spec §7; strict gate in tests/scripts).
        self._segmenter = segmenter if segmenter is not None else SurrogateSegmenter(cfg.camera)
        self._scene: list[SemanticRegion] = list(scene) if scene else []
        self._fail_cost = float(cfg.failure_cost)
        self._fail_radius = float(cfg.failure_radius_m)
        self._sim_threshold = float(cfg.propagation_sim_threshold)
        self._propagate_cost = float(cfg.propagation_cost)
        self._crop_half_extent = float(cfg.crop_half_extent_m)

    @property
    def costmap(self) -> Costmap:
        return self._costmap

    @property
    def scene(self) -> list[SemanticRegion]:
        return self._scene

    def set_scene(self, scene: list[SemanticRegion]) -> None:
        self._scene = list(scene)

    # ------------------------------------------------------------ online ops

    def observe(self, pose_xy: np.ndarray, heading: float) -> int:
        """Segment the visible scene and paint the visual prior; returns #regions seen."""
        regions = self._segmenter.segment(pose_xy, heading, self._scene)
        self._costmap.integrate_visual(regions)
        return len(regions)

    def _appearance_at(self, world_xy: np.ndarray) -> np.ndarray | None:
        """Best appearance embedding at a world point: the *perceived* map feature first, then
        scene truth. Preferring the painted costmap embedding keeps propagation in the same
        feature space the segmenter used (pixel features under RgbdSegmenter, class-hash under
        the surrogate) — otherwise a real-pixel map would never match a class-hash query."""
        perceived = self._costmap.embedding_at(world_xy)
        if perceived is not None:
            return perceived
        for region in self._scene:
            if region.rect.contains(np.asarray(world_xy, dtype=np.float64)):
                return region.embedding
        return None

    def mark_failure(self, world_xy: np.ndarray, embedding: np.ndarray | None = None) -> dict:
        """Overwrite the failure site untraversable (sticky) and propagate to homogeneous cells.

        Returns ``{"stamped", "propagated"}`` cell counts for telemetry/tests.
        """
        if embedding is None:
            embedding = self._appearance_at(world_xy)
        embedding = (
            np.asarray(embedding, dtype=np.float64)
            if embedding is not None
            else np.zeros(self._costmap_embed_dim())
        )
        stamped = self._costmap.overwrite_physical(
            world_xy, self._fail_cost, embedding, self._fail_radius
        )
        propagated = 0
        if float(np.linalg.norm(embedding)) > 0.0:
            propagated = self._costmap.propagate_similar(
                embedding, self._propagate_cost, self._sim_threshold
            )
        return {"stamped": stamped, "propagated": propagated}

    def _costmap_embed_dim(self) -> int:
        from kino_vla.map.appearance import EMBED_DIM

        return EMBED_DIM

    # ------------------------------------------------------------ planner hand-off

    def crop_for_planner(self, pose_xy: np.ndarray) -> MapCrop:
        return self._costmap.crop(pose_xy, self._crop_half_extent)

    def nav_hazards(self) -> list[tuple[np.ndarray, float]]:
        """Cluster untraversable cells into (centre, radius) avoid discs for the planner.

        Greedy, deterministic single-link clustering at the configured merge radius — a
        coarse but reproducible bridge from the grid to the FSM's circle-based detour
        planner (spec §7 "map crop served to planner context").
        """
        threshold = float(self._cfg.hazard_threshold)
        merge_radius = float(self._cfg.hazard_merge_radius_m)
        pts = self._costmap.untraversable_points(threshold)
        clusters: list[list[np.ndarray]] = []
        centers: list[np.ndarray] = []
        for p in pts:
            placed = False
            for k, c in enumerate(centers):
                if float(np.linalg.norm(p - c)) <= merge_radius:
                    clusters[k].append(p)
                    centers[k] = np.mean(clusters[k], axis=0)
                    placed = True
                    break
            if not placed:
                clusters.append([p])
                centers.append(p.copy())
        hazards: list[tuple[np.ndarray, float]] = []
        cell_half = 0.5 * float(self._cfg.costmap.resolution_m) * np.sqrt(2.0)
        for members, c in zip(clusters, centers, strict=True):
            radius = cell_half + max((float(np.linalg.norm(m - c)) for m in members), default=0.0)
            hazards.append((c, radius))
        return hazards
