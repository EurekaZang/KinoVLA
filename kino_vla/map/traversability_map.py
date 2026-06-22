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


# The perception front-end the map builds when none is injected and the config omits a
# ``segmenter`` key. Spec §7 wants real open-vocab CLIP, so that is the DEFAULT. The shared
# CI / fast-test / surrogate-demo config (configs/map/traversability_v0.yaml) pins
# ``segmenter: surrogate`` to keep those paths GPU/model-free (CLAUDE.md QA 5.1, the demo gate);
# the Isaac CLIP demos load it with ``segmenter: clip``.
DEFAULT_SEGMENTER = "clip"


def make_segmenter(kind: str, camera_cfg: Config) -> Segmenter:
    """Build the §7 perception front-end named by ``kind``.

    - ``"clip"`` → :class:`~kino_vla.map.clip_segmentation.ClipSegmenter`: real open-vocab CLIP
      image features (512-d) + CLIP labelling — the default front-end (spec §7).
    - ``"rgbd"`` → :class:`~kino_vla.map.rgbd.RgbdSegmenter`: real pinhole RGB-D back-projection
      with the network-free pixel encoder (64-d).
    - ``"surrogate"`` → :class:`~kino_vla.map.segmentation.SurrogateSegmenter`: the cheap
      centre-in-cone class-hash stand-in (64-d), for the GPU/model-free CI path.

    Imports for the CLIP / RGB-D fronts are lazy, so selecting ``surrogate`` never pulls in
    torch/transformers (keeps the fast suite and the surrogate demo light).
    """
    kind = kind.lower()
    if kind == "surrogate":
        return SurrogateSegmenter(camera_cfg)
    if kind == "rgbd":
        from kino_vla.map.rgbd import RgbdSegmenter

        return RgbdSegmenter(camera_cfg)
    if kind == "clip":
        from kino_vla.map.clip_segmentation import ClipSegmenter

        return ClipSegmenter(camera_cfg)
    raise ValueError(f"unknown segmenter {kind!r}; expected 'clip', 'rgbd', or 'surrogate'")


class TraversabilityMap:
    """Online semantic traversability map maintained in the odometry frame."""

    def __init__(
        self,
        cfg: Config,
        scene: list[SemanticRegion] | None = None,
        segmenter: Segmenter | None = None,
    ) -> None:
        self._cfg = cfg
        # Perception front-end. An explicit ``segmenter`` wins; otherwise the config's
        # ``segmenter`` key selects one, defaulting to real CLIP (DEFAULT_SEGMENTER, spec §7) —
        # the RGB-D and surrogate fronts are the same .segment drop-in. The costmap's feature
        # width follows the segmenter's encoder (64 for the class/pixel surrogates, 512 for real
        # CLIP), so the propagation machinery runs unchanged on whichever features it produces.
        self._segmenter = (
            segmenter
            if segmenter is not None
            else make_segmenter(str(cfg.get("segmenter", DEFAULT_SEGMENTER)), cfg.camera)
        )
        self._costmap = Costmap(cfg.costmap, embed_dim=getattr(self._segmenter, "embed_dim", None))
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
    def sim_threshold(self) -> float:
        """CLIP cosine-similarity threshold for 'the same appearance/feature' (spec §7)."""
        return self._sim_threshold

    def feature_at(self, world_xy: np.ndarray) -> np.ndarray | None:
        """The CLIP appearance feature perceived at a world point (None if never observed).

        Public wrapper over the perceived-first lookup so the planner can enforce the user's
        hard rule: a nav waypoint is forbidden from landing where the feature matches one the
        robot has already failed in (the feature-veto, replacing the advisory-only map text).
        """
        return self._appearance_at(world_xy)

    def region_feature(self, world_xy: np.ndarray) -> np.ndarray | None:
        """The DOMINANT perceived CLIP feature of the scene region containing ``world_xy`` (the
        appearance of the region the robot failed in), averaged over the region's observed cells —
        robust when the exact failure cell was not imaged. None if no region / nothing observed."""
        wp = np.asarray(world_xy, dtype=np.float64)
        for region in self._scene:
            if region.rect.contains(wp):
                return self._costmap.dominant_feature_in(region.rect)
        return None

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
        dim = self._costmap_embed_dim()
        # Guard a dim mismatch: a scene-truth surrogate feature (64) cannot stamp the CLIP costmap
        # (512). Drop it to a zero feature of the costmap width (the mark lands; no propagation).
        if embedding is not None and int(np.asarray(embedding).shape[-1]) != dim:
            embedding = None
        embedding = (
            np.asarray(embedding, dtype=np.float64) if embedding is not None else np.zeros(dim)
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
        return self._costmap.embed_dim

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
