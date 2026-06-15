"""Surrogate open-vocab segmenter + RGB-D depth back-projection (spec §7).

Real pipeline: SAM segments the RGB frame into regions, CLIP embeds each, and the
depth channel back-projects every region into the odometry frame as a 3D patch. Here
(no camera at the dev/CI tier, CLAUDE.md deviation #16) the "scene" is given as a list
of ``SemanticRegion`` world footprints; the segmenter models the two things that matter
for the map subsystem's logic:

- **Field of view / range gating:** only regions whose footprint is within the camera
  cone and range are observed this frame — so the map is built incrementally as the
  robot moves and looks around (and a marked region persists once the robot turns away).
- **Depth corruption (O7):** ``depth_bias_m`` displaces a region's back-projected
  footprint radially along the camera bearing, modelling the over-exposure / IR /
  specular depth failures the spec requires O7 to inject alongside the visual remap —
  so the map cannot trivially "see through" the deception via the clean D channel.

Swapping in real SAM+CLIP+depth means replacing ``segment`` with one that consumes an
RGB-D frame and returns the same ``ObservedRegion`` list.
"""

from __future__ import annotations

import numpy as np

from kino_vla.map.types import ObservedRegion, SemanticRegion
from kino_vla.utils.config import Config
from kino_vla.utils.geometry import Rect, unit, wrap_angle


class SurrogateSegmenter:
    """Field-of-view-gated segmenter with depth back-projection and O7 depth corruption."""

    def __init__(self, cfg: Config) -> None:
        self._fov = float(cfg.fov_rad)
        self._max_range = float(cfg.max_range_m)

    def visible(self, pose_xy: np.ndarray, heading: float, region: SemanticRegion) -> bool:
        """True when the region centre is within the camera cone and range."""
        delta = region.center - np.asarray(pose_xy, dtype=np.float64)
        dist = float(np.linalg.norm(delta))
        if dist > self._max_range or dist == 0.0:
            return dist == 0.0  # standing on it counts as observed
        bearing = float(np.arctan2(delta[1], delta[0]))
        return abs(wrap_angle(bearing - heading)) <= 0.5 * self._fov

    def segment(
        self, pose_xy: np.ndarray, heading: float, scene: list[SemanticRegion]
    ) -> list[ObservedRegion]:
        """Return the regions visible from ``pose`` as depth-back-projected observations."""
        pose_xy = np.asarray(pose_xy, dtype=np.float64)
        observed: list[ObservedRegion] = []
        for region in scene:
            if not self.visible(pose_xy, heading, region):
                continue
            center = region.center
            if region.depth_bias_m != 0.0:
                bearing = unit(center - pose_xy)
                center = center + region.depth_bias_m * bearing
            footprint = Rect(
                cx=float(center[0]),
                cy=float(center[1]),
                hx=region.rect.hx,
                hy=region.rect.hy,
            )
            observed.append(
                ObservedRegion(
                    footprint=footprint,
                    embedding=region.embedding,
                    appearance_class=region.appearance_class,
                    visual_cost=region.visual_cost,
                )
            )
        return observed
