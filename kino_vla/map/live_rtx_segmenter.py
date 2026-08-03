"""Live Isaac RTX camera → §7 semantic-map perception (real, not synthetic).

Closes the perception loop the synthetic ``ClipSegmenter`` left open. A robot-mounted RGB + depth +
semantic-segmentation camera images the textured terrain; for every tagged hazard region the real
 camera pixels are CLIP-embedded + CLIP-labelled, and the region geometry is recovered from the
camera's measured ``distance_to_image_plane`` depth and real intrinsics/pose.
So WHICH pixels are a hazard (the semantic mask), WHAT it is (the open-vocab CLIP label + 512-d
feature) and WHERE it is (the ground-ray footprint) all come from the real camera — not from a
ground-truth rect or a label-keyed synthetic texture.

Drop-in for the :class:`~kino_vla.map.traversability_map.Segmenter` protocol.  Using the measured
depth is essential for O7: a timestamped depth fault must displace the observed map footprint while
the RGB appearance remains unchanged.
"""

from __future__ import annotations

import numpy as np

from kino_vla.map.clip_appearance import ClipAppearanceEncoder
from kino_vla.map.clip_segmentation import MATERIAL_VOCAB
from kino_vla.map.rgbd import CameraExtrinsics, CameraIntrinsics
from kino_vla.map.types import ObservedRegion, SemanticRegion
from kino_vla.utils.geometry import Rect

_SKIP_CLASSES = {"BACKGROUND", "UNLABELLED", "UNLABELED", ""}


class LiveRtxSegmenter:
    """Ground the §7 map from the real RTX camera: semantic mask + real-CLIP label + ray∩ground."""

    def __init__(
        self,
        backend: object,
        *,
        min_px: int = 40,
        visual_cost: float = 0.2,
        vocabulary: dict[str, str] | None = None,
    ) -> None:
        self._backend = backend
        self._enc = ClipAppearanceEncoder(vocabulary=vocabulary or MATERIAL_VOCAB)
        self._min_px = int(min_px)
        self._visual_cost = float(visual_cost)
        # Last frame's perception, for live visualisation (dashboard shows what the camera sees).
        self.last_rgb: np.ndarray | None = None
        self.last_regions: list[tuple[str, float, Rect]] = []  # (clip_label, prob, footprint)

    @property
    def embed_dim(self) -> int:
        return self._enc.embed_dim  # 512 (real CLIP ViT-B/32)

    def _ground_xy(self, cap: dict) -> tuple[np.ndarray, np.ndarray]:
        """Back-project measured image-plane depth to start-frame xy for every valid pixel."""
        h, w = cap["rgb"].shape[:2]
        eye, target = (
            np.asarray(cap["eye"], dtype=np.float64),
            np.asarray(cap["target"], dtype=np.float64),
        )
        fwd = target - eye
        heading = float(np.arctan2(fwd[1], fwd[0]))
        pitch = float(np.arctan2(-fwd[2], float(np.hypot(fwd[0], fwd[1]))))
        k = cap["K"]
        intr = CameraIntrinsics(
            width=w,
            height=h,
            fx=float(k[0, 0]),
            fy=float(k[1, 1]),
            cx=float(k[0, 2]),
            cy=float(k[1, 2]),
        )
        extr = CameraExtrinsics.look(eye[:2], heading, float(eye[2]), pitch)
        us = np.arange(w, dtype=np.float64) + 0.5
        vs = np.arange(h, dtype=np.float64) + 0.5
        grid_u, grid_v = np.meshgrid(us, vs)
        ray_camera = np.stack(
            [(grid_u - intr.cx) / intr.fx, (grid_v - intr.cy) / intr.fy, np.ones_like(grid_u)],
            axis=-1,
        )
        # Isaac's distance_to_image_plane is optical-axis Z depth, so multiplying the
        # unnormalised [x/z,y/z,1] ray gives camera-frame xyz directly.
        ray_world = ray_camera @ extr.rot_wc.T
        depth = np.asarray(cap["depth"], dtype=np.float64)
        world_xy = extr.pos[:2] + depth[..., None] * ray_world[..., :2]
        finite = (
            np.isfinite(depth)
            & (depth > 0.0)
            & (depth < 30.0)
            & np.isfinite(world_xy).all(-1)
        )
        return world_xy, finite

    def segment(
        self, pose_xy: np.ndarray, heading: float, scene: list[SemanticRegion]
    ) -> list[ObservedRegion]:
        """Read the live camera and back-project every tagged hazard region (ignores ``scene``)."""
        cap = self._backend.capture_perception()
        if cap is None or cap.get("eye") is None:
            return []
        rgb, seg = cap["rgb"], cap["seg"]
        self.last_rgb = rgb
        self.last_regions = []
        world_xy, finite = self._ground_xy(cap)
        observed: list[ObservedRegion] = []
        for sid, lab in cap["id_to_labels"].items():
            cls = (lab.get("class") if isinstance(lab, dict) else str(lab)) or ""
            if cls.upper() in _SKIP_CLASSES:
                continue
            mask = (seg == int(sid)) & finite
            if int(mask.sum()) < self._min_px:
                continue
            pts = world_xy[mask]
            xlo, xhi = np.percentile(
                pts[:, 0], [5, 95]
            )  # robust extent (drop grazing-ray outliers)
            ylo, yhi = np.percentile(pts[:, 1], [5, 95])
            footprint = Rect(
                cx=float(0.5 * (xlo + xhi)),
                cy=float(0.5 * (ylo + yhi)),
                hx=float(max(0.1, 0.5 * (xhi - xlo))),
                hy=float(max(0.1, 0.5 * (yhi - ylo))),
            )
            ys, xs = np.where(mask)
            crop = rgb[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1, :3]
            label, prob, _all = self._enc.classify(crop)
            self.last_regions.append((label, float(prob), footprint))
            observed.append(
                ObservedRegion(
                    footprint=footprint,
                    embedding=self._enc.embed(crop),  # real 512-d CLIP feature
                    appearance_class=label,  # real open-vocab CLIP label
                    visual_cost=self._visual_cost,
                )
            )
        return observed
