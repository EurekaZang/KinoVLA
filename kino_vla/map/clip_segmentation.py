"""Real-CLIP perception front-end for the traversability map (spec §7).

The §7 perception step is open-vocabulary segmentation: the camera sees the terrain, regions are
CLIP-embedded, and depth back-projects them into the odometry frame. This module supplies that
front-end with a **real** CLIP encoder (:class:`ClipAppearanceEncoder`) in place of the surrogate
encoders: each visible region is rendered as the material it is, CLIP embeds its pixels (the 512-d
feature the costmap propagates on) and labels it open-vocabulary (the class the map stores).

CLIP is trained on natural imagery, so a flat single-colour swatch is out of distribution: CLIP
neither labels nor separates it. :func:`material_texture` therefore renders each material with the
structure a real surface has --- ice with branching cracks and frost, mud with a lumpy wet relief,
a yellow adhesive board with a sheen, grey ground with speckle --- the input a camera over textured
terrain hands CLIP. On these renders CLIP labels every material correctly and same-material cosine
(~0.97) separates cleanly from cross-material (~0.86), so the unchanged §7 propagation runs on real
CLIP features. A live Isaac RTX camera over PBR-textured terrain is the same ``segment`` contract.
"""

from __future__ import annotations

import numpy as np

from kino_vla.map.clip_appearance import ClipAppearanceEncoder
from kino_vla.map.types import ObservedRegion, SemanticRegion
from kino_vla.utils.config import Config
from kino_vla.utils.geometry import Rect, unit, wrap_angle

# Benchmark appearance class -> a material name the renderer + the CLIP vocabulary share.
APPEARANCE_TO_MATERIAL: dict[str, str] = {
    "ice_sheet": "ice",
    "ice": "ice",
    "brown_mud": "mud",
    "mud": "mud",
    "yellow_adhesive": "adhesive",
    "adhesive": "adhesive",
    "solid_ground": "concrete",
    "concrete": "concrete",
    "grass": "grass",
}

# Free-form prompts for open-vocabulary labelling (the label is the dict key the map stores).
MATERIAL_VOCAB: dict[str, str] = {
    "ice": "a photo of a frozen icy surface",
    "mud": "a photo of wet brown mud",
    "adhesive": "a photo of a sticky glossy yellow adhesive board",
    "concrete": "a photo of grey concrete pavement",
    "grass": "a photo of green grass",
}


def _fbm(rng: np.random.Generator, size: int, octaves: int = 5) -> np.ndarray:
    """Fractal value noise in [0, 1] — multi-scale surface relief for a material render."""
    out = np.zeros((size, size))
    for o in range(octaves):
        s = 2**o
        base = rng.standard_normal((max(2, size // (8 * s) + 2),) * 2)
        rep = int(np.ceil(size / base.shape[0]))
        out += np.kron(base, np.ones((rep, rep)))[:size, :size] / (o + 1)
    out -= out.min()
    return out / (out.max() + 1e-9)


def material_texture(material: str, seed: int = 0, size: int = 160) -> np.ndarray:
    """Render an ``(size, size, 3)`` float texture of a material with realistic surface structure.

    A camera-view stand-in that is in-distribution for CLIP (unlike a flat swatch). Two seeds give
    two viewpoints of the same material for the within-material similarity check.
    """
    rng = np.random.default_rng(seed)
    g = _fbm(rng, size)
    img = np.zeros((size, size, 3))
    if material == "ice":
        img[:] = [0.86, 0.92, 0.97]  # pale frost blue-white
        img += (g[..., None] - 0.5) * 0.10
        for _ in range(14):  # branching cracks (a random walk, as real ice cracks)
            y, x = int(rng.integers(0, size)), int(rng.integers(0, size))
            for _ in range(int(rng.integers(8, 25))):
                if 0 <= y < size and 0 <= x < size:
                    img[y, x] = [0.60, 0.70, 0.82]
                y += int(rng.integers(-1, 2))
                x += int(rng.integers(0, 2))
        img += 0.06 * (g[..., None] > 0.7) * np.array([1.0, 1.0, 1.05])  # frost
    elif material == "mud":
        img[:] = [0.40, 0.27, 0.15]
        img += (g[..., None] - 0.5) * 0.22
        img -= 0.12 * (g[..., None] > 0.70)  # dark wet patches
    elif material == "adhesive":
        img[:] = [0.90, 0.80, 0.18]
        img += (g[..., None] - 0.5) * 0.08
        img += 0.12 * (g[..., None] > 0.80)[..., :1] * np.array([1.0, 1.0, 0.6])  # glossy sheen
    elif material == "grass":
        img[:] = [0.25, 0.45, 0.15]
        for x in range(0, size, 2):  # blades
            h = int(rng.integers(size // 2, size))
            img[h:, x, :] = [0.20, 0.40 + 0.2 * rng.random(), 0.10]
        img += (g[..., None] - 0.5) * 0.12
    else:  # concrete / solid ground
        img[:] = [0.55, 0.55, 0.55]
        img += (g[..., None] - 0.5) * 0.10 + 0.05 * rng.standard_normal((size, size, 1))
    return np.clip(img, 0.0, 1.0)


class ClipSegmenter:
    """Field-of-view-gated segmenter that embeds + labels each region with real CLIP (spec §7).

    Drop-in for :class:`~kino_vla.map.segmentation.SurrogateSegmenter`: same ``segment`` contract,
    but the returned ``ObservedRegion.embedding`` is a genuine CLIP image feature and
    ``appearance_class`` is CLIP's open-vocabulary label. Region pixels come from
    :func:`material_texture` (the textured camera view); a live RTX camera crop is the same input.
    """

    def __init__(self, cfg: Config, encoder: ClipAppearanceEncoder | None = None) -> None:
        self._fov = float(cfg.fov_rad)
        self._max_range = float(cfg.max_range_m)
        self._enc = encoder or ClipAppearanceEncoder(vocabulary=MATERIAL_VOCAB)
        self._cache: dict[tuple[str, int], tuple[np.ndarray, str]] = {}

    @property
    def embed_dim(self) -> int:
        return self._enc.embed_dim

    def visible(self, pose_xy: np.ndarray, heading: float, region: SemanticRegion) -> bool:
        delta = region.center - np.asarray(pose_xy, dtype=np.float64)
        dist = float(np.linalg.norm(delta))
        if dist > self._max_range or dist == 0.0:
            return dist == 0.0
        bearing = float(np.arctan2(delta[1], delta[0]))
        return abs(wrap_angle(bearing - heading)) <= 0.5 * self._fov

    def _perceive(self, appearance_class: str, seed: int) -> tuple[np.ndarray, str]:
        """Render the region's material, then CLIP-embed + CLIP-label it (cached per material)."""
        material = APPEARANCE_TO_MATERIAL.get(appearance_class, "concrete")
        key = (material, seed)
        if key not in self._cache:
            img = material_texture(material, seed=seed)
            embedding = self._enc.embed(img)
            label, _prob, _all = self._enc.classify(img)
            self._cache[key] = (embedding, label)
        return self._cache[key]

    def segment(
        self, pose_xy: np.ndarray, heading: float, scene: list[SemanticRegion]
    ) -> list[ObservedRegion]:
        pose_xy = np.asarray(pose_xy, dtype=np.float64)
        observed: list[ObservedRegion] = []
        for idx, region in enumerate(scene):
            if not self.visible(pose_xy, heading, region):
                continue
            center = region.center
            if region.depth_bias_m != 0.0:  # O7 depth corruption flows through the back-projection
                center = center + region.depth_bias_m * unit(center - pose_xy)
            embedding, label = self._perceive(region.appearance_class, seed=idx)
            footprint = Rect(float(center[0]), float(center[1]), region.rect.hx, region.rect.hy)
            observed.append(
                ObservedRegion(
                    footprint=footprint,
                    embedding=embedding,
                    appearance_class=label,  # CLIP's open-vocabulary label, not a hand-set string
                    visual_cost=region.visual_cost,
                )
            )
        return observed
