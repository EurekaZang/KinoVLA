"""Network-free pixel appearance encoder — real-pixel perception for the map (spec §7).

CLAUDE.md §6 #22 (M5 strict-GPU gap): the appearance encoder must consume real image pixels,
not a hash of a class string. This is a **deterministic colour encoder** over an RGB crop,
satisfying the same drop-in contract as ``appearance.appearance_embedding``: an L2-normalised
vector whose cosine similarity tracks visual homogeneity (same material → cosine ≈ 1,
different → ≈ 0), so the UNMODIFIED §7 propagation machinery runs on pixels, not labels.

Honest input scope on this box: the *binding* blocker is that the live Isaac RTX camera
cannot run here — ``AppLauncher --enable_cameras`` crashes during app init in Vulkan plugin
registration on this RTX 3060 / driver 595 / Ubuntu 26.04 / Isaac 5.1 stack (three probes:
outputs/gpu_audit/cam_probe*.log). So the encoder is exercised on a procedural *render* of
the material (``render_material_swatch``: diffuse colour + lighting gradient + grain — what a
camera would hand it), wiring it into the real ``Costmap.propagate_similar`` (see
tests/test_map_pixel_perception.py). The upgrade path — a live RTX camera on working hardware
(scripts/isaac_m5_perception_check.py) or a CLIP image encoder — is a drop-in: the contract
(RGB crop → unit feature) is identical. It is NOT CLIP (no open-vocabulary semantics).

Pure numpy: a soft-binned 3-D RGB histogram (distinct flat materials land in disjoint colour
bins ⇒ near-orthogonal) and, optionally, two gradient-texture features. Soft (trilinear)
binning makes the histogram robust to the lighting gradient across a swatch, so two views of
one material stay near-collinear above the 0.9 propagation bar. With ``bins=4`` and no texture
the feature is exactly ``EMBED_DIM`` (64) — the costmap embedding width.
"""

from __future__ import annotations

import numpy as np

# Diffuse colours rendered for each material plate (kept here so the encoder and the Isaac
# renderer agree). Chosen near distinct 4-bin RGB cell centres so lighting variation cannot
# push a material across a bin boundary into a neighbour's cell.
PIXEL_MATERIALS: dict[str, tuple[float, float, float]] = {
    "ice": (0.80, 0.88, 0.98),  # pale blue-white
    "mud": (0.42, 0.28, 0.14),  # brown
    "adhesive": (0.90, 0.80, 0.18),  # yellow board (O4 glue trap)
    "solid_ground": (0.50, 0.50, 0.50),  # neutral grey
}


class PixelAppearanceEncoder:
    """RGB crop → L2-normalised colour(/texture) feature (drop-in for CLIP, spec §7).

    With ``bins=4`` and ``include_texture=False`` the feature is exactly ``EMBED_DIM`` (64)
    dimensional, so it drops straight into the map's costmap embedding array — the §7
    propagation machinery then runs on real pixels with no other change. ``include_texture``
    appends two gradient-roughness features (dim 66) for uses that do not feed the costmap."""

    def __init__(
        self, bins: int = 4, include_texture: bool = False, texture_weight: float = 0.15
    ) -> None:
        self.bins = int(bins)
        self.include_texture = bool(include_texture)
        self.texture_weight = float(texture_weight)
        self.dim = self.bins**3 + (2 if self.include_texture else 0)

    def embed(self, rgb: np.ndarray) -> np.ndarray:
        """Encode an ``(H, W, 3)`` RGB crop (uint8 or float) to a ``(dim,)`` unit vector."""
        x = np.asarray(rgb, dtype=np.float64)
        if x.ndim != 3 or x.shape[2] < 3:
            raise ValueError(f"expected (H, W, >=3) RGB, got {x.shape}")
        x = x[..., :3]
        if x.max() > 1.5:  # uint8 / 0..255 → 0..1
            x = x / 255.0
        x = np.clip(x, 0.0, 1.0)

        hist = self._soft_histogram(x.reshape(-1, 3))
        if self.include_texture:
            gray = x.mean(axis=2)
            tex = np.array(
                [np.abs(np.diff(gray, axis=1)).mean(), np.abs(np.diff(gray, axis=0)).mean()]
            )
            vec = np.concatenate([hist, self.texture_weight * tex])
        else:
            vec = hist
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0.0 else vec

    def _soft_histogram(self, pixels: np.ndarray) -> np.ndarray:
        """Trilinear-soft 3-D RGB histogram, sum-normalised. Each pixel splits its mass
        across the 8 surrounding bin centres so a lighting gradient spreads consistently
        between two views of one material (robust within-material cosine)."""
        b = self.bins
        # Continuous bin coordinate in [0, b-1] with bin centres at integer positions.
        coord = np.clip(pixels * b - 0.5, 0.0, b - 1.0)  # (P, 3)
        lo = np.floor(coord).astype(int)
        hi = np.minimum(lo + 1, b - 1)
        frac = coord - lo  # (P, 3) in [0, 1]
        hist = np.zeros((b, b, b), dtype=np.float64)
        for dr in (0, 1):
            wr = frac[:, 0] if dr else 1.0 - frac[:, 0]
            ir = hi[:, 0] if dr else lo[:, 0]
            for dg in (0, 1):
                wg = frac[:, 1] if dg else 1.0 - frac[:, 1]
                ig = hi[:, 1] if dg else lo[:, 1]
                for db in (0, 1):
                    wb = frac[:, 2] if db else 1.0 - frac[:, 2]
                    ib = hi[:, 2] if db else lo[:, 2]
                    w = wr * wg * wb
                    np.add.at(hist, (ir, ig, ib), w)
        total = hist.sum()
        return (hist / total).ravel() if total > 0.0 else hist.ravel()


def solid_color_image(rgb: tuple[float, float, float], size: int = 32) -> np.ndarray:
    """A flat ``(size, size, 3)`` image of one colour — the CI/test stand-in for a render."""
    img = np.empty((size, size, 3), dtype=np.float64)
    img[:] = np.asarray(rgb, dtype=np.float64)
    return img


def render_material_swatch(name: str, seed: int = 0, size: int = 48) -> np.ndarray:
    """A material swatch under directional light + surface grain — a camera-view stand-in.

    Real Isaac RTX rendering is hardware-blocked on this box (AppLauncher --enable_cameras
    crashes in Vulkan plugin registration — outputs/gpu_audit/cam_probe.log), so the pixel
    encoder is exercised against this network-free procedural render of the material's diffuse
    colour with a lighting gradient and per-pixel grain — the kind of input a real camera (or
    a CLIP image path) would hand it — rather than the live Isaac camera feed. Two seeds give
    two viewpoints of the same material (different lighting/grain) for the same-material check.
    """
    rng = np.random.default_rng(seed)
    img = np.empty((size, size, 3), dtype=np.float64)
    img[:] = np.asarray(PIXEL_MATERIALS[name], dtype=np.float64)
    gx = np.linspace(-0.10, 0.10, size)[None, :, None]  # directional light across x
    gy = np.linspace(-0.06, 0.06, size)[:, None, None]  # and across y
    img = img + gx + gy + 0.03 * rng.standard_normal((size, size, 3))
    return np.clip(img, 0.0, 1.0)
