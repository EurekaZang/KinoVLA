"""Real pinhole RGB-D back-projection for the semantic map (spec §7).

Spec §7 grounds regions by *RGB-D depth back-projection*: "结合 RGB-D 深度反投影，将 2D 像素
区域持久化为机器人里程计坐标系下的 3D 区域" — segment the RGB frame, then use the depth
channel to unproject every segmented pixel into the odometry frame as a 3D patch. The
``SurrogateSegmenter`` (segmentation.py) shortcut skips pixel space entirely: it takes the
scene's *world* rectangles and only displaces their centres radially for O7. This module is
the real geometry the spec asks for — a pinhole camera model with explicit intrinsics and
extrinsics, a ground-plane RGB-D render (the synthetic depth sensor), and the inverse
unprojection that turns a depth image back into odometry-frame footprints.

Procedural render vs a live camera: this module's renderer is camera-free by design (CPU/CI,
deterministic, no RTX needed) and is the default for the live loop. A live Isaac RTX camera now
works on this box too (driver 580 + CUDA 12.8 nvrtc resolved the §6 #24/#26/#28 block;
scripts/m5_rtx_camera_perception.py runs the encoder on real RGB pixels). A live
``Camera.get_rgba`` + depth annotator drops straight into :func:`backproject_regions` unchanged
— the back-projection is the same pinhole math whether the (rgb, depth) come from this render
or the RTX camera.

The O7 depth corruption is modelled *in the depth channel* (a per-pixel range bias on the
deceptive region), so it flows through the real unprojection: the back-projected footprint
lands displaced along the camera ray, exactly the failure the spec requires (the clean D
channel "sees through" nothing — vision lied, and physics later overwrites the true site).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from kino_vla.map.appearance import EMBED_DIM, _seed_from_name
from kino_vla.map.pixel_appearance import PIXEL_MATERIALS, PixelAppearanceEncoder
from kino_vla.map.types import ObservedRegion, SemanticRegion
from kino_vla.utils.config import Config
from kino_vla.utils.geometry import Rect

# Appearance-class → diffuse RGB, the simulated world's material assignment (the renderer
# legitimately knows it, like a PhysX/RTX material on a plate; the *encoder* then reads only
# pixels). Aliases the operators' class names onto the encoder's PIXEL_MATERIALS palette.
_MATERIAL_ALIASES: dict[str, str] = {
    "ice": "ice",
    "ice_sheet": "ice",
    "thin_ice": "ice",
    "mud": "mud",
    "brown_mud": "mud",
    "compliance": "mud",
    "adhesive": "adhesive",
    "yellow_adhesive": "adhesive",
    "tether": "adhesive",
    "solid_ground": "solid_ground",
    "dry_concrete": "solid_ground",
    "concrete": "solid_ground",
}


def material_color(appearance_class: str) -> tuple[float, float, float]:
    """Diffuse RGB the renderer paints for a material class (deterministic).

    Known classes map onto the encoder's calibrated palette; an unknown class gets a stable
    pseudo-colour from its name hash so distinct materials still render distinctly.
    """
    alias = _MATERIAL_ALIASES.get(appearance_class)
    if alias is not None:
        return PIXEL_MATERIALS[alias]
    rng = np.random.default_rng(_seed_from_name(appearance_class))
    c = 0.25 + 0.5 * rng.random(3)  # keep away from 0/1 so lighting never clips a channel flat
    return (float(c[0]), float(c[1]), float(c[2]))


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics (pixels). Image x is right, y is down, z is forward (optical axis)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_rad: float) -> CameraIntrinsics:
        """Square-pixel intrinsics from a horizontal field of view; principal point centred."""
        fx = 0.5 * width / math.tan(0.5 * hfov_rad)
        return cls(
            width=int(width),
            height=int(height),
            fx=float(fx),
            fy=float(fx),  # square pixels
            cx=0.5 * width,
            cy=0.5 * height,
        )


@dataclass(frozen=True)
class CameraExtrinsics:
    """Camera pose in the odometry/world frame: position and world-from-camera rotation."""

    pos: np.ndarray  # (3,) world position of the optical centre
    rot_wc: np.ndarray  # (3, 3) columns are the camera x(right)/y(down)/z(forward) axes in world

    @classmethod
    def look(
        cls, pose_xy: np.ndarray, heading: float, height_m: float, pitch_rad: float
    ) -> CameraExtrinsics:
        """Body-mounted camera at ``height_m`` over ``pose_xy``, yaw ``heading``, tilted down
        by ``pitch_rad`` (a positive pitch looks at the ground ahead)."""
        pose_xy = np.asarray(pose_xy, dtype=np.float64)
        cphi, sphi = math.cos(pitch_rad), math.sin(pitch_rad)
        cpsi, spsi = math.cos(heading), math.sin(heading)
        forward = np.array([cpsi * cphi, spsi * cphi, -sphi], dtype=np.float64)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)  # completes a right-handed x=right,y=down,z=forward frame
        rot_wc = np.column_stack([right, down, forward])
        pos = np.array([pose_xy[0], pose_xy[1], float(height_m)], dtype=np.float64)
        return cls(pos=pos, rot_wc=rot_wc)


@dataclass(frozen=True)
class RgbdFrame:
    """A rendered RGB-D frame plus the per-pixel rays needed to invert it."""

    rgb: np.ndarray  # (H, W, 3) float in [0, 1]
    range_m: np.ndarray  # (H, W) measured range along the pixel ray (incl. O7 corruption)
    region_id: np.ndarray  # (H, W) int: index into the scene list, or -1 (unsegmented ground)
    ray_w: np.ndarray  # (H, W, 3) unit ray directions in the world frame
    pos: np.ndarray  # (3,) camera optical centre in world


def _pixel_rays(intr: CameraIntrinsics, extr: CameraExtrinsics) -> np.ndarray:
    """Unit world-frame ray directions for every pixel centre (H, W, 3)."""
    us = np.arange(intr.width, dtype=np.float64) + 0.5
    vs = np.arange(intr.height, dtype=np.float64) + 0.5
    grid_u, grid_v = np.meshgrid(us, vs)  # (H, W)
    xc = (grid_u - intr.cx) / intr.fx
    yc = (grid_v - intr.cy) / intr.fy
    ray_c = np.stack([xc, yc, np.ones_like(xc)], axis=-1)  # (H, W, 3) camera frame
    ray_w = ray_c @ extr.rot_wc.T  # rotate into world
    ray_w /= np.linalg.norm(ray_w, axis=-1, keepdims=True)
    return ray_w


def render_ground_scene(
    intr: CameraIntrinsics, extr: CameraExtrinsics, scene: list[SemanticRegion]
) -> RgbdFrame:
    """Render an RGB-D frame of a flat (z=0) ground populated with ``scene`` regions.

    A ground-plane ray cast (the synthetic depth sensor): each pixel ray is intersected with
    z=0, classified by which region rectangle contains the hit, shaded with the material
    colour under a mild directional light, and given a measured range (plus the O7 depth bias
    on the deceptive region). This is the (rgb, depth) a real RGB-D camera would hand the map.
    """
    ray_w = _pixel_rays(intr, extr)
    pos = extr.pos
    rz = ray_w[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -pos[2] / rz  # range to the ground plane along the unit ray
    valid = (rz < -1e-6) & (t > 0.0)
    t = np.where(valid, t, np.inf)
    hit_xy = pos[:2] + t[..., None] * ray_w[..., :2]  # (H, W, 2); garbage where invalid

    region_id = np.full((intr.height, intr.width), -1, dtype=np.int64)
    for idx, region in enumerate(scene):
        r = region.rect
        inside = (
            valid
            & (np.abs(hit_xy[..., 0] - r.cx) <= r.hx)
            & (np.abs(hit_xy[..., 1] - r.cy) <= r.hy)
            & (region_id == -1)  # first region wins on overlap (scenes are disjoint)
        )
        region_id[inside] = idx

    # Shade: base material colour under a smooth left→right light so two viewpoints of one
    # material differ slightly (tests within-material robustness) but stay near-collinear.
    light = 0.92 + 0.08 * (np.arange(intr.width, dtype=np.float64) / max(intr.width - 1, 1))
    rgb = np.empty((intr.height, intr.width, 3), dtype=np.float64)
    rgb[:] = material_color("solid_ground")  # unsegmented ground background
    for idx, region in enumerate(scene):
        mask = region_id == idx
        if mask.any():
            rgb[mask] = material_color(region.appearance_class)
    rgb = np.clip(rgb * light[None, :, None], 0.0, 1.0)

    range_m = t.copy()
    for idx, region in enumerate(scene):
        if region.depth_bias_m != 0.0:
            mask = region_id == idx
            range_m[mask] += region.depth_bias_m  # O7: corrupt the measured depth channel

    return RgbdFrame(rgb=rgb, range_m=range_m, region_id=region_id, ray_w=ray_w, pos=pos)


def backproject_regions(
    frame: RgbdFrame, scene: list[SemanticRegion], encoder: PixelAppearanceEncoder
) -> list[ObservedRegion]:
    """Unproject each segmented region's pixels to an odometry-frame footprint + embedding.

    For every region present in the frame, the segmented pixels are back-projected through the
    *measured* depth (``pos + range * ray``) into world points; their (x, y) extent is the
    odometry-frame footprint, and the rendered RGB of those pixels is encoded to the appearance
    feature. With an uncorrupted depth channel the unprojection reconstructs the true ground
    patch exactly; the O7 range bias displaces it along the ray (the deception the map inherits
    until physics overwrites the true site).
    """
    observed: list[ObservedRegion] = []
    for idx, region in enumerate(scene):
        mask = frame.region_id == idx
        n = int(mask.sum())
        if n == 0:
            continue
        rng = frame.range_m[mask][:, None]  # (N, 1)
        rays = frame.ray_w[mask]  # (N, 3)
        pts = frame.pos[None, :] + rng * rays  # (N, 3) world points
        xs, ys = pts[:, 0], pts[:, 1]
        footprint = Rect(
            cx=0.5 * float(xs.min() + xs.max()),
            cy=0.5 * float(ys.min() + ys.max()),
            hx=0.5 * float(xs.max() - xs.min()),
            hy=0.5 * float(ys.max() - ys.min()),
        )
        pixels = frame.rgb[mask].reshape(
            n, 1, 3
        )  # region pixels as a (N,1,3) image for the encoder
        observed.append(
            ObservedRegion(
                footprint=footprint,
                embedding=encoder.embed(pixels),
                appearance_class=region.appearance_class,
                visual_cost=region.visual_cost,
            )
        )
    return observed


class RgbdSegmenter:
    """Real RGB-D segmenter: render → segment → pinhole back-project (drop-in for
    ``SurrogateSegmenter``). Implements the same ``segment(pose, heading, scene)`` contract,
    so ``TraversabilityMap`` runs the §7 pipeline on genuine depth geometry."""

    def __init__(self, cfg: Config) -> None:
        self._intr = CameraIntrinsics.from_hfov(
            int(cfg.image_width), int(cfg.image_height), float(cfg.fov_rad)
        )
        self._height_m = float(cfg.mount_height_m)
        self._pitch = float(cfg.pitch_rad)
        self._encoder = PixelAppearanceEncoder(bins=int(cfg.encoder_bins))
        if self._encoder.dim != EMBED_DIM:
            raise ValueError(
                f"encoder dim {self._encoder.dim} != costmap EMBED_DIM {EMBED_DIM}; "
                f"set encoder_bins so bins**3 == {EMBED_DIM}"
            )

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self._intr

    def render(self, pose_xy: np.ndarray, heading: float, scene: list[SemanticRegion]) -> RgbdFrame:
        extr = CameraExtrinsics.look(pose_xy, heading, self._height_m, self._pitch)
        return render_ground_scene(self._intr, extr, scene)

    def segment(
        self, pose_xy: np.ndarray, heading: float, scene: list[SemanticRegion]
    ) -> list[ObservedRegion]:
        """Render the scene from the pose and back-project the visible regions."""
        frame = self.render(pose_xy, heading, scene)
        return backproject_regions(frame, scene, self._encoder)
