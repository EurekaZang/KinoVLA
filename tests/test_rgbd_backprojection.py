"""Real RGB-D pinhole back-projection (spec §7) — kino_vla/map/rgbd.py.

These gate the §7 grounding step the SurrogateSegmenter shortcut skipped: a pinhole camera
renders a ground-plane RGB-D frame and the depth channel is unprojected back into the
odometry frame. The camera is procedural, but the geometry tests cover round-trip accuracy,
cross-view consistency, O7 depth-corruption flow-through, and the full chain into the costmap.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kino_vla.map.appearance import EMBED_DIM, cosine_similarity
from kino_vla.map.rgbd import (
    CameraExtrinsics,
    CameraIntrinsics,
    RgbdSegmenter,
)
from kino_vla.map.traversability_map import TraversabilityMap
from kino_vla.map.types import SemanticRegion
from kino_vla.utils.config import load_config
from kino_vla.utils.geometry import Rect

CFG = load_config("map/traversability_v0.yaml")


def _segmenter() -> RgbdSegmenter:
    return RgbdSegmenter(CFG.camera)


# ----------------------------------------------------------------- camera model


def test_intrinsics_from_hfov_centred_principal_point() -> None:
    intr = CameraIntrinsics.from_hfov(96, 72, math.radians(90.0))
    assert intr.cx == pytest.approx(48.0) and intr.cy == pytest.approx(36.0)
    # 90° hfov over 96 px ⇒ fx = (W/2)/tan(45°) = 48.
    assert intr.fx == pytest.approx(48.0, abs=1e-2)
    assert intr.fx == intr.fy  # square pixels


def test_extrinsics_axes_orthonormal_and_look_down() -> None:
    extr = CameraExtrinsics.look(np.array([0.0, 0.0]), heading=0.0, height_m=0.45, pitch_rad=0.6)
    r = extr.rot_wc
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-9)  # orthonormal
    forward = r[:, 2]
    assert forward[0] > 0.0 and forward[2] < 0.0  # looks forward (+x) and down (−z)


# ----------------------------------------------------------------- back-projection geometry


def test_roundtrip_footprint_recovers_true_region() -> None:
    """An uncorrupted depth render back-projects to the true ground footprint (sub-cell)."""
    seg = _segmenter()
    region = SemanticRegion(Rect(2.5, 0.0, 0.6, 0.6), "ice_sheet")
    obs = seg.segment(np.array([0.0, 0.0]), 0.0, [region])
    assert len(obs) == 1
    fp = obs[0].footprint
    assert fp.cx == pytest.approx(2.5, abs=0.1)
    assert fp.cy == pytest.approx(0.0, abs=0.1)
    assert fp.hx == pytest.approx(0.6, abs=0.1)
    assert fp.hy == pytest.approx(0.6, abs=0.1)


def test_cross_view_consistency_same_world_footprint() -> None:
    """The §7 odometry-frame invariance, through real geometry: two camera poses viewing the
    same region back-project to the same world footprint (the cure for 'turn around and forget',
    here proven by unprojection rather than a stored rect)."""
    seg = _segmenter()
    region = SemanticRegion(Rect(2.5, 0.0, 0.6, 0.6), "ice_sheet")
    a = seg.segment(np.array([0.0, 0.0]), 0.0, [region])[0].footprint
    pose2 = np.array([0.5, -1.0])
    heading2 = math.atan2(0.0 - (-1.0), 2.5 - 0.5)
    b = seg.segment(pose2, heading2, [region])[0].footprint
    assert math.hypot(a.cx - b.cx, a.cy - b.cy) < 0.15  # within ~half a cell


def test_o7_depth_corruption_displaces_footprint_along_ray() -> None:
    """O7 lives in the depth channel: a range bias pushes the back-projected footprint farther
    from the camera along the viewing ray (the deception the clean D channel cannot remove)."""
    seg = _segmenter()
    bias = 0.6
    true_rect = Rect(2.5, 0.0, 0.6, 0.6)
    clean = seg.segment(np.array([0.0, 0.0]), 0.0, [SemanticRegion(true_rect, "solid_ground")])[0]
    corrupt = seg.segment(
        np.array([0.0, 0.0]), 0.0, [SemanticRegion(true_rect, "solid_ground", depth_bias_m=bias)]
    )[0]
    # Clean recovers the truth; corrupt is displaced outward (farther in +x) by ~the bias.
    assert clean.footprint.cx == pytest.approx(2.5, abs=0.1)
    disp = corrupt.footprint.cx - clean.footprint.cx
    assert disp > 0.3  # pushed away from the camera
    assert disp == pytest.approx(bias, abs=0.2)  # magnitude tracks the injected range error


def test_region_out_of_view_is_not_observed() -> None:
    """FOV gating falls out of the render: a region behind the camera yields no pixels."""
    seg = _segmenter()
    behind = SemanticRegion(Rect(-3.0, 0.0, 0.6, 0.6), "ice_sheet")
    assert seg.segment(np.array([0.0, 0.0]), 0.0, [behind]) == []


# ----------------------------------------------------------------- appearance from real pixels


def test_material_separation_from_rendered_pixels() -> None:
    seg = _segmenter()
    pose, heading = np.array([0.0, 0.0]), 0.0
    rect = Rect(2.5, 0.0, 0.6, 0.6)
    ice_a = seg.segment(pose, heading, [SemanticRegion(rect, "ice_sheet")])[0].embedding
    pose2 = np.array([0.6, -0.8])
    heading2 = math.atan2(0.0 - (-0.8), 2.5 - 0.6)
    ice_b = seg.segment(pose2, heading2, [SemanticRegion(rect, "ice_sheet")])[0].embedding
    mud = seg.segment(pose, heading, [SemanticRegion(rect, "brown_mud")])[0].embedding
    bar = float(CFG.propagation_sim_threshold)
    assert cosine_similarity(ice_a, ice_b) >= bar  # two views of one material would propagate
    assert cosine_similarity(ice_a, mud) < bar  # different material would not


def test_embedding_is_costmap_shaped() -> None:
    seg = _segmenter()
    obs = seg.segment(np.array([0.0, 0.0]), 0.0, [SemanticRegion(Rect(2.5, 0.0, 0.6, 0.6), "ice")])
    assert obs[0].embedding.shape == (EMBED_DIM,)
    assert float(np.linalg.norm(obs[0].embedding)) == pytest.approx(1.0, abs=1e-9)


def test_encoder_bins_must_match_costmap_width() -> None:
    bad = load_config("map/traversability_v0.yaml", overrides={"camera.encoder_bins": 3})
    with pytest.raises(ValueError, match="EMBED_DIM"):
        RgbdSegmenter(bad.camera)  # 3**3 = 27 ≠ 64


# ----------------------------------------------------------------- full §7 chain on RGB-D


def test_full_chain_propagation_through_rgbd() -> None:
    """End-to-end §7 on real depth geometry: render → segment → back-project → costmap →
    physical overwrite → CLIP-similarity propagation, condemning the homogeneous ice sheet
    from one stepped cell while sparing the dissimilar mud (the 'physics writes the map' claim,
    now from pinhole-unprojected pixels rather than known world rects)."""
    ice = SemanticRegion(Rect(2.6, -0.9, 0.7, 0.7), "ice_sheet")
    mud = SemanticRegion(Rect(2.6, 1.1, 0.5, 0.5), "brown_mud")
    nav = TraversabilityMap(CFG, scene=[ice, mud], segmenter=_segmenter())

    # Observe the scene from a couple of poses so both regions are painted into the map.
    for pose, heading in (
        (np.array([0.0, 0.0]), 0.0),
        (np.array([0.0, 0.6]), -0.2),
        (np.array([0.2, -0.4]), 0.1),
    ):
        nav.observe(pose, heading)

    ice_corner = np.array([2.9, -1.2])  # one stepped-through ice cell
    far_ice = np.array([2.4, -0.7])  # elsewhere on the same homogeneous sheet
    assert nav.costmap.embedding_at(ice_corner) is not None  # the failure site was perceived

    res = nav.mark_failure(ice_corner)
    assert res["stamped"] > 0
    assert res["propagated"] > 0
    assert nav.costmap.is_physical(ice_corner)  # sticky overwrite at the site
    assert nav.costmap.cost_at(far_ice) >= float(CFG.propagation_cost)  # whole sheet condemned
    assert nav.costmap.cost_at(np.array([2.6, 1.1])) < float(CFG.propagation_cost)  # mud spared
