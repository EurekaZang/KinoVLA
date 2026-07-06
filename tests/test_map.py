"""M5 semantic-traversability-map gates (spec §7; CLAUDE.md M5 exit criteria).

Covers the map subsystem and its two named exit criteria:
- **turn-around persistence**: a marked region stays marked after a 360° rotation;
- **propagation**: stepping through one homogeneous cell down-weights the whole region.
Plus appearance-embedding properties, segmenter FOV/depth-corruption, costmap stickiness,
and the planner hand-off (crop + avoid discs).
"""

from __future__ import annotations

import numpy as np
import pytest

from kino_vla.map import (
    Costmap,
    SemanticRegion,
    SurrogateSegmenter,
    TraversabilityMap,
    appearance_embedding,
    cosine_similarity,
)
from kino_vla.utils.config import load_config
from kino_vla.utils.geometry import Rect

MAP_CFG = load_config("map/traversability_v0.yaml")


# ------------------------------------------------------------ perception front-end selection


def test_default_segmenter_is_clip_but_shared_config_pins_surrogate():
    """The code default front-end is real CLIP (spec §7); the shared CI/demo config opts down to
    the surrogate so the fast suite and the demo gate stay GPU/model-free (CLAUDE.md QA 5.1)."""
    from kino_vla.map.traversability_map import DEFAULT_SEGMENTER

    assert DEFAULT_SEGMENTER == "clip"
    assert MAP_CFG.get("segmenter") == "surrogate"
    # the shared config builds the cheap front-end (no torch/CLIP) ⇒ a 64-d costmap
    assert type(TraversabilityMap(MAP_CFG).costmap.embed_dim) is int
    assert TraversabilityMap(MAP_CFG).costmap.embed_dim == 64


def test_make_segmenter_selects_front_end_lazily():
    """make_segmenter maps the config string to the front-end class; the clip/rgbd imports are
    lazy so *constructing* them does not load torch/CLIP (only .segment / .embed_dim would)."""
    from kino_vla.map.clip_segmentation import ClipSegmenter
    from kino_vla.map.rgbd import RgbdSegmenter
    from kino_vla.map.traversability_map import make_segmenter

    cam = MAP_CFG.camera
    assert isinstance(make_segmenter("surrogate", cam), SurrogateSegmenter)
    assert isinstance(make_segmenter("rgbd", cam), RgbdSegmenter)
    assert isinstance(make_segmenter("clip", cam), ClipSegmenter)  # cheap: CLIP loads lazily
    # a {'segmenter': 'clip'} override routes through the same selector (what the Isaac demo uses)
    clip_cfg = load_config("map/traversability_v0.yaml", {"segmenter": "clip"})
    seg = make_segmenter(str(clip_cfg.get("segmenter")), clip_cfg.camera)
    assert isinstance(seg, ClipSegmenter)
    with pytest.raises(ValueError):
        make_segmenter("bogus", cam)


# ------------------------------------------------------------ appearance embeddings


def test_same_class_is_collinear():
    a = appearance_embedding("brown_mud")
    b = appearance_embedding("brown_mud")
    assert cosine_similarity(a, b) == 1.0


def test_distinct_classes_are_near_orthogonal():
    classes = ["brown_mud", "yellow_adhesive", "ice", "solid_ground", "dry_concrete"]
    embs = {c: appearance_embedding(c) for c in classes}
    for i, ci in enumerate(classes):
        for cj in classes[i + 1 :]:
            assert cosine_similarity(embs[ci], embs[cj]) < 0.5


def test_jitter_keeps_class_similar_but_not_identical():
    base = appearance_embedding("brown_mud")
    neighbour = appearance_embedding("brown_mud", jitter=0.15, instance=3)
    sim = cosine_similarity(base, neighbour)
    assert 0.9 < sim < 1.0  # still clearly the same surface, not bit-identical


def test_embeddings_are_unit_norm_and_deterministic():
    a = appearance_embedding("ice")
    b = appearance_embedding("ice")
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0)


# ------------------------------------------------------------ costmap


def test_costmap_physical_overwrite_is_sticky_against_visual():
    cm = Costmap(MAP_CFG.costmap)
    pt = np.array([3.0, 0.0])
    emb = appearance_embedding("ice")
    cm.overwrite_physical(pt, cost=1.0, embedding=emb, radius_m=0.5)
    assert cm.cost_at(pt) == 1.0
    assert cm.is_physical(pt)
    # A later low-cost visual pass over the same cell must NOT lower the physical mark.
    from kino_vla.map.types import ObservedRegion

    cm.integrate_visual([ObservedRegion(Rect(3.0, 0.0, 0.5, 0.5), emb, "ice", visual_cost=0.0)])
    assert cm.cost_at(pt) == 1.0


def test_costmap_crop_window():
    cm = Costmap(MAP_CFG.costmap)
    cm.overwrite_physical(np.array([2.0, 1.0]), 1.0, appearance_embedding("ice"), 0.4)
    crop = cm.crop(np.array([2.0, 1.0]), half_extent_m=1.0)
    assert crop.cost.max() == 1.0
    assert crop.physical.any()


# ------------------------------------------------------------ segmenter


def test_segmenter_fov_gating():
    seg = SurrogateSegmenter(MAP_CFG.camera)
    ahead = SemanticRegion(Rect(3.0, 0.0, 0.5, 0.5), "ice")
    behind = SemanticRegion(Rect(-3.0, 0.0, 0.5, 0.5), "ice")
    seen = seg.segment(np.array([0.0, 0.0]), 0.0, [ahead, behind])
    # Only the region in front of a forward-facing camera is observed.
    assert len(seen) == 1
    assert seg.visible(np.array([0.0, 0.0]), 0.0, ahead)
    assert not seg.visible(np.array([0.0, 0.0]), 0.0, behind)


def test_segmenter_depth_corruption_displaces_footprint():
    seg = SurrogateSegmenter(MAP_CFG.camera)
    clean = SemanticRegion(Rect(3.0, 0.0, 0.5, 0.5), "solid_ground", depth_bias_m=0.0)
    corrupt = SemanticRegion(Rect(3.0, 0.0, 0.5, 0.5), "solid_ground", depth_bias_m=1.0)
    fc = seg.segment(np.array([0.0, 0.0]), 0.0, [clean])[0]
    fk = seg.segment(np.array([0.0, 0.0]), 0.0, [corrupt])[0]
    # Positive depth bias pushes the back-projected footprint farther along the bearing.
    assert fk.footprint.cx > fc.footprint.cx + 0.9


# ------------------------------------------------------------ EXIT CRITERION: persistence


def test_turn_around_persistence():
    """A marked region stays marked after a full rotation + re-approach (spec §7)."""
    scene = [SemanticRegion(Rect(3.0, 0.0, 1.0, 1.0), "ice_sheet")]
    tm = TraversabilityMap(MAP_CFG, scene=scene)
    tm.observe(np.array([0.0, 0.0]), 0.0)
    tm.mark_failure(np.array([3.0, 0.0]))
    cost0 = tm.costmap.cost_at(np.array([3.0, 0.0]))
    assert cost0 >= 0.99
    # Rotate through a full turn and walk away, re-observing from every heading.
    for k in range(36):
        heading = 2.0 * np.pi * k / 36.0
        tm.observe(np.array([0.0, 0.0]), heading)
    for x in np.linspace(0.0, -4.0, 10):  # re-approach from the far side
        tm.observe(np.array([x, 0.0]), np.pi)
    assert tm.costmap.cost_at(np.array([3.0, 0.0])) == cost0  # unchanged, sticky


# ------------------------------------------------------------ EXIT CRITERION: propagation


def test_propagation_downweights_homogeneous_region():
    """Stepping through one ice cell down-weights the whole visually homogeneous sheet."""
    scene = [
        SemanticRegion(Rect(3.5, 0.0, 1.5, 1.5), "ice_sheet"),
        SemanticRegion(Rect(3.5, 4.0, 1.0, 1.0), "dry_concrete"),
    ]
    tm = TraversabilityMap(MAP_CFG, scene=scene)
    for heading in np.linspace(-np.pi, np.pi, 16):
        tm.observe(np.array([0.0, 0.0]), heading)
    far_before = tm.costmap.cost_at(np.array([4.8, 1.2]))
    res = tm.mark_failure(np.array([2.2, 0.0]))  # one cell of the sheet
    far_after = tm.costmap.cost_at(np.array([4.8, 1.2]))  # far corner, same sheet
    concrete = tm.costmap.cost_at(np.array([3.5, 4.0]))  # dissimilar patch
    assert res["propagated"] > 0
    assert far_after > far_before  # the homogeneous sheet was down-weighted
    assert far_after >= float(MAP_CFG.propagation_cost) - 1e-9
    assert concrete < 0.1  # the dissimilar region was NOT touched


def test_propagation_respects_similarity_threshold():
    """A visually dissimilar neighbour is not down-weighted (no over-propagation)."""
    scene = [
        SemanticRegion(Rect(3.0, 0.0, 0.6, 0.6), "ice"),
        SemanticRegion(Rect(3.0, 1.5, 0.6, 0.6), "yellow_adhesive"),
    ]
    tm = TraversabilityMap(MAP_CFG, scene=scene)
    for heading in np.linspace(-np.pi, np.pi, 12):
        tm.observe(np.array([0.0, 0.0]), heading)
    tm.mark_failure(np.array([3.0, 0.0]))
    assert tm.costmap.cost_at(np.array([3.0, 1.5])) < 0.1


# ------------------------------------------------------------ planner hand-off


def test_nav_hazards_cover_marked_region():
    scene = [SemanticRegion(Rect(3.0, 0.0, 1.0, 1.0), "ice_sheet")]
    tm = TraversabilityMap(MAP_CFG, scene=scene)
    tm.observe(np.array([0.0, 0.0]), 0.0)
    tm.mark_failure(np.array([3.0, 0.0]))
    hazards = tm.nav_hazards()
    assert hazards
    # Some avoid disc must contain the failure site.
    assert any(float(np.linalg.norm(c - np.array([3.0, 0.0]))) <= r for c, r in hazards)


# ------------------ EXIT CRITERION: O3 thin-ice propagation in a closed loop ----------


def test_o3_thin_ice_propagation_closed_loop():
    """Literal spec §3/§7 wording: stepping through one O3 thin-ice cell, in a sim rollout,
    down-weights the whole visually-homogeneous sheet (not just the broken cell)."""
    from kino_vla.loop import run_episode
    from kino_vla.shield.cbf_shield import CbfShield
    from kino_vla.sim.operators import Collapse, OperatorStack
    from kino_vla.sim.surrogate import SurrogateBackend
    from kino_vla.vla.fsm_recovery import FsmRecovery
    from tests._monitor_stub import StubMonitor

    sim_cfg = load_config("sim/surrogate.yaml")
    goal = np.array([6.0, 0.0])
    backend = SurrogateBackend(sim_cfg, np.array([0.0, 0.0]), 0.0)
    # O3 collapse on a sub-cell; the visually-homogeneous ice sheet is wider than it.
    ice = Collapse(region=Rect(3.0, 0.0, 0.8, 0.8), mu_collapsed=0.08, trigger_dwell_s=0.2)
    ops = OperatorStack([ice])
    monitor = StubMonitor()
    policy = FsmRecovery(load_config("recovery/fsm_v0.yaml"), goal_xy=goal, dt=backend.dt)
    shield = CbfShield(load_config("shield/cbf_v0.yaml"))
    scene = [SemanticRegion(Rect(3.0, 0.0, 1.5, 1.5), "ice_sheet")]
    nav_map = TraversabilityMap(MAP_CFG, scene=scene)
    result = run_episode(
        backend,
        ops,
        monitor,
        policy,
        shield,
        seed=0,
        goal_xy=goal,
        goal_tol_m=0.3,
        max_time_s=30.0,
        nav_map=nav_map,
    )
    assert result.monitor_fired  # the O3 collapse produced a slip the monitor caught
    assert nav_map.costmap.n_physical > 0  # the stepped cell was overwritten
    # A far corner of the SAME homogeneous sheet (never directly stepped) is down-weighted.
    far = nav_map.costmap.cost_at(np.array([3.9, 1.0]))
    assert far >= float(MAP_CFG.propagation_cost) - 1e-9


# --- #47 rolling (egocentric) costmap for long-distance / multi-patch ------------------------
def test_costmap_recenter_preserves_marks_in_world_frame():
    """The egocentric roll moves the grid with the robot but a mark stays at its WORLD coordinate
    (integer-cell shift, no resampling); a point beyond the start-centred grid becomes representable
    once the window rolls forward (long-distance). It only rolls near an edge (hysteresis)."""
    cm = Costmap(MAP_CFG.costmap)  # extent [16,10] ⇒ x∈[-8,8]
    cm.overwrite_physical(np.array([6.0, 0.0]), 1.0, np.zeros(cm.embed_dim), 0.4)
    n0 = cm.n_physical
    assert cm.cost_at(np.array([6.0, 0.0])) == 1.0 and n0 > 0
    assert cm.recenter(np.array([7.5, 0.0]), margin_m=2.0), "near the +x edge ⇒ rolls forward"
    assert cm.cost_at(np.array([6.0, 0.0])) == 1.0, "mark invariant in world frame after the roll"
    assert cm.n_physical == n0, "the on-window mark survived the roll"
    assert cm.cost_at(np.array([10.0, 0.0])) == 0.0, "x=10 now representable (was off the old grid)"
    assert not cm.recenter(np.array([7.5, 0.0]), margin_m=2.0), "no roll when comfortably inside"


def test_costmap_recenter_drops_cells_far_behind():
    cm = Costmap(MAP_CFG.costmap)  # x∈[-8,8]
    cm.overwrite_physical(np.array([-7.0, 0.0]), 1.0, np.zeros(cm.embed_dim), 0.3)
    assert cm.cost_at(np.array([-7.0, 0.0])) == 1.0
    cm.recenter(np.array([7.5, 0.0]), margin_m=2.0)  # window jumps forward
    assert cm.cost_at(np.array([-7.0, 0.0])) == 0.0, "the cell far behind left the window"


def test_traversability_map_recenter_respects_rolling_flag():
    """recenter is a no-op unless the map is configured ``rolling`` (default OFF ⇒ fixed grid)."""
    tm_fixed = TraversabilityMap(MAP_CFG, scene=[])
    assert tm_fixed.recenter(np.array([7.5, 0.0])) is False
    rolling_cfg = load_config("map/traversability_v0.yaml", overrides={"rolling": True})
    tm_roll = TraversabilityMap(rolling_cfg, scene=[])
    assert tm_roll.recenter(np.array([7.5, 0.0])) is True
