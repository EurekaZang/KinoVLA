"""M5 strict closure: the real §7 map propagates hazards from REAL pixel-encoder embeddings.

CLAUDE.md §6 #22 — the appearance encoder must consume real pixels, not a class hash. The
live Isaac RTX camera is hardware-blocked on this box (three AppLauncher ``--enable_cameras``
probe crashes — outputs/gpu_audit/cam_probe*.log), exactly as real CLIP is proxy-blocked, so
the real-pixel input here is a procedural material render (lighting + grain — a camera
stand-in). It drives the UNMODIFIED §7 map machinery (Costmap.integrate_visual /
overwrite_physical / propagate_similar) and verifies the §7 claim end-to-end from pixels:
"step through one thin-ice cell ⇒ the whole visually-homogeneous sheet is down-weighted, a
different material is spared." The encoder is a drop-in for CLIP — only the encoder changed,
not the map.
"""

from __future__ import annotations

import numpy as np

from kino_vla.map.costmap import Costmap
from kino_vla.map.pixel_appearance import PixelAppearanceEncoder, render_material_swatch
from kino_vla.map.types import ObservedRegion
from kino_vla.utils.config import load_config
from kino_vla.utils.geometry import Rect


def _region(name: str, enc: PixelAppearanceEncoder, seed: int, cx: float, cy: float):
    emb = enc.embed(render_material_swatch(name, seed=seed))
    return ObservedRegion(
        footprint=Rect(cx, cy, 0.5, 0.5), embedding=emb, appearance_class=name, visual_cost=0.1
    )


def test_map_propagates_hazard_from_pixel_embeddings():
    cfg = load_config("map/traversability_v0.yaml")
    enc = PixelAppearanceEncoder()  # 64-dim, drops straight into the costmap embedding array
    bar = float(cfg.propagation_sim_threshold)
    pc = float(cfg.propagation_cost)
    cm = Costmap(cfg.costmap)

    ice_a, ice_b = np.array([2.0, 0.0]), np.array([2.0, 2.0])  # two views of one ice sheet
    mud, ground = np.array([-2.0, 0.0]), np.array([-2.0, 2.0])
    cm.integrate_visual(
        [
            _region("ice", enc, 1, *ice_a),
            _region("ice", enc, 2, *ice_b),
            _region("mud", enc, 3, *mud),
            _region("solid_ground", enc, 4, *ground),
        ]
    )

    # A broken thin-ice cell (physical failure) at ice_a; propagate by PIXEL similarity.
    ice_query = enc.embed(render_material_swatch("ice", seed=5))
    cm.overwrite_physical(ice_a, cost=1.0, embedding=ice_query, radius_m=0.3)
    n = cm.propagate_similar(ice_query, cost=pc, sim_threshold=bar)

    assert n > 0, "the homogeneous ice sheet should propagate from the broken cell"
    assert cm.cost_at(ice_b) >= pc, "the visually-homogeneous ice neighbour is condemned (§7)"
    assert cm.cost_at(mud) < pc, "a different material must be spared"
    assert cm.cost_at(ground) < pc, "firm ground must be spared"


def test_pixel_embedding_is_costmap_shaped():
    """The pixel encoder output must match the costmap embedding width (drop-in for CLIP)."""
    from kino_vla.map.appearance import EMBED_DIM

    enc = PixelAppearanceEncoder()
    emb = enc.embed(render_material_swatch("adhesive", seed=0))
    assert emb.shape == (EMBED_DIM,)
