#!/usr/bin/env python
"""Semantic traversability map on REAL CLIP (spec §7) — open-vocab labelling + physics propagation.

Demonstrates the §7 map running end to end on a genuine CLIP encoder (no class-hash / histogram
surrogate): the robot observes a scene of textured materials, CLIP labels each region
open-vocabulary and embeds its pixels, and a single attributed thin-ice failure propagates to the
whole homogeneous ice sheet via real CLIP-feature cosine while sparing the mud and the adhesive
board. Prints PASS when (a) CLIP labels every region correctly and (b) the propagation respects
material boundaries.

    python scripts/clip_semantic_map.py
"""

from __future__ import annotations

import numpy as np

from kino_vla.map.clip_appearance import ClipAppearanceEncoder
from kino_vla.map.clip_segmentation import MATERIAL_VOCAB, ClipSegmenter
from kino_vla.map.traversability_map import TraversabilityMap
from kino_vla.map.types import SemanticRegion
from kino_vla.utils.config import load_config
from kino_vla.utils.geometry import Rect


def main() -> int:
    cfg = load_config("map/traversability_v0.yaml", {"propagation_sim_threshold": 0.92})

    # Scene: a large ice sheet flanked by a mud patch and a yellow adhesive board, all within one
    # 90-degree camera view from the origin. Their physics differ; appearance is what CLIP reads.
    scene = [
        SemanticRegion(Rect(3.0, 0.0, 1.5, 1.2), "ice_sheet"),
        SemanticRegion(Rect(4.0, 1.8, 0.9, 0.9), "brown_mud"),
        SemanticRegion(Rect(4.0, -1.8, 0.9, 0.9), "yellow_adhesive"),
    ]
    truth = {"ice_sheet": "ice", "brown_mud": "mud", "yellow_adhesive": "adhesive"}

    encoder = ClipAppearanceEncoder(vocabulary=MATERIAL_VOCAB)
    print(f"[clip-map] encoder = real CLIP ({encoder.model_id}), feature dim = {encoder.embed_dim}")
    segmenter = ClipSegmenter(cfg.camera, encoder=encoder)
    nav_map = TraversabilityMap(cfg, scene=scene, segmenter=segmenter)
    assert nav_map.costmap.embed_dim == encoder.embed_dim, "costmap must carry CLIP-width features"

    # 1) Observe: CLIP labels each region open-vocabulary and paints the costmap.
    n_seen = nav_map.observe(np.array([0.0, 0.0]), heading=0.0)
    print(f"[clip-map] observed {n_seen} regions; open-vocabulary labels from CLIP:")
    labels_ok = True
    for region in scene:
        material = truth[region.appearance_class]
        img = segmenter._perceive(region.appearance_class, seed=scene.index(region))[1]
        hit = img == material
        labels_ok &= hit
        verdict = "OK" if hit else "WRONG"
        print(f"           {region.appearance_class:16s} -> CLIP label '{img}'  {verdict}")

    # 2) Physics propagation: one attributed thin-ice failure condemns the homogeneous ice sheet.
    fail_xy = np.array([2.5, 0.3])  # a cell inside the ice sheet
    res = nav_map.mark_failure(fail_xy)
    print(
        f"[clip-map] thin-ice failure at {fail_xy.tolist()} -> "
        f"stamped {res['stamped']} cells, propagated {res['propagated']} via CLIP similarity"
    )

    # 3) Verify the mark respected CLIP material boundaries.
    ice_cost = nav_map.costmap.cost_at(np.array([3.5, -0.5]))  # elsewhere on the ice sheet
    mud_cost = nav_map.costmap.cost_at(np.array([4.0, 1.8]))  # the mud patch
    adh_cost = nav_map.costmap.cost_at(np.array([4.0, -1.8]))  # the adhesive board
    print(
        f"[clip-map] propagated cost: ice-sheet-elsewhere={ice_cost:.2f}  "
        f"mud={mud_cost:.2f}  adhesive={adh_cost:.2f}"
    )

    ice_condemned = ice_cost >= cfg.propagation_cost - 1e-6
    others_spared = mud_cost < 0.5 and adh_cost < 0.5
    ok = labels_ok and ice_condemned and others_spared and res["propagated"] > 0
    print(
        "PASS: real-CLIP semantic map labels every material and the thin-ice failure condemns the "
        "ice sheet while sparing mud and adhesive"
        if ok
        else "FAIL: CLIP labels or propagation did not respect material boundaries"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
