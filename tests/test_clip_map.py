"""Real-CLIP semantic traversability map (spec §7): open-vocab labelling + physics propagation.

Marked ``slow`` — it loads a real CLIP backbone (``openai/clip-vit-base-patch32``), the
deliverable the surrogate encoders stood in for. Skipped gracefully when CLIP/transformers or the
cached weights are unavailable, so the fast suite never depends on the network.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def _encoder_or_skip():
    try:
        from kino_vla.map.clip_segmentation import MATERIAL_VOCAB, ClipAppearanceEncoder

        enc = ClipAppearanceEncoder(vocabulary=MATERIAL_VOCAB)
        assert enc.embed_dim == 512  # real CLIP ViT-B/32 projection width
        return enc
    except Exception as exc:  # noqa: BLE001 - missing model/weights/transformers ⇒ skip, not fail
        pytest.skip(f"real CLIP unavailable: {exc}")


def test_clip_labels_materials_open_vocabulary():
    """CLIP names each benchmark surface from its pixels (the §7 open-vocab segmentation)."""
    from kino_vla.map.clip_segmentation import material_texture

    enc = _encoder_or_skip()
    for material in ("ice", "mud", "adhesive", "concrete"):
        label, prob, _ = enc.classify(material_texture(material, seed=0))
        assert label == material, f"CLIP mislabelled {material} as {label} (p={prob:.2f})"


def test_clip_features_separate_same_from_different_material():
    """CLIP image-feature cosine tracks visual homogeneity: same > different (propagation)."""
    from kino_vla.map.clip_segmentation import material_texture

    enc = _encoder_or_skip()
    mats = ("ice", "mud", "adhesive", "concrete")
    emb = {m: [enc.embed(material_texture(m, seed=s)) for s in (0, 1)] for m in mats}
    same = min(float(emb[m][0] @ emb[m][1]) for m in mats)
    diff = max(float(emb[a][0] @ emb[b][0]) for i, a in enumerate(mats) for b in mats[i + 1 :])
    assert same > diff, f"same-material cosine {same:.3f} must exceed cross-material {diff:.3f}"
    assert same - diff > 0.05  # a usable propagation-threshold margin


def test_clip_map_propagation_respects_material_boundaries():
    """One thin-ice failure condemns the homogeneous ice sheet but spares mud/adhesive (§7), on
    real CLIP features end to end through the costmap."""
    from kino_vla.map.clip_segmentation import ClipSegmenter
    from kino_vla.map.traversability_map import TraversabilityMap
    from kino_vla.map.types import SemanticRegion
    from kino_vla.utils.config import load_config
    from kino_vla.utils.geometry import Rect

    enc = _encoder_or_skip()
    cfg = load_config("map/traversability_v0.yaml", {"propagation_sim_threshold": 0.92})
    scene = [
        SemanticRegion(Rect(3.0, 0.0, 1.5, 1.2), "ice_sheet"),
        SemanticRegion(Rect(4.0, 1.8, 0.9, 0.9), "brown_mud"),
        SemanticRegion(Rect(4.0, -1.8, 0.9, 0.9), "yellow_adhesive"),
    ]
    nav = TraversabilityMap(cfg, scene=scene, segmenter=ClipSegmenter(cfg.camera, encoder=enc))
    assert nav.costmap.embed_dim == 512  # the costmap carries real CLIP features
    nav.observe(np.array([0.0, 0.0]), heading=0.0)
    res = nav.mark_failure(np.array([2.5, 0.3]))  # a cell on the ice sheet
    assert res["propagated"] > 0
    ice_elsewhere = nav.costmap.cost_at(np.array([3.5, -0.5]))
    assert ice_elsewhere >= cfg.propagation_cost - 1e-6  # the rest of the ice sheet is condemned
    assert nav.costmap.cost_at(np.array([4.0, 1.8])) < 0.5  # mud spared
    assert nav.costmap.cost_at(np.array([4.0, -1.8])) < 0.5  # adhesive spared
