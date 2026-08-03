from __future__ import annotations

import inspect

import numpy as np

from kino_vla.eval.evidence_routed_v2 import observable_features, proprio_summary
from kino_vla.eval.material_support import MATERIAL_CLASSES, fit_material_prototypes


def _models():
    anchors = {
        "adhesion": [0.9, 0.8, 0.2],
        "compliant_terrain": [0.4, 0.3, 0.2],
        "low_friction": [0.8, 0.9, 1.0],
        "solid_ground": [0.5, 0.5, 0.5],
    }
    return {
        name: fit_material_prototypes(
            [
                {
                    "sample_id": name,
                    "appearance_id": name,
                    "semantic_class": name,
                    "material_rgb": anchors[name],
                }
            ],
            target_class=name,
        )
        for name in MATERIAL_CLASSES
    }


def test_proprio_summary_is_fixed_and_finite():
    window = np.arange(55, dtype=np.float32).reshape(5, 11)
    summary = proprio_summary(window)
    assert summary.shape == (66,)
    assert np.isfinite(summary).all()


def test_observable_features_use_only_rgb_proprio_and_frozen_models():
    signature = list(inspect.signature(observable_features).parameters)
    assert signature == ["rgb", "proprio_window", "material_models", "material_classes"]
    image = np.full((20, 20, 3), [0.88, 0.78, 0.22], dtype=np.float32)
    features = observable_features(
        image, np.zeros((25, 11), dtype=np.float32), _models(), MATERIAL_CLASSES
    )
    assert features.proprio.shape == (66,)
    assert features.material.shape == (16,)
    assert features.joint.shape == (82,)
    assert np.allclose(features.joint[-16:], features.material)
