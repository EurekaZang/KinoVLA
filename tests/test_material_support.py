import numpy as np

from kino_vla.eval.material_support import (
    MATERIAL_CLASSES,
    dominant_chromatic_rgb,
    fit_material_prototypes,
    matched_material_pair_logits,
    material_evidence_features,
    material_support_distance,
)


def test_dominant_chromatic_rgb_suppresses_neutral_background():
    image = np.full((2, 20, 20, 3), 0.5, dtype=np.float32)
    image[:, 5:15, 5:15] = [0.42, 0.28, 0.14]
    assert np.allclose(dominant_chromatic_rgb(image), [0.42, 0.28, 0.14])


def test_material_support_uses_nearest_calibration_appearance():
    rows = [
        {
            "sample_id": "a",
            "appearance_id": "brown",
            "semantic_class": "compliant_terrain",
            "material_rgb": [0.42, 0.28, 0.14],
        },
        {
            "sample_id": "b",
            "appearance_id": "gray",
            "semantic_class": "compliant_terrain",
            "material_rgb": [0.30, 0.28, 0.26],
        },
        {
            "sample_id": "c",
            "appearance_id": "ice",
            "semantic_class": "low_friction",
            "material_rgb": [0.8, 0.9, 1.0],
        },
    ]
    model = fit_material_prototypes(rows, target_class="compliant_terrain")
    assert model["n_appearances"] == 2
    assert material_support_distance([0.40, 0.28, 0.15], model) < 0.03
    assert material_support_distance([0.8, 0.9, 1.0], model) > 0.5


def test_material_evidence_exposes_relative_train_prototype_support():
    anchors = {
        "adhesion": [0.9, 0.8, 0.2],
        "compliant_terrain": [0.4, 0.3, 0.2],
        "low_friction": [0.8, 0.9, 1.0],
        "solid_ground": [0.5, 0.5, 0.5],
    }
    models = {}
    for name in MATERIAL_CLASSES:
        models[name] = fit_material_prototypes(
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
    value = material_evidence_features([0.88, 0.79, 0.22], models)
    assert value.shape == (16,)
    assert np.isclose(value[3], 0.66)
    distances = value[4:8]
    margins = value[8:12]
    assignment = value[12:16]
    assert int(np.argmin(distances)) == MATERIAL_CLASSES.index("adhesion")
    assert margins[MATERIAL_CLASSES.index("adhesion")] == 0.0
    assert int(np.argmax(assignment)) == MATERIAL_CLASSES.index("adhesion")
    assert np.isclose(assignment.sum(), 1.0)
    categories = ["adhesion", "compliant_terrain", "low_friction", "nominal"]
    logits = matched_material_pair_logits([0.88, 0.79, 0.22], models, categories)
    assert categories[int(np.argmax(logits))] == "adhesion"
    assert logits[categories.index("low_friction")] < logits[categories.index("adhesion")]
