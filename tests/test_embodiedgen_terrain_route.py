from __future__ import annotations

import pytest

from kino_vla.sim.embodiedgen_terrain_route import (
    episode_layer_text,
    floor_material_layer_text,
)


def test_floor_material_layer_authors_explicit_readback() -> None:
    text = floor_material_layer_text(static_friction=0.8, dynamic_friction=0.6)
    assert 'prepend apiSchemas = ["PhysicsMaterialAPI"]' in text
    assert "float physics:staticFriction = 0.8" in text
    assert "float physics:dynamicFriction = 0.6" in text
    assert "kino:operatorLayerMayOverride = 1" in text


def test_floor_material_rejects_unphysical_order() -> None:
    with pytest.raises(ValueError, match="dynamic_friction"):
        floor_material_layer_text(static_friction=0.5, dynamic_friction=0.7)


def test_episode_places_material_override_before_base(tmp_path) -> None:
    output = tmp_path / "terrain"
    output.mkdir()
    material = output / "nominal_floor_material.usda"
    base = tmp_path / "base" / "episode_v4.usda"
    base.parent.mkdir()
    text = episode_layer_text(material_layer=material, base_episode=base, output_dir=output)
    assert text.index("nominal_floor_material.usda") < text.index("../base/episode_v4.usda")
