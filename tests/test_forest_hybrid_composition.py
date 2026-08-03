from __future__ import annotations

import pytest

from kino_vla.sim.forest_hybrid_composition import (
    _deep_merge,
    _load_composition_config,
    _author_naturalized_route_mesh,
    _author_feathered_route_mesh,
    _procedural_ground_height,
    _route_blend_mask,
    surrounding_rectangles,
)

import numpy as np
from kino_vla.sim.visual_shell_composition import route_surface_geometry


def test_surrounding_rectangles_partition_around_route() -> None:
    route = route_surface_geometry(
        {
            "waypoints_xy_m": [[0.0, 0.0], [4.0, 0.0]],
            "surface_width_m": 1.4,
            "endpoint_margin_m": 0.75,
            "floor_thickness_m": 0.08,
        }
    )
    rectangles = surrounding_rectangles([[-3.0, -8.0], [10.0, 8.0]], route)
    assert [item["id"] for item in rectangles] == [
        "left_bank",
        "right_bank",
        "entry_ground",
        "exit_ground",
    ]
    route_area = route["surface_length_m"] * route["surface_width_m"]
    surrounding_area = sum(
        (item["xmax"] - item["xmin"]) * (item["ymax"] - item["ymin"])
        for item in rectangles
    )
    assert surrounding_area + route_area == pytest.approx(13.0 * 16.0)


def test_surrounding_rectangles_reject_extent_that_does_not_contain_route() -> None:
    route = route_surface_geometry(
        {
            "waypoints_xy_m": [[0.0, 0.0], [4.0, 0.0]],
            "surface_width_m": 1.4,
            "endpoint_margin_m": 0.75,
            "floor_thickness_m": 0.08,
        }
    )
    with pytest.raises(ValueError, match="strictly contain"):
        surrounding_rectangles([[0.0, -1.0], [4.0, 1.0]], route)


def test_deep_merge_preserves_base_and_replaces_nested_values() -> None:
    result = _deep_merge(
        {"background": {"intensity": 420.0, "type": "pano"}, "scene_id": "forest"},
        {"background": {"intensity": 280.0}, "output": {"directory": "v3"}},
    )
    assert result == {
        "background": {"intensity": 280.0, "type": "pano"},
        "scene_id": "forest",
        "output": {"directory": "v3"},
    }


def test_visual_ground_variation_is_flat_at_route_boundary() -> None:
    profile = {
        "amplitude_m": 0.05,
        "wavelength_m": 2.2,
        "route_boundary_fade_m": 0.8,
        "phase_rad": 0.3,
    }
    bounds = (-0.75, -0.7, 4.75, 0.7)
    assert _procedural_ground_height(
        1.0, 0.7, base_z=-0.002, profile=profile, route_bounds=bounds
    ) == pytest.approx(-0.002)


def test_visual_ground_variation_is_deterministic_and_nonflat_away_from_route() -> None:
    profile = {
        "amplitude_m": 0.05,
        "wavelength_m": 2.2,
        "route_boundary_fade_m": 0.8,
        "phase_rad": 0.3,
    }
    bounds = (-0.75, -0.7, 4.75, 0.7)
    first = _procedural_ground_height(
        2.2, 2.4, base_z=-0.002, profile=profile, route_bounds=bounds
    )
    second = _procedural_ground_height(
        2.2, 2.4, base_z=-0.002, profile=profile, route_bounds=bounds
    )
    assert first == second
    assert first != pytest.approx(-0.002)


def test_recursive_composition_config_loading_preserves_v4_overrides() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    config, lineage = _load_composition_config(
        root / "configs/data/kinofail_forest_hybrid_composition_dev_v4.json", root=root
    )
    assert config["schema_version"].endswith("v4-development")
    assert config["appearance"]["route_material_id"] == "train_ground054"
    assert len(config["near_field"]["instances"]) == 28
    assert len(lineage) == 1


def test_naturalized_route_helper_is_exposed_for_usd_compile() -> None:
    assert callable(_author_naturalized_route_mesh)
    assert callable(_author_feathered_route_mesh)


def test_opaque_route_mask_blends_across_physical_boundary() -> None:
    route = route_surface_geometry(
        {
            "waypoints_xy_m": [[0.0, 0.0], [4.0, 0.0]],
            "surface_width_m": 1.4,
            "endpoint_margin_m": 0.75,
            "floor_thickness_m": 0.08,
        }
    )
    x = np.full((1, 4), 2.0)
    y = np.asarray([[0.0, 0.7, 0.9, 1.2]])
    mask = _route_blend_mask(
        x,
        y,
        route_geometry=route,
        profile={
            "core_half_width_m": 0.43,
            "outer_half_width_m": 1.02,
            "width_variation_m": 0.0,
            "centerline_wander_m": 0.0,
            "noise_amplitude_m": 0.0,
            "wavelength_m": 4.2,
            "phase_rad": 0.0,
        },
    )
    assert mask[0, 0] == pytest.approx(1.0)
    assert 0.0 < mask[0, 1] < 1.0
    assert mask[0, 1] > mask[0, 2]
    assert mask[0, 3] == pytest.approx(0.0)


def test_v12_opaque_config_inherits_polyhaven_materials() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    config, lineage = _load_composition_config(
        root / "configs/data/kinofail_forest_hybrid_composition_dev_v12.json", root=root
    )
    assert config["near_field"]["opaque_composited_ground"]["enabled"] is True
    assert config["appearance"]["route_material_id"] == "train_polyhaven_dirt_2k"
    assert config["appearance"]["surrounding_material_id"] == (
        "train_polyhaven_forest_leaves02_2k"
    )
    assert len(lineage) >= 2


def test_v14_collision_policy_classifies_every_used_prop_asset() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    config, _ = _load_composition_config(
        root / "configs/data/kinofail_forest_hybrid_composition_dev_v14.json", root=root
    )
    policy = config["near_field"]["prop_collision_policy"]
    classified = set(policy["rigid_asset_ids"]) | set(
        policy["flexible_noncolliding_asset_ids"]
    )
    used = {item["asset_id"] for item in config["near_field"]["instances"]}
    assert classified == used
    assert not set(policy["rigid_asset_ids"]) & set(
        policy["flexible_noncolliding_asset_ids"]
    )


def test_v15_and_v16_freeze_train_and_heldout_appearance_splits() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    v15, _ = _load_composition_config(
        root / "configs/data/kinofail_forest_hybrid_composition_dev_v15.json",
        root=root,
    )
    v16, _ = _load_composition_config(
        root / "configs/data/kinofail_forest_hybrid_composition_dev_v16.json",
        root=root,
    )

    assert v15["appearance"]["route_expected_split"] == "train"
    assert v15["appearance"]["surrounding_expected_split"] == "train"
    assert v16["appearance"]["route_expected_split"] == "train"
    assert v16["appearance"]["surrounding_expected_split"] == "val"
    assert v15["background"]["panorama"]["sha256"] != v16["background"][
        "panorama"
    ]["sha256"]


def test_v17_changes_only_heldout_lighting_after_v16_visual_failure() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    v16, _ = _load_composition_config(
        root / "configs/data/kinofail_forest_hybrid_composition_dev_v16.json",
        root=root,
    )
    v17, _ = _load_composition_config(
        root / "configs/data/kinofail_forest_hybrid_composition_dev_v17.json",
        root=root,
    )

    assert v17["background"] == v16["background"]
    assert v17["route"] == v16["route"]
    assert v17["near_field"] == v16["near_field"]
    assert v17["appearance"]["distant_light_intensity"] == 70.0
    assert v16["appearance"]["distant_light_intensity"] == 35.0
    assert v17["appearance"]["surrounding_expected_split"] == "val"


def test_v18_uses_preflight_radiance_normalization_without_physics_changes() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    v17, _ = _load_composition_config(
        root / "configs/data/kinofail_forest_hybrid_composition_dev_v17.json",
        root=root,
    )
    v18, _ = _load_composition_config(
        root / "configs/data/kinofail_forest_hybrid_composition_dev_v18.json",
        root=root,
    )

    assert v18["background"]["intensity"] == 455.0
    assert v18["background"]["panorama"] == v17["background"]["panorama"]
    assert v18["route"] == v17["route"]
    assert v18["near_field"] == v17["near_field"]
    assert v18["appearance"] == v17["appearance"]
