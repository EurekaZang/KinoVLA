from __future__ import annotations

import numpy as np
import pytest

from kino_vla.sim.realistic_route_protocol_v2 import (
    RouteFrameV2,
    scene_route_binding,
)


def _forest() -> dict:
    return {
        "passed": True,
        "schema_version": "kinofail.forest-hybrid-compiled-scene.v7-development",
        "scene_id": "forest_g06",
        "files": {"episode.usda": "hash"},
        "route": {"waypoints_xy_m": [[0.0, -0.5], [2.5, -0.5]]},
        "physics_contract": {
            "route_geometry": {
                "direction_xy": [1.0, 0.0],
                "left_xy": [0.0, 1.0],
                "route_length_m": 2.5,
                "surface_width_m": 1.4,
            }
        },
    }


def _embodiedgen() -> dict:
    return {
        "passed": True,
        "schema_version": "kinofail.embodiedgen-compiled-scene.v4-development",
        "scene_id": "indoor_kitchen_31",
        "files": {"episode_v4.usda": "hash"},
        "route": {
            "route_selection": "robust_straight_experiment_corridor_v2",
            "collector_start_waypoint_index": 1,
            "waypoints_xy_m": [[-0.4, -0.8], [0.2, -0.8], [2.05, -0.8]],
        },
        "appearance_contract": {
            "schema": "kinofail.render-only-route-surface.v1",
            "visual_intervention_only": True,
            "collision_authored": False,
            "geometry": {
                "direction_xy": [1.0, 0.0],
                "left_xy": [0.0, 1.0],
                "route_length_m": 2.45,
                "surface_width_m": 1.2,
                "endpoint_margin_m": 0.45,
            },
        },
    }


def test_forest_binding_preserves_v1_paths() -> None:
    binding = scene_route_binding(_forest())
    assert binding.source_kind == "forest_hybrid"
    assert binding.episode_filename == "episode.usda"
    assert binding.floor_prim_path("/World/KinoIndoor") == "/World/KinoIndoor/Collision/Floor"
    assert binding.route_surface_prim_path("/World/KinoIndoor") == (
        "/World/KinoIndoor/Appearance/RouteSurface"
    )


def test_embodiedgen_v4_binding_uses_render_only_surface() -> None:
    binding = scene_route_binding(_embodiedgen())
    assert binding.source_kind == "embodiedgen_v4"
    assert binding.episode_filename == "episode_v4.usda"
    assert binding.route_surface_collision_authored is False
    assert binding.collector_start_waypoint_index == 1
    assert binding.collector_start_progress_m == pytest.approx(0.6)
    assert binding.frame.origin_xy_m == pytest.approx((0.2, -0.8))
    assert binding.frame.route_length_m == pytest.approx(2.30)
    assert binding.route_surface_prim_path("/World/KinoIndoor") == (
        "/World/KinoIndoor/AppearanceV4/RenderOnlyRouteSurface"
    )
    assert binding.frame.project(np.asarray([0.6, -0.7])) == pytest.approx((0.4, 0.1))


def test_embodiedgen_route_surface_must_be_visual_only() -> None:
    compiled = _embodiedgen()
    compiled["appearance_contract"]["collision_authored"] = True
    with pytest.raises(ValueError, match="must not author collision"):
        scene_route_binding(compiled)


def test_curved_route_is_rejected_for_axis_aligned_terrain_backend() -> None:
    compiled = _embodiedgen()
    compiled["route"]["waypoints_xy_m"][1][1] = -0.7
    with pytest.raises(ValueError, match="requires a straight route"):
        RouteFrameV2.from_compiled_audit(compiled)


def test_route_length_mismatch_is_rejected() -> None:
    compiled = _embodiedgen()
    compiled["appearance_contract"]["geometry"]["route_length_m"] = 2.4
    with pytest.raises(ValueError, match="disagree on length"):
        scene_route_binding(compiled)
