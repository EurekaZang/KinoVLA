from __future__ import annotations

from kino_vla.eval.realistic_route_scene_contract import (
    audit_realistic_route_scene_contract,
)


def _embodiedgen() -> dict:
    return {
        "passed": True,
        "schema_version": "kinofail.embodiedgen-terrain-route-compiled-scene.v2-development",
        "scene_id": "indoor_office_27",
        "files": {"episode_terrain_v2.usda": "hash"},
        "route": {
            "collector_start_waypoint_index": 1,
            "waypoints_xy_m": [[0.0, 0.0], [0.55, 0.0], [2.45, 0.0]],
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
        "physics_contract": {
            "owned_by": "Kino-Fail",
            "embodiedgen_collision_used": False,
            "proxy_count": 11,
            "nominal_floor_material": {
                "owned_by": "Kino-Fail",
                "api_schema": "PhysicsMaterialAPI",
                "static_friction": 0.8,
                "dynamic_friction": 0.6,
                "operator_layer_may_override": True,
            },
            "dense_route_surface": {
                "collision_authored": False,
                "opaque_composited_ground": True,
                "vertex_count": 2635,
                "quad_count": 2520,
                "maximum_spacing_m": 0.04,
            },
        },
    }


def test_embodiedgen_terrain_v2_contract_passes() -> None:
    result = audit_realistic_route_scene_contract(_embodiedgen())
    assert result["passed"] is True
    assert result["source_kind"] == "embodiedgen_terrain_v2"


def test_contract_fails_without_kino_collision_authority() -> None:
    compiled = _embodiedgen()
    compiled["physics_contract"]["embodiedgen_collision_used"] = True
    result = audit_realistic_route_scene_contract(compiled)
    assert result["passed"] is False
    assert result["checks"]["kino_owns_embodiedgen_collision_contract"] is False


def test_contract_fails_coarse_operator_surface() -> None:
    compiled = _embodiedgen()
    compiled["physics_contract"]["dense_route_surface"]["maximum_spacing_m"] = 0.08
    result = audit_realistic_route_scene_contract(compiled)
    assert result["passed"] is False
    assert result["checks"]["dense_render_surface_is_operator_ready"] is False
