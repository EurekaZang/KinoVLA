"""Cross-domain scene contract used by realistic terrain-operator pair audits."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kino_vla.sim.realistic_route_protocol_v2 import scene_route_binding


SCHEMA_VERSION = "kinofail.realistic-route-scene-contract.v1"


def audit_realistic_route_scene_contract(compiled: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the collision/render contract without assuming a forest schema."""
    try:
        binding = scene_route_binding(compiled)
    except (KeyError, TypeError, ValueError) as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "passed": False,
            "source_kind": None,
            "checks": {"route_binding_valid": False},
            "error": str(error),
        }

    checks: dict[str, bool] = {
        "route_binding_valid": True,
        "compiled_scene_passed": compiled.get("passed") is True,
        "route_surface_collision_not_authored": binding.route_surface_collision_authored
        is False,
        "positive_operator_route_length": binding.frame.route_length_m > 0.0,
        "positive_route_surface_width": binding.frame.surface_width_m > 0.0,
    }
    if binding.source_kind == "forest_hybrid":
        near_field = compiled.get("near_field_contract", {})
        checks.update(
            {
                "forest_prop_collision_proxies_complete": near_field.get(
                    "prop_collision_proxies_complete"
                )
                is True,
                "forest_physics_contract_present": bool(compiled.get("physics_contract")),
            }
        )
    elif binding.source_kind == "embodiedgen_terrain_v2":
        physics = compiled.get("physics_contract", {})
        nominal = physics.get("nominal_floor_material", {})
        dense = physics.get("dense_route_surface", {})
        static = float(nominal.get("static_friction", -1.0))
        dynamic = float(nominal.get("dynamic_friction", -1.0))
        checks.update(
            {
                "kino_owns_embodiedgen_collision_contract": physics.get("owned_by")
                == "Kino-Fail"
                and physics.get("embodiedgen_collision_used") is False,
                "collision_proxy_contract_nonempty": int(physics.get("proxy_count", 0)) >= 5,
                "explicit_floor_physics_material": nominal.get("owned_by") == "Kino-Fail"
                and nominal.get("api_schema") == "PhysicsMaterialAPI"
                and 0.0 <= dynamic <= static <= 2.0,
                "operator_may_override_floor_material": nominal.get(
                    "operator_layer_may_override"
                )
                is True,
                "dense_render_surface_is_operator_ready": dense.get("collision_authored")
                is False
                and dense.get("opaque_composited_ground") is True
                and int(dense.get("vertex_count", 0)) >= 1000
                and int(dense.get("quad_count", 0)) > 0
                and float(dense.get("maximum_spacing_m", 1.0)) <= 0.04 + 1.0e-9,
            }
        )
    else:
        checks["supported_operator_scene_source"] = False

    return {
        "schema_version": SCHEMA_VERSION,
        "passed": all(checks.values()),
        "source_kind": binding.source_kind,
        "scene_id": binding.scene_id,
        "checks": checks,
        "binding": {
            "floor_prim_suffix": binding.floor_prim_suffix,
            "route_surface_prim_suffix": binding.route_surface_prim_suffix,
            "collector_start_waypoint_index": binding.collector_start_waypoint_index,
            "available_route_length_m": binding.frame.route_length_m,
            "surface_width_m": binding.frame.surface_width_m,
        },
    }
