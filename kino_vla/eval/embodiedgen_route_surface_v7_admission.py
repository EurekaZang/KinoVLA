"""Pure admission logic for held-out route-surface EmbodiedGen scenes.

The v7 gate joins source geometry, inherited corridor physics, render-only
terrain appearance, RTX scene QA, and articulated-Go2 QA.  File hashing stays
in the command-line auditor so this decision contract remains unit-testable
without Isaac Sim.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


SCHEMA_VERSION = "kinofail.embodiedgen-route-surface-v7-admission-audit.v1"
COMPILED_SCHEMA = "kinofail.embodiedgen-compiled-scene.v4-development"
CORRIDOR_SCHEMA = "kinofail.embodiedgen-compiled-scene.v2"
SOURCE_PREFLIGHT_SCHEMA = "kinofail.embodiedgen-source-geometry-preflight.v1"
RTX_SCHEMA = "kinofail.embodiedgen-rtx-scene-qa.v5-development"
GO2_SCHEMA = "kinofail.embodiedgen-articulated-go2-qa.v2"
APPEARANCE_SCHEMA = "kinofail.render-only-route-surface.v1"
LIGHTING_SCHEMA = "kinofail.bidirectional-route-envelope-lighting.v1"
VIEW_SCHEMA = "kinofail.embodiedgen-phase-aware-scene-views.v5"
PLACEMENT_CONTRACT = "route_relative_from_collector_start_v2"
ALLOWED_HELDOUT_SPLITS = ("val", "test")


def _finite_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def evaluate_route_surface_v7_admission(
    *,
    source_preflight: dict[str, Any],
    compiled: dict[str, Any],
    corridor: dict[str, Any],
    rtx: dict[str, Any],
    go2: dict[str, Any],
    episode_sha256: str,
    compiled_sha256: str,
    corridor_sha256: str,
    source_manifest_sha256: str,
    base_audit_verified: bool,
    material_lock_verified: bool,
    material_maps_verified: bool,
    allowed_material_splits: Iterable[str] = ALLOWED_HELDOUT_SPLITS,
) -> dict[str, Any]:
    """Evaluate the immutable evidence bindings for one held-out scene cell."""

    route = compiled.get("route", {})
    appearance = compiled.get("appearance_contract", {})
    lighting = compiled.get("lighting_contract", {})
    material = appearance.get("material", {})
    structural = rtx.get("structural_checks", {})
    visual = rtx.get("visual_checks", {})
    go2_checks = go2.get("checks", {})
    clip = rtx.get("camera_intrinsics", {}).get("calibrated_clipping_range_m", [])
    near_clip = _finite_float(clip[0] if clip else None, math.inf)
    tracking_budget = _finite_float(
        route.get("maximum_admissible_tracking_error_m"), -math.inf
    )
    measured_clearance = _finite_float(route.get("measured_min_clearance_m"), -math.inf)
    robust_clearance = _finite_float(route.get("robust_centerline_clearance_m"), math.inf)
    observed_tracking = _finite_float(go2.get("max_route_deviation_m"), math.inf)
    allowed_splits = set(allowed_material_splits)

    checks = {
        "source_preflight_schema": source_preflight.get("schema_version")
        == SOURCE_PREFLIGHT_SCHEMA,
        "source_preflight_passed": source_preflight.get("passed") is True,
        "source_manifest_hash_verified": bool(source_manifest_sha256)
        and compiled.get("source_manifest_sha256") == source_manifest_sha256
        and source_preflight.get("source_manifest_sha256") == source_manifest_sha256,
        "compiled_schema_v4": compiled.get("schema_version") == COMPILED_SCHEMA,
        "compiled_passed": compiled.get("passed") is True,
        "corridor_schema_v2": corridor.get("schema_version") == CORRIDOR_SCHEMA,
        "corridor_passed": corridor.get("passed") is True,
        "corridor_hash_verified": bool(corridor_sha256)
        and compiled.get("corridor_v2_audit_sha256") == corridor_sha256,
        "base_audit_hash_verified": bool(base_audit_verified),
        "scene_identity_preserved": bool(compiled.get("scene_id"))
        and compiled.get("scene_id") == corridor.get("scene_id")
        and compiled.get("scene_id") == source_preflight.get("scene_id"),
        "route_inherited_unchanged": compiled.get("route") == corridor.get("route"),
        "operator_inherited_unchanged": compiled.get("operator")
        == corridor.get("operator"),
        "physics_inherited_unchanged": compiled.get("physics_contract")
        == corridor.get("physics_contract"),
        "route_relative_o4_placement": compiled.get("operator", {}).get(
            "placement_contract"
        )
        == PLACEMENT_CONTRACT,
        "robust_clearance_contract": tracking_budget > 0.0
        and measured_clearance + 1.0e-6 >= robust_clearance,
        "episode_hash_verified": bool(episode_sha256)
        and compiled.get("files", {}).get("episode_v4.usda") == episode_sha256,
        "appearance_schema": appearance.get("schema") == APPEARANCE_SCHEMA,
        "appearance_is_visual_only": appearance.get("visual_intervention_only") is True,
        "appearance_authors_no_collision": appearance.get("collision_authored") is False,
        "appearance_render_purpose": appearance.get("purpose") == "render",
        "heldout_material_split": material.get("split") in allowed_splits,
        "material_lock_hash_verified": bool(material_lock_verified),
        "material_map_hashes_verified": bool(material_maps_verified),
        "lighting_schema": lighting.get("schema") == LIGHTING_SCHEMA,
        "lighting_is_visual_only": lighting.get("visual_intervention_only") is True,
        "rtx_schema_v5": rtx.get("schema_version") == RTX_SCHEMA,
        "rtx_passed": rtx.get("passed") is True,
        "rtx_binds_compiled": rtx.get("compiled_audit_sha256") == compiled_sha256,
        "rtx_binds_episode": rtx.get("episode_usd_sha256") == episode_sha256,
        "rtx_structural_contract_passed": bool(structural)
        and all(value is True for value in structural.values()),
        "rtx_visual_contract_passed": bool(visual)
        and all(value is True for value in visual.values()),
        "seven_phase_aware_views": rtx.get("view_contract_schema") == VIEW_SCHEMA
        and len(rtx.get("views", [])) == 7,
        "near_clip_calibrated": 0.009 <= near_clip <= 0.011,
        "go2_schema_v2": go2.get("schema_version") == GO2_SCHEMA,
        "go2_passed": go2.get("passed") is True,
        "go2_binds_compiled": go2.get("compiled_audit_sha256") == compiled_sha256,
        "go2_binds_episode": go2.get("episode_usd_sha256") == episode_sha256,
        "go2_checks_passed": bool(go2_checks)
        and all(value is True for value in go2_checks.values()),
        "go2_tracking_within_frozen_budget": observed_tracking <= tracking_budget,
        "go2_stable_tilt": _finite_float(go2.get("max_tilt_rad"), math.inf) <= 0.55,
        "go2_stable_height": _finite_float(
            go2.get("min_base_height_m"), -math.inf
        )
        >= 0.20,
        "go2_no_fall": go2_checks.get("robot_did_not_fall") is True,
        "actual_body_fixed_front_camera": go2.get("camera", {}).get("role")
        == "actual body-fixed Go2 front RTX RGB",
    }
    issues = [name for name, passed in checks.items() if not passed]
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "issues": issues,
        "primary_failed_gate": issues[0] if issues else None,
        "material": {
            "id": material.get("id"),
            "semantic_family": material.get("semantic_family"),
            "split": material.get("split"),
            "allowed_heldout_splits": sorted(allowed_splits),
        },
        "clearance_contract_m": {
            "maximum_tracking_error": tracking_budget,
            "required_robust_centerline_clearance": robust_clearance,
            "measured_centerline_clearance": measured_clearance,
            "observed_go2_tracking_error": observed_tracking,
        },
    }
