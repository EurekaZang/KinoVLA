from __future__ import annotations

from copy import deepcopy

from kino_vla.eval.embodiedgen_route_surface_v7_admission import (
    evaluate_route_surface_v7_admission,
)


def _evidence(split: str = "val") -> dict:
    route = {
        "maximum_admissible_tracking_error_m": 0.3,
        "measured_min_clearance_m": 0.9,
        "robust_centerline_clearance_m": 0.74,
    }
    operator = {"placement_contract": "route_relative_from_collector_start_v2"}
    physics = {"owned_by": "Kino-Fail", "embodiedgen_collision_used": False}
    corridor = {
        "schema_version": "kinofail.embodiedgen-compiled-scene.v2",
        "scene_id": "indoor_office_01",
        "passed": True,
        "route": route,
        "operator": operator,
        "physics_contract": physics,
    }
    compiled = {
        "schema_version": "kinofail.embodiedgen-compiled-scene.v4-development",
        "scene_id": "indoor_office_01",
        "source_manifest_sha256": "manifest",
        "corridor_v2_audit_sha256": "corridor",
        "passed": True,
        "route": deepcopy(route),
        "operator": deepcopy(operator),
        "physics_contract": deepcopy(physics),
        "appearance_contract": {
            "schema": "kinofail.render-only-route-surface.v1",
            "visual_intervention_only": True,
            "collision_authored": False,
            "purpose": "render",
            "material": {"id": f"{split}_ground", "split": split},
        },
        "lighting_contract": {
            "schema": "kinofail.bidirectional-route-envelope-lighting.v1",
            "visual_intervention_only": True,
        },
        "files": {"episode_v4.usda": "episode"},
    }
    source_preflight = {
        "schema_version": "kinofail.embodiedgen-source-geometry-preflight.v1",
        "scene_id": "indoor_office_01",
        "source_manifest_sha256": "manifest",
        "passed": True,
    }
    rtx = {
        "schema_version": "kinofail.embodiedgen-rtx-scene-qa.v5-development",
        "passed": True,
        "compiled_audit_sha256": "compiled",
        "episode_usd_sha256": "episode",
        "structural_checks": {"visual_only": True},
        "visual_checks": {"sensor_valid": True},
        "view_contract_schema": "kinofail.embodiedgen-phase-aware-scene-views.v5",
        "views": [{} for _ in range(7)],
        "camera_intrinsics": {"calibrated_clipping_range_m": [0.01, 1000.0]},
    }
    go2 = {
        "schema_version": "kinofail.embodiedgen-articulated-go2-qa.v2",
        "passed": True,
        "compiled_audit_sha256": "compiled",
        "episode_usd_sha256": "episode",
        "max_route_deviation_m": 0.2,
        "max_tilt_rad": 0.3,
        "min_base_height_m": 0.25,
        "checks": {"robot_did_not_fall": True, "front_camera_rigid_mount": True},
        "camera": {"role": "actual body-fixed Go2 front RTX RGB"},
    }
    return {
        "source_preflight": source_preflight,
        "compiled": compiled,
        "corridor": corridor,
        "rtx": rtx,
        "go2": go2,
        "episode_sha256": "episode",
        "compiled_sha256": "compiled",
        "corridor_sha256": "corridor",
        "source_manifest_sha256": "manifest",
        "base_audit_verified": True,
        "material_lock_verified": True,
        "material_maps_verified": True,
    }


def test_clean_heldout_scene_passes() -> None:
    result = evaluate_route_surface_v7_admission(**_evidence())
    assert result["passed"], result


def test_train_material_is_not_heldout_confirmation() -> None:
    result = evaluate_route_surface_v7_admission(**_evidence("train"))
    assert not result["passed"]
    assert "heldout_material_split" in result["issues"]


def test_visual_surface_cannot_change_inherited_physics() -> None:
    evidence = _evidence()
    evidence["compiled"]["physics_contract"]["proxy_count"] = 9
    result = evaluate_route_surface_v7_admission(**evidence)
    assert not result["passed"]
    assert "physics_inherited_unchanged" in result["issues"]


def test_stale_rtx_and_excess_go2_tracking_fail_closed() -> None:
    evidence = _evidence()
    evidence["rtx"]["episode_usd_sha256"] = "stale"
    evidence["go2"]["max_route_deviation_m"] = 0.31
    result = evaluate_route_surface_v7_admission(**evidence)
    assert not result["passed"]
    assert "rtx_binds_episode" in result["issues"]
    assert "go2_tracking_within_frozen_budget" in result["issues"]


def test_default_near_clip_regression_is_rejected() -> None:
    evidence = _evidence()
    evidence["rtx"]["camera_intrinsics"]["calibrated_clipping_range_m"][0] = 1.0
    result = evaluate_route_surface_v7_admission(**evidence)
    assert not result["passed"]
    assert "near_clip_calibrated" in result["issues"]
