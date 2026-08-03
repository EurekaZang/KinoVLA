from __future__ import annotations

from kino_vla.eval.embodiedgen_source_preflight_v1 import (
    MAX_ABS_INDOOR_COORDINATE_M,
    MIN_VISUAL_FACES,
    MIN_VISUAL_MESHES,
    evaluate_source_geometry_preflight,
)


def _scan() -> dict:
    return {
        "stage_opened": True,
        "default_prim_present": True,
        "up_axis": "Z",
        "metres_per_unit": 1.0,
        "mesh_count": MIN_VISUAL_MESHES,
        "point_count": 800,
        "face_count": MIN_VISUAL_FACES,
        "invalid_transform_prims": [],
        "invalid_bound_prims": [],
        "floor_candidates": [{"path": "/World/room_floor"}],
    }


def test_clean_source_passes_at_frozen_compiler_thresholds() -> None:
    result = evaluate_source_geometry_preflight(_scan(), source_integrity_passed=True)
    assert result["passed"], result
    assert result["thresholds"]["maximum_absolute_indoor_coordinate_m"] == (
        MAX_ABS_INDOOR_COORDINATE_M
    )


def test_implausible_transform_is_rejected_before_compile() -> None:
    scan = _scan()
    scan["invalid_transform_prims"] = [
        {"path": "/World/window", "translation_xyz_m": [1.0e23, 0.0, 1.0]}
    ]
    result = evaluate_source_geometry_preflight(scan, source_integrity_passed=True)
    assert not result["passed"]
    assert "implausible_world_transform" in result["issues"]


def test_low_face_source_is_rejected_before_compile() -> None:
    scan = _scan()
    scan["face_count"] = MIN_VISUAL_FACES - 1
    result = evaluate_source_geometry_preflight(scan, source_integrity_passed=True)
    assert not result["passed"]
    assert "insufficient_visual_faces" in result["issues"]


def test_integrity_and_metric_floor_are_mandatory() -> None:
    scan = _scan()
    scan["floor_candidates"] = []
    result = evaluate_source_geometry_preflight(scan, source_integrity_passed=False)
    assert not result["passed"]
    assert result["issues"] == ["source_integrity_failed", "metric_floor_missing"]


def test_unreadable_stage_fails_closed_without_type_error() -> None:
    scan = _scan()
    scan.update(
        {
            "stage_opened": False,
            "default_prim_present": False,
            "up_axis": None,
            "metres_per_unit": None,
            "mesh_count": 0,
            "face_count": 0,
            "floor_candidates": [],
        }
    )
    result = evaluate_source_geometry_preflight(scan, source_integrity_passed=True)
    assert not result["passed"]
    assert result["primary_failed_gate"] == "stage_unreadable"
    assert "source_not_authored_in_metres" in result["issues"]
