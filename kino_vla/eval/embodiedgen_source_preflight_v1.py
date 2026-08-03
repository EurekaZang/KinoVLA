"""Pure admission logic for EmbodiedGen source-geometry preflight audits.

The USD traversal lives in the Isaac-side command-line entrypoint.  Keeping the
decision logic here makes the frozen thresholds independently testable without
requiring Kit or a GPU.
"""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "kinofail.embodiedgen-source-geometry-preflight.v1"
MIN_VISUAL_MESHES = 5
MIN_VISUAL_FACES = 1000
MAX_ABS_INDOOR_COORDINATE_M = 100.0
METRES_PER_UNIT = 1.0


def evaluate_source_geometry_preflight(
    scan: dict[str, Any],
    *,
    source_integrity_passed: bool,
) -> dict[str, Any]:
    """Evaluate a USD scan using the compiler's existing geometry thresholds."""

    try:
        metres_per_unit = float(scan.get("metres_per_unit"))
    except (TypeError, ValueError):
        metres_per_unit = float("inf")
    checks = {
        "source_integrity_passed": bool(source_integrity_passed),
        "stage_opened": bool(scan.get("stage_opened")),
        "default_prim_present": bool(scan.get("default_prim_present")),
        "z_up": scan.get("up_axis") == "Z",
        "metres_per_unit_is_one": abs(metres_per_unit - METRES_PER_UNIT) <= 1.0e-6,
        "all_world_transforms_plausible": not scan.get("invalid_transform_prims", []),
        "all_metric_bounds_plausible": not scan.get("invalid_bound_prims", []),
        "minimum_visual_meshes": int(scan.get("mesh_count", 0)) >= MIN_VISUAL_MESHES,
        "minimum_visual_faces": int(scan.get("face_count", 0)) >= MIN_VISUAL_FACES,
        "metric_floor_present": bool(scan.get("floor_candidates", [])),
    }
    issue_names = {
        "source_integrity_passed": "source_integrity_failed",
        "stage_opened": "stage_unreadable",
        "default_prim_present": "default_prim_missing",
        "z_up": "source_not_z_up",
        "metres_per_unit_is_one": "source_not_authored_in_metres",
        "all_world_transforms_plausible": "implausible_world_transform",
        "all_metric_bounds_plausible": "implausible_metric_bounds",
        "minimum_visual_meshes": "insufficient_visual_meshes",
        "minimum_visual_faces": "insufficient_visual_faces",
        "metric_floor_present": "metric_floor_missing",
    }
    issues = [issue_names[name] for name, passed in checks.items() if not passed]
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "issues": issues,
        "primary_failed_gate": issues[0] if issues else None,
        "thresholds": {
            "minimum_visual_meshes": MIN_VISUAL_MESHES,
            "minimum_visual_faces": MIN_VISUAL_FACES,
            "maximum_absolute_indoor_coordinate_m": MAX_ABS_INDOOR_COORDINATE_M,
            "metres_per_unit": METRES_PER_UNIT,
        },
    }
