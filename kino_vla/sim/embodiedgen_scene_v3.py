"""Pitch-aware lighting refinement for frozen EmbodiedGen corridor-v2 scenes.

The corridor-v2 compiler and every file referenced by its sealed batches remain
immutable.  This module consumes a passed v2 audit, preserves its route and
Kino-owned collision layer, and replaces only the Kino-owned lighting layer.
The resulting episode is intended for development until a lighting candidate is
selected and frozen on scenes excluded from future confirmation batches.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "kinofail.embodiedgen-compiled-scene.v3-development"
PITCH_AWARE_VIEW_SCHEMA = "kinofail.embodiedgen-pitch-aware-views.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _direction_and_left(
    origin: tuple[float, float], target: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    dx = float(target[0]) - float(origin[0])
    dy = float(target[1]) - float(origin[1])
    length = math.hypot(dx, dy)
    if length <= 1.0e-9:
        raise ValueError("camera view requires distinct route points")
    direction = (dx / length, dy / length)
    return direction, (-direction[1], direction[0])


def pitch_aware_camera_views(
    route: list[list[float]], floor_z_m: float
) -> list[dict[str, Any]]:
    """Return four canonical and three body-pitch stress views.

    The three stress views share the Go2 camera height and progressively shorten
    the ground intersection distance, approximating 45, 70 and 80 degree body
    pitch without requiring a robot simulation.  They are scene-admission views,
    not extra benchmark samples.
    """
    if len(route) < 5:
        raise ValueError("pitch-aware QA requires the five-point corridor-v2 route")
    points = [tuple(float(value) for value in point) for point in route]
    start, second = points[0], points[1]
    middle_index = len(points) // 2
    middle = points[middle_index]
    middle_next = points[middle_index + 1]
    reverse_origin, reverse_target = points[-2], points[-3]
    entry_direction, entry_left = _direction_and_left(start, second)
    middle_direction, middle_left = _direction_and_left(middle, middle_next)
    reverse_direction, reverse_left = _direction_and_left(reverse_origin, reverse_target)

    def xyz(point: tuple[float, float], z: float) -> list[float]:
        return [float(point[0]), float(point[1]), float(z)]

    views = [
        {
            "name": "01_entry_oblique",
            "eye_xyz_m": [
                start[0] + 0.42 * entry_left[0],
                start[1] + 0.42 * entry_left[1],
                floor_z_m + 1.30,
            ],
            "target_xyz_m": [
                start[0] + 1.40 * entry_direction[0],
                start[1] + 1.40 * entry_direction[1],
                floor_z_m + 0.62,
            ],
            "kind": "canonical_review",
        },
        {
            "name": "02_route_mid",
            "eye_xyz_m": [
                middle[0] - 0.30 * middle_direction[0] - 0.42 * middle_left[0],
                middle[1] - 0.30 * middle_direction[1] - 0.42 * middle_left[1],
                floor_z_m + 1.25,
            ],
            "target_xyz_m": [
                middle[0] + 1.25 * middle_direction[0],
                middle[1] + 1.25 * middle_direction[1],
                floor_z_m + 0.58,
            ],
            "kind": "canonical_review",
        },
        {
            "name": "03_reverse",
            "eye_xyz_m": [
                reverse_origin[0] + 0.36 * reverse_left[0],
                reverse_origin[1] + 0.36 * reverse_left[1],
                floor_z_m + 1.30,
            ],
            "target_xyz_m": [
                reverse_origin[0] + 1.25 * reverse_direction[0],
                reverse_origin[1] + 1.25 * reverse_direction[1],
                floor_z_m + 0.60,
            ],
            "kind": "canonical_review",
        },
        {
            "name": "04_go2_front_height",
            "eye_xyz_m": xyz(middle, floor_z_m + 0.42),
            "target_xyz_m": xyz(middle_next, floor_z_m + 0.34),
            "kind": "go2_front_height_proxy",
        },
    ]
    eye = xyz(middle, floor_z_m + 0.42)
    for index, (label, distance_m) in enumerate(
        (("pitch45", 0.42), ("pitch70", 0.153), ("pitch80", 0.074)), start=5
    ):
        target = [
            middle[0] + distance_m * middle_direction[0],
            middle[1] + distance_m * middle_direction[1],
            floor_z_m,
        ]
        views.append(
            {
                "name": f"{index:02d}_go2_{label}",
                "eye_xyz_m": list(eye),
                "target_xyz_m": target,
                "kind": "go2_body_pitch_stress",
                "nominal_pitch_down_deg": math.degrees(math.atan2(0.42, distance_m)),
            }
        )
    return views


def evaluate_pitch_aware_metrics(
    records: list[dict[str, Any]], canonical_pairwise_l1: dict[str, float]
) -> dict[str, bool]:
    """Apply frozen candidate-independent image-quality gates."""
    stress = [record for record in records if record["kind"] == "go2_body_pitch_stress"]
    return {
        "seven_views_captured": len(records) == 7,
        "three_body_pitch_stress_views": len(stress) == 3,
        "non_degenerate_luminance": all(
            record["metrics"]["std_luminance"] >= 0.035 for record in records
        ),
        "usable_dynamic_range": all(
            record["metrics"]["p99_luminance"]
            - record["metrics"]["p01_luminance"]
            >= 0.12
            for record in records
        ),
        "not_black_or_white": all(
            record["metrics"]["black_fraction"] < 0.90
            and record["metrics"]["white_fraction"] < 0.90
            for record in records
        ),
        "appearance_complexity": all(
            record["metrics"]["quantized_color_count_5bit"] >= 64
            for record in records
        ),
        "canonical_viewpoints_not_stale": bool(canonical_pairwise_l1)
        and min(canonical_pairwise_l1.values()) >= 0.01,
        "go2_height_view_present": any(
            record["kind"] == "go2_front_height_proxy" for record in records
        ),
    }


def refine_embodiedgen_kinofail_scene_v3(
    corridor_v2_audit: str | Path,
    output_dir: str | Path,
    *,
    candidate_id: str,
    dome_intensity: float,
    ceiling_sphere_intensity: float,
) -> dict[str, Any]:
    """Replace only corridor-v2 lighting with an auditable candidate layer."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

    if not candidate_id or any(value <= 0.0 for value in (dome_intensity, ceiling_sphere_intensity)):
        raise ValueError("candidate id and positive light intensities are required")
    parent_path = Path(corridor_v2_audit).resolve()
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("passed") is not True or parent.get("schema_version") != "kinofail.embodiedgen-compiled-scene.v2":
        raise ValueError("input is not a passed corridor-v2 compilation")
    parent_dir = parent_path.parent
    base_path = Path(parent["base_compiled_audit"]).resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if _sha256(base_path) != parent["base_compiled_audit_sha256"]:
        raise ValueError("corridor-v2 parent no longer binds its base compilation")
    base_dir = base_path.parent
    for name, expected in parent["files"].items():
        path = parent_dir / name
        if not path.is_file():
            path = base_dir / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"stale corridor-v2 file: {path}")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    floor_z = float(parent["visual_metrics"]["floor_z_m"])
    waypoints = parent["route"]["waypoints_xy_m"]
    room_height = max(
        float(proxy["center_xyz_m"][2]) + float(proxy["size_xyz_m"][2]) / 2.0
        for proxy in parent["physics_contract"]["proxies"]
    ) - floor_z
    light_z = floor_z + min(max(room_height - 0.25, 2.3), 2.75)

    lighting_path = output / "lighting_v3.usda"
    lighting_stage = Usd.Stage.CreateNew(str(lighting_path))
    root = UsdGeom.Xform.Define(lighting_stage, "/KinoScene")
    lighting_stage.SetDefaultPrim(root.GetPrim())
    scope = UsdGeom.Scope.Define(lighting_stage, "/KinoScene/Lighting")
    scope.GetPrim().CreateAttribute("kino:ownedBy", Sdf.ValueTypeNames.String).Set("Kino-Fail")
    scope.GetPrim().CreateAttribute("kino:developmentCandidate", Sdf.ValueTypeNames.String).Set(candidate_id)
    dome = UsdLux.DomeLight.Define(lighting_stage, "/KinoScene/Lighting/PitchAwareDome")
    dome.CreateIntensityAttr(float(dome_intensity))
    dome.CreateExposureAttr(0.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.94))
    sphere_positions = []
    for index, (x, y) in enumerate(waypoints):
        sphere_positions.append([float(x), float(y), light_z])
        sphere = UsdLux.SphereLight.Define(
            lighting_stage, f"/KinoScene/Lighting/CeilingFill_{index}"
        )
        sphere.CreateIntensityAttr(float(ceiling_sphere_intensity))
        sphere.CreateExposureAttr(0.0)
        sphere.CreateRadiusAttr(0.10)
        sphere.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.90))
        UsdGeom.XformCommonAPI(sphere).SetTranslate(Gf.Vec3d(float(x), float(y), light_z))
    lighting_stage.GetRootLayer().Save()

    route_path = parent_dir / "route_operator_v2.usda"
    episode_path = output / "episode_v3.usda"
    episode_layer = Sdf.Layer.CreateNew(str(episode_path))
    episode_layer.subLayerPaths = [
        os.path.relpath(base_dir / "visual.usda", output),
        os.path.relpath(base_dir / "collision.usda", output),
        lighting_path.name,
        os.path.relpath(route_path, output),
    ]
    episode_layer.defaultPrim = "KinoScene"
    episode_layer.Save()

    lighting_contract = {
        "owned_by": "Kino-Fail",
        "development_candidate_id": candidate_id,
        "source_lights_retained": True,
        "neutral_dome_intensity": float(dome_intensity),
        "ceiling_sphere_lights": {
            "count": len(sphere_positions),
            "intensity": float(ceiling_sphere_intensity),
            "radius_m": 0.10,
            "positions_xyz_m": sphere_positions,
        },
        "pitch_aware_view_schema": PITCH_AWARE_VIEW_SCHEMA,
        "visual_intervention_only": True,
    }
    files = {
        "visual.usda": _sha256(base_dir / "visual.usda"),
        "collision.usda": _sha256(base_dir / "collision.usda"),
        lighting_path.name: _sha256(lighting_path),
        route_path.name: _sha256(route_path),
        episode_path.name: _sha256(episode_path),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "scene_id": parent["scene_id"],
        "source_kind": parent["source_kind"],
        "source_manifest": parent["source_manifest"],
        "source_manifest_sha256": parent["source_manifest_sha256"],
        "base_compiled_audit": str(base_path),
        "base_compiled_audit_sha256": _sha256(base_path),
        "corridor_v2_audit": str(parent_path),
        "corridor_v2_audit_sha256": _sha256(parent_path),
        "visual_metrics": parent["visual_metrics"],
        "physics_contract": parent["physics_contract"],
        "lighting_contract": lighting_contract,
        "route": parent["route"],
        "operator": parent["operator"],
        "corridor_search": parent["corridor_search"],
        "files": files,
        "passed": True,
        "issues": [],
        "admission_state": "pitch_aware_lighting_v3_development_pending_rtx_qa",
    }
    audit_path = output / "compiled_scene_audit.json"
    audit_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {**result, "audit_path": str(audit_path), "episode_usd": str(episode_path)}
