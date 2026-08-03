#!/usr/bin/env python3
"""Fail-closed offline USD audit for an EmbodiedGen terrain-route v2 package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "kinofail.embodiedgen-terrain-route-v2-offline-audit.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _bound_file(path_value: Any, expected_sha256: Any) -> tuple[Path | None, bool]:
    if not isinstance(path_value, str) or not isinstance(expected_sha256, str):
        return None, False
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    return path, path.is_file() and _sha256(path) == expected_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    from kino_vla.sim.realistic_route_protocol_v2 import (
        EMBODIEDGEN_TERRAIN_V2_SCHEMA,
        resolve_bound_episode,
        scene_route_binding,
    )

    compiled_path = args.compiled_audit.resolve()
    compiled = _json(compiled_path)
    checks: dict[str, bool] = {
        "supported_schema": compiled.get("schema_version")
        == EMBODIEDGEN_TERRAIN_V2_SCHEMA,
        "compiler_audit_passed": compiled.get("passed") is True,
        "development_boundary_retained": compiled.get("development_only") is True,
        "not_a0_a7_evidence": compiled.get("counts_as_a0_a7_evidence") is False,
    }

    try:
        binding = scene_route_binding(compiled)
        episode = resolve_bound_episode(compiled_path, compiled)
        route_binding_valid = binding.source_kind == "embodiedgen_terrain_v2"
    except (KeyError, TypeError, ValueError, FileNotFoundError):
        binding = None
        episode = compiled_path.parent / "episode_terrain_v2.usda"
        route_binding_valid = False
    checks["route_binding_valid"] = route_binding_valid

    files = compiled.get("files", {})
    for filename in ("episode_terrain_v2.usda", "terrain_operator_surface.usda"):
        path = compiled_path.parent / filename
        checks[f"{filename}_hash_verified"] = (
            path.is_file() and files.get(filename) == _sha256(path)
        )

    base_path, base_verified = _bound_file(
        compiled.get("base_compiled_audit"), compiled.get("base_compiled_audit_sha256")
    )
    base = _json(base_path) if base_verified and base_path is not None else {}
    checks["base_audit_hash_verified"] = base_verified
    checks["base_scene_passed"] = base.get("passed") is True
    if base_path is not None:
        base_episode = base_path.parent / "episode_v4.usda"
        checks["base_episode_hash_verified"] = (
            base_episode.is_file()
            and base.get("files", {}).get("episode_v4.usda") == _sha256(base_episode)
        )
    else:
        checks["base_episode_hash_verified"] = False

    material = compiled.get("appearance_contract", {}).get("material", {})
    map_checks: dict[str, dict[str, Any]] = {}
    for map_name in ("basecolor", "normal", "roughness"):
        record = material.get("maps", {}).get(map_name, {})
        path, verified = _bound_file(record.get("absolute_path"), record.get("sha256"))
        checks[f"material_{map_name}_hash_verified"] = verified
        map_checks[map_name] = {
            "path": str(path) if path is not None else None,
            "sha256": record.get("sha256"),
            "verified": verified,
        }

    physics = compiled.get("physics_contract", {})
    nominal_material = physics.get("nominal_floor_material", {})
    dense = physics.get("dense_route_surface", {})
    checks.update(
        {
            "kino_owned_collision_contract": physics.get("owned_by") == "Kino-Fail"
            and physics.get("embodiedgen_collision_used") is False,
            "collision_proxy_contract_nonempty": int(physics.get("proxy_count", 0)) >= 5,
            "floor_material_contract_valid": nominal_material.get("owned_by") == "Kino-Fail"
            and nominal_material.get("api_schema") == "PhysicsMaterialAPI"
            and 0.0
            <= float(nominal_material.get("dynamic_friction", -1.0))
            <= float(nominal_material.get("static_friction", -1.0))
            <= 2.0,
            "dense_surface_contract_valid": dense.get("collision_authored") is False
            and dense.get("opaque_composited_ground") is True
            and int(dense.get("vertex_count", 0)) >= 1000
            and int(dense.get("quad_count", 0)) > 0
            and float(dense.get("maximum_spacing_m", math.inf)) <= 0.04 + 1.0e-9,
        }
    )

    stage = Usd.Stage.Open(str(episode)) if episode.is_file() else None
    checks["usd_composition_opened"] = stage is not None
    floor = stage.GetPrimAtPath("/KinoScene/Collision/Floor") if stage else None
    surface = (
        stage.GetPrimAtPath("/KinoScene/AppearanceV4/RenderOnlyRouteSurface")
        if stage
        else None
    )
    floor_valid = bool(floor and floor.IsValid())
    surface_valid = bool(surface and surface.IsValid())
    checks["floor_has_collision_api"] = floor_valid and floor.HasAPI(UsdPhysics.CollisionAPI)
    checks["floor_has_material_api"] = floor_valid and floor.HasAPI(UsdPhysics.MaterialAPI)
    checks["floor_static_friction_readback"] = floor_valid and abs(
        float(floor.GetAttribute("physics:staticFriction").Get())
        - float(nominal_material.get("static_friction", math.inf))
    ) <= 1.0e-6
    checks["floor_dynamic_friction_readback"] = floor_valid and abs(
        float(floor.GetAttribute("physics:dynamicFriction").Get())
        - float(nominal_material.get("dynamic_friction", math.inf))
    ) <= 1.0e-6

    mesh = UsdGeom.Mesh(surface) if surface_valid else None
    points = list(mesh.GetPointsAttr().Get() or []) if mesh else []
    face_counts = list(mesh.GetFaceVertexCountsAttr().Get() or []) if mesh else []
    material_binding = None
    if surface_valid:
        bound, _ = UsdShade.MaterialBindingAPI(surface).ComputeBoundMaterial()
        if bound and bound.GetPrim().IsValid():
            material_binding = str(bound.GetPath())
    checks.update(
        {
            "surface_is_render_only": surface_valid
            and UsdGeom.Imageable(surface).GetPurposeAttr().Get() == UsdGeom.Tokens.render
            and surface.GetAttribute("kino:collisionAuthored").Get() is False,
            "surface_opaque_marker_readback": surface_valid
            and surface.GetAttribute("kino:opaqueCompositedGround").Get() is True,
            "surface_dense_marker_readback": surface_valid
            and surface.GetAttribute("kino:denseTerrainOperatorSurface").Get() is True,
            "surface_vertex_count_matches_audit": len(points)
            == int(dense.get("vertex_count", -1)),
            "surface_quad_count_matches_audit": len(face_counts)
            == int(dense.get("quad_count", -1))
            and all(int(count) == 4 for count in face_counts),
            "surface_has_inherited_pbr_material": material_binding is not None,
        }
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "scene_id": compiled.get("scene_id"),
        "passed": all(checks.values()),
        "checks": checks,
        "compiled_audit": {
            "path": str(compiled_path),
            "sha256": _sha256(compiled_path),
        },
        "episode": {
            "path": str(episode),
            "sha256": _sha256(episode) if episode.is_file() else None,
        },
        "route_binding": (
            {
                "source_kind": binding.source_kind,
                "collector_start_waypoint_index": binding.collector_start_waypoint_index,
                "collector_start_progress_m": binding.collector_start_progress_m,
                "available_route_length_m": binding.frame.route_length_m,
            }
            if binding is not None
            else None
        ),
        "material_maps": map_checks,
        "pbr_material_prim": material_binding,
        "counts_as_a0_a7_evidence": False,
        "remaining_gates": [
            "nominal_articulated_go2_route_preflight",
            "frozen_native_v3_operator_pair",
            "scene_registry_binding",
            "formal_corpus_collection",
            "new_a0_a7_replication",
        ],
    }
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite offline audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "scene_id": result["scene_id"], "passed": result["passed"]}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
