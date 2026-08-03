"""Compile an operator-ready dense terrain surface over an EmbodiedGen v4 scene."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from kino_vla.sim.embodiedgen_terrain_route import episode_layer_text
from kino_vla.sim.realistic_route_protocol_v2 import scene_route_binding


SCHEMA_VERSION = "kinofail.embodiedgen-terrain-route-compiled-scene.v2-development"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: float) -> str:
    return f"{float(value):.9g}"


def dense_route_grid(
    *, route: Mapping[str, Any], geometry: Mapping[str, Any], max_spacing_m: float
) -> dict[str, Any]:
    if not 0.015 <= max_spacing_m <= 0.10:
        raise ValueError("max_spacing_m must be in [0.015, 0.10]")
    points = np.asarray(route["waypoints_xy_m"], dtype=np.float64)
    direction = np.asarray(geometry["direction_xy"], dtype=np.float64)
    left = np.asarray(geometry["left_xy"], dtype=np.float64)
    route_length = float(geometry["route_length_m"])
    surface_length = float(geometry["surface_length_m"])
    surface_width = float(geometry["surface_width_m"])
    endpoint_margin = float(geometry["endpoint_margin_m"])
    z = float(np.asarray(geometry["vertices_xyz_m"], dtype=np.float64)[:, 2].mean())
    if abs(surface_length - (route_length + 2.0 * endpoint_margin)) > 1.0e-6:
        raise ValueError("surface length does not match route length plus endpoint margins")
    n_long = int(math.ceil(surface_length / max_spacing_m)) + 1
    n_lat = int(math.ceil(surface_width / max_spacing_m)) + 1
    origin = points[0] - endpoint_margin * direction - 0.5 * surface_width * left
    vertices: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    uv_repeat = np.asarray(geometry.get("uv_repeat", [1.0, 1.0]), dtype=np.float64)
    for i in range(n_long):
        u = i / (n_long - 1)
        for j in range(n_lat):
            v = j / (n_lat - 1)
            xy = origin + u * surface_length * direction + v * surface_width * left
            vertices.append((float(xy[0]), float(xy[1]), z))
            uvs.append((float(u * uv_repeat[0]), float(v * uv_repeat[1])))
    faces: list[tuple[int, int, int, int]] = []
    for i in range(n_long - 1):
        for j in range(n_lat - 1):
            a = i * n_lat + j
            b = (i + 1) * n_lat + j
            faces.append((a, b, b + 1, a + 1))
    return {
        "vertices": vertices,
        "uvs": uvs,
        "faces": faces,
        "n_long": n_long,
        "n_lat": n_lat,
        "actual_longitudinal_spacing_m": surface_length / (n_long - 1),
        "actual_lateral_spacing_m": surface_width / (n_lat - 1),
    }


def operator_surface_layer_text(
    *,
    grid: Mapping[str, Any],
    static_friction: float,
    dynamic_friction: float,
) -> str:
    if not (0.0 <= dynamic_friction <= static_friction <= 2.0):
        raise ValueError("require 0 <= dynamic_friction <= static_friction <= 2")
    vertices = ", ".join(
        f"({_fmt(x)}, {_fmt(y)}, {_fmt(z)})" for x, y, z in grid["vertices"]
    )
    uvs = ", ".join(f"({_fmt(u)}, {_fmt(v)})" for u, v in grid["uvs"])
    indices = ", ".join(str(index) for face in grid["faces"] for index in face)
    counts = ", ".join("4" for _ in grid["faces"])
    return f'''#usda 1.0

over "KinoScene"
{{
    over "Collision"
    {{
        over "Floor" (
            prepend apiSchemas = ["PhysicsMaterialAPI"]
        )
        {{
            custom bool kino:operatorLayerMayOverride = 1
            custom string kino:terrainAdapter = "embodiedgen_terrain_route_v2"
            float physics:dynamicFriction = {_fmt(dynamic_friction)}
            float physics:staticFriction = {_fmt(static_friction)}
        }}
    }}

    over "AppearanceV4"
    {{
        over "RenderOnlyRouteSurface"
        {{
            custom bool kino:collisionAuthored = 0
            custom bool kino:denseTerrainOperatorSurface = 1
            custom bool kino:opaqueCompositedGround = 1
            custom int kino:terrainGridLongitudinalVertices = {grid['n_long']}
            custom int kino:terrainGridLateralVertices = {grid['n_lat']}
            int[] faceVertexCounts = [{counts}]
            int[] faceVertexIndices = [{indices}]
            normal3f[] normals = [(0, 0, 1)] (
                interpolation = "constant"
            )
            point3f[] points = [{vertices}]
            texCoord2f[] primvars:st = [{uvs}] (
                interpolation = "vertex"
            )
            uniform token purpose = "render"
            uniform token subdivisionScheme = "none"
        }}
    }}
}}
'''


def compile_embodiedgen_terrain_route_v2(
    *,
    base_audit_path: Path,
    output_dir: Path,
    expected_base_audit_sha256: str | None = None,
    static_friction: float = 0.8,
    dynamic_friction: float = 0.6,
    max_spacing_m: float = 0.04,
) -> dict[str, Any]:
    base_audit_path = base_audit_path.resolve()
    output_dir = output_dir.resolve()
    base_hash = _sha256(base_audit_path)
    if expected_base_audit_sha256 is not None and base_hash != expected_base_audit_sha256:
        raise ValueError("base compiled-audit hash does not match frozen input")
    base = json.loads(base_audit_path.read_text(encoding="utf-8"))
    binding = scene_route_binding(base)
    if binding.source_kind != "embodiedgen_v4":
        raise ValueError("v2 compiler requires an admitted EmbodiedGen v4 base scene")
    base_episode = base_audit_path.parent / binding.episode_filename
    if _sha256(base_episode) != base.get("files", {}).get(binding.episode_filename):
        raise ValueError("base episode hash does not match compiled audit")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = dict(base["appearance_contract"]["geometry"])
    geometry["uv_repeat"] = base["appearance_contract"].get("uv_repeat", [1.0, 1.0])
    grid = dense_route_grid(
        route=base["route"], geometry=geometry, max_spacing_m=max_spacing_m
    )
    surface_layer = output_dir / "terrain_operator_surface.usda"
    surface_layer.write_text(
        operator_surface_layer_text(
            grid=grid,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
        ),
        encoding="utf-8",
    )
    episode = output_dir / "episode_terrain_v2.usda"
    episode.write_text(
        episode_layer_text(
            material_layer=surface_layer,
            base_episode=base_episode,
            output_dir=output_dir,
        ).replace("episode_terrain_v1", "episode_terrain_v2"),
        encoding="utf-8",
    )
    checks = {
        "base_scene_passed": base.get("passed") is True,
        "base_episode_hash_verified": True,
        "straight_route_bound": binding.frame.route_length_m > 0.0,
        "route_surface_is_visual_only": binding.route_surface_collision_authored is False,
        "explicit_nominal_floor_material_authored": True,
        "dense_operator_surface_authored": len(grid["vertices"]) >= 1000,
        "operator_surface_spacing_at_most_requested": max(
            grid["actual_longitudinal_spacing_m"],
            grid["actual_lateral_spacing_m"],
        )
        <= max_spacing_m + 1.0e-9,
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "development_only": True,
        "scene_registry_eligible": False,
        "counts_as_a0_a7_evidence": False,
        "scene_id": base["scene_id"],
        "source_kind": "embodiedgen_v4_with_dense_kino_terrain_surface",
        "base_compiled_audit": str(base_audit_path),
        "base_compiled_audit_sha256": base_hash,
        "route": base["route"],
        "appearance_contract": base["appearance_contract"],
        "physics_contract": {
            **base["physics_contract"],
            "nominal_floor_material": {
                "owned_by": "Kino-Fail",
                "prim_path": "/KinoScene/Collision/Floor",
                "api_schema": "PhysicsMaterialAPI",
                "static_friction": static_friction,
                "dynamic_friction": dynamic_friction,
                "operator_layer_may_override": True,
            },
            "dense_route_surface": {
                "prim_path": "/KinoScene/AppearanceV4/RenderOnlyRouteSurface",
                "collision_authored": False,
                "opaque_composited_ground": True,
                "vertex_count": len(grid["vertices"]),
                "quad_count": len(grid["faces"]),
                "n_long": grid["n_long"],
                "n_lat": grid["n_lat"],
                "maximum_spacing_m": max(
                    grid["actual_longitudinal_spacing_m"],
                    grid["actual_lateral_spacing_m"],
                ),
            },
        },
        "files": {
            "terrain_operator_surface.usda": _sha256(surface_layer),
            "episode_terrain_v2.usda": _sha256(episode),
        },
        "checks": checks,
        "remaining_gates": [
            "offline_usd_composition_audit",
            "nominal_articulated_go2_route_preflight",
            "fresh-process_operator_pair",
            "formal_native_v3_collector",
            "scene_registry_binding",
        ],
    }
    (output_dir / "compiled_scene_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
