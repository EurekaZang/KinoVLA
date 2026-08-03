"""Render-only route-surface and bidirectional lighting for EmbodiedGen scenes.

Version 4 is development-only.  It consumes a frozen corridor-v2 scene and
adds two new USD layers: a collision-free, metre-scaled PBR route surface and
seven route-envelope ceiling panels.  The source visual, collision, route and
operator layers are never edited.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "kinofail.embodiedgen-compiled-scene.v4-development"
ROUTE_SURFACE_SCHEMA = "kinofail.render-only-route-surface.v1"
LIGHTING_SCHEMA = "kinofail.bidirectional-route-envelope-lighting.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def route_surface_geometry(
    route: list[list[float]],
    floor_z_m: float,
    *,
    width_m: float = 1.20,
    endpoint_margin_m: float = 0.45,
    lift_m: float = 0.003,
) -> dict[str, Any]:
    """Return a straight render surface that contains the frozen route."""
    if len(route) < 2:
        raise ValueError("route surface requires at least two waypoints")
    if width_m <= 0.0 or endpoint_margin_m < 0.0 or lift_m <= 0.0:
        raise ValueError("route surface dimensions must be positive")
    points = [(float(row[0]), float(row[1])) for row in route]
    dx = points[-1][0] - points[0][0]
    dy = points[-1][1] - points[0][1]
    route_length = math.hypot(dx, dy)
    if route_length <= 1.0e-9:
        raise ValueError("route endpoints must be distinct")
    direction = (dx / route_length, dy / route_length)
    left = (-direction[1], direction[0])
    maximum_deviation = max(
        abs((x - points[0][0]) * left[0] + (y - points[0][1]) * left[1])
        for x, y in points
    )
    if maximum_deviation > 1.0e-5:
        raise ValueError("v4 route surface requires the straight corridor-v2 route")

    start = (
        points[0][0] - endpoint_margin_m * direction[0],
        points[0][1] - endpoint_margin_m * direction[1],
    )
    end = (
        points[-1][0] + endpoint_margin_m * direction[0],
        points[-1][1] + endpoint_margin_m * direction[1],
    )
    half_width = width_m / 2.0
    z = float(floor_z_m) + lift_m
    vertices = [
        [start[0] - half_width * left[0], start[1] - half_width * left[1], z],
        [end[0] - half_width * left[0], end[1] - half_width * left[1], z],
        [end[0] + half_width * left[0], end[1] + half_width * left[1], z],
        [start[0] + half_width * left[0], start[1] + half_width * left[1], z],
    ]
    return {
        "vertices_xyz_m": vertices,
        "direction_xy": list(direction),
        "left_xy": list(left),
        "route_length_m": route_length,
        "surface_length_m": route_length + 2.0 * endpoint_margin_m,
        "surface_width_m": width_m,
        "endpoint_margin_m": endpoint_margin_m,
        "lift_above_floor_m": lift_m,
        "maximum_waypoint_line_deviation_m": maximum_deviation,
    }


def route_envelope_light_positions(
    route: list[list[float]], light_z_m: float, *, endpoint_extension_m: float = 0.45
) -> list[list[float]]:
    """Return the five route lights plus one extension beyond either endpoint."""
    geometry = route_surface_geometry(route, 0.0, endpoint_margin_m=endpoint_extension_m)
    direction = geometry["direction_xy"]
    first = [float(value) for value in route[0]]
    last = [float(value) for value in route[-1]]
    return [
        [
            first[0] - endpoint_extension_m * direction[0],
            first[1] - endpoint_extension_m * direction[1],
            float(light_z_m),
        ],
        *[[float(x), float(y), float(light_z_m)] for x, y in route],
        [
            last[0] + endpoint_extension_m * direction[0],
            last[1] + endpoint_extension_m * direction[1],
            float(light_z_m),
        ],
    ]


def _load_material(lock_path: Path, material_id: str) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    matches = [row for row in lock["materials"] if row["id"] == material_id]
    if len(matches) != 1:
        raise ValueError(f"material id must resolve exactly once: {material_id}")
    material = matches[0]
    asset_root = Path(lock["asset_root"]).resolve()
    resolved_maps: dict[str, dict[str, Any]] = {}
    for name in ("basecolor", "normal", "roughness"):
        spec = material["maps"][name]
        path = asset_root / spec["path"]
        if not path.is_file() or _sha256(path) != spec["sha256"]:
            raise ValueError(f"stale route-surface material map: {path}")
        resolved_maps[name] = {**spec, "absolute_path": str(path)}
    return {
        "id": material["id"],
        "semantic_family": material["semantic_family"],
        "split": material["split"],
        "license": material["license"],
        "physical_size_m": material["physical_size_m"],
        "maps": resolved_maps,
    }


def refine_embodiedgen_kinofail_scene_v4(
    corridor_v2_audit: str | Path,
    output_dir: str | Path,
    *,
    material_id: str,
    material_lock: str | Path = REPO_ROOT
    / "outputs/assets/terrain_pbr_v1/terrain_assets.lock.json",
    dome_intensity: float = 350.0,
    panel_intensity: float = 450.0,
    panel_exposure: float = 7.0,
    route_surface_width_m: float = 1.20,
) -> dict[str, Any]:
    """Author collision-free appearance and route-envelope lighting layers."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

    if min(dome_intensity, panel_intensity) <= 0.0:
        raise ValueError("positive light intensities are required")
    parent_path = Path(corridor_v2_audit).resolve()
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        parent.get("passed") is not True
        or parent.get("schema_version") != "kinofail.embodiedgen-compiled-scene.v2"
    ):
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
    lock_path = Path(material_lock).resolve()
    material = _load_material(lock_path, material_id)
    floor_z = float(parent["visual_metrics"]["floor_z_m"])
    route = parent["route"]["waypoints_xy_m"]
    surface = route_surface_geometry(route, floor_z, width_m=route_surface_width_m)

    appearance_path = output / "appearance_v4.usda"
    appearance_stage = Usd.Stage.CreateNew(str(appearance_path))
    root = UsdGeom.Xform.Define(appearance_stage, "/KinoScene")
    appearance_stage.SetDefaultPrim(root.GetPrim())
    scope = UsdGeom.Scope.Define(appearance_stage, "/KinoScene/AppearanceV4")
    scope.GetPrim().CreateAttribute("kino:ownedBy", Sdf.ValueTypeNames.String).Set(
        "Kino-Fail"
    )
    scope.GetPrim().CreateAttribute(
        "kino:visualInterventionOnly", Sdf.ValueTypeNames.Bool
    ).Set(True)
    mesh = UsdGeom.Mesh.Define(
        appearance_stage, "/KinoScene/AppearanceV4/RenderOnlyRouteSurface"
    )
    mesh.CreatePointsAttr([Gf.Vec3f(*row) for row in surface["vertices_xyz_m"]])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * 4)
    mesh.SetNormalsInterpolation("vertex")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreatePurposeAttr(UsdGeom.Tokens.render)
    mesh.GetPrim().CreateAttribute(
        "kino:collisionAuthored", Sdf.ValueTypeNames.Bool
    ).Set(False)
    mesh.GetPrim().CreateAttribute("kino:materialId", Sdf.ValueTypeNames.String).Set(
        material_id
    )
    tile_u = surface["surface_length_m"] / float(material["physical_size_m"][0])
    tile_v = surface["surface_width_m"] / float(material["physical_size_m"][1])
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st.Set([(0.0, 0.0), (tile_u, 0.0), (tile_u, tile_v), (0.0, tile_v)])

    material_path = "/KinoScene/AppearanceV4/Looks/RouteSurfacePBR"
    usd_material = UsdShade.Material.Define(appearance_stage, material_path)
    pbr = UsdShade.Shader.Define(appearance_stage, f"{material_path}/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    reader = UsdShade.Shader.Define(appearance_stage, f"{material_path}/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    def texture(name: str, file_path: str, output_name: str, output_type: Any):
        shader = UsdShade.Shader.Define(appearance_stage, f"{material_path}/{name}")
        shader.CreateIdAttr("UsdUVTexture")
        shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(file_path))
        shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
            "sRGB" if name == "basecolorTex" else "raw"
        )
        shader.CreateOutput(output_name, output_type)
        return shader

    basecolor = texture(
        "basecolorTex",
        material["maps"]["basecolor"]["absolute_path"],
        "rgb",
        Sdf.ValueTypeNames.Float3,
    )
    normal = texture(
        "normalTex",
        material["maps"]["normal"]["absolute_path"],
        "rgb",
        Sdf.ValueTypeNames.Float3,
    )
    roughness = texture(
        "roughnessTex",
        material["maps"]["roughness"]["absolute_path"],
        "r",
        Sdf.ValueTypeNames.Float,
    )
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        basecolor.ConnectableAPI(), "rgb"
    )
    pbr.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
        normal.ConnectableAPI(), "rgb"
    )
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        roughness.ConnectableAPI(), "r"
    )
    usd_material.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(usd_material)
    appearance_stage.GetRootLayer().Save()

    room_height = max(
        float(proxy["center_xyz_m"][2]) + float(proxy["size_xyz_m"][2]) / 2.0
        for proxy in parent["physics_contract"]["proxies"]
    ) - floor_z
    light_z = floor_z + min(max(room_height - 0.25, 2.3), 2.75)
    light_positions = route_envelope_light_positions(route, light_z)
    lighting_path = output / "lighting_v4.usda"
    lighting_stage = Usd.Stage.CreateNew(str(lighting_path))
    root = UsdGeom.Xform.Define(lighting_stage, "/KinoScene")
    lighting_stage.SetDefaultPrim(root.GetPrim())
    lighting_scope = UsdGeom.Scope.Define(lighting_stage, "/KinoScene/Lighting")
    lighting_scope.GetPrim().CreateAttribute(
        "kino:ownedBy", Sdf.ValueTypeNames.String
    ).Set("Kino-Fail")
    dome = UsdLux.DomeLight.Define(
        lighting_stage, "/KinoScene/Lighting/RouteEnvelopeDome"
    )
    dome.CreateIntensityAttr(float(dome_intensity))
    dome.CreateExposureAttr(0.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.94))
    for index, (x, y, z) in enumerate(light_positions):
        panel = UsdLux.RectLight.Define(
            lighting_stage, f"/KinoScene/Lighting/RouteEnvelopePanel_{index}"
        )
        panel.CreateIntensityAttr(float(panel_intensity))
        panel.CreateExposureAttr(float(panel_exposure))
        panel.CreateWidthAttr(0.85)
        panel.CreateHeightAttr(0.45)
        panel.CreateColorAttr(Gf.Vec3f(1.0, 0.94, 0.88))
        UsdGeom.XformCommonAPI(panel).SetTranslate(Gf.Vec3d(x, y, z))
    lighting_stage.GetRootLayer().Save()

    route_path = parent_dir / "route_operator_v2.usda"
    episode_path = output / "episode_v4.usda"
    episode_layer = Sdf.Layer.CreateNew(str(episode_path))
    episode_layer.subLayerPaths = [
        os.path.relpath(base_dir / "visual.usda", output),
        os.path.relpath(base_dir / "collision.usda", output),
        lighting_path.name,
        appearance_path.name,
        os.path.relpath(route_path, output),
    ]
    episode_layer.defaultPrim = "KinoScene"
    episode_layer.Save()

    appearance_contract = {
        "schema": ROUTE_SURFACE_SCHEMA,
        "visual_intervention_only": True,
        "collision_authored": False,
        "purpose": "render",
        "material_lock": {"path": str(lock_path), "sha256": _sha256(lock_path)},
        "material": material,
        "geometry": surface,
        "uv_repeat": [tile_u, tile_v],
    }
    lighting_contract = {
        "schema": LIGHTING_SCHEMA,
        "owned_by": "Kino-Fail",
        "source_lights_retained": True,
        "neutral_dome_intensity": float(dome_intensity),
        "route_envelope_panels": {
            "count": len(light_positions),
            "intensity": float(panel_intensity),
            "exposure": float(panel_exposure),
            "positions_xyz_m": light_positions,
            "endpoint_extension_m": 0.45,
        },
        "visual_intervention_only": True,
    }
    files = {
        "visual.usda": _sha256(base_dir / "visual.usda"),
        "collision.usda": _sha256(base_dir / "collision.usda"),
        lighting_path.name: _sha256(lighting_path),
        appearance_path.name: _sha256(appearance_path),
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
        "appearance_contract": appearance_contract,
        "lighting_contract": lighting_contract,
        "route": parent["route"],
        "operator": parent["operator"],
        "corridor_search": parent["corridor_search"],
        "files": files,
        "passed": True,
        "issues": [],
        "admission_state": "route_surface_lighting_v4_development_pending_rtx_qa",
    }
    audit_path = output / "compiled_scene_audit.json"
    audit_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {**result, "audit_path": str(audit_path), "episode_usd": str(episode_path)}
