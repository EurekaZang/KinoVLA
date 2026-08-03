"""Compile an EmbodiedGen panorama mesh as a render-only Kino-Fail scene shell.

The reconstructed mesh supplies visual context only.  A small, explicit Kino-owned
route floor is the sole collision authority.  This separation prevents reconstruction
artifacts from silently changing the physical fault that an A0--A7 episode measures.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import trimesh


SCHEMA_VERSION = "kinofail.visual-shell-compiled-scene.v1-development"
VISUAL_SHELL_CONTRACT = "kinofail.render-only-embodiedgen-shell.v1"
ROUTE_COLLISION_CONTRACT = "kinofail.kino-owned-route-collision.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coordinate_matrix(contract: Mapping[str, Any]) -> np.ndarray:
    """Return and validate the frozen right-handed metric-to-Isaac transform."""
    matrix = np.asarray(contract["matrix"], dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("coordinate matrix must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-9):
        raise ValueError("coordinate matrix must be orthonormal")
    determinant = float(np.linalg.det(matrix))
    if not np.isclose(determinant, 1.0, atol=1.0e-9):
        raise ValueError("coordinate matrix must preserve handedness")
    if not np.isclose(determinant, float(contract["determinant"]), atol=1.0e-9):
        raise ValueError("coordinate determinant differs from contract")
    return matrix


def transform_vertices(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(vertices, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("mesh vertices must be finite Nx3 values")
    return values @ np.asarray(matrix, dtype=np.float64).T


def route_surface_geometry(route: Mapping[str, Any]) -> dict[str, Any]:
    """Build the rectangular nominal route used by appearance and collision layers."""
    points = np.asarray(route["waypoints_xy_m"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("route requires at least two XY waypoints")
    delta = points[-1] - points[0]
    route_length = float(np.linalg.norm(delta))
    if route_length <= 0.0:
        raise ValueError("route endpoints must differ")
    direction = delta / route_length
    left = np.array([-direction[1], direction[0]], dtype=np.float64)
    deviations = np.abs((points - points[0]) @ left)
    if float(deviations.max()) > 1.0e-6:
        raise ValueError("development visual-shell route must be straight")
    width = float(route["surface_width_m"])
    margin = float(route["endpoint_margin_m"])
    thickness = float(route["floor_thickness_m"])
    if width <= 0.0 or margin < 0.0 or thickness <= 0.0:
        raise ValueError("route dimensions must be positive")
    start = points[0] - margin * direction
    end = points[-1] + margin * direction
    half_width = width / 2.0
    corners = np.asarray(
        [
            start - half_width * left,
            end - half_width * left,
            end + half_width * left,
            start + half_width * left,
        ],
        dtype=np.float64,
    )
    center = (start + end) / 2.0
    return {
        "corners_xy_m": corners.tolist(),
        "center_xy_m": center.tolist(),
        "direction_xy": direction.tolist(),
        "left_xy": left.tolist(),
        "route_length_m": route_length,
        "surface_length_m": route_length + 2.0 * margin,
        "surface_width_m": width,
        "maximum_waypoint_deviation_m": float(deviations.max()),
    }


def _resolve_frozen(root: Path, spec: Mapping[str, Any], label: str) -> Path:
    path = (root / str(spec["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen {label}: {path}")
    actual = sha256_file(path)
    if actual != spec["sha256"]:
        raise ValueError(f"stale frozen {label}: {path} ({actual})")
    return path


def _load_material(root: Path, lock_path: Path, material_id: str) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    matches = [row for row in lock["materials"] if row["id"] == material_id]
    if len(matches) != 1:
        raise ValueError(f"material id must resolve exactly once: {material_id}")
    source = matches[0]
    asset_root = (root / lock["asset_root"]).resolve()
    maps: dict[str, dict[str, Any]] = {}
    for name in ("basecolor", "normal", "roughness"):
        spec = source["maps"][name]
        path = asset_root / spec["path"]
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ValueError(f"missing or stale material map: {path}")
        maps[name] = {**spec, "absolute_path": str(path)}
    return {
        "id": source["id"],
        "split": source["split"],
        "domains": source["domains"],
        "semantic_family": source["semantic_family"],
        "physical_size_m": source["physical_size_m"],
        "license": source["license"],
        "maps": maps,
    }


def _author_visual_shell(
    output_path: Path,
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdShade, Vt

    stage = Usd.Stage.CreateNew(str(output_path))
    root = UsdGeom.Xform.Define(stage, "/KinoScene")
    stage.SetDefaultPrim(root.GetPrim())
    scope = UsdGeom.Scope.Define(stage, "/KinoScene/VisualShell")
    scope.GetPrim().CreateAttribute("kino:ownedBy", Sdf.ValueTypeNames.String).Set(
        "EmbodiedGen-derived visual; Kino-Fail composition"
    )
    scope.GetPrim().CreateAttribute(
        "kino:visualInterventionOnly", Sdf.ValueTypeNames.Bool
    ).Set(True)
    mesh = UsdGeom.Mesh.Define(stage, "/KinoScene/VisualShell/ForestContext")
    # Vt's zero-copy NumPy bridge avoids millions of slow Boost.Python scalar
    # conversions and, importantly, preserves the USD float/int element types.
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32))
    )
    mesh.CreateFaceVertexIndicesAttr(
        Vt.IntArray.FromNumpy(faces.astype(np.int32).reshape(-1))
    )
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreatePurposeAttr(UsdGeom.Tokens.render)
    display_color = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
    )
    display_color.Set(Vt.Vec3fArray.FromNumpy(colors.astype(np.float32)))
    mesh.GetPrim().CreateAttribute(
        "kino:collisionAuthored", Sdf.ValueTypeNames.Bool
    ).Set(False)
    mesh.GetPrim().CreateAttribute(
        "kino:rigidBodyAuthored", Sdf.ValueTypeNames.Bool
    ).Set(False)

    material = UsdShade.Material.Define(stage, "/KinoScene/VisualShell/Looks/VertexColor")
    shader = UsdShade.Shader.Define(
        stage, "/KinoScene/VisualShell/Looks/VertexColor/PBR"
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.82)
    reader = UsdShade.Shader.Define(
        stage, "/KinoScene/VisualShell/Looks/VertexColor/displayColorReader"
    )
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh).Bind(material)
    stage.GetRootLayer().Save()


def _author_collision(output_path: Path, route: Mapping[str, Any], geometry: Mapping[str, Any]) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(output_path))
    root = UsdGeom.Xform.Define(stage, "/KinoScene")
    stage.SetDefaultPrim(root.GetPrim())
    group = UsdGeom.Xform.Define(stage, "/KinoScene/Collision")
    group.GetPrim().CreateAttribute("kino:ownedBy", Sdf.ValueTypeNames.String).Set(
        "Kino-Fail"
    )
    group.GetPrim().CreateAttribute(
        "kino:embodiedGenCollisionUsed", Sdf.ValueTypeNames.Bool
    ).Set(False)
    floor = UsdGeom.Cube.Define(stage, "/KinoScene/Collision/Floor")
    floor.CreateSizeAttr(1.0)
    UsdGeom.Imageable(floor.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    top = float(route["floor_top_z_m"])
    thickness = float(route["floor_thickness_m"])
    center = geometry["center_xy_m"]
    xform = UsdGeom.XformCommonAPI(floor)
    xform.SetTranslate(Gf.Vec3d(float(center[0]), float(center[1]), top - thickness / 2.0))
    xform.SetScale(
        Gf.Vec3f(
            float(geometry["surface_length_m"]),
            float(geometry["surface_width_m"]),
            thickness,
        )
    )
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    material = UsdPhysics.MaterialAPI.Apply(floor.GetPrim())
    material.CreateStaticFrictionAttr(float(route["nominal_static_friction"]))
    material.CreateDynamicFrictionAttr(float(route["nominal_dynamic_friction"]))
    floor.GetPrim().CreateAttribute("kino:semantic", Sdf.ValueTypeNames.String).Set(
        "nominal_route_substrate"
    )
    floor.GetPrim().CreateAttribute(
        "kino:operatorLayerMayOverride", Sdf.ValueTypeNames.Bool
    ).Set(True)
    stage.GetRootLayer().Save()


def _author_appearance(
    output_path: Path,
    *,
    route: Mapping[str, Any],
    geometry: Mapping[str, Any],
    appearance: Mapping[str, Any],
    material: Mapping[str, Any],
) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

    stage = Usd.Stage.CreateNew(str(output_path))
    root = UsdGeom.Xform.Define(stage, "/KinoScene")
    stage.SetDefaultPrim(root.GetPrim())
    scope = UsdGeom.Scope.Define(stage, "/KinoScene/Appearance")
    scope.GetPrim().CreateAttribute("kino:ownedBy", Sdf.ValueTypeNames.String).Set(
        "Kino-Fail"
    )
    scope.GetPrim().CreateAttribute(
        "kino:physicsIndependent", Sdf.ValueTypeNames.Bool
    ).Set(True)
    surface = UsdGeom.Mesh.Define(stage, "/KinoScene/Appearance/RouteSurface")
    z = float(route["floor_top_z_m"]) + float(route["appearance_lift_m"])
    corners = geometry["corners_xy_m"]
    surface.CreatePointsAttr([Gf.Vec3f(float(x), float(y), z) for x, y in corners])
    surface.CreateFaceVertexCountsAttr([4])
    surface.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    surface.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * 4)
    surface.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    surface.CreateSubdivisionSchemeAttr("none")
    surface.CreateDoubleSidedAttr(True)
    surface.CreatePurposeAttr(UsdGeom.Tokens.render)
    surface.GetPrim().CreateAttribute(
        "kino:collisionAuthored", Sdf.ValueTypeNames.Bool
    ).Set(False)
    surface.GetPrim().CreateAttribute("kino:materialId", Sdf.ValueTypeNames.String).Set(
        material["id"]
    )
    tile_u = float(geometry["surface_length_m"]) / float(material["physical_size_m"][0])
    tile_v = float(geometry["surface_width_m"]) / float(material["physical_size_m"][1])
    st = UsdGeom.PrimvarsAPI(surface).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st.Set([(0.0, 0.0), (tile_u, 0.0), (tile_u, tile_v), (0.0, tile_v)])

    material_path = "/KinoScene/Appearance/Looks/RoutePBR"
    usd_material = UsdShade.Material.Define(stage, material_path)
    pbr = UsdShade.Shader.Define(stage, f"{material_path}/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    reader = UsdShade.Shader.Define(stage, f"{material_path}/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    def texture(name: str, map_name: str, output_name: str, output_type: Any):
        shader = UsdShade.Shader.Define(stage, f"{material_path}/{name}")
        shader.CreateIdAttr("UsdUVTexture")
        shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(material["maps"][map_name]["absolute_path"])
        )
        shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
            "sRGB" if map_name == "basecolor" else "raw"
        )
        shader.CreateOutput(output_name, output_type)
        return shader

    basecolor = texture("basecolorTex", "basecolor", "rgb", Sdf.ValueTypeNames.Float3)
    normal = texture("normalTex", "normal", "rgb", Sdf.ValueTypeNames.Float3)
    roughness = texture("roughnessTex", "roughness", "r", Sdf.ValueTypeNames.Float)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        basecolor.ConnectableAPI(), "rgb"
    )
    pbr.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
        normal.ConnectableAPI(), "rgb"
    )
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        roughness.ConnectableAPI(), "r"
    )
    pbr.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    usd_material.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(surface).Bind(usd_material)

    dome = UsdLux.DomeLight.Define(stage, "/KinoScene/Appearance/ForestDome")
    dome.CreateIntensityAttr(float(appearance["dome_light_intensity"]))
    sun = UsdLux.DistantLight.Define(stage, "/KinoScene/Appearance/ForestSun")
    sun.CreateIntensityAttr(float(appearance["distant_light_intensity"]))
    sun.CreateAngleAttr(float(appearance["distant_light_angle_deg"]))
    UsdGeom.XformCommonAPI(sun).SetRotate(
        Gf.Vec3f(*[float(value) for value in appearance["distant_light_rotation_xyz_deg"]])
    )
    stage.GetRootLayer().Save()


def _author_route(output_path: Path, route: Mapping[str, Any]) -> None:
    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(output_path))
    root = UsdGeom.Xform.Define(stage, "/KinoScene")
    stage.SetDefaultPrim(root.GetPrim())
    prim = UsdGeom.Xform.Define(stage, "/KinoScene/Route").GetPrim()
    flat = [float(value) for point in route["waypoints_xy_m"] for value in point]
    prim.CreateAttribute("kino:waypointsXY", Sdf.ValueTypeNames.DoubleArray).Set(flat)
    prim.CreateAttribute("kino:requiredClearanceM", Sdf.ValueTypeNames.Double).Set(
        float(route["required_clearance_m"])
    )
    prim.CreateAttribute("kino:operatorCapabilitiesAudited", Sdf.ValueTypeNames.Bool).Set(
        False
    )
    stage.GetRootLayer().Save()


def _inspect_layers(paths: Mapping[str, Path]) -> dict[str, Any]:
    from pxr import Usd, UsdGeom, UsdPhysics

    shell_stage = Usd.Stage.Open(str(paths["visual_shell.usdc"]))
    shell = shell_stage.GetPrimAtPath("/KinoScene/VisualShell/ForestContext")
    shell_api_names = list(shell.GetAppliedSchemas()) if shell.IsValid() else []
    collision_stage = Usd.Stage.Open(str(paths["collision.usda"]))
    floor = collision_stage.GetPrimAtPath("/KinoScene/Collision/Floor")
    appearance_stage = Usd.Stage.Open(str(paths["appearance.usda"]))
    surface = appearance_stage.GetPrimAtPath("/KinoScene/Appearance/RouteSurface")
    return {
        "shell_prim_present": shell.IsValid(),
        "shell_purpose_render": shell.IsValid()
        and UsdGeom.Imageable(shell).GetPurposeAttr().Get() == UsdGeom.Tokens.render,
        "shell_has_no_collision_api": shell.IsValid()
        and not shell.HasAPI(UsdPhysics.CollisionAPI),
        "shell_has_no_rigid_body_api": shell.IsValid()
        and not shell.HasAPI(UsdPhysics.RigidBodyAPI),
        "shell_applied_schemas": shell_api_names,
        "kino_floor_prim_present": floor.IsValid(),
        "kino_floor_has_collision_api": floor.IsValid()
        and floor.HasAPI(UsdPhysics.CollisionAPI),
        "route_surface_prim_present": surface.IsValid(),
        "route_surface_has_no_collision_api": surface.IsValid()
        and not surface.HasAPI(UsdPhysics.CollisionAPI),
    }


def compile_visual_shell_scene(config_path: Path, *, root: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "kinofail.visual-shell-isaac-composition.v1-development":
        raise ValueError("unsupported visual-shell composition contract")
    sources = config["source"]
    mesh_audit_path = _resolve_frozen(root, sources["mesh_stage_audit"], "mesh audit")
    mesh_path = _resolve_frozen(root, sources["metric_visual_mesh"], "metric mesh")
    translated_path = _resolve_frozen(
        root, sources["translated_view_audit"], "translated-view audit"
    )
    review_path = _resolve_frozen(
        root, sources["translated_view_review"], "translated-view review"
    )
    mesh_audit = json.loads(mesh_audit_path.read_text(encoding="utf-8"))
    translated = json.loads(translated_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if not mesh_audit.get("passed") or not translated.get("passed"):
        raise ValueError("source mesh or translated-view machine audit did not pass")
    if review.get("may_proceed_to_isaac_visual_shell_composition") is not True:
        raise ValueError("translated-view review did not authorize development composition")
    if review.get("scene_registry_eligible") is not False:
        raise ValueError("development review must not pre-admit the scene")

    lock_path = _resolve_frozen(
        root, config["appearance"]["material_lock"], "material lock"
    )
    material = _load_material(
        root, lock_path, str(config["appearance"]["route_material_id"])
    )
    if config["domain"] not in material["domains"] or material["split"] != config["split"]:
        raise ValueError("route material domain/split differs from scene contract")

    loaded = trimesh.load(mesh_path, process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError("visual shell must contain one Trimesh")
    vertices_metric = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    colors = np.asarray(loaded.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
    if len(colors) != len(vertices_metric):
        raise ValueError("visual shell must have one RGB color per vertex")
    matrix = coordinate_matrix(config["coordinate_contract"])
    vertices_isaac = transform_vertices(vertices_metric, matrix)
    geometry = route_surface_geometry(config["route"])

    output = (root / config["output"]["directory"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite composition attempt: {output}")
    output.mkdir(parents=True)
    paths = {
        "visual_shell.usdc": output / "visual_shell.usdc",
        "collision.usda": output / "collision.usda",
        "appearance.usda": output / "appearance.usda",
        "route.usda": output / "route.usda",
        "episode.usda": output / "episode.usda",
    }
    _author_visual_shell(
        paths["visual_shell.usdc"],
        vertices=vertices_isaac,
        faces=faces,
        colors=colors,
    )
    _author_collision(paths["collision.usda"], config["route"], geometry)
    _author_appearance(
        paths["appearance.usda"],
        route=config["route"],
        geometry=geometry,
        appearance=config["appearance"],
        material=material,
    )
    _author_route(paths["route.usda"], config["route"])

    from pxr import Sdf

    episode = Sdf.Layer.CreateNew(str(paths["episode.usda"]))
    episode.subLayerPaths = [
        "visual_shell.usdc",
        "collision.usda",
        "appearance.usda",
        "route.usda",
    ]
    episode.defaultPrim = "KinoScene"
    episode.Save()
    layer_checks = _inspect_layers(paths)
    checks = {
        "source_mesh_stage_passed": bool(mesh_audit["passed"]),
        "translated_view_machine_passed": bool(translated["passed"]),
        "translated_view_review_allows_development": bool(
            review["may_proceed_to_isaac_visual_shell_composition"]
        ),
        "source_review_does_not_pre_admit_scene": review["scene_registry_eligible"] is False,
        "right_handed_coordinate_transform": np.isclose(np.linalg.det(matrix), 1.0),
        "route_matches_go2_forward_axis": np.allclose(
            np.asarray(geometry["direction_xy"]), np.asarray([1.0, 0.0])
        ),
        "material_is_train_wild": material["split"] == "train"
        and "wild" in material["domains"],
        "shell_prim_present": layer_checks["shell_prim_present"],
        "shell_purpose_render": layer_checks["shell_purpose_render"],
        "shell_has_no_collision_api": layer_checks["shell_has_no_collision_api"],
        "shell_has_no_rigid_body_api": layer_checks["shell_has_no_rigid_body_api"],
        "kino_floor_prim_present": layer_checks["kino_floor_prim_present"],
        "kino_floor_has_collision_api": layer_checks["kino_floor_has_collision_api"],
        "route_surface_prim_present": layer_checks["route_surface_prim_present"],
        "route_surface_has_no_collision_api": layer_checks[
            "route_surface_has_no_collision_api"
        ],
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "scene_id": config["scene_id"],
        "source_scene_id": config["source_scene_id"],
        "development_only": True,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "source": {
            "mesh_stage_audit": str(mesh_audit_path),
            "metric_visual_mesh": str(mesh_path),
            "translated_view_audit": str(translated_path),
            "translated_view_review": str(review_path),
        },
        "visual_shell_contract": {
            "schema": VISUAL_SHELL_CONTRACT,
            "vertices": int(len(vertices_isaac)),
            "faces": int(len(faces)),
            "bounds_isaac_xyz_m": {
                "minimum": vertices_isaac.min(axis=0).tolist(),
                "maximum": vertices_isaac.max(axis=0).tolist(),
            },
            "purpose": "render",
            "collision_authored": False,
            "rigid_body_authored": False,
            "coordinate_matrix": matrix.tolist(),
        },
        "physics_contract": {
            "schema": ROUTE_COLLISION_CONTRACT,
            "owned_by": "Kino-Fail",
            "embodiedgen_collision_used": False,
            "sole_collision_prim": "/KinoScene/Collision/Floor",
            "route_geometry": geometry,
            "nominal_friction": {
                "static": config["route"]["nominal_static_friction"],
                "dynamic": config["route"]["nominal_dynamic_friction"],
            },
            "operator_layer_may_override": True,
        },
        "appearance_contract": {
            "material": material,
            "physics_independent": True,
            "swappable_without_physics_change": True,
        },
        "route": config["route"],
        "layer_checks": layer_checks,
        "checks": {key: bool(value) for key, value in checks.items()},
        "files": {name: sha256_file(path) for name, path in paths.items()},
        "passed": all(bool(value) for value in checks.values()),
        "scene_registry_eligible": False,
        "counts_as_a0_a7_evidence": False,
        "operator_capabilities_audited": [],
        "remaining_gates": [
            "body_fixed_go2_front_rtx_qa",
            "articulated_go2_route_traversal_qa",
            "operator_capability_audit",
            "independent_human_scene_review",
            "scene_registry_binding",
            "paired_realistic_a0_a7_collection",
        ],
        "admission_state": (
            "isaac_layers_compiled_pending_go2_rtx_physics_operator_qa"
            if all(bool(value) for value in checks.values())
            else "isaac_layer_compilation_failed"
        ),
    }
    audit_path = output / "compiled_scene_audit.json"
    audit_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**result, "audit_path": str(audit_path), "episode_usd": str(paths["episode.usda"])}
