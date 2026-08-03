"""Compile a route-complete forest hybrid for realistic Kino-Fail development.

The single-panorama reconstruction is deliberately not used along the robot path: its
validated translation envelope is smaller than the episode.  EmbodiedGen supplies an
infinite equirectangular background while metric PBR ground and audited CC0 props supply
near-field parallax.  Kino-Fail remains the sole owner of route collision and operators.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from kino_vla.sim.visual_shell_composition import (
    _author_collision,
    _author_route,
    _load_material,
    _resolve_frozen,
    route_surface_geometry,
    sha256_file,
)


SCHEMA_VERSION = "kinofail.forest-hybrid-compiled-scene.v2-development"


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def surrounding_rectangles(
    extent_xy_m: list[list[float]], route_geometry: Mapping[str, Any]
) -> list[dict[str, float]]:
    """Partition the near field around the route surface without overlap or gaps."""
    extent = np.asarray(extent_xy_m, dtype=np.float64)
    corners = np.asarray(route_geometry["corners_xy_m"], dtype=np.float64)
    if extent.shape != (2, 2) or not np.isfinite(extent).all():
        raise ValueError("near-field extent must be finite [[xmin,ymin],[xmax,ymax]]")
    xmin, ymin = extent[0]
    xmax, ymax = extent[1]
    rxmin, rymin = corners.min(axis=0)
    rxmax, rymax = corners.max(axis=0)
    if not (xmin < rxmin < rxmax < xmax and ymin < rymin < rymax < ymax):
        raise ValueError("near-field extent must strictly contain the route surface")

    def rect(name: str, ax: float, ay: float, bx: float, by: float) -> dict[str, float]:
        if not (bx > ax and by > ay):
            raise ValueError(f"empty surrounding rectangle: {name}")
        return {
            "id": name,
            "xmin": float(ax),
            "ymin": float(ay),
            "xmax": float(bx),
            "ymax": float(by),
        }

    return [
        rect("left_bank", xmin, rymax, xmax, ymax),
        rect("right_bank", xmin, ymin, xmax, rymin),
        rect("entry_ground", xmin, rymin, rxmin, rymax),
        rect("exit_ground", rxmax, rymin, xmax, rymax),
    ]


def _author_pbr_material(
    stage: Any,
    material_path: str,
    material: Mapping[str, Any],
    *,
    opacity: float = 1.0,
    opacity_primvar: str | None = None,
) -> Any:
    from pxr import Sdf, UsdShade

    usd_material = UsdShade.Material.Define(stage, material_path)
    pbr = UsdShade.Shader.Define(stage, f"{material_path}/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    if opacity_primvar is None:
        pbr.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    else:
        opacity_reader = UsdShade.Shader.Define(stage, f"{material_path}/opacityReader")
        opacity_reader.CreateIdAttr("UsdPrimvarReader_float")
        opacity_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(opacity_primvar)
        opacity_reader.CreateOutput("result", Sdf.ValueTypeNames.Float)
        pbr.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(
            opacity_reader.ConnectableAPI(), "result"
        )
    reader = UsdShade.Shader.Define(stage, f"{material_path}/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    def texture(name: str, map_name: str, output_name: str, output_type: Any) -> Any:
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

    base = texture("basecolorTex", "basecolor", "rgb", Sdf.ValueTypeNames.Float3)
    normal = texture("normalTex", "normal", "rgb", Sdf.ValueTypeNames.Float3)
    rough = texture("roughnessTex", "roughness", "r", Sdf.ValueTypeNames.Float)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        base.ConnectableAPI(), "rgb"
    )
    pbr.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
        normal.ConnectableAPI(), "rgb"
    )
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        rough.ConnectableAPI(), "r"
    )
    pbr.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    usd_material.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return usd_material


def _author_ground_mesh(
    stage: Any,
    *,
    path: str,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    z: float,
    material: Mapping[str, Any],
    usd_material: Any,
    height_variation: Mapping[str, Any] | None = None,
    route_bounds: tuple[float, float, float, float] | None = None,
    texture_extent_xy: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    mesh = UsdGeom.Mesh.Define(stage, path)
    variation_enabled = bool(height_variation and height_variation.get("enabled", False))
    if variation_enabled:
        if route_bounds is None:
            raise ValueError("route bounds are required for surrounding-ground variation")
        spacing = float(height_variation["grid_spacing_m"])
        if spacing <= 0.0:
            raise ValueError("ground variation grid spacing must be positive")
        x_count = max(2, int(np.ceil((xmax - xmin) / spacing)) + 1)
        y_count = max(2, int(np.ceil((ymax - ymin) / spacing)) + 1)
        xs = np.linspace(xmin, xmax, x_count)
        ys = np.linspace(ymin, ymax, y_count)
        points: list[Any] = []
        texcoords: list[tuple[float, float]] = []
        heights: list[float] = []
        for yy in ys:
            for xx in xs:
                zz = _procedural_ground_height(
                    float(xx),
                    float(yy),
                    base_z=z,
                    profile=height_variation,
                    route_bounds=route_bounds,
                )
                points.append(Gf.Vec3f(float(xx), float(yy), zz))
                heights.append(zz)
                if texture_extent_xy is None:
                    texcoords.append(
                        (
                            float(xx) / float(material["physical_size_m"][0]),
                            float(yy) / float(material["physical_size_m"][1]),
                        )
                    )
                else:
                    txmin, tymin, txmax, tymax = texture_extent_xy
                    texcoords.append(
                        (
                            (float(xx) - txmin) / (txmax - txmin),
                            (float(yy) - tymin) / (tymax - tymin),
                        )
                    )
        indices: list[int] = []
        for row in range(y_count - 1):
            for column in range(x_count - 1):
                lower_left = row * x_count + column
                indices.extend(
                    [
                        lower_left,
                        lower_left + 1,
                        lower_left + x_count + 1,
                        lower_left + x_count,
                    ]
                )
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([4] * ((x_count - 1) * (y_count - 1)))
        mesh.CreateFaceVertexIndicesAttr(indices)
    else:
        heights = [z] * 4
        if texture_extent_xy is None:
            texcoords = [
                (0.0, 0.0),
                ((xmax - xmin) / float(material["physical_size_m"][0]), 0.0),
                (
                    (xmax - xmin) / float(material["physical_size_m"][0]),
                    (ymax - ymin) / float(material["physical_size_m"][1]),
                ),
                (0.0, (ymax - ymin) / float(material["physical_size_m"][1])),
            ]
        else:
            texcoords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        mesh.CreatePointsAttr(
            [
                Gf.Vec3f(xmin, ymin, z),
                Gf.Vec3f(xmax, ymin, z),
                Gf.Vec3f(xmax, ymax, z),
                Gf.Vec3f(xmin, ymax, z),
            ]
        )
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)])
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.constant)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreatePurposeAttr(UsdGeom.Tokens.render)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.vertex if variation_enabled else UsdGeom.Tokens.faceVarying,
    )
    st.Set(texcoords)
    mesh.GetPrim().CreateAttribute("kino:collisionAuthored", Sdf.ValueTypeNames.Bool).Set(False)
    UsdShade.MaterialBindingAPI(mesh).Bind(usd_material)
    return {
        "path": path,
        "variation_enabled": variation_enabled,
        "vertex_count": len(heights),
        "minimum_z_m": float(min(heights)),
        "maximum_z_m": float(max(heights)),
    }


def _route_blend_mask(
    world_x: np.ndarray,
    world_y: np.ndarray,
    *,
    route_geometry: Mapping[str, Any],
    profile: Mapping[str, Any],
    noise: np.ndarray | None = None,
) -> np.ndarray:
    """Return a deterministic soft trail mask with no edge at the collision boundary."""
    core = float(profile["core_half_width_m"])
    outer = float(profile["outer_half_width_m"])
    physical = 0.5 * float(route_geometry["surface_width_m"])
    width_variation = float(profile.get("width_variation_m", 0.0))
    if not 0.0 < core < physical < outer - abs(width_variation):
        raise ValueError(
            "opaque trail blend must place the physical route boundary strictly inside "
            "the texture transition"
        )
    wavelength = float(profile["wavelength_m"])
    if wavelength <= 0.0:
        raise ValueError("opaque trail wavelength must be positive")
    center = np.asarray(route_geometry["center_xy_m"], dtype=np.float64)
    direction = np.asarray(route_geometry["direction_xy"], dtype=np.float64)
    left = np.asarray(route_geometry["left_xy"], dtype=np.float64)
    rel_x = world_x - float(center[0])
    rel_y = world_y - float(center[1])
    along = rel_x * float(direction[0]) + rel_y * float(direction[1])
    lateral = rel_x * float(left[0]) + rel_y * float(left[1])
    phase = float(profile.get("phase_rad", 0.0))
    centerline = float(profile.get("centerline_wander_m", 0.0)) * (
        0.68 * np.sin(2.0 * np.pi * along / wavelength + phase)
        + 0.32 * np.sin(2.0 * np.pi * along / (0.43 * wavelength) - 0.7 * phase)
    )
    local_outer = outer + width_variation * (
        0.61 * np.sin(2.0 * np.pi * along / (0.81 * wavelength) - 0.4 * phase)
        + 0.39 * np.sin(2.0 * np.pi * along / (0.37 * wavelength) + 1.3)
    )
    distance = np.abs(lateral - centerline)
    if noise is not None:
        if noise.shape != distance.shape:
            raise ValueError("opaque trail noise must match the world-coordinate grid")
        distance = distance + float(profile.get("noise_amplitude_m", 0.0)) * noise
    transition = np.clip((local_outer - distance) / (local_outer - core), 0.0, 1.0)
    return transition * transition * (3.0 - 2.0 * transition)


def _tile_material_map(
    path: str,
    *,
    physical_size_m: list[float],
    pixels_per_m: float,
    output_size: tuple[int, int],
    mode: str,
) -> np.ndarray:
    from PIL import Image

    source = Image.open(path).convert(mode)
    tile_width = max(8, int(round(float(physical_size_m[0]) * pixels_per_m)))
    tile_height = max(8, int(round(float(physical_size_m[1]) * pixels_per_m)))
    source = source.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
    array = np.asarray(source)
    if array.ndim == 2:
        array = array[:, :, None]
    width, height = output_size
    repeats_y = int(np.ceil(height / tile_height))
    repeats_x = int(np.ceil(width / tile_width))
    return np.tile(array, (repeats_y, repeats_x, 1))[:height, :width]


def _build_opaque_composite_material(
    output_directory: Path,
    *,
    config: Mapping[str, Any],
    route_geometry: Mapping[str, Any],
    route_material: Mapping[str, Any],
    surrounding_material: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bake two tiled PBR materials into one opaque, metric, softly blended ground map."""
    from PIL import Image

    profile = config["near_field"]["opaque_composited_ground"]
    extent = np.asarray(config["near_field"]["extent_xy_m"], dtype=np.float64)
    xmin, ymin = extent[0]
    xmax, ymax = extent[1]
    span_x, span_y = float(xmax - xmin), float(ymax - ymin)
    max_dimension = int(profile["max_texture_dimension_px"])
    if max_dimension < 512:
        raise ValueError("opaque composite texture must be at least 512 pixels")
    pixels_per_m = max_dimension / max(span_x, span_y)
    width = max(2, int(round(span_x * pixels_per_m)))
    height = max(2, int(round(span_y * pixels_per_m)))
    xs = np.linspace(xmin, xmax, width, endpoint=False, dtype=np.float64)
    ys = np.linspace(ymin, ymax, height, endpoint=False, dtype=np.float64)
    world_x, world_y = np.meshgrid(xs, ys)
    seed = int(profile["seed"])
    rng = np.random.default_rng(seed)
    coarse_size = int(profile.get("noise_grid_size", 13))
    coarse = rng.integers(0, 256, size=(coarse_size, coarse_size), dtype=np.uint8)
    noise_image = Image.fromarray(coarse, mode="L").resize(
        (width, height), Image.Resampling.BICUBIC
    )
    noise = np.asarray(noise_image, dtype=np.float32) / 127.5 - 1.0
    mask = _route_blend_mask(
        world_x,
        world_y,
        route_geometry=route_geometry,
        profile=profile,
        noise=noise,
    ).astype(np.float32)
    alpha = mask[:, :, None]
    maps_directory = output_directory / "appearance_maps"
    maps_directory.mkdir(parents=True, exist_ok=False)
    generated: dict[str, dict[str, Any]] = {}

    for map_name, mode in (("basecolor", "RGB"), ("normal", "RGB"), ("roughness", "L")):
        route_map = _tile_material_map(
            route_material["maps"][map_name]["absolute_path"],
            physical_size_m=route_material["physical_size_m"],
            pixels_per_m=pixels_per_m,
            output_size=(width, height),
            mode=mode,
        ).astype(np.float32) / 255.0
        bank_map = _tile_material_map(
            surrounding_material["maps"][map_name]["absolute_path"],
            physical_size_m=surrounding_material["physical_size_m"],
            pixels_per_m=pixels_per_m,
            output_size=(width, height),
            mode=mode,
        ).astype(np.float32) / 255.0
        if map_name == "basecolor":
            route_linear = np.where(
                route_map <= 0.04045,
                route_map / 12.92,
                ((route_map + 0.055) / 1.055) ** 2.4,
            )
            bank_linear = np.where(
                bank_map <= 0.04045,
                bank_map / 12.92,
                ((bank_map + 0.055) / 1.055) ** 2.4,
            )
            mixed_linear = alpha * route_linear + (1.0 - alpha) * bank_linear
            mixed = np.where(
                mixed_linear <= 0.0031308,
                12.92 * mixed_linear,
                1.055 * np.power(mixed_linear, 1.0 / 2.4) - 0.055,
            )
        elif map_name == "normal":
            mixed_vector = alpha * (2.0 * route_map - 1.0) + (1.0 - alpha) * (
                2.0 * bank_map - 1.0
            )
            norm = np.linalg.norm(mixed_vector, axis=2, keepdims=True)
            mixed = 0.5 * (mixed_vector / np.maximum(norm, 1.0e-8) + 1.0)
        else:
            mixed = alpha * route_map + (1.0 - alpha) * bank_map
        encoded = np.clip(np.rint(255.0 * mixed), 0.0, 255.0).astype(np.uint8)
        if mode == "L":
            encoded = encoded[:, :, 0]
        map_path = maps_directory / f"ground_{map_name}.png"
        Image.fromarray(encoded, mode=mode).save(map_path, optimize=True)
        generated[map_name] = {
            "absolute_path": str(map_path.resolve()),
            "sha256": sha256_file(map_path),
        }

    mask_path = maps_directory / "route_blend_mask.png"
    Image.fromarray(np.rint(255.0 * mask).astype(np.uint8), mode="L").save(
        mask_path, optimize=True
    )
    physical_half_width = 0.5 * float(route_geometry["surface_width_m"])
    center = np.asarray(route_geometry["center_xy_m"], dtype=np.float64)
    left = np.asarray(route_geometry["left_xy"], dtype=np.float64)
    physical_lateral = np.abs(
        (world_x - center[0]) * left[0] + (world_y - center[1]) * left[1]
    )
    boundary_band = np.abs(physical_lateral - physical_half_width) <= max(
        1.5 / pixels_per_m, 0.01
    )
    boundary_values = mask[boundary_band]
    composite = {
        "id": f"opaque_{route_material['id']}__{surrounding_material['id']}",
        "split": route_material["split"],
        "domains": sorted(set(route_material["domains"]) & set(surrounding_material["domains"])),
        "physical_size_m": [span_x, span_y],
        "maps": generated,
    }
    audit = {
        "enabled": True,
        "single_opaque_texture_set": True,
        "source_route_material_id": route_material["id"],
        "source_surrounding_material_id": surrounding_material["id"],
        "resolution_px": [width, height],
        "pixels_per_m": pixels_per_m,
        "seed": seed,
        "mask_path": str(mask_path.resolve()),
        "mask_sha256": sha256_file(mask_path),
        "mask_range": [float(mask.min()), float(mask.max())],
        "physical_route_boundary_weight_range": [
            float(boundary_values.min()),
            float(boundary_values.max()),
        ],
        "physical_route_boundary_weight_mean": float(boundary_values.mean()),
        "transparent_pixels": 0,
        "generated_maps": generated,
    }
    return composite, audit


def _procedural_ground_height(
    x: float,
    y: float,
    *,
    base_z: float,
    profile: Mapping[str, Any],
    route_bounds: tuple[float, float, float, float],
) -> float:
    """Smooth deterministic visual relief that is exactly flat at the route boundary."""
    rxmin, rymin, rxmax, rymax = route_bounds
    dx = max(rxmin - x, 0.0, x - rxmax)
    dy = max(rymin - y, 0.0, y - rymax)
    distance = float(np.hypot(dx, dy))
    fade_distance = float(profile["route_boundary_fade_m"])
    if fade_distance <= 0.0:
        raise ValueError("route-boundary fade distance must be positive")
    u = min(max(distance / fade_distance, 0.0), 1.0)
    smooth_fade = u * u * (3.0 - 2.0 * u)
    wavelength = float(profile["wavelength_m"])
    if wavelength <= 0.0:
        raise ValueError("ground variation wavelength must be positive")
    phase = float(profile.get("phase_rad", 0.0))
    wave = (
        0.52 * np.sin(2.0 * np.pi * x / wavelength + phase)
        + 0.31 * np.sin(2.0 * np.pi * y / (wavelength * 0.73) - 0.4 * phase)
        + 0.17
        * np.sin(2.0 * np.pi * (0.61 * x + 0.79 * y) / (wavelength * 0.47) + 1.7)
    )
    return float(base_z + float(profile["amplitude_m"]) * smooth_fade * wave)


def _author_naturalized_route_mesh(
    stage: Any,
    *,
    path: str,
    route_geometry: Mapping[str, Any],
    z: float,
    material: Mapping[str, Any],
    usd_material: Any,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    segment_length = float(profile["segment_length_m"])
    if segment_length <= 0.0:
        raise ValueError("naturalized route segment length must be positive")
    minimum_overlap = float(profile["minimum_bank_overlap_m"])
    maximum_overlap = float(profile["maximum_bank_overlap_m"])
    if not 0.0 <= minimum_overlap <= maximum_overlap:
        raise ValueError("invalid naturalized route bank-overlap range")
    length = float(route_geometry["surface_length_m"])
    width = float(route_geometry["surface_width_m"])
    center = np.asarray(route_geometry["center_xy_m"], dtype=np.float64)
    direction = np.asarray(route_geometry["direction_xy"], dtype=np.float64)
    left = np.asarray(route_geometry["left_xy"], dtype=np.float64)
    start = center - 0.5 * length * direction
    count = max(2, int(np.ceil(length / segment_length)) + 1)
    distances = np.linspace(0.0, length, count)
    phase = float(profile.get("phase_rad", 0.0))

    points: list[Any] = []
    texcoords: list[tuple[float, float]] = []
    half_widths: list[float] = []
    for distance in distances:
        normalized = float(distance / max(length, 1.0e-12))
        wave = 0.5 + 0.32 * np.sin(2.0 * np.pi * 1.7 * normalized + phase)
        wave += 0.18 * np.sin(2.0 * np.pi * 4.1 * normalized - 0.7 * phase)
        wave = float(np.clip(wave, 0.0, 1.0))
        overlap = minimum_overlap + (maximum_overlap - minimum_overlap) * wave
        half_width = 0.5 * width + overlap
        half_widths.append(half_width)
        route_center = start + float(distance) * direction
        for sign in (-1.0, 1.0):
            xy = route_center + sign * half_width * left
            points.append(Gf.Vec3f(float(xy[0]), float(xy[1]), z))
            texcoords.append(
                (
                    float(xy[0]) / float(material["physical_size_m"][0]),
                    float(xy[1]) / float(material["physical_size_m"][1]),
                )
            )
    indices = []
    for index in range(count - 1):
        lower = 2 * index
        indices.extend([lower, lower + 2, lower + 3, lower + 1])
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * (count - 1))
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreatePurposeAttr(UsdGeom.Tokens.render)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set(texcoords)
    mesh.GetPrim().CreateAttribute("kino:collisionAuthored", Sdf.ValueTypeNames.Bool).Set(False)
    mesh.GetPrim().CreateAttribute("kino:naturalizedVisualEdge", Sdf.ValueTypeNames.Bool).Set(True)
    UsdShade.MaterialBindingAPI(mesh).Bind(usd_material)
    return {
        "path": path,
        "naturalized_visual_edge": True,
        "vertex_count": len(points),
        "minimum_half_width_m": float(min(half_widths)),
        "maximum_half_width_m": float(max(half_widths)),
        "physical_route_half_width_m": 0.5 * width,
    }


def _author_feathered_route_mesh(
    stage: Any,
    *,
    path: str,
    route_geometry: Mapping[str, Any],
    z: float,
    material: Mapping[str, Any],
    usd_material: Any,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """One non-overlapping mesh with a vertex-opacity transition into the forest bank."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    segment_length = float(profile["segment_length_m"])
    minimum_overlap = float(profile["minimum_bank_overlap_m"])
    maximum_overlap = float(profile["maximum_bank_overlap_m"])
    if segment_length <= 0.0 or not 0.0 < minimum_overlap <= maximum_overlap:
        raise ValueError("invalid feathered route profile")
    length = float(route_geometry["surface_length_m"])
    physical_half_width = 0.5 * float(route_geometry["surface_width_m"])
    center = np.asarray(route_geometry["center_xy_m"], dtype=np.float64)
    direction = np.asarray(route_geometry["direction_xy"], dtype=np.float64)
    left = np.asarray(route_geometry["left_xy"], dtype=np.float64)
    start = center - 0.5 * length * direction
    count = max(2, int(np.ceil(length / segment_length)) + 1)
    distances = np.linspace(0.0, length, count)
    phase = float(profile.get("phase_rad", 0.0))
    points: list[Any] = []
    texcoords: list[tuple[float, float]] = []
    opacities: list[float] = []
    outer_half_widths: list[float] = []
    for distance in distances:
        normalized = float(distance / max(length, 1.0e-12))
        wave = 0.5 + 0.32 * np.sin(2.0 * np.pi * 1.7 * normalized + phase)
        wave += 0.18 * np.sin(2.0 * np.pi * 4.1 * normalized - 0.7 * phase)
        wave = float(np.clip(wave, 0.0, 1.0))
        overlap = minimum_overlap + (maximum_overlap - minimum_overlap) * wave
        outer_half_width = physical_half_width + overlap
        outer_half_widths.append(outer_half_width)
        route_center = start + float(distance) * direction
        for offset, alpha in (
            (-outer_half_width, 0.0),
            (-physical_half_width, 1.0),
            (physical_half_width, 1.0),
            (outer_half_width, 0.0),
        ):
            xy = route_center + offset * left
            points.append(Gf.Vec3f(float(xy[0]), float(xy[1]), z))
            texcoords.append(
                (
                    float(xy[0]) / float(material["physical_size_m"][0]),
                    float(xy[1]) / float(material["physical_size_m"][1]),
                )
            )
            opacities.append(alpha)
    indices = []
    for station in range(count - 1):
        first = 4 * station
        second = 4 * (station + 1)
        for strip in range(3):
            indices.extend(
                [first + strip, second + strip, second + strip + 1, first + strip + 1]
            )
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * (3 * (count - 1)))
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreatePurposeAttr(UsdGeom.Tokens.render)
    api = UsdGeom.PrimvarsAPI(mesh)
    st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(texcoords)
    opacity = api.CreatePrimvar(
        "routeOpacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex
    )
    opacity.Set(opacities)
    mesh.GetPrim().CreateAttribute("kino:collisionAuthored", Sdf.ValueTypeNames.Bool).Set(False)
    mesh.GetPrim().CreateAttribute("kino:featheredVisualEdge", Sdf.ValueTypeNames.Bool).Set(True)
    UsdShade.MaterialBindingAPI(mesh).Bind(usd_material)
    return {
        "path": path,
        "feathered_visual_edge": True,
        "single_nonoverlapping_mesh": True,
        "vertex_count": len(points),
        "physical_route_half_width_m": physical_half_width,
        "minimum_outer_half_width_m": float(min(outer_half_widths)),
        "maximum_outer_half_width_m": float(max(outer_half_widths)),
        "opacity_range": [float(min(opacities)), float(max(opacities))],
    }


def _author_appearance(
    output_path: Path,
    *,
    config: Mapping[str, Any],
    route_geometry: Mapping[str, Any],
    route_material: Mapping[str, Any],
    surrounding_material: Mapping[str, Any],
    surrounding: list[Mapping[str, float]],
) -> dict[str, Any]:
    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(output_path))
    root = UsdGeom.Xform.Define(stage, "/KinoScene")
    stage.SetDefaultPrim(root.GetPrim())
    scope = UsdGeom.Scope.Define(stage, "/KinoScene/Appearance")
    scope.GetPrim().CreateAttribute("kino:ownedBy", Sdf.ValueTypeNames.String).Set("Kino-Fail")
    scope.GetPrim().CreateAttribute("kino:physicsIndependent", Sdf.ValueTypeNames.Bool).Set(True)
    opaque_profile = config["near_field"].get("opaque_composited_ground")
    opaque_enabled = bool(opaque_profile and opaque_profile.get("enabled", False))
    ground = UsdGeom.Scope.Define(stage, "/KinoScene/Appearance/SurroundingGround")
    ground.GetPrim().CreateAttribute("kino:routeHolePreserved", Sdf.ValueTypeNames.Bool).Set(
        not opaque_enabled
    )
    ground.GetPrim().CreateAttribute(
        "kino:singleOpaqueContinuousGround", Sdf.ValueTypeNames.Bool
    ).Set(opaque_enabled)
    surrounding_usd_material = _author_pbr_material(
        stage,
        "/KinoScene/Appearance/Looks/SurroundingForestGroundPBR",
        surrounding_material,
    )
    corners = np.asarray(route_geometry["corners_xy_m"], dtype=np.float64)
    route_bounds = (
        float(corners[:, 0].min()),
        float(corners[:, 1].min()),
        float(corners[:, 0].max()),
        float(corners[:, 1].max()),
    )
    surrounding_audits = []
    if not opaque_enabled:
        for item in surrounding:
            surrounding_audits.append(
                _author_ground_mesh(
                    stage,
                    path=f"/KinoScene/Appearance/SurroundingGround/{item['id']}",
                    xmin=float(item["xmin"]),
                    ymin=float(item["ymin"]),
                    xmax=float(item["xmax"]),
                    ymax=float(item["ymax"]),
                    z=float(config["near_field"]["ground_z_m"]),
                    material=surrounding_material,
                    usd_material=surrounding_usd_material,
                    height_variation=config["near_field"].get("visual_height_variation"),
                    route_bounds=route_bounds,
                )
            )
    route_profile = config["near_field"].get("naturalized_route_visual")
    blend_bands = route_profile.get("blend_bands", []) if route_profile else []
    opaque_audit = None
    if opaque_enabled:
        composite_material, opaque_audit = _build_opaque_composite_material(
            output_path.parent,
            config=config,
            route_geometry=route_geometry,
            route_material=route_material,
            surrounding_material=surrounding_material,
        )
        route_usd_material = _author_pbr_material(
            stage,
            "/KinoScene/Appearance/Looks/OpaqueCompositeGroundPBR",
            composite_material,
        )
        extent = config["near_field"]["extent_xy_m"]
        texture_extent = (
            float(extent[0][0]),
            float(extent[0][1]),
            float(extent[1][0]),
            float(extent[1][1]),
        )
        route_audit = _author_ground_mesh(
            stage,
            path="/KinoScene/Appearance/RouteSurface",
            xmin=texture_extent[0],
            ymin=texture_extent[1],
            xmax=texture_extent[2],
            ymax=texture_extent[3],
            z=float(config["route"]["floor_top_z_m"])
            + float(config["route"]["appearance_lift_m"]),
            material=composite_material,
            usd_material=route_usd_material,
            height_variation=config["near_field"].get("visual_height_variation"),
            route_bounds=route_bounds,
            texture_extent_xy=texture_extent,
        )
        route_surface = stage.GetPrimAtPath("/KinoScene/Appearance/RouteSurface")
        route_surface.CreateAttribute(
            "kino:opaqueCompositedGround", Sdf.ValueTypeNames.Bool
        ).Set(True)
        route_surface.CreateAttribute(
            "kino:visualMaterialIndependentOfOperator", Sdf.ValueTypeNames.Bool
        ).Set(True)
        route_audit.update(
            {
                "opaque_composited_ground": True,
                "single_continuous_mesh": True,
                "texture_composite": opaque_audit,
            }
        )
    elif route_profile and route_profile.get("enabled", False) and route_profile.get(
        "feathered_edge", False
    ):
        route_usd_material = _author_pbr_material(
            stage,
            "/KinoScene/Appearance/Looks/RouteGroundPBR_Feathered",
            route_material,
            opacity_primvar="routeOpacity",
        )
        route_audit = _author_feathered_route_mesh(
            stage,
            path="/KinoScene/Appearance/RouteSurface",
            route_geometry=route_geometry,
            z=float(config["route"]["floor_top_z_m"])
            + float(config["route"]["appearance_lift_m"]),
            material=route_material,
            usd_material=route_usd_material,
            profile=route_profile,
        )
    elif route_profile and route_profile.get("enabled", False) and blend_bands:
        route_parent = UsdGeom.Xform.Define(stage, "/KinoScene/Appearance/RouteSurface")
        band_audits = []
        base_z = float(config["route"]["floor_top_z_m"]) + float(
            config["route"]["appearance_lift_m"]
        )
        for index, band in enumerate(blend_bands):
            band_id = str(band["id"])
            band_material = _author_pbr_material(
                stage,
                f"/KinoScene/Appearance/Looks/RouteGroundPBR_{band_id}",
                route_material,
                opacity=float(band["opacity"]),
            )
            band_profile = {
                **route_profile,
                "minimum_bank_overlap_m": float(band["minimum_bank_overlap_m"]),
                "maximum_bank_overlap_m": float(band["maximum_bank_overlap_m"]),
            }
            band_audit = _author_naturalized_route_mesh(
                stage,
                path=f"/KinoScene/Appearance/RouteSurface/{band_id}",
                route_geometry=route_geometry,
                z=base_z + float(band["z_offset_m"]),
                material=route_material,
                usd_material=band_material,
                profile=band_profile,
            )
            band_audits.append(
                {
                    **band_audit,
                    "id": band_id,
                    "opacity": float(band["opacity"]),
                    "z_order": index,
                }
            )
        route_audit = {
            "path": str(route_parent.GetPath()),
            "naturalized_visual_edge": True,
            "alpha_blend_band_count": len(band_audits),
            "bands": band_audits,
        }
    elif route_profile and route_profile.get("enabled", False):
        route_usd_material = _author_pbr_material(
            stage, "/KinoScene/Appearance/Looks/RouteGroundPBR", route_material
        )
        route_audit = _author_naturalized_route_mesh(
            stage,
            path="/KinoScene/Appearance/RouteSurface",
            route_geometry=route_geometry,
            z=float(config["route"]["floor_top_z_m"])
            + float(config["route"]["appearance_lift_m"]),
            material=route_material,
            usd_material=route_usd_material,
            profile=route_profile,
        )
    else:
        route_usd_material = _author_pbr_material(
            stage, "/KinoScene/Appearance/Looks/RouteGroundPBR", route_material
        )
        route_audit = _author_ground_mesh(
            stage,
            path="/KinoScene/Appearance/RouteSurface",
            xmin=float(corners[:, 0].min()),
            ymin=float(corners[:, 1].min()),
            xmax=float(corners[:, 0].max()),
            ymax=float(corners[:, 1].max()),
            z=float(config["route"]["floor_top_z_m"])
            + float(config["route"]["appearance_lift_m"]),
            material=route_material,
            usd_material=route_usd_material,
        )
    stage.GetPrimAtPath("/KinoScene/Appearance/RouteSurface").CreateAttribute(
        "kino:operatorTopologyMayHide", Sdf.ValueTypeNames.Bool
    ).Set(True)
    stage.GetRootLayer().Save()
    return {
        "surrounding": surrounding_audits,
        "route": route_audit,
        "opaque_composite": opaque_audit,
        "visual_height_variation_enabled": bool(
            route_audit.get("variation_enabled", False)
            or any(item["variation_enabled"] for item in surrounding_audits)
        ),
    }


def _author_background(output_path: Path, config: Mapping[str, Any], panorama: Path) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

    stage = Usd.Stage.CreateNew(str(output_path))
    root = UsdGeom.Xform.Define(stage, "/KinoScene")
    stage.SetDefaultPrim(root.GetPrim())
    scope = UsdGeom.Scope.Define(stage, "/KinoScene/Background")
    scope.GetPrim().CreateAttribute("kino:source", Sdf.ValueTypeNames.String).Set(
        config["background"].get(
            "source_label", "EmbodiedGen V2 generated equirectangular panorama"
        )
    )
    scope.GetPrim().CreateAttribute("kino:infiniteBackground", Sdf.ValueTypeNames.Bool).Set(True)
    dome = UsdLux.DomeLight.Define(stage, "/KinoScene/Background/EmbodiedGenDome")
    dome.CreateTextureFileAttr(Sdf.AssetPath(str(panorama)))
    dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    dome.CreateIntensityAttr(float(config["background"]["intensity"]))
    UsdGeom.XformCommonAPI(dome).SetRotate(
        Gf.Vec3f(*[float(value) for value in config["background"]["rotation_xyz_deg"]])
    )
    sun = UsdLux.DistantLight.Define(stage, "/KinoScene/Background/ForestSun")
    sun.CreateIntensityAttr(float(config["appearance"]["distant_light_intensity"]))
    sun.CreateAngleAttr(float(config["appearance"]["distant_light_angle_deg"]))
    UsdGeom.XformCommonAPI(sun).SetRotate(
        Gf.Vec3f(*[float(value) for value in config["appearance"]["distant_light_rotation_xyz_deg"]])
    )
    stage.GetRootLayer().Save()


def _forest_material_specs(asset: Mapping[str, Any]) -> list[dict[str, Any]]:
    files = {Path(record["path"]).name: record["path"] for record in asset["files"]}

    def file(name: str, *, optional: bool = False) -> str | None:
        value = files.get(name)
        if value is None and not optional:
            raise ValueError(f"asset {asset['id']} lacks required compatibility texture {name}")
        return value

    asset_id = asset["id"]
    if asset_id == "pine_roots":
        return [
            {
                "id": "pine_roots_a",
                "target_contains": "pine_roots_a/pine_roots_a",
                "basecolor": file("pine_roots_a_diff_1k.png"),
                "normal": file("pine_roots_a_nor_gl_1k.png"),
                "roughness": file("pine_roots_a_rough_1k.png"),
            },
            {
                "id": "pine_roots_b",
                "target_contains": "pine_roots_b/pine_roots_b",
                "basecolor": file("pine_roots_b_diff_1k.png"),
                "normal": file("pine_roots_b_nor_gl_1k.png"),
                "roughness": file("pine_roots_b_rough_1k.png"),
            },
        ]
    if asset_id == "rock_moss_set_01":
        return [
            {
                "id": asset_id,
                "target_contains": None,
                "basecolor": file("rock_moss_set_01_diff_1k.jpg"),
                "normal": file("rock_moss_set_01_nor_gl_1k.exr"),
                "roughness": file("rock_moss_set_01_rough_1k.jpg"),
            }
        ]
    if asset_id == "fir_sapling":
        return [
            {
                "id": "fir_branches",
                "target_contains": "fir_sapling_branches",
                "basecolor": file("fir_sapling_branches_diff_1k.png"),
                "normal": file("fir_sapling_branches_nor_gl_1k.png"),
                "roughness": file("fir_sapling_branches_rough_1k.png"),
            },
            {
                "id": "fir_twigs",
                "target_contains": "fir_sapling_twigs",
                "basecolor": file("fir_sapling_twigs_diff_1k.png"),
                "normal": file("fir_sapling_twigs_nor_gl_1k.png"),
                "roughness": file("fir_sapling_twigs_rough_1k.png"),
                "opacity": file("fir_sapling_twigs_alpha_1k.png"),
            },
        ]
    if asset_id == "fern_02":
        return [
            {
                "id": asset_id,
                "target_contains": None,
                "basecolor": file("fern_02_diff_1k.jpg"),
                "normal": file("fern_02_nor_gl_1k.exr"),
                "roughness": file("fern_02_rough_1k.exr"),
                "opacity": file("fern_02_alpha_1k.png"),
            }
        ]
    if asset_id == "shrub_03":
        return [
            {
                "id": asset_id,
                "target_contains": None,
                "basecolor": file("shrub_03_diff_1k.jpg"),
                "normal": file("shrub_03_nor_gl_1k.exr"),
                "roughness": file("shrub_03_rough_1k.exr"),
                "opacity": file("shrub_03_alpha_1k.png"),
            }
        ]
    if asset_id == "tree_stump_01":
        return [
            {
                "id": asset_id,
                "target_contains": None,
                "basecolor": file("tree_stump_01_diff_1k.jpg"),
                "normal": file("tree_stump_01_nor_gl_1k.exr"),
                "roughness": file("tree_stump_01_rough_1k.exr"),
            }
        ]
    raise ValueError(f"no Isaac material compatibility contract for {asset_id}")


def _author_prop_preview_material(stage: Any, path: str, spec: Mapping[str, Any]) -> Any:
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PBR")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.78)
    reader = UsdShade.Shader.Define(stage, f"{path}/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    def texture(name: str, file_path: str, output: str, output_type: Any, color_space: str) -> Any:
        node = UsdShade.Shader.Define(stage, f"{path}/{name}")
        node.CreateIdAttr("UsdUVTexture")
        node.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(file_path))
        node.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        node.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        node.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        node.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
        node.CreateOutput(output, output_type)
        return node

    base = texture("basecolorTex", spec["basecolor"], "rgb", Sdf.ValueTypeNames.Float3, "sRGB")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        base.ConnectableAPI(), "rgb"
    )
    if spec.get("normal"):
        normal = texture("normalTex", spec["normal"], "rgb", Sdf.ValueTypeNames.Float3, "raw")
        shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
            normal.ConnectableAPI(), "rgb"
        )
    if spec.get("roughness"):
        rough = texture(
            "roughnessTex", spec["roughness"], "r", Sdf.ValueTypeNames.Float, "raw"
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
            rough.ConnectableAPI(), "r"
        )
    if spec.get("opacity"):
        opacity = texture(
            "opacityTex", spec["opacity"], "r", Sdf.ValueTypeNames.Float, "raw"
        )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(
            opacity.ConnectableAPI(), "r"
        )
        shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.18)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _author_props(
    output_path: Path, config: Mapping[str, Any], lock: Mapping[str, Any]
) -> dict[str, Any]:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    assets = {record["id"]: record for record in lock["assets"]}
    stage = Usd.Stage.CreateNew(str(output_path))
    root = UsdGeom.Xform.Define(stage, "/KinoScene")
    stage.SetDefaultPrim(root.GetPrim())
    scope = UsdGeom.Scope.Define(stage, "/KinoScene/NearFieldProps")
    scope.GetPrim().CreateAttribute("kino:visualOnly", Sdf.ValueTypeNames.Bool).Set(True)
    compatibility_mode = config["near_field"].get("material_compatibility_mode", "source")
    compatibility_materials: dict[tuple[str, str], Any] = {}
    if compatibility_mode == "isaac_preview_override_v1":
        looks = UsdGeom.Scope.Define(stage, "/KinoScene/NearFieldLooks")
        looks.GetPrim().CreateAttribute("kino:isaacCompatibilityOverride", Sdf.ValueTypeNames.Bool).Set(
            True
        )
        for asset in lock["assets"]:
            for spec in _forest_material_specs(asset):
                material = _author_prop_preview_material(
                    stage,
                    f"/KinoScene/NearFieldLooks/{asset['id']}_{spec['id']}",
                    spec,
                )
                compatibility_materials[(asset["id"], spec["id"])] = (material, spec)
    elif compatibility_mode != "source":
        raise ValueError(f"unsupported forest material compatibility mode: {compatibility_mode}")
    bound_compatibility_targets = []
    for item in config["near_field"]["instances"]:
        if item["asset_id"] not in assets:
            raise ValueError(f"unknown forest prop asset: {item['asset_id']}")
        prim = UsdGeom.Xform.Define(stage, f"/KinoScene/NearFieldProps/{item['id']}")
        prim.GetPrim().GetReferences().AddReference(assets[item["asset_id"]]["primary_path"])
        xform = UsdGeom.XformCommonAPI(prim)
        xform.SetTranslate(Gf.Vec3d(*[float(value) for value in item["xyz_m"]]))
        xform.SetRotate(Gf.Vec3f(*[float(value) for value in item["rotation_xyz_deg"]]))
        scale = float(item["scale"])
        xform.SetScale(Gf.Vec3f(scale, scale, scale))
        prim.GetPrim().CreateAttribute("kino:assetId", Sdf.ValueTypeNames.String).Set(
            item["asset_id"]
        )
        prim.GetPrim().CreateAttribute("kino:collisionAuthored", Sdf.ValueTypeNames.Bool).Set(False)
        if compatibility_mode == "isaac_preview_override_v1":
            instance_path = str(prim.GetPath())
            for target in Usd.PrimRange(prim.GetPrim()):
                if target.IsA(UsdGeom.Mesh):
                    UsdGeom.Mesh(target).CreateDoubleSidedAttr(True)
                if not (target.IsA(UsdGeom.Mesh) or target.IsA(UsdGeom.Subset)):
                    continue
                relative = str(target.GetPath())[len(instance_path) + 1 :]
                matches = []
                for (asset_id, _), (material, spec) in compatibility_materials.items():
                    if asset_id != item["asset_id"]:
                        continue
                    token = spec.get("target_contains")
                    if token is None or token in relative:
                        matches.append((material, spec))
                if not matches:
                    continue
                # Specific subset/mesh selectors win over an asset-wide fallback.
                material, spec = sorted(
                    matches, key=lambda pair: pair[1].get("target_contains") is not None
                )[-1]
                UsdShade.MaterialBindingAPI.Apply(target).Bind(material)
                bound_compatibility_targets.append(
                    {"prim": str(target.GetPath()), "material": str(material.GetPath())}
                )
    stage.GetRootLayer().Save()
    return {
        "mode": compatibility_mode,
        "material_count": len(compatibility_materials),
        "bound_target_count": len(bound_compatibility_targets),
        "bindings": bound_compatibility_targets,
    }


def _resolved_prop_instances(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    instances = [dict(item) for item in config["near_field"]["instances"]]
    overrides = config["near_field"].get("prop_transform_overrides", {})
    known = {item["id"] for item in instances}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(f"prop transform overrides reference unknown instances: {unknown}")
    return [_deep_merge(item, overrides.get(item["id"], {})) for item in instances]


def _author_prop_collision(
    output_path: Path,
    *,
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    route_geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Author conservative static proxies and explicit flexible-vegetation exemptions."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(output_path))
    root = UsdGeom.Xform.Define(stage, "/KinoScene")
    stage.SetDefaultPrim(root.GetPrim())
    scope = UsdGeom.Scope.Define(stage, "/KinoScene/PropCollision")
    scope.GetPrim().CreateAttribute("kino:ownedBy", Sdf.ValueTypeNames.String).Set("Kino-Fail")
    policy = config["near_field"].get("prop_collision_policy", {})
    if not policy.get("enabled", False):
        scope.GetPrim().CreateAttribute("kino:complete", Sdf.ValueTypeNames.Bool).Set(False)
        stage.GetRootLayer().Save()
        return {
            "enabled": False,
            "classification_complete": False,
            "proxy_count": 0,
            "minimum_route_clearance_m": None,
            "instances": [],
        }
    rigid = set(policy["rigid_asset_ids"])
    flexible = set(policy["flexible_noncolliding_asset_ids"])
    if rigid & flexible:
        raise ValueError("rigid and flexible prop asset classes must be disjoint")
    assets = {record["id"]: record for record in lock["assets"]}
    all_asset_ids = {item["asset_id"] for item in config["near_field"]["instances"]}
    if rigid | flexible != all_asset_ids:
        raise ValueError("prop collision policy must classify every used asset id exactly once")
    horizontal_inset = float(policy.get("horizontal_proxy_inset", 0.82))
    vertical_inset = float(policy.get("vertical_proxy_inset", 0.90))
    if not 0.5 <= horizontal_inset <= 1.0 or not 0.5 <= vertical_inset <= 1.0:
        raise ValueError("prop collision proxy insets must lie in [0.5, 1.0]")
    route_center = np.asarray(route_geometry["center_xy_m"], dtype=np.float64)
    route_left = np.asarray(route_geometry["left_xy"], dtype=np.float64)
    route_half_width = 0.5 * float(route_geometry["surface_width_m"])
    required_clearance = float(config["route"]["required_clearance_m"])
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    proxy_count = 0
    instance_audits = []
    all_clearances = []
    for item in config["near_field"]["instances"]:
        instance_path = f"/KinoScene/PropCollision/{item['id']}"
        parent = UsdGeom.Xform.Define(stage, instance_path)
        parent.GetPrim().CreateAttribute("kino:assetId", Sdf.ValueTypeNames.String).Set(
            item["asset_id"]
        )
        if item["asset_id"] in flexible:
            parent.GetPrim().CreateAttribute(
                "kino:flexibleNoncollidingVegetation", Sdf.ValueTypeNames.Bool
            ).Set(True)
            instance_audits.append(
                {
                    "id": item["id"],
                    "asset_id": item["asset_id"],
                    "classification": "flexible_noncolliding_vegetation",
                    "proxy_count": 0,
                }
            )
            continue
        xform = UsdGeom.XformCommonAPI(parent)
        xform.SetTranslate(Gf.Vec3d(*[float(value) for value in item["xyz_m"]]))
        xform.SetRotate(Gf.Vec3f(*[float(value) for value in item["rotation_xyz_deg"]]))
        scale = float(item["scale"])
        xform.SetScale(Gf.Vec3f(scale, scale, scale))
        source_stage = Usd.Stage.Open(assets[item["asset_id"]]["primary_path"])
        source_meshes = [prim for prim in source_stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
        instance_clearances = []
        yaw = np.deg2rad(float(item["rotation_xyz_deg"][2]))
        rotation = np.asarray(
            [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
            dtype=np.float64,
        )
        translation = np.asarray(item["xyz_m"][:2], dtype=np.float64)
        for index, source_mesh in enumerate(source_meshes):
            bounds = bbox_cache.ComputeWorldBound(source_mesh).ComputeAlignedBox()
            minimum = np.asarray(bounds.GetMin(), dtype=np.float64)
            maximum = np.asarray(bounds.GetMax(), dtype=np.float64)
            center = 0.5 * (minimum + maximum)
            size = np.maximum(maximum - minimum, 0.02)
            size[:2] *= horizontal_inset
            size[2] *= vertical_inset
            proxy_path = f"{instance_path}/proxy_{index:02d}"
            cube = UsdGeom.Cube.Define(stage, proxy_path)
            cube.CreateSizeAttr(1.0)
            cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            cube_xform = UsdGeom.XformCommonAPI(cube)
            cube_xform.SetTranslate(Gf.Vec3d(*[float(value) for value in center]))
            cube_xform.SetScale(Gf.Vec3f(*[float(value) for value in size]))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            cube.GetPrim().CreateAttribute("kino:sourceMesh", Sdf.ValueTypeNames.String).Set(
                str(source_mesh.GetPath())
            )
            local_corners = np.asarray(
                [
                    [center[0] + sx * 0.5 * size[0], center[1] + sy * 0.5 * size[1]]
                    for sx in (-1.0, 1.0)
                    for sy in (-1.0, 1.0)
                ],
                dtype=np.float64,
            )
            world_corners = translation + scale * local_corners @ rotation.T
            lateral = (world_corners - route_center) @ route_left
            if float(lateral.min()) <= 0.0 <= float(lateral.max()):
                minimum_lateral = 0.0
            else:
                minimum_lateral = float(np.min(np.abs(lateral)))
            clearance = minimum_lateral - route_half_width
            instance_clearances.append(clearance)
            all_clearances.append(clearance)
            proxy_count += 1
        instance_audits.append(
            {
                "id": item["id"],
                "asset_id": item["asset_id"],
                "classification": "rigid_static_proxy",
                "proxy_count": len(source_meshes),
                "minimum_route_clearance_m": float(min(instance_clearances)),
            }
        )
    minimum_clearance = float(min(all_clearances)) if all_clearances else None
    complete = bool(
        proxy_count > 0
        and minimum_clearance is not None
        and minimum_clearance >= required_clearance
        and len(instance_audits) == len(config["near_field"]["instances"])
    )
    scope.GetPrim().CreateAttribute("kino:complete", Sdf.ValueTypeNames.Bool).Set(complete)
    scope.GetPrim().CreateAttribute("kino:proxyCount", Sdf.ValueTypeNames.Int).Set(proxy_count)
    stage.GetRootLayer().Save()
    return {
        "enabled": True,
        "classification_complete": len(instance_audits)
        == len(config["near_field"]["instances"]),
        "proxy_count": proxy_count,
        "minimum_route_clearance_m": minimum_clearance,
        "required_route_clearance_m": required_clearance,
        "instances": instance_audits,
        "complete": complete,
    }


def _load_composition_config(
    config_path: Path, *, root: Path, stack: tuple[Path, ...] = ()
) -> tuple[dict[str, Any], list[Path]]:
    config_path = config_path.resolve()
    if config_path in stack:
        raise ValueError(f"forest hybrid config inheritance cycle: {config_path}")
    requested = json.loads(config_path.read_text(encoding="utf-8"))
    schema = requested.get("schema_version")
    if schema == "kinofail.forest-hybrid-isaac-composition.v2-development":
        return requested, []
    if schema not in {
        "kinofail.forest-hybrid-isaac-composition.v3-development",
        "kinofail.forest-hybrid-isaac-composition.v4-development",
        "kinofail.forest-hybrid-isaac-composition.v5-development",
        "kinofail.forest-hybrid-isaac-composition.v6-development",
        "kinofail.forest-hybrid-isaac-composition.v7-development",
    }:
        raise ValueError("unsupported forest-hybrid composition schema")
    base_path = _resolve_frozen(root, requested["extends"], "forest hybrid base config")
    base, lineage = _load_composition_config(
        base_path, root=root, stack=(*stack, config_path)
    )
    merged = _deep_merge(base, requested["overrides"])
    merged["schema_version"] = schema
    return merged, [base_path, *lineage]


def compile_forest_hybrid_scene(config_path: Path, *, root: Path) -> dict[str, Any]:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    config_path = config_path.resolve()
    requested_config = json.loads(config_path.read_text(encoding="utf-8"))
    requested_schema = requested_config.get("schema_version")
    config, inheritance_lineage = _load_composition_config(config_path, root=root)
    base_config_path = inheritance_lineage[0] if inheritance_lineage else None
    if requested_schema not in {
        "kinofail.forest-hybrid-isaac-composition.v2-development",
        "kinofail.forest-hybrid-isaac-composition.v3-development",
        "kinofail.forest-hybrid-isaac-composition.v4-development",
        "kinofail.forest-hybrid-isaac-composition.v5-development",
        "kinofail.forest-hybrid-isaac-composition.v6-development",
        "kinofail.forest-hybrid-isaac-composition.v7-development",
    }:
        raise ValueError("unsupported forest-hybrid composition schema")
    config["near_field"]["instances"] = _resolved_prop_instances(config)
    panorama = _resolve_frozen(root, config["background"]["panorama"], "panorama")
    preflight_path = _resolve_frozen(root, config["background"]["preflight"], "panorama preflight")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("passed") is not True:
        raise ValueError("EmbodiedGen panorama preflight did not pass")
    prop_lock_path = _resolve_frozen(
        root, config["near_field"]["prop_asset_lock"], "forest prop lock"
    )
    prop_lock = json.loads(prop_lock_path.read_text(encoding="utf-8"))
    if prop_lock.get("schema_version") != "kinofail.forest-prop-lock.v1":
        raise ValueError("unsupported forest prop lock")
    for asset in prop_lock["assets"]:
        for record in asset["files"]:
            path = Path(record["path"])
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise ValueError(f"missing or stale forest prop file: {path}")
    route_lock_spec = config["appearance"].get(
        "route_material_lock", config["appearance"]["material_lock"]
    )
    surrounding_lock_spec = config["appearance"].get(
        "surrounding_material_lock", config["appearance"]["material_lock"]
    )
    route_material_lock_path = _resolve_frozen(
        root, route_lock_spec, "route terrain material lock"
    )
    surrounding_material_lock_path = _resolve_frozen(
        root, surrounding_lock_spec, "surrounding terrain material lock"
    )
    route_material = _load_material(
        root, route_material_lock_path, config["appearance"]["route_material_id"]
    )
    surrounding_material = _load_material(
        root,
        surrounding_material_lock_path,
        config["appearance"].get(
            "surrounding_material_id", config["appearance"]["route_material_id"]
        ),
    )
    route_geometry = route_surface_geometry(config["route"])
    surrounding = surrounding_rectangles(config["near_field"]["extent_xy_m"], route_geometry)

    output = (root / config["output"]["directory"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite forest hybrid attempt: {output}")
    output.mkdir(parents=True)
    paths = {
        "background.usda": output / "background.usda",
        "props.usda": output / "props.usda",
        "prop_collision.usda": output / "prop_collision.usda",
        "appearance.usda": output / "appearance.usda",
        "collision.usda": output / "collision.usda",
        "route.usda": output / "route.usda",
        "episode.usda": output / "episode.usda",
    }
    _author_background(paths["background.usda"], config, panorama)
    prop_material_audit = _author_props(paths["props.usda"], config, prop_lock)
    prop_collision_audit = _author_prop_collision(
        paths["prop_collision.usda"],
        config=config,
        lock=prop_lock,
        route_geometry=route_geometry,
    )
    appearance_audit = _author_appearance(
        paths["appearance.usda"],
        config=config,
        route_geometry=route_geometry,
        route_material=route_material,
        surrounding_material=surrounding_material,
        surrounding=surrounding,
    )
    _author_collision(paths["collision.usda"], config["route"], route_geometry)
    _author_route(paths["route.usda"], config["route"])
    episode = Sdf.Layer.CreateNew(str(paths["episode.usda"]))
    episode.subLayerPaths = [
        "background.usda",
        "props.usda",
        "prop_collision.usda",
        "appearance.usda",
        "collision.usda",
        "route.usda",
    ]
    episode.defaultPrim = "KinoScene"
    episode.Save()

    stage = Usd.Stage.Open(str(paths["episode.usda"]))
    collision_prims = [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    rigid_prims = [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    route_surface = stage.GetPrimAtPath("/KinoScene/Appearance/RouteSurface")
    surrounding_scope = stage.GetPrimAtPath("/KinoScene/Appearance/SurroundingGround")
    prop_scope = stage.GetPrimAtPath("/KinoScene/NearFieldProps")
    dome = stage.GetPrimAtPath("/KinoScene/Background/EmbodiedGenDome")
    checks = {
        "embodiedgen_panorama_preflight_passed": preflight.get("passed") is True,
        "failed_single_panorama_mesh_not_reused": config["background"]["use_reconstructed_mesh"]
        is False,
        "infinite_background_present": dome.IsValid(),
        "surrounding_near_field_present": surrounding_scope.IsValid(),
        "route_surface_present": route_surface.IsValid(),
        "route_hole_partition_has_four_rectangles": (
            bool(config["near_field"].get("opaque_composited_ground", {}).get("enabled", False))
            or len(surrounding) == 4
        ),
        "near_field_strictly_contains_route": True,
        "metric_visual_props_present": prop_scope.IsValid()
        and len(config["near_field"]["instances"]) >= 12,
        "all_props_resolve_from_frozen_lock": all(
            item["asset_id"] in {record["id"] for record in prop_lock["assets"]}
            for item in config["near_field"]["instances"]
        ),
        "isaac_preview_material_overrides_bound": (
            prop_material_audit["mode"] != "isaac_preview_override_v1"
            or prop_material_audit["bound_target_count"] >= len(config["near_field"]["instances"])
        ),
        "all_collision_prims_are_kino_owned": bool(collision_prims)
        and "/KinoScene/Collision/Floor" in collision_prims
        and all(
            path == "/KinoScene/Collision/Floor"
            or path.startswith("/KinoScene/PropCollision/")
            for path in collision_prims
        ),
        "prop_collision_classification_complete": (
            not config["near_field"].get("prop_collision_policy", {}).get("enabled", False)
            or prop_collision_audit["classification_complete"]
        ),
        "rigid_prop_collision_proxies_complete": (
            not config["near_field"].get("prop_collision_policy", {}).get("enabled", False)
            or prop_collision_audit["complete"]
        ),
        "no_rigid_body_props": not rigid_prims,
        "route_material_matches_preregistered_split_and_domain": (
            route_material["split"]
            == config["appearance"].get("route_expected_split", "train")
            and "wild" in route_material["domains"]
        ),
        "surrounding_material_matches_preregistered_split_and_domain": (
            surrounding_material["split"]
            == config["appearance"].get("surrounding_expected_split", "train")
            and "wild" in surrounding_material["domains"]
        ),
        "naturalistic_surrounding_relief_authored": (
            not config["near_field"].get("visual_height_variation", {}).get("enabled", False)
            or appearance_audit["visual_height_variation_enabled"]
        ),
        "route_visual_blend_contract_authored": (
            not (
                config["near_field"].get("naturalized_route_visual", {}).get(
                    "blend_bands"
                )
                or config["near_field"].get("naturalized_route_visual", {}).get(
                    "feathered_edge", False
                )
            )
            or appearance_audit["route"].get("alpha_blend_band_count", 0) >= 3
            or appearance_audit["route"].get("feathered_visual_edge", False)
            or appearance_audit["route"].get("opaque_composited_ground", False)
        ),
        "opaque_ground_has_no_transparent_pixels": (
            appearance_audit.get("opaque_composite") is None
            or appearance_audit["opaque_composite"].get("transparent_pixels") == 0
        ),
        "collision_boundary_lies_inside_texture_transition": (
            appearance_audit.get("opaque_composite") is None
            or 0.02
            < appearance_audit["opaque_composite"].get(
                "physical_route_boundary_weight_mean", -1.0
            )
            < 0.98
        ),
    }
    result = {
        "schema_version": (
            "kinofail.forest-hybrid-compiled-scene.v7-development"
            if requested_schema.endswith("v7-development")
            else (
                "kinofail.forest-hybrid-compiled-scene.v6-development"
                if requested_schema.endswith("v6-development")
                else (
                    "kinofail.forest-hybrid-compiled-scene.v5-development"
                    if requested_schema.endswith("v5-development")
                    else (
                        "kinofail.forest-hybrid-compiled-scene.v4-development"
                        if requested_schema.endswith("v4-development")
                        else (
                            "kinofail.forest-hybrid-compiled-scene.v3-development"
                            if requested_schema.endswith("v3-development")
                            else SCHEMA_VERSION
                        )
                    )
                )
            )
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "scene_id": config["scene_id"],
        "source_scene_id": config["source_scene_id"],
        "development_only": True,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "extends": str(base_config_path) if base_config_path else None,
            "extends_sha256": sha256_file(base_config_path) if base_config_path else None,
            "inheritance_lineage": [str(path) for path in inheritance_lineage],
        },
        "background_contract": {
            "type": config["background"]["type"],
            "panorama": str(panorama),
            "panorama_sha256": sha256_file(panorama),
            "reconstructed_mesh_used": False,
            "parallax_claim": "none_at_infinity_background",
        },
        "near_field_contract": {
            "metric_pbr_ground": True,
            "route_hole_rectangles": surrounding,
            "visual_ground_topology": (
                "single_opaque_continuous_mesh"
                if appearance_audit.get("opaque_composite") is not None
                else "route_plus_four_surrounding_meshes"
            ),
            "prop_instances": config["near_field"]["instances"],
            "prop_asset_lock": str(prop_lock_path),
            "material_compatibility": prop_material_audit,
            "prop_collision": prop_collision_audit,
            "appearance_mesh": appearance_audit,
            "route_material_id": route_material["id"],
            "surrounding_material_id": surrounding_material["id"],
            "route_material_split": route_material["split"],
            "surrounding_material_split": surrounding_material["split"],
            "route_expected_split": config["appearance"].get(
                "route_expected_split", "train"
            ),
            "surrounding_expected_split": config["appearance"].get(
                "surrounding_expected_split", "train"
            ),
            "route_material_lock": str(route_material_lock_path),
            "surrounding_material_lock": str(surrounding_material_lock_path),
            "prop_collision_proxies_complete": prop_collision_audit.get("complete", False),
        },
        "physics_contract": {
            "owned_by": "Kino-Fail",
            "collision_prims": collision_prims,
            "route_geometry": route_geometry,
            "operator_layer_may_override": True,
        },
        "route": config["route"],
        "checks": {key: bool(value) for key, value in checks.items()},
        "files": {name: sha256_file(path) for name, path in paths.items()},
        "passed": all(bool(value) for value in checks.values()),
        "scene_registry_eligible": False,
        "counts_as_a0_a7_evidence": False,
        "remaining_gates": [
            "body_fixed_go2_full_route_rtx_coverage",
            "articulated_go2_route_traversal_qa",
            "operator_nominal_anomaly_pair",
            "three_synchronized_appearance_views",
            *(
                []
                if prop_collision_audit.get("complete", False)
                else ["prop_collision_proxy_audit"]
            ),
            "physical_go2_fixture_camera_calibration",
            "independent_human_scene_review",
            "formal_schedule_binding",
        ],
    }
    audit_path = output / "compiled_scene_audit.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**result, "audit_path": str(audit_path), "episode_usd": str(paths["episode.usda"])}
