"""Causal, texture-independent visual cues for the realistic C1 matched battery.

The base scanned-PBR material is deliberately shared between the O2 and O4
counterfactual renders.  The only class-dependent signal is physical appearance:
an undulating/depressed soft surface versus overlapping protective-film panels
with creases.  All authored prims are visual-only and therefore cannot perturb
the shared pre-contact proprioceptive prefix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CauseVisual:
    root_path: str
    texture_target_paths: tuple[str, ...]
    cue_kind: str


@dataclass(frozen=True)
class C1CauseVisuals:
    compliant: CauseVisual
    adhesion: CauseVisual
    shared_edge_material_path: str


def _world_xy(frame: Any, progress_m: float, lateral_m: float) -> tuple[float, float]:
    value = frame.point(float(progress_m), float(lateral_m))
    return float(value[0]), float(value[1])


def _shared_edge_material(stage: Any, path: str, seed: int) -> Any:
    from pxr import Gf, Sdf, UsdShade

    phase = (int(seed) % 997) / 997.0
    value = 0.34 + 0.08 * phase
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PBR")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(value, value * 0.98, value * 0.94)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.56)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.02)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    return material


def _tag(prim: Any, label: str) -> None:
    try:
        from isaacsim.core.utils.semantics import add_update_semantics
    except ImportError:  # pragma: no cover - older Isaac
        from omni.isaac.core.utils.semantics import add_update_semantics

    add_update_semantics(
        prim, semantic_label=label, type_label="class"
    )


def _author_compliant(
    stage: Any,
    frame: Any,
    *,
    root_path: str,
    start_progress_m: float,
    length_m: float,
    half_width_m: float,
    base_z_m: float,
    seed: int,
    edge_material: Any,
) -> CauseVisual:
    from pxr import Gf, UsdGeom, UsdShade

    root = UsdGeom.Xform.Define(stage, root_path)
    mesh_path = f"{root_path}/SoftSurface"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    # A dense, subdivided sheet produces a genuinely smooth compliant surface.
    # The earlier coarse mesh plus thin ridge cubes projected to the same
    # line-like cue as O4, which made the two causes non-identifiable under
    # camera-pose nuisance.
    n_progress, n_lateral = 25, 15
    phase = 2.0 * math.pi * ((int(seed) % 1049) / 1049.0)
    points = []
    for i in range(n_progress):
        u = i / (n_progress - 1)
        progress = start_progress_m + u * length_m
        for j in range(n_lateral):
            v = -1.0 + 2.0 * j / (n_lateral - 1)
            lateral = v * half_width_m
            x, y = _world_xy(frame, progress, lateral)
            ripple = math.sin(2.5 * math.pi * u + phase) * math.cos(
                1.2 * math.pi * v - 0.3 * phase
            )
            z = base_z_m + 0.012 + 0.0025 * ripple
            points.append(Gf.Vec3f(x, y, z))
    counts: list[int] = []
    indices: list[int] = []
    for i in range(n_progress - 1):
        for j in range(n_lateral - 1):
            a = i * n_lateral + j
            b = a + n_lateral
            counts.append(4)
            indices.extend((a, b, b + 1, a + 1))
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("catmullClark")
    mesh.CreateDoubleSidedAttr(True)
    _tag(mesh.GetPrim(), "c1_visible_cause")
    texture_targets = [mesh_path]
    # Broad smooth pillows are a generic geometric cue for a yielding sheet.
    # Unlike the O4 film creases, they contain no thin line primitives.
    for index, (u, v, sx, sy, sz) in enumerate(
        (
            (0.38, -0.25, 0.19, 0.24, 0.030),
            (0.70, 0.22, 0.16, 0.21, 0.024),
        )
    ):
        x, y = _world_xy(
            frame,
            start_progress_m + u * length_m,
            v * half_width_m,
        )
        path = f"{root_path}/SoftPillow_{index}"
        pillow = UsdGeom.Sphere.Define(stage, path)
        pillow.CreateRadiusAttr(1.0)
        xf = UsdGeom.XformCommonAPI(pillow)
        xf.SetTranslate(Gf.Vec3d(x, y, base_z_m + 0.004))
        xf.SetScale(Gf.Vec3f(sx, sy, sz))
        _tag(pillow.GetPrim(), "c1_visible_cause")
        texture_targets.append(path)
    return CauseVisual(
        root_path=str(root.GetPrim().GetPath()),
        texture_target_paths=tuple(texture_targets),
        cue_kind="soft_surface_deformation",
    )


def _author_adhesion(
    stage: Any,
    frame: Any,
    *,
    root_path: str,
    start_progress_m: float,
    length_m: float,
    half_width_m: float,
    base_z_m: float,
    seed: int,
    edge_material: Any,
) -> CauseVisual:
    from pxr import Gf, UsdGeom, UsdShade

    root = UsdGeom.Xform.Define(stage, root_path)
    route_yaw = math.degrees(float(frame.heading_rad))
    jitter = ((int(seed) % 607) / 607.0 - 0.5) * 8.0
    texture_targets: list[str] = []
    panels = (
        (0.31, -0.11, 0.52, 0.78, -2.0),
        (0.62, 0.10, 0.46, 0.72, 3.5),
        (0.84, -0.04, 0.31, 0.64, -4.0),
    )
    for index, (u, lateral_fraction, panel_length, width_fraction, yaw) in enumerate(
        panels
    ):
        x, y = _world_xy(
            frame,
            start_progress_m + u * length_m,
            lateral_fraction * half_width_m,
        )
        path = f"{root_path}/FilmPanel_{index}"
        panel = UsdGeom.Cube.Define(stage, path)
        panel.CreateSizeAttr(1.0)
        xf = UsdGeom.XformCommonAPI(panel)
        xf.SetTranslate(
            Gf.Vec3d(x, y, base_z_m + 0.016 + 0.0018 * index)
        )
        xf.SetRotate(
            Gf.Vec3f(0.0, 0.0, route_yaw + yaw + 0.15 * jitter)
        )
        xf.SetScale(
            Gf.Vec3f(
                panel_length,
                max(0.18, width_fraction * 2.0 * half_width_m),
                0.0035,
            )
        )
        _tag(panel.GetPrim(), "c1_visible_cause")
        texture_targets.append(path)

    for index, (u, v, length, width, yaw) in enumerate(
        (
            (0.38, 0.16, 0.38, 0.008, -13.0),
            (0.57, -0.14, 0.46, 0.010, 8.0),
            (0.78, 0.08, 0.31, 0.007, -4.0),
            (0.90, -0.25, 0.22, 0.009, 17.0),
        )
    ):
        x, y = _world_xy(
            frame,
            start_progress_m + u * length_m,
            v * half_width_m,
        )
        crease = UsdGeom.Cube.Define(stage, f"{root_path}/Crease_{index}")
        crease.CreateSizeAttr(1.0)
        xf = UsdGeom.XformCommonAPI(crease)
        xf.SetTranslate(Gf.Vec3d(x, y, base_z_m + 0.024))
        xf.SetRotate(
            Gf.Vec3f(0.0, 0.0, route_yaw + yaw - 0.15 * jitter)
        )
        xf.SetScale(Gf.Vec3f(length, width, 0.0025))
        UsdShade.MaterialBindingAPI.Apply(crease.GetPrim()).Bind(edge_material)
        _tag(crease.GetPrim(), "c1_visible_cause")
    return CauseVisual(
        root_path=str(root.GetPrim().GetPath()),
        texture_target_paths=tuple(texture_targets),
        cue_kind="overlapping_film_and_creases",
    )


def author_c1_cause_visuals(
    stage: Any,
    frame: Any,
    *,
    start_progress_m: float = 0.62,
    length_m: float = 0.90,
    half_width_m: float = 0.45,
    base_z_m: float = 0.006,
    seed: int = 0,
) -> C1CauseVisuals:
    """Author the paired visual-only causes and leave both invisible."""
    from pxr import UsdGeom

    edge_path = "/World/Looks/C1SharedCauseEdge"
    edge_material = _shared_edge_material(stage, edge_path, seed)
    compliant = _author_compliant(
        stage,
        frame,
        root_path="/World/C1CompliantCause",
        start_progress_m=start_progress_m,
        length_m=length_m,
        half_width_m=half_width_m,
        base_z_m=base_z_m,
        seed=seed,
        edge_material=edge_material,
    )
    adhesion = _author_adhesion(
        stage,
        frame,
        root_path="/World/C1AdhesionCause",
        start_progress_m=start_progress_m,
        length_m=length_m,
        half_width_m=half_width_m,
        base_z_m=base_z_m,
        seed=seed,
        edge_material=edge_material,
    )
    UsdGeom.Imageable(
        stage.GetPrimAtPath(compliant.root_path)
    ).MakeInvisible()
    UsdGeom.Imageable(
        stage.GetPrimAtPath(adhesion.root_path)
    ).MakeInvisible()
    return C1CauseVisuals(
        compliant=compliant,
        adhesion=adhesion,
        shared_edge_material_path=edge_path,
    )


def show_c1_cause(stage: Any, visuals: C1CauseVisuals, cause: str) -> CauseVisual:
    """Show exactly one cause root and return its binding targets."""
    from pxr import Usd, UsdGeom

    if cause not in {"O2_compliance", "O4_tether"}:
        raise ValueError(f"unsupported C1 cause {cause!r}")
    selected = (
        visuals.compliant if cause == "O2_compliance" else visuals.adhesion
    )
    for value in (visuals.compliant, visuals.adhesion):
        visible = value is selected
        root = stage.GetPrimAtPath(value.root_path)
        # Author visibility on every imageable descendant.  Root-only
        # visibility was not reliably propagated through the RTX/Fabric
        # population, so O4 crease children leaked into O2 renders.
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(
                    UsdGeom.Tokens.inherited
                    if visible
                    else UsdGeom.Tokens.invisible
                )
    return selected
