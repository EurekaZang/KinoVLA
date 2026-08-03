"""Geometry-only checks for development EmbodiedGen visual-shell meshes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import trimesh

from kino_vla.eval.visual_shell_preflight import sha256_file


SCHEMA_VERSION = "kinofail.embodiedgen-visual-shell-mesh-preflight.v1"


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("mesh scene contains no geometry")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"unsupported mesh type: {type(loaded).__name__}")
    return loaded


def create_metric_visual_mesh(
    raw_path: Path, output_path: Path, *, target_height_m: float
) -> dict[str, float]:
    """Create a metric visual-shell copy using EmbodiedGen's robust height convention."""

    mesh = _load_mesh(raw_path).copy()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("raw mesh vertices must be finite Nx3 values")
    low, high = np.percentile(vertices[:, 1], [1.0, 99.0])
    raw_height = float(high - low)
    if raw_height <= 0.0:
        raise ValueError("raw mesh has zero robust vertical extent")
    scale = float(target_height_m) / raw_height
    # EmbodiedGen restores its scene orientation with a 180-degree rotation around Y.
    vertices[:, 0] *= -1.0
    vertices[:, 2] *= -1.0
    vertices[:, 1] -= low
    vertices *= scale
    mesh.vertices = vertices
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    return {
        "raw_height_p01_p99": raw_height,
        "target_height_m": float(target_height_m),
        "scale": scale,
    }


def evaluate_visual_shell_mesh(
    mesh_path: Path,
    *,
    thresholds: Mapping[str, float | int],
    target_height_m: float | None = None,
) -> dict[str, Any]:
    mesh = _load_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    finite_vertices = np.isfinite(vertices).all(axis=1)
    nonfinite_fraction = float(1.0 - finite_vertices.mean()) if len(vertices) else 1.0

    robust_height = float("nan")
    bounds = None
    if len(vertices) and finite_vertices.any():
        finite = vertices[finite_vertices]
        robust_height = float(np.percentile(finite[:, 1], 99) - np.percentile(finite[:, 1], 1))
        bounds = {
            "minimum": finite.min(axis=0).tolist(),
            "maximum": finite.max(axis=0).tolist(),
            "extent": np.ptp(finite, axis=0).tolist(),
        }

    valid_face_indices = (
        faces.ndim == 2
        and faces.shape[1] == 3
        and len(vertices) > 0
        and np.all((faces >= 0) & (faces < len(vertices)))
    )
    if valid_face_indices and len(faces):
        a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        double_area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
        degenerate_fraction = float(np.mean(~np.isfinite(double_area) | (double_area <= 1e-12)))
    else:
        degenerate_fraction = 1.0

    colors = getattr(mesh.visual, "vertex_colors", None)
    colored_vertex_fraction = (
        float(min(len(colors), len(vertices)) / len(vertices))
        if colors is not None and len(vertices)
        else 0.0
    )
    metrics = {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "colored_vertex_fraction": colored_vertex_fraction,
        "nonfinite_vertex_fraction": nonfinite_fraction,
        "degenerate_face_fraction": degenerate_fraction,
        "robust_height_axis_y_p01_p99": robust_height,
        "bounds": bounds,
    }
    checks = {
        "minimum_vertices": metrics["vertices"] >= int(thresholds["minimum_vertices"]),
        "minimum_faces": metrics["faces"] >= int(thresholds["minimum_faces"]),
        "colored_vertices": colored_vertex_fraction
        >= float(thresholds["minimum_colored_vertex_fraction"]),
        "finite_vertices": nonfinite_fraction
        <= float(thresholds["maximum_nonfinite_vertex_fraction"]),
        "nondegenerate_faces": degenerate_fraction
        <= float(thresholds["maximum_degenerate_face_fraction"]),
    }
    if target_height_m is None:
        checks["raw_height_plausible"] = (
            np.isfinite(robust_height)
            and robust_height >= float(thresholds["minimum_raw_percentile_height"])
            and robust_height <= float(thresholds["maximum_raw_percentile_height"])
        )
    else:
        relative_error = abs(robust_height - float(target_height_m)) / float(target_height_m)
        metrics["target_height_m"] = float(target_height_m)
        metrics["metric_height_relative_error"] = float(relative_error)
        checks["metric_height"] = relative_error <= float(
            thresholds["metric_height_relative_tolerance"]
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "mesh": {"path": str(mesh_path), "sha256": sha256_file(mesh_path)},
        "metrics": metrics,
        "thresholds": dict(thresholds),
        "checks": checks,
        "passed": all(bool(value) for value in checks.values()),
        "interpretation": (
            "Development geometry gate only. Passing does not establish route traversability, "
            "collision fidelity, Go2 camera validity, operator capability, corpus eligibility, "
            "A0-A7 reproduction, or sim-to-real transfer."
        ),
    }
