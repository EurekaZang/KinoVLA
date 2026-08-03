from __future__ import annotations

import numpy as np
import pytest
import trimesh

from kino_vla.eval.visual_shell_mesh import (
    create_metric_visual_mesh,
    evaluate_visual_shell_mesh,
)
from scripts.generate_embodiedgen_visual_shell_mesh import _install_mesh_only_sr_shim


def _thresholds() -> dict[str, float | int]:
    return {
        "minimum_vertices": 8,
        "minimum_faces": 12,
        "minimum_colored_vertex_fraction": 0.95,
        "maximum_nonfinite_vertex_fraction": 0.0,
        "maximum_degenerate_face_fraction": 0.01,
        "minimum_raw_percentile_height": 0.1,
        "maximum_raw_percentile_height": 100.0,
        "metric_height_relative_tolerance": 0.02,
    }


def test_mesh_preflight_and_metric_copy(tmp_path) -> None:
    mesh = trimesh.creation.box(extents=(2.0, 4.0, 3.0))
    mesh.visual.vertex_colors = np.tile([80, 120, 160, 255], (len(mesh.vertices), 1))
    raw = tmp_path / "raw.ply"
    metric = tmp_path / "metric.ply"
    mesh.export(raw)

    raw_audit = evaluate_visual_shell_mesh(raw, thresholds=_thresholds())
    transform = create_metric_visual_mesh(raw, metric, target_height_m=8.0)
    metric_audit = evaluate_visual_shell_mesh(
        metric, thresholds=_thresholds(), target_height_m=8.0
    )

    assert raw_audit["passed"] is True
    assert transform["scale"] > 1.0
    assert metric_audit["passed"] is True
    assert metric_audit["metrics"]["metric_height_relative_error"] < 0.02


def test_mesh_preflight_rejects_too_few_faces(tmp_path) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=0)
    mesh.visual.vertex_colors = np.tile([80, 120, 160, 255], (len(mesh.vertices), 1))
    path = tmp_path / "small.ply"
    mesh.export(path)
    thresholds = _thresholds()
    thresholds["minimum_faces"] = 1000

    audit = evaluate_visual_shell_mesh(path, thresholds=thresholds)

    assert audit["passed"] is False
    assert audit["checks"]["minimum_faces"] is False


def test_mesh_only_sr_shim_fails_if_unreachable_component_is_called() -> None:
    _install_mesh_only_sr_shim()
    from embodied_gen.models.sr_model import ImageRealESRGAN

    model = ImageRealESRGAN(outscale=4)
    with pytest.raises(RuntimeError, match="mesh-only compatibility contract violated"):
        model(object())
