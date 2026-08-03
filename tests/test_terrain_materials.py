from __future__ import annotations

import hashlib

import pytest
from PIL import Image

from kino_vla.sim.terrain_materials import (
    PBRMapSet,
    TerrainAppearanceBinding,
    audit_terrain_asset_lock,
    resolve_pbr_maps,
)


def _image(path, color) -> None:
    Image.new("RGB", (8, 8), color).save(path)


def _entry(root, path) -> dict:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def test_resolve_prefers_opengl_normal_and_finds_required_maps(tmp_path) -> None:
    material = tmp_path / "mat"
    material.mkdir()
    _image(material / "Asset_Color.jpg", "brown")
    _image(material / "Asset_NormalDX.jpg", "blue")
    _image(material / "Asset_NormalGL.jpg", "cyan")
    _image(material / "Asset_Roughness.jpg", "gray")
    _image(material / "Asset_Displacement.jpg", "black")
    maps = resolve_pbr_maps("mat", material)
    assert maps.normal.name == "Asset_NormalGL.jpg"
    assert maps.displacement is not None


def test_asset_lock_audit_detects_hash_mismatch(tmp_path) -> None:
    material = tmp_path / "mat"
    material.mkdir()
    paths = {}
    for key, name in (
        ("basecolor", "Asset_Color.jpg"),
        ("normal", "Asset_NormalGL.jpg"),
        ("roughness", "Asset_Roughness.jpg"),
    ):
        path = material / name
        _image(path, "gray")
        paths[key] = _entry(tmp_path, path)
    lock = {
        "materials": [
            {
                "id": "mat",
                "license": "CC0",
                "maps": {**paths, "displacement": None, "ambient_occlusion": None},
            }
        ]
    }
    assert audit_terrain_asset_lock(lock, asset_root=tmp_path)["passed"]
    (material / "Asset_Color.jpg").write_bytes(b"corrupt")
    audit = audit_terrain_asset_lock(lock, asset_root=tmp_path)
    assert not audit["passed"]
    assert audit["materials"]["mat"]["hash_mismatch"] == ["basecolor"]


def test_metric_omnipbr_parameters_keep_visual_physics_separate(tmp_path) -> None:
    maps = PBRMapSet(
        material_id="mat",
        directory=tmp_path,
        basecolor=tmp_path / "color.jpg",
        normal=tmp_path / "normal_gl.jpg",
        roughness=tmp_path / "roughness.jpg",
        displacement=None,
        ambient_occlusion=None,
    )
    binding = TerrainAppearanceBinding(
        material_id="mat",
        appearance_id="app_1",
        maps=maps,
        physical_size_m=(2.0, 4.0),
        uv_scale=1.2,
        rotation_deg=90.0,
        surface_state="wet",
        uv_offset=(0.25, 0.75),
        albedo_brightness_multiplier=1.1,
        normal_strength=0.8,
        roughness_multiplier=1.15,
    )
    params = binding.omnipbr_parameters()
    assert params["project_uvw"]
    assert params["world_or_object"]
    assert params["texture_scale"] == (0.6, 0.3)
    assert params["texture_rotate"] == 90.0
    assert params["texture_translate"] == (0.25, 0.75)
    assert params["bump_factor"] == 0.8
    assert params["albedo_brightness"] == pytest.approx(0.68 * 1.1)
    assert params["flip_tangent_v"] is False
    assert params["reflection_roughness_constant"] < 0.2
    assert "mu" not in params and "friction" not in params
