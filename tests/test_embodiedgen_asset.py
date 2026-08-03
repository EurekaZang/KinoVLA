from __future__ import annotations

import json
import subprocess
from pathlib import Path

from kino_vla.sim.embodiedgen_asset import (
    EmbodiedGenRoomSource,
    audit_embodiedgen_room_manifest,
    write_embodiedgen_room_manifest,
)


def _source(tmp_path: Path) -> EmbodiedGenRoomSource:
    usd = tmp_path / "usd" / "export_scene" / "export_scene.usda"
    textures = usd.parent / "textures"
    textures.mkdir(parents=True)
    usd.write_text(
        '#usda 1.0\n(defaultPrim = "World")\ndef Xform "World" {}\n', encoding="utf-8"
    )
    (textures / "desk_DIFFUSE.png").write_bytes(b"diffuse")
    (textures / "desk_NORMAL.png").write_bytes(b"normal")
    (textures / "desk_ROUGHNESS.png").write_bytes(b"roughness")
    return EmbodiedGenRoomSource(
        scene_id="indoor_office_02",
        asset_root=tmp_path,
        visual_usd=usd,
        room_type="Office",
        seed=20260721,
        complexity="simple",
        generation_command=(
            "room-cli",
            "-m",
            "embodied_gen.scripts.room_gen.gen_room",
            "--output-root",
            "outputs/scenes",
            "--room-type",
            "office",
            "--seed",
            "20260721",
            "--complexity",
            "simple",
        ),
    )


def test_manifest_freezes_official_source_files_and_visual_only_trust(tmp_path: Path) -> None:
    source = _source(tmp_path)
    path = tmp_path / "source_manifest.json"
    manifest = write_embodiedgen_room_manifest(source, path)
    assert manifest["generator"]["release"] == "v2.0.0"
    assert manifest["generation"]["routing"] == "explicit_no_gpt"
    assert not manifest["trust_boundary"]["source_collision_authoritative"]
    assert len(manifest["asset"]["files"]) == 4
    audit = audit_embodiedgen_room_manifest(path)
    assert audit["passed"], audit
    assert audit["texture_file_count"] == 3


def test_manifest_detects_asset_tampering(tmp_path: Path) -> None:
    source = _source(tmp_path)
    path = tmp_path / "source_manifest.json"
    write_embodiedgen_room_manifest(source, path)
    (source.visual_usd.parent / "textures" / "desk_DIFFUSE.png").write_bytes(b"tampered")
    audit = audit_embodiedgen_room_manifest(path)
    assert not audit["passed"]
    assert any(issue.startswith("asset_file_") for issue in audit["issues"])


def test_manifest_detects_uninventoried_package_file(tmp_path: Path) -> None:
    source = _source(tmp_path)
    path = tmp_path / "source_manifest.json"
    write_embodiedgen_room_manifest(source, path)
    (source.visual_usd.parent / "textures" / "late_file.png").write_bytes(b"not frozen")
    audit = audit_embodiedgen_room_manifest(path)
    assert not audit["passed"]
    assert "asset_file_inventory_not_exact" in audit["issues"]


def test_manifest_rejects_physics_authority_escalation(tmp_path: Path) -> None:
    source = _source(tmp_path)
    path = tmp_path / "source_manifest.json"
    write_embodiedgen_room_manifest(source, path)
    manifest = json.loads(path.read_text())
    manifest["trust_boundary"]["source_collision_authoritative"] = True
    path.write_text(json.dumps(manifest), encoding="utf-8")
    audit = audit_embodiedgen_room_manifest(path)
    assert not audit["passed"]
    assert "unsafe_trust_boundary_source_collision_authoritative" in audit["issues"]


def test_manifest_rejects_gpt_routed_formal_source(tmp_path: Path) -> None:
    source = _source(tmp_path)
    source = EmbodiedGenRoomSource(
        **{
            **source.__dict__,
            "generation_command": ("room-cli", "--prompt", "make an office"),
        }
    )
    path = tmp_path / "source_manifest.json"
    try:
        write_embodiedgen_room_manifest(source, path)
    except ValueError as exc:
        assert "freeze room type" in str(exc) or "GPT routing" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("GPT-routed source was accepted")


def test_generation_entrypoint_exposes_frozen_explicit_cli() -> None:
    result = subprocess.run(
        ["python", "scripts/generate_embodiedgen_room_source.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--embodiedgen-root" in result.stdout
    assert "--room-type" in result.stdout
    assert "--seed" in result.stdout
    assert "--complexity" in result.stdout
    assert "--skip-generation" in result.stdout
