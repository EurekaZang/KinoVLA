"""Auditable EmbodiedGen RoomGen visual-source contract for Kino-Fail.

EmbodiedGen supplies scene appearance and layout only.  Kino-Fail owns collision, traversability,
physical parameters, anomaly operators, sensors, and causal labels.  This module makes that
boundary machine-checkable before a generated room is admitted to a benchmark episode.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "kinofail.embodiedgen-room-source.v1"
OFFICIAL_REPOSITORY = "https://github.com/HorizonRobotics/EmbodiedGen"
OFFICIAL_RELEASE = "v2.0.0"
OFFICIAL_COMMIT = "cc3015ca5ccdacf94df3428d9e65f79375982216"
OFFICIAL_INFINIGEN_COMMIT = "892947be71c4ff102c6a558d088ffe149561c11f"
LICENSE_SPDX = "Apache-2.0"
SOURCE_KIND = "embodiedgen_v2_roomgen_visual_layout_only"
COMPATIBILITY_SHIM_ID = "gin_namespace_filesystem_reader_v2"
COMPATIBILITY_SHIM_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts/compat/embodiedgen_sitecustomize/sitecustomize.py"
)

ROOM_TYPES = frozenset(
    {"Bedroom", "LivingRoom", "Kitchen", "Bathroom", "DiningRoom", "Office", "House"}
)
ROOM_TYPE_CLI = {
    "Bedroom": "bedroom",
    "LivingRoom": "livingRoom",
    "Kitchen": "kitchen",
    "Bathroom": "bathroom",
    "DiningRoom": "diningRoom",
    "Office": "office",
    "House": "house",
}
COMPLEXITIES = frozenset({"minimalist", "simple", "medium", "detail"})
USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})
TEXTURE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".exr", ".hdr", ".tif", ".tiff"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_asset_path(value: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _command_option(command: list[str], flag: str) -> str | None:
    try:
        index = command.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return command[index + 1]


@dataclass(frozen=True)
class EmbodiedGenRoomSource:
    """Frozen provenance needed to admit one RoomGen output as a visual source."""

    scene_id: str
    asset_root: Path
    visual_usd: Path
    room_type: str
    seed: int
    complexity: str
    generation_command: tuple[str, ...]
    repository: str = OFFICIAL_REPOSITORY
    release: str = OFFICIAL_RELEASE
    commit: str = OFFICIAL_COMMIT
    infinigen_commit: str = OFFICIAL_INFINIGEN_COMMIT
    license_spdx: str = LICENSE_SPDX
    centered_on_export: bool = True


def _asset_package_files(asset_root: Path, visual_usd: Path) -> list[Path]:
    """Return the self-contained USD package files, rejecting links that escape its root."""
    asset_root = asset_root.resolve()
    visual_usd = visual_usd.resolve()
    if not visual_usd.is_relative_to(asset_root):
        raise ValueError("visual USD must live below asset_root")
    package_root = visual_usd.parent
    files: list[Path] = []
    for path in sorted(package_root.rglob("*")):
        if path.is_symlink():
            resolved = path.resolve()
            if not resolved.is_relative_to(asset_root):
                raise ValueError(f"asset symlink escapes root: {path}")
        if path.is_file():
            files.append(path)
    return files


def write_embodiedgen_room_manifest(
    source: EmbodiedGenRoomSource,
    output_path: str | Path,
) -> dict[str, Any]:
    """Hash a generated RoomGen USD package and write its immutable source manifest."""
    asset_root = source.asset_root.resolve()
    visual_usd = source.visual_usd.resolve()
    output_path = Path(output_path)
    if source.room_type not in ROOM_TYPES:
        raise ValueError(f"unsupported RoomGen room type: {source.room_type}")
    if source.complexity not in COMPLEXITIES:
        raise ValueError(f"unsupported RoomGen complexity: {source.complexity}")
    if not isinstance(source.seed, int) or isinstance(source.seed, bool):
        raise ValueError("RoomGen seed must be an integer")
    if not visual_usd.is_file() or visual_usd.suffix.lower() not in USD_SUFFIXES:
        raise FileNotFoundError(f"RoomGen visual USD missing: {visual_usd}")
    if source.repository != OFFICIAL_REPOSITORY:
        raise ValueError("unreviewed EmbodiedGen repository")
    if source.release != OFFICIAL_RELEASE or source.commit != OFFICIAL_COMMIT:
        raise ValueError("EmbodiedGen source is not the frozen v2.0.0 release")
    if source.infinigen_commit != OFFICIAL_INFINIGEN_COMMIT:
        raise ValueError("Infinigen source is not the frozen EmbodiedGen V2 submodule commit")
    if source.license_spdx != LICENSE_SPDX:
        raise ValueError("unexpected EmbodiedGen license")
    if not source.centered_on_export:
        raise ValueError("RoomGen export must use --center-scene")

    command = list(source.generation_command)
    command_text = " ".join(command)
    if (
        _command_option(command, "--room-type") != ROOM_TYPE_CLI[source.room_type]
        or _command_option(command, "--seed") != str(source.seed)
        or _command_option(command, "--complexity") != source.complexity
    ):
        raise ValueError("generation command does not freeze room type, seed, and complexity")
    if "--prompt" in command or "--prompt=" in command_text:
        raise ValueError("formal RoomGen sources must use explicit parameters, not GPT routing")

    package_files = _asset_package_files(asset_root, visual_usd)
    file_rows = []
    for path in package_files:
        relative = path.relative_to(asset_root).as_posix()
        file_rows.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    main_usd = visual_usd.relative_to(asset_root).as_posix()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "scene_id": source.scene_id,
        "source_kind": SOURCE_KIND,
        "generator": {
            "name": "EmbodiedGen RoomGen",
            "repository": source.repository,
            "release": source.release,
            "commit": source.commit,
            "infinigen_commit": source.infinigen_commit,
            "license_spdx": source.license_spdx,
        },
        "generation": {
            "routing": "explicit_no_gpt",
            "room_type": source.room_type,
            "seed": source.seed,
            "complexity": source.complexity,
            "centered_on_export": source.centered_on_export,
            "command": command,
            "compatibility_shim": {
                "id": COMPATIBILITY_SHIM_ID,
                "path": "scripts/compat/embodiedgen_sitecustomize/sitecustomize.py",
                "sha256": _sha256(COMPATIBILITY_SHIM_PATH),
                "modifies_upstream_source": False,
            },
        },
        "asset": {"main_usd": main_usd, "files": file_rows},
        "trust_boundary": {
            "appearance_and_layout_source": "EmbodiedGen RoomGen",
            "source_collision_authoritative": False,
            "source_mass_inertia_authoritative": False,
            "source_contact_parameters_authoritative": False,
            "kinofail_collision_required": True,
            "kinofail_route_audit_required": True,
            "kinofail_operator_physics_required": True,
        },
        "admission_state": "source_integrity_only_pending_isaac_scene_qa",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def audit_embodiedgen_room_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Recompute source integrity and enforce the visual-only physics trust boundary."""
    manifest_path = Path(manifest_path)
    issues: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "issues": [f"manifest_unreadable:{type(exc).__name__}"]}

    if manifest.get("schema_version") != SCHEMA_VERSION:
        issues.append("schema_version_mismatch")
    scene_id = manifest.get("scene_id")
    if not isinstance(scene_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9_\-]*", scene_id) is None:
        issues.append("invalid_scene_id")
    if manifest.get("source_kind") != SOURCE_KIND:
        issues.append("source_kind_mismatch")

    generator = manifest.get("generator", {})
    expected_generator = {
        "name": "EmbodiedGen RoomGen",
        "repository": OFFICIAL_REPOSITORY,
        "release": OFFICIAL_RELEASE,
        "commit": OFFICIAL_COMMIT,
        "infinigen_commit": OFFICIAL_INFINIGEN_COMMIT,
        "license_spdx": LICENSE_SPDX,
    }
    for key, expected in expected_generator.items():
        if generator.get(key) != expected:
            issues.append(f"generator_{key}_mismatch")

    generation = manifest.get("generation", {})
    if generation.get("routing") != "explicit_no_gpt":
        issues.append("non_reproducible_generation_routing")
    if generation.get("room_type") not in ROOM_TYPES:
        issues.append("invalid_room_type")
    if generation.get("complexity") not in COMPLEXITIES:
        issues.append("invalid_complexity")
    seed = generation.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        issues.append("invalid_seed")
    if generation.get("centered_on_export") is not True:
        issues.append("export_not_centered")
    shim = generation.get("compatibility_shim", {})
    if shim.get("id") != COMPATIBILITY_SHIM_ID:
        issues.append("compatibility_shim_id_mismatch")
    if shim.get("modifies_upstream_source") is not False:
        issues.append("compatibility_shim_upstream_mutation")
    if not COMPATIBILITY_SHIM_PATH.is_file():
        issues.append("compatibility_shim_missing")
    elif shim.get("sha256") != _sha256(COMPATIBILITY_SHIM_PATH):
        issues.append("compatibility_shim_hash_mismatch")
    command = generation.get("command", [])
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        issues.append("invalid_generation_command")
        command = []
    if "--prompt" in command or any(item.startswith("--prompt=") for item in command):
        issues.append("gpt_routed_generation_forbidden")
    room_type = generation.get("room_type")
    for flag, value in (
        ("--room-type", ROOM_TYPE_CLI.get(room_type)),
        ("--seed", seed),
        ("--complexity", generation.get("complexity")),
    ):
        if value is None or _command_option(command, flag) != str(value):
            issues.append(f"generation_command_missing_{flag.removeprefix('--').replace('-', '_')}")

    trust = manifest.get("trust_boundary", {})
    for key in (
        "source_collision_authoritative",
        "source_mass_inertia_authoritative",
        "source_contact_parameters_authoritative",
    ):
        if trust.get(key) is not False:
            issues.append(f"unsafe_trust_boundary_{key}")
    for key in (
        "kinofail_collision_required",
        "kinofail_route_audit_required",
        "kinofail_operator_physics_required",
    ):
        if trust.get(key) is not True:
            issues.append(f"missing_trust_boundary_{key}")

    asset_root = manifest_path.parent.resolve()
    asset = manifest.get("asset", {})
    main_usd = asset.get("main_usd")
    if not isinstance(main_usd, str) or not _is_relative_asset_path(main_usd):
        issues.append("invalid_main_usd_path")
        main_path = None
    else:
        main_path = (asset_root / main_usd).resolve()
        if not main_path.is_relative_to(asset_root):
            issues.append("main_usd_escapes_asset_root")
        elif not main_path.is_file():
            issues.append("main_usd_missing")
        elif main_path.suffix.lower() not in USD_SUFFIXES:
            issues.append("main_asset_not_usd")

    files = asset.get("files", [])
    if not isinstance(files, list) or not files:
        issues.append("asset_file_inventory_empty")
        files = []
    seen: set[str] = set()
    texture_files = 0
    total_bytes = 0
    inventoried_main_usd = False
    for row in files:
        if not isinstance(row, dict):
            issues.append("malformed_asset_file_row")
            continue
        relative = row.get("path")
        if not isinstance(relative, str) or not _is_relative_asset_path(relative):
            issues.append("invalid_asset_file_path")
            continue
        if relative in seen:
            issues.append("duplicate_asset_file_path")
            continue
        seen.add(relative)
        path = (asset_root / relative).resolve()
        if not path.is_relative_to(asset_root):
            issues.append("asset_file_escapes_root")
            continue
        if not path.is_file():
            issues.append(f"asset_file_missing:{relative}")
            continue
        size = path.stat().st_size
        total_bytes += size
        if row.get("bytes") != size:
            issues.append(f"asset_file_size_mismatch:{relative}")
        digest = row.get("sha256")
        if not isinstance(digest, str) or digest != _sha256(path):
            issues.append(f"asset_file_hash_mismatch:{relative}")
        if path.suffix.lower() in TEXTURE_SUFFIXES:
            texture_files += 1
        if relative == main_usd:
            inventoried_main_usd = True
    if main_usd and not inventoried_main_usd:
        issues.append("main_usd_not_in_inventory")
    if texture_files < 3:
        issues.append("insufficient_pbr_texture_inventory")
    if main_path is not None and main_path.is_file():
        try:
            actual_files = _asset_package_files(asset_root, main_path)
        except ValueError:
            issues.append("asset_package_symlink_escape")
        else:
            actual_relatives = {path.relative_to(asset_root).as_posix() for path in actual_files}
            if actual_relatives != seen:
                issues.append("asset_file_inventory_not_exact")

    return {
        "passed": not issues,
        "issues": sorted(set(issues)),
        "scene_id": scene_id,
        "main_usd": str(main_path) if main_path is not None else None,
        "file_count": len(seen),
        "texture_file_count": texture_files,
        "total_bytes": total_bytes,
        "admission_state": (
            "source_integrity_passed_pending_isaac_scene_qa"
            if not issues
            else "source_integrity_failed"
        ),
    }


def compile_embodiedgen_visual_wrapper(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    translation_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_xyz_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
) -> dict[str, Any]:
    """Reference a verified RoomGen package as a visual-only Kino-Fail USD layer."""
    from pxr import Gf, Sdf, Usd, UsdGeom

    manifest_path = Path(manifest_path).resolve()
    audit = audit_embodiedgen_room_manifest(manifest_path)
    if not audit["passed"]:
        raise ValueError(f"EmbodiedGen source audit failed: {audit}")
    if scale <= 0.0:
        raise ValueError("visual source scale must be positive")
    source_usd = Path(str(audit["main_usd"]))
    source_layer = Sdf.Layer.FindOrOpen(str(source_usd))
    if source_layer is None:
        raise ValueError(f"USD source cannot be opened: {source_usd}")
    if not source_layer.defaultPrim:
        raise ValueError("RoomGen USD source has no default prim")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "visual.usda"
    layer = Sdf.Layer.CreateNew(str(output_path))
    stage = Usd.Stage.Open(layer)
    root = UsdGeom.Xform.Define(stage, "/KinoScene")
    stage.SetDefaultPrim(root.GetPrim())
    background = UsdGeom.Xform.Define(stage, "/KinoScene/EmbodiedGenVisual")
    reference_path = Path(os.path.relpath(source_usd, output_dir)).as_posix()
    background.GetPrim().GetReferences().AddReference(reference_path)
    xform = UsdGeom.XformCommonAPI(background)
    xform.SetTranslate(Gf.Vec3d(*translation_xyz_m))
    xform.SetRotate(Gf.Vec3f(*rotation_xyz_deg))
    xform.SetScale(Gf.Vec3f(scale, scale, scale))
    root.GetPrim().CreateAttribute("kino:sourceKind", Sdf.ValueTypeNames.String).Set(SOURCE_KIND)
    root.GetPrim().CreateAttribute("kino:sourceCollisionAuthoritative", Sdf.ValueTypeNames.Bool).Set(
        False
    )
    root.GetPrim().CreateAttribute("kino:sourceManifestSha256", Sdf.ValueTypeNames.String).Set(
        _sha256(manifest_path)
    )
    if not layer.Save():
        raise RuntimeError(f"failed to save visual wrapper: {output_path}")
    return {
        "visual_usd": str(output_path),
        "visual_usd_sha256": _sha256(output_path),
        "source_usd": str(source_usd),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "reference_path": reference_path,
        "source_collision_authoritative": False,
    }
