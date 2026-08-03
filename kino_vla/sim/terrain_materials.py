"""Licensed PBR terrain asset synchronization, locking, and runtime resolution.

The repository tracks the small human-readable catalog; large texture maps live below ``outputs/``
and are reproducibly recreated from their source URLs.  A lock file freezes the archive and map
SHA-256 values used by a benchmark run.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from PIL import Image, ImageDraw, ImageFont

from kino_vla.utils.config import CONFIGS_DIR

if TYPE_CHECKING:
    from pxr import Usd

ASSET_LOCK_SCHEMA = "kinofail.terrain-assets.v1"
REQUIRED_MAPS = ("basecolor", "normal", "roughness")


@dataclass(frozen=True)
class PBRMapSet:
    """Resolved maps for one tileable terrain material."""

    material_id: str
    directory: Path
    basecolor: Path
    normal: Path
    roughness: Path
    displacement: Path | None
    ambient_occlusion: Path | None


@dataclass(frozen=True)
class TerrainAppearanceBinding:
    """One episode's PBR binding, independent of collision and physics material."""

    material_id: str
    appearance_id: str
    maps: PBRMapSet
    physical_size_m: tuple[float, float]
    uv_scale: float
    rotation_deg: float
    surface_state: str
    uv_offset: tuple[float, float] = (0.0, 0.0)
    albedo_brightness_multiplier: float = 1.0
    normal_strength: float = 1.0
    roughness_multiplier: float = 1.0

    def omnipbr_parameters(self) -> dict[str, Any]:
        """Return metric triplanar OmniPBR inputs, including state-level photometric changes."""
        state = {
            "clean": (1.00, 1.00, 0.50),
            "dusty": (0.88, 0.72, 0.82),
            "scuffed": (0.94, 0.88, 0.60),
            "damp": (0.78, 0.62, 0.34),
            "oil_stained": (0.62, 0.48, 0.20),
            "wet": (0.68, 0.42, 0.16),
            "dry": (1.00, 0.92, 0.62),
            "leaf_litter": (0.90, 0.82, 0.72),
            "muddy": (0.72, 0.60, 0.38),
        }
        if self.surface_state not in state:
            raise ValueError(f"unsupported terrain surface state {self.surface_state!r}")
        brightness, roughness_influence, roughness_constant = state[self.surface_state]
        brightness = max(0.05, min(2.0, brightness * self.albedo_brightness_multiplier))
        roughness_influence = max(0.0, min(1.0, roughness_influence * self.roughness_multiplier))
        roughness_constant = max(0.0, min(1.0, roughness_constant * self.roughness_multiplier))
        repeat_x = self.uv_scale / float(self.physical_size_m[0])
        repeat_y = self.uv_scale / float(self.physical_size_m[1])
        return {
            "diffuse_texture": str(self.maps.basecolor.resolve()),
            "normalmap_texture": str(self.maps.normal.resolve()),
            "reflectionroughness_texture": str(self.maps.roughness.resolve()),
            "project_uvw": True,
            "world_or_object": True,
            "texture_scale": (repeat_x, repeat_y),
            "texture_rotate": float(self.rotation_deg),
            "texture_translate": tuple(float(value) for value in self.uv_offset),
            "flip_tangent_v": False,  # catalog locks OpenGL normal maps
            "albedo_brightness": brightness,
            "reflection_roughness_texture_influence": roughness_influence,
            "reflection_roughness_constant": roughness_constant,
            "bump_factor": float(self.normal_strength),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_catalog(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = CONFIGS_DIR / resolved
    with resolved.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise TypeError(f"top level of {resolved} must be a mapping")
    return value


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": "Kino-Fail-Benchmark/1.0"})


def _download(url: str, destination: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(_request(url), timeout=120) as response:  # noqa: S310
                with destination.open("wb") as stream:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
            return
        except Exception as error:  # transient CDN/TLS failures are retried as one operation
            last_error = error
            destination.unlink(missing_ok=True)
            if attempt < 4:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"download failed after 5 attempts: {url}") from last_error


def _ambientcg_metadata(asset_id: str) -> dict[str, Any]:
    query = urllib.parse.urlencode({"id": asset_id})
    url = f"https://ambientcg.com/api/v2/full_json?{query}"
    payload: dict[str, Any] | None = None
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(_request(url), timeout=60) as response:  # noqa: S310
                payload = json.load(response)
            break
        except Exception as error:  # ambientCG occasionally closes a TLS connection early
            last_error = error
            if attempt < 4:
                time.sleep(0.5 * (attempt + 1))
    if payload is None:
        raise RuntimeError(f"metadata request failed after 5 attempts: {asset_id}") from last_error
    assets = payload.get("foundAssets", [])
    if len(assets) != 1 or assets[0].get("assetId") != asset_id:
        raise RuntimeError(f"ambientCG metadata did not resolve unique asset {asset_id!r}")
    asset = assets[0]
    return {
        "asset_id": asset_id,
        "release_date": asset.get("releaseDate"),
        "creation_method": asset.get("creationMethod"),
        "dimension_cm": [asset.get("dimensionX"), asset.get("dimensionY")],
        "maps_reported": list(asset.get("maps") or []),
        "tags": list(asset.get("tags") or []),
        "metadata_url": url,
    }


def _safe_extract_images(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    allowed = {".jpg", ".jpeg", ".png", ".exr", ".tif", ".tiff", ".txt", ".md"}
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if member.is_dir():
                continue
            target = destination / Path(member.filename).name
            if target.suffix.lower() not in allowed:
                continue
            if target.resolve().parent != root:
                raise ValueError(f"unsafe archive member {member.filename!r}")
            with bundle.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)


def _find_map(directory: Path, tokens: Sequence[str], *, reject: Sequence[str] = ()) -> Path | None:
    candidates = []
    for path in sorted(directory.iterdir()):
        lower = path.name.lower()
        if (
            path.is_file()
            and any(token in lower for token in tokens)
            and not any(token in lower for token in reject)
        ):
            candidates.append(path)
    return candidates[0] if candidates else None


def resolve_pbr_maps(material_id: str, directory: str | Path) -> PBRMapSet:
    """Resolve the standard ambientCG map names, preferring OpenGL normals for USD/RTX."""
    directory = Path(directory)
    basecolor = _find_map(directory, ("_color", "basecolor", "_diffuse"))
    normal = _find_map(directory, ("normalgl", "normal_gl")) or _find_map(
        directory, ("normal",), reject=("normaldx", "normal_dx")
    )
    roughness = _find_map(directory, ("roughness",))
    displacement = _find_map(directory, ("displacement", "height"))
    ambient_occlusion = _find_map(directory, ("ambientocclusion", "_ao"))
    missing = [
        name
        for name, value in (("basecolor", basecolor), ("normal", normal), ("roughness", roughness))
        if value is None
    ]
    if missing:
        raise FileNotFoundError(
            f"{material_id}: missing required PBR maps {missing} in {directory}"
        )
    assert basecolor is not None and normal is not None and roughness is not None
    return PBRMapSet(
        material_id=material_id,
        directory=directory,
        basecolor=basecolor,
        normal=normal,
        roughness=roughness,
        displacement=displacement,
        ambient_occlusion=ambient_occlusion,
    )


def sync_terrain_materials(
    config: str | Path = "data/kinofail_realistic.yaml",
    *,
    asset_root: str | Path = "outputs/assets/terrain_pbr_v1",
    material_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Download/extract the catalog subset and write an exact, auditable asset lock."""
    spec = _load_catalog(config)
    catalog = [dict(value) for value in spec["materials"]]
    requested = set(material_ids or [str(value["id"]) for value in catalog])
    unknown = requested - {str(value["id"]) for value in catalog}
    if unknown:
        raise KeyError(f"unknown material ids: {sorted(unknown)}")
    root = Path(asset_root)
    root.mkdir(parents=True, exist_ok=True)
    locked: list[dict[str, Any]] = []
    for material in catalog:
        material_id = str(material["id"])
        if material_id not in requested:
            continue
        target = root / material_id
        provenance_path = target / "archive_provenance.json"
        maps: PBRMapSet
        archive_sha: str | None = None
        archive_bytes: int | None = None
        if target.exists():
            try:
                maps = resolve_pbr_maps(material_id, target)
            except FileNotFoundError:
                shutil.rmtree(target)
            else:
                print(f"[terrain-assets] reuse {material_id}")
                if provenance_path.exists():
                    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                    archive_sha = provenance.get("archive_sha256")
                    archive_bytes = provenance.get("archive_bytes")
        if not target.exists():
            with tempfile.TemporaryDirectory(prefix="kinofail_pbr_") as tmp_dir:
                archive = Path(tmp_dir) / f"{material_id}.zip"
                print(f"[terrain-assets] download {material_id} <- {material['archive_url']}")
                _download(str(material["archive_url"]), archive)
                archive_sha = _sha256(archive)
                archive_bytes = archive.stat().st_size
                _safe_extract_images(archive, target)
                provenance_path.write_text(
                    json.dumps(
                        {
                            "archive_url": material["archive_url"],
                            "archive_sha256": archive_sha,
                            "archive_bytes": archive_bytes,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            maps = resolve_pbr_maps(material_id, target)
        if archive_sha is None or archive_bytes is None:
            # An interrupted earlier run may have extracted valid maps before the lock was written.
            # Re-fetch only the small 1K archive to recover exact source provenance.
            with tempfile.TemporaryDirectory(prefix="kinofail_pbr_provenance_") as tmp_dir:
                archive = Path(tmp_dir) / f"{material_id}.zip"
                print(f"[terrain-assets] recover archive provenance for {material_id}")
                _download(str(material["archive_url"]), archive)
                archive_sha = _sha256(archive)
                archive_bytes = archive.stat().st_size
            provenance_path.write_text(
                json.dumps(
                    {
                        "archive_url": material["archive_url"],
                        "archive_sha256": archive_sha,
                        "archive_bytes": archive_bytes,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        files = {
            "basecolor": maps.basecolor,
            "normal": maps.normal,
            "roughness": maps.roughness,
            "displacement": maps.displacement,
            "ambient_occlusion": maps.ambient_occlusion,
        }
        locked.append(
            {
                "id": material_id,
                "source_asset_id": material["source_asset_id"],
                "semantic_family": material["semantic_family"],
                "split": material["split"],
                "domains": material["domains"],
                "source": material["source"],
                "archive_url": material["archive_url"],
                "license": material["license"],
                "physical_size_m": material["physical_size_m"],
                "archive_sha256": archive_sha,
                "archive_bytes": archive_bytes,
                "source_metadata": _ambientcg_metadata(str(material["source_asset_id"])),
                "maps": {
                    key: (
                        {
                            "path": str(value.relative_to(root)),
                            "sha256": _sha256(value),
                            "bytes": value.stat().st_size,
                        }
                        if value is not None
                        else None
                    )
                    for key, value in files.items()
                },
            }
        )
    lock = {
        "schema_version": ASSET_LOCK_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "asset_root": str(root),
        "catalog_config": str(config),
        "materials": locked,
    }
    lock["audit"] = audit_terrain_asset_lock(lock, asset_root=root)
    lock_path = root / "terrain_assets.lock.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return lock


def audit_terrain_asset_lock(lock: Mapping[str, Any], *, asset_root: str | Path) -> dict[str, Any]:
    """Verify licenses, required map presence, and hashes for an asset lock."""
    root = Path(asset_root)
    issues: list[str] = []
    material_results: dict[str, Any] = {}
    for material in lock["materials"]:
        material_id = str(material["id"])
        missing: list[str] = []
        hash_mismatch: list[str] = []
        if material["license"] not in {"CC0", "CC-BY-4.0"}:
            issues.append(f"unapproved_license:{material_id}")
        for map_name in REQUIRED_MAPS:
            entry = material["maps"].get(map_name)
            if not entry:
                missing.append(map_name)
                continue
            path = root / entry["path"]
            if not path.exists():
                missing.append(map_name)
            elif _sha256(path) != entry["sha256"]:
                hash_mismatch.append(map_name)
        if missing:
            issues.append(f"missing_maps:{material_id}")
        if hash_mismatch:
            issues.append(f"hash_mismatch:{material_id}")
        material_results[material_id] = {
            "passed": not missing and not hash_mismatch,
            "missing": missing,
            "hash_mismatch": hash_mismatch,
        }
    return {
        "passed": not issues,
        "issues": issues,
        "n_materials": len(lock["materials"]),
        "materials": material_results,
    }


def load_terrain_asset_lock(path: str | Path) -> dict[str, Any]:
    """Load and validate the basic schema of a frozen terrain asset lock."""
    resolved = Path(path)
    with resolved.open(encoding="utf-8") as stream:
        lock = json.load(stream)
    if lock.get("schema_version") != ASSET_LOCK_SCHEMA:
        raise ValueError(f"unsupported terrain asset lock schema: {lock.get('schema_version')!r}")
    return lock


def appearance_binding_from_record(
    episode: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
    asset_root: str | Path | None = None,
) -> TerrainAppearanceBinding:
    """Resolve a design record into the exact maps and OmniPBR episode parameters."""
    root = Path(asset_root or lock["asset_root"])
    by_id = {str(value["id"]): value for value in lock["materials"]}
    material_id = str(episode["material_family"])
    if material_id not in by_id:
        raise KeyError(f"material {material_id!r} is absent from the terrain asset lock")
    material = by_id[material_id]
    map_entries = material["maps"]

    def path_for(name: str) -> Path | None:
        entry = map_entries.get(name)
        return root / entry["path"] if entry else None

    basecolor = path_for("basecolor")
    normal = path_for("normal")
    roughness = path_for("roughness")
    if basecolor is None or normal is None or roughness is None:
        raise FileNotFoundError(f"locked material {material_id!r} lacks required maps")
    maps = PBRMapSet(
        material_id=material_id,
        directory=root / material_id,
        basecolor=basecolor,
        normal=normal,
        roughness=roughness,
        displacement=path_for("displacement"),
        ambient_occlusion=path_for("ambient_occlusion"),
    )
    return TerrainAppearanceBinding(
        material_id=material_id,
        appearance_id=str(episode["appearance_id"]),
        maps=maps,
        physical_size_m=tuple(float(value) for value in material["physical_size_m"]),
        uv_scale=float(episode["uv_scale"]),
        rotation_deg=float(episode["uv_rotation_deg"]),
        surface_state=str(episode["surface_state"]),
        uv_offset=tuple(float(value) for value in episode.get("uv_offset", (0.0, 0.0))),
        albedo_brightness_multiplier=float(episode.get("albedo_brightness_multiplier", 1.0)),
        normal_strength=float(episode.get("normal_strength", 1.0)),
        roughness_multiplier=float(episode.get("roughness_multiplier", 1.0)),
    )


def bind_omnipbr_material(
    stage: Usd.Stage,
    surface_prim_path: str,
    binding: TerrainAppearanceBinding,
    *,
    material_root: str = "/World/Looks",
) -> dict[str, Any]:
    """Author and bind a metric, world-projected OmniPBR material inside Isaac Sim.

    This mutates *only* the visual material binding.  PhysX friction/compliance remains on the
    collision prim and is owned by the failure operator, preserving the causal separation.
    """
    from omni.usd.commands import CreateMdlMaterialPrimCommand
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    prim = stage.GetPrimAtPath(surface_prim_path)
    if not prim.IsValid():
        raise ValueError(f"cannot bind terrain appearance: invalid prim {surface_prim_path}")
    safe_id = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in binding.appearance_id)
    material_path = f"{material_root}/kino_terrain_{safe_id}"
    if not stage.GetPrimAtPath(material_path).IsValid():
        CreateMdlMaterialPrimCommand(
            mtl_url="OmniPBR.mdl",
            mtl_name="OmniPBR",
            mtl_path=material_path,
            stage=stage,
            select_new_prim=False,
        ).do()
    shader = UsdShade.Shader(stage.GetPrimAtPath(f"{material_path}/Shader"))
    if not shader.GetPrim().IsValid():
        raise RuntimeError(f"OmniPBR shader was not created at {material_path}/Shader")
    type_names = {
        "diffuse_texture": Sdf.ValueTypeNames.Asset,
        "normalmap_texture": Sdf.ValueTypeNames.Asset,
        "reflectionroughness_texture": Sdf.ValueTypeNames.Asset,
        "project_uvw": Sdf.ValueTypeNames.Bool,
        "world_or_object": Sdf.ValueTypeNames.Bool,
        "texture_scale": Sdf.ValueTypeNames.Float2,
        "texture_rotate": Sdf.ValueTypeNames.Float,
        "texture_translate": Sdf.ValueTypeNames.Float2,
        "flip_tangent_v": Sdf.ValueTypeNames.Bool,
        "albedo_brightness": Sdf.ValueTypeNames.Float,
        "reflection_roughness_texture_influence": Sdf.ValueTypeNames.Float,
        "reflection_roughness_constant": Sdf.ValueTypeNames.Float,
        "bump_factor": Sdf.ValueTypeNames.Float,
    }
    params = binding.omnipbr_parameters()
    for name, value in params.items():
        shader_input = shader.GetInput(name)
        if not shader_input:
            shader_input = shader.CreateInput(name, type_names[name])
        if type_names[name] == Sdf.ValueTypeNames.Asset:
            value = Sdf.AssetPath(str(value))
        shader_input.Set(value)
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    bound_paths: list[str] = []
    # Isaac's CuboidCfg may author the visible Gprim below an Xform and give that descendant its
    # own PreviewSurface.  Bind the root *and* every Gprim explicitly, otherwise the legacy child
    # material can remain visible even when the root carries the right Kino metadata.
    for candidate in Usd.PrimRange(prim):
        if candidate == prim or candidate.IsA(UsdGeom.Gprim):
            UsdShade.MaterialBindingAPI.Apply(candidate).Bind(
                material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )
            bound_paths.append(str(candidate.GetPath()))
    prim.CreateAttribute("kino:appearanceId", Sdf.ValueTypeNames.String).Set(binding.appearance_id)
    prim.CreateAttribute("kino:materialFamily", Sdf.ValueTypeNames.String).Set(binding.material_id)
    prim.CreateAttribute("kino:visualPhysicsSeparated", Sdf.ValueTypeNames.Bool).Set(True)
    return {
        "surface_prim_path": surface_prim_path,
        "material_path": material_path,
        "appearance_id": binding.appearance_id,
        "material_id": binding.material_id,
        "bound_prim_paths": bound_paths,
        "parameters": params,
    }


def render_material_contact_sheet(
    lock: Mapping[str, Any], *, asset_root: str | Path, output_path: str | Path
) -> Path:
    """Render basecolor/normal/roughness tiles for fast human asset review."""
    root = Path(asset_root)
    materials = list(lock["materials"])
    tile = 180
    label_h = 48
    columns = 3
    rows = len(materials)
    canvas = Image.new("RGB", (columns * tile, rows * (tile + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row, material in enumerate(materials):
        for column, key in enumerate(("basecolor", "normal", "roughness")):
            entry = material["maps"][key]
            image = Image.open(root / entry["path"]).convert("RGB")
            image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
            x = column * tile + (tile - image.width) // 2
            y = row * (tile + label_h) + (tile - image.height) // 2
            canvas.paste(image, (x, y))
        label_y = row * (tile + label_h) + tile + 4
        draw.text(
            (6, label_y),
            f"{material['id']} | {material['semantic_family']} | {material['split']} | CC0",
            fill="black",
            font=font,
        )
    for column, key in enumerate(("BASECOLOR", "NORMAL-GL", "ROUGHNESS")):
        draw.rectangle((column * tile, 0, column * tile + 75, 16), fill="white")
        draw.text((column * tile + 4, 3), key, fill="black", font=font)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=92)
    return destination
