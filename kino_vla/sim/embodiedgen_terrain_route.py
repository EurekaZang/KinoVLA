"""Compile a versioned terrain-operator layer over an admitted EmbodiedGen v4 scene."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from kino_vla.sim.realistic_route_protocol_v2 import scene_route_binding


SCHEMA_VERSION = "kinofail.embodiedgen-terrain-route-compiled-scene.v1-development"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_asset(source: Path, destination_parent: Path) -> str:
    return Path(os.path.relpath(source.resolve(), destination_parent.resolve())).as_posix()


def floor_material_layer_text(*, static_friction: float, dynamic_friction: float) -> str:
    if not (0.0 <= dynamic_friction <= static_friction <= 2.0):
        raise ValueError("require 0 <= dynamic_friction <= static_friction <= 2")
    return f'''#usda 1.0

over "KinoScene"
{{
    over "Collision"
    {{
        over "Floor" (
            prepend apiSchemas = ["PhysicsMaterialAPI"]
        )
        {{
            custom bool kino:operatorLayerMayOverride = 1
            custom string kino:terrainAdapter = "embodiedgen_terrain_route_v1"
            float physics:dynamicFriction = {dynamic_friction:.9g}
            float physics:staticFriction = {static_friction:.9g}
        }}
    }}
}}
'''


def episode_layer_text(*, material_layer: Path, base_episode: Path, output_dir: Path) -> str:
    material_ref = _relative_asset(material_layer, output_dir)
    base_ref = _relative_asset(base_episode, output_dir)
    return f'''#usda 1.0
(
    defaultPrim = "KinoScene"
    subLayers = [
        @{material_ref}@,
        @{base_ref}@
    ]
)
'''


def compile_embodiedgen_terrain_route(
    *,
    base_audit_path: Path,
    output_dir: Path,
    expected_base_audit_sha256: str | None = None,
    static_friction: float = 0.8,
    dynamic_friction: float = 0.6,
) -> dict[str, Any]:
    base_audit_path = base_audit_path.resolve()
    output_dir = output_dir.resolve()
    base_hash = _sha256(base_audit_path)
    if expected_base_audit_sha256 is not None and base_hash != expected_base_audit_sha256:
        raise ValueError("base compiled-audit hash does not match frozen input")
    base = json.loads(base_audit_path.read_text(encoding="utf-8"))
    binding = scene_route_binding(base)
    if binding.source_kind != "embodiedgen_v4":
        raise ValueError("terrain route compiler requires an admitted EmbodiedGen v4 base scene")
    base_episode = base_audit_path.parent / binding.episode_filename
    if _sha256(base_episode) != base.get("files", {}).get(binding.episode_filename):
        raise ValueError("base episode hash does not match compiled audit")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    material_layer = output_dir / "nominal_floor_material.usda"
    material_layer.write_text(
        floor_material_layer_text(
            static_friction=static_friction, dynamic_friction=dynamic_friction
        ),
        encoding="utf-8",
    )
    episode = output_dir / "episode_terrain_v1.usda"
    episode.write_text(
        episode_layer_text(
            material_layer=material_layer,
            base_episode=base_episode,
            output_dir=output_dir,
        ),
        encoding="utf-8",
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "development_only": True,
        "scene_registry_eligible": False,
        "counts_as_a0_a7_evidence": False,
        "scene_id": base["scene_id"],
        "source_kind": "embodiedgen_v4_with_kino_terrain_physics_layer",
        "base_compiled_audit": str(base_audit_path),
        "base_compiled_audit_sha256": base_hash,
        "route": base["route"],
        "appearance_contract": base["appearance_contract"],
        "physics_contract": {
            **base["physics_contract"],
            "nominal_floor_material": {
                "owned_by": "Kino-Fail",
                "prim_path": "/KinoScene/Collision/Floor",
                "api_schema": "PhysicsMaterialAPI",
                "static_friction": static_friction,
                "dynamic_friction": dynamic_friction,
                "operator_layer_may_override": True,
            },
        },
        "files": {
            "nominal_floor_material.usda": _sha256(material_layer),
            "episode_terrain_v1.usda": _sha256(episode),
        },
        "checks": {
            "base_scene_passed": base.get("passed") is True,
            "base_episode_hash_verified": True,
            "straight_route_bound": binding.frame.route_length_m > 0.0,
            "route_surface_is_visual_only": binding.route_surface_collision_authored is False,
            "explicit_nominal_floor_material_authored": True,
            "operator_override_declared": True,
        },
        "remaining_gates": [
            "offline_usd_composition_audit",
            "nominal_articulated_go2_route_preflight",
            "fresh-process_operator_pair",
            "formal_native_v3_collector",
            "scene_registry_binding",
        ],
    }
    audit_path = output_dir / "compiled_scene_audit.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
