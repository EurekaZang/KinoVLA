#!/usr/bin/env python3
"""Admit one held-out route-surface scene for frozen O4 collection."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kino_vla.eval.embodiedgen_route_surface_v7_admission import (
    SCHEMA_VERSION,
    evaluate_route_surface_v7_admission,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bound_file(path_value: Any, expected_sha256: Any) -> tuple[Path | None, bool]:
    if not isinstance(path_value, str) or not isinstance(expected_sha256, str):
        return None, False
    path = Path(path_value).resolve()
    return path, path.is_file() and _sha256(path) == expected_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-preflight", type=Path, required=True)
    parser.add_argument("--compiled-audit", type=Path, required=True)
    parser.add_argument("--rtx-audit", type=Path, required=True)
    parser.add_argument("--go2-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source_preflight_path = args.source_preflight.resolve()
    compiled_path = args.compiled_audit.resolve()
    rtx_path = args.rtx_audit.resolve()
    go2_path = args.go2_audit.resolve()
    source_preflight = _json(source_preflight_path)
    compiled = _json(compiled_path)
    rtx = _json(rtx_path)
    go2 = _json(go2_path)

    episode_path = compiled_path.parent / "episode_v4.usda"
    episode_sha256 = _sha256(episode_path) if episode_path.is_file() else ""
    corridor_path, corridor_verified = _bound_file(
        compiled.get("corridor_v2_audit"), compiled.get("corridor_v2_audit_sha256")
    )
    corridor = _json(corridor_path) if corridor_path and corridor_path.is_file() else {}
    base_path, base_verified = _bound_file(
        compiled.get("base_compiled_audit"), compiled.get("base_compiled_audit_sha256")
    )
    source_manifest_path, source_manifest_verified = _bound_file(
        compiled.get("source_manifest"), compiled.get("source_manifest_sha256")
    )
    appearance = compiled.get("appearance_contract", {})
    lock = appearance.get("material_lock", {})
    lock_path, lock_verified = _bound_file(lock.get("path"), lock.get("sha256"))
    material_maps_verified = True
    map_evidence: dict[str, Any] = {}
    maps = appearance.get("material", {}).get("maps", {})
    if not maps:
        material_maps_verified = False
    for map_name, record in maps.items():
        map_path, verified = _bound_file(record.get("absolute_path"), record.get("sha256"))
        map_evidence[map_name] = {
            "path": str(map_path) if map_path else None,
            "sha256": record.get("sha256"),
            "verified": verified,
        }
        material_maps_verified = material_maps_verified and verified

    decision = evaluate_route_surface_v7_admission(
        source_preflight=source_preflight,
        compiled=compiled,
        corridor=corridor,
        rtx=rtx,
        go2=go2,
        episode_sha256=episode_sha256,
        compiled_sha256=_sha256(compiled_path),
        corridor_sha256=_sha256(corridor_path) if corridor_verified and corridor_path else "",
        source_manifest_sha256=(
            _sha256(source_manifest_path)
            if source_manifest_verified and source_manifest_path
            else ""
        ),
        base_audit_verified=base_verified,
        material_lock_verified=lock_verified,
        material_maps_verified=material_maps_verified,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "scene_id": compiled.get("scene_id"),
        "evidence": {
            "source_preflight": {
                "path": str(source_preflight_path),
                "sha256": _sha256(source_preflight_path),
            },
            "compiled_audit": {
                "path": str(compiled_path),
                "sha256": _sha256(compiled_path),
            },
            "corridor_v2_audit": {
                "path": str(corridor_path) if corridor_path else None,
                "sha256": _sha256(corridor_path) if corridor_verified and corridor_path else None,
            },
            "base_compiled_audit": {
                "path": str(base_path) if base_path else None,
                "verified": base_verified,
            },
            "source_manifest": {
                "path": str(source_manifest_path) if source_manifest_path else None,
                "verified": source_manifest_verified,
            },
            "episode": {"path": str(episode_path), "sha256": episode_sha256 or None},
            "rtx_audit": {"path": str(rtx_path), "sha256": _sha256(rtx_path)},
            "go2_audit": {"path": str(go2_path), "sha256": _sha256(go2_path)},
            "material_lock": {
                "path": str(lock_path) if lock_path else None,
                "verified": lock_verified,
            },
            "material_maps": map_evidence,
        },
        **decision,
        "admission_state": (
            "route_surface_v7_admitted_for_frozen_o4_collection"
            if decision["passed"]
            else "route_surface_v7_rejected"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite admission audit: {args.out}")
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
