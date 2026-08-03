#!/usr/bin/env python3
"""Reject malformed or visually trivial EmbodiedGen USDs before scene compilation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kino_vla.eval.embodiedgen_source_preflight_v1 import (
    MAX_ABS_INDOOR_COORDINATE_M,
    SCHEMA_VERSION,
    evaluate_source_geometry_preflight,
)
from kino_vla.sim.embodiedgen_asset import audit_embodiedgen_room_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _invalid(values: list[float]) -> bool:
    return any(
        not math.isfinite(value) or abs(value) > MAX_ABS_INDOOR_COORDINATE_M
        for value in values
    )


def _scan_usd(main_usd: Path) -> dict[str, Any]:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(main_usd))
    if stage is None:
        return {
            "stage_opened": False,
            "default_prim_present": False,
            "up_axis": None,
            "metres_per_unit": None,
            "mesh_count": 0,
            "point_count": 0,
            "face_count": 0,
            "invalid_transform_prims": [],
            "invalid_bound_prims": [],
            "floor_candidates": [],
        }

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    mesh_count = 0
    point_count = 0
    face_count = 0
    invalid_transform_prims: list[dict[str, Any]] = []
    invalid_bound_prims: list[dict[str, Any]] = []
    floor_candidates: list[dict[str, Any]] = []

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_count += 1
        mesh = UsdGeom.Mesh(prim)
        point_count += len(mesh.GetPointsAttr().Get() or [])
        face_count += len(mesh.GetFaceVertexCountsAttr().Get() or [])
        path = str(prim.GetPath())

        try:
            transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            translation = [float(value) for value in transform.ExtractTranslation()]
        except Exception as exc:  # noqa: BLE001 - malformed source is audit evidence
            invalid_transform_prims.append(
                {"path": path, "error": f"{type(exc).__name__}:{exc}"}
            )
            continue
        if _invalid(translation):
            invalid_transform_prims.append({"path": path, "translation_xyz_m": translation})
            continue

        try:
            bound = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            low, high = bound.GetMin(), bound.GetMax()
            bounds = [float(low[index]) for index in range(3)] + [
                float(high[index]) for index in range(3)
            ]
        except Exception as exc:  # noqa: BLE001 - malformed source is audit evidence
            invalid_bound_prims.append(
                {"path": path, "error": f"{type(exc).__name__}:{exc}"}
            )
            continue
        if _invalid(bounds):
            invalid_bound_prims.append(
                {"path": path, "low_xyz_m": bounds[:3], "high_xyz_m": bounds[3:]}
            )
            continue
        if "floor" in path.lower() and bounds[5] - bounds[2] < 0.25:
            floor_candidates.append(
                {
                    "path": path,
                    "low_xyz_m": bounds[:3],
                    "high_xyz_m": bounds[3:],
                    "area_xy_m2": (bounds[3] - bounds[0]) * (bounds[4] - bounds[1]),
                }
            )

    return {
        "stage_opened": True,
        "default_prim_present": bool(stage.GetDefaultPrim()),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)).upper(),
        "metres_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "mesh_count": mesh_count,
        "point_count": point_count,
        "face_count": face_count,
        "invalid_transform_prims": invalid_transform_prims,
        "invalid_bound_prims": invalid_bound_prims,
        "floor_candidates": floor_candidates,
    }


def audit_source(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    source_audit = audit_embodiedgen_room_manifest(manifest_path)
    main_usd_text = source_audit.get("main_usd")
    main_usd = Path(main_usd_text) if isinstance(main_usd_text, str) else None
    if main_usd is not None and main_usd.is_file():
        scan = _scan_usd(main_usd)
    else:
        scan = {
            "stage_opened": False,
            "default_prim_present": False,
            "up_axis": None,
            "metres_per_unit": None,
            "mesh_count": 0,
            "point_count": 0,
            "face_count": 0,
            "invalid_transform_prims": [],
            "invalid_bound_prims": [],
            "floor_candidates": [],
        }
    decision = evaluate_source_geometry_preflight(
        scan, source_integrity_passed=bool(source_audit.get("passed"))
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "scene_id": manifest.get("scene_id"),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "main_usd": str(main_usd) if main_usd is not None else None,
        "main_usd_sha256": (
            _sha256(main_usd) if main_usd is not None and main_usd.is_file() else None
        ),
        "source_integrity_audit": source_audit,
        "scan": scan,
        **decision,
        "admission_state": (
            "source_geometry_preflight_passed_pending_scene_compile"
            if decision["passed"]
            else "source_geometry_preflight_failed"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = audit_source(Path(args.source_manifest))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
