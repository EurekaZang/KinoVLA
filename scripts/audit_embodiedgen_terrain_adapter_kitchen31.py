#!/usr/bin/env python3
"""Audit the first EmbodiedGen terrain-adapter calibration, including failed attempts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pxr import Usd, UsdGeom, UsdPhysics


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_embodiedgen_terrain_adapter_kitchen31_dev_v1.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/scene_sources/embodiedgen_v2/Kitchen_seed20260861/route_surface_v7_confirmation/test_ground073/terrain_adapter_v3_dev/calibration_audit.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _json(config_path)
    boundary = config["evidence_boundary"]
    checks: dict[str, bool] = {
        "supported_schema": config.get("schema_version")
        == "kinofail.embodiedgen-terrain-adapter-calibration.v1-development",
        "development_only": boundary.get("development_only") is True,
        "dynamic_bridge_disclosed": boundary.get("uses_dynamic_bridge_collector") is True,
        "not_registry_evidence": boundary.get("scene_registry_eligible") is False,
        "not_a0_a7_evidence": boundary.get("counts_as_a0_a7_evidence") is False,
    }
    frozen_inputs: list[dict[str, str]] = []
    payloads: dict[str, dict[str, Any]] = {}
    for name, spec in config["frozen_inputs"].items():
        path = _resolve(spec["path"])
        checks[f"frozen_{name}_hash"] = path.is_file() and _sha256(path) == spec["sha256"]
        if path.suffix == ".json" and path.is_file():
            payloads[name] = _json(path)
        frozen_inputs.append({"name": name, "path": str(path), "sha256": spec["sha256"]})
    checks["base_scene_passed"] = payloads["base_compiled_audit"].get("passed") is True
    checks["route_surface_admission_passed"] = payloads["route_surface_admission"].get(
        "passed"
    ) is True
    terrain = payloads["terrain_compiled_audit"]
    dense = terrain.get("physics_contract", {}).get("dense_route_surface", {})
    checks["terrain_compilation_passed"] = terrain.get("passed") is True
    checks["dense_surface_at_least_1000_vertices"] = int(dense.get("vertex_count", 0)) >= 1000
    checks["dense_surface_spacing_at_most_4cm"] = float(
        dense.get("maximum_spacing_m", 1.0)
    ) <= 0.04 + 1.0e-9
    checks["dense_surface_authors_no_collision"] = dense.get("collision_authored") is False

    episode = _resolve(config["frozen_inputs"]["terrain_episode"]["path"])
    stage = Usd.Stage.Open(str(episode))
    floor = stage.GetPrimAtPath("/KinoScene/Collision/Floor") if stage else None
    surface = (
        stage.GetPrimAtPath("/KinoScene/AppearanceV4/RenderOnlyRouteSurface")
        if stage
        else None
    )
    mesh = UsdGeom.Mesh(surface) if surface and surface.IsValid() else None
    checks["usd_composition_opened"] = stage is not None
    checks["single_floor_has_collision_api"] = bool(
        floor and floor.IsValid() and floor.HasAPI(UsdPhysics.CollisionAPI)
    )
    checks["floor_has_explicit_material_api"] = bool(
        floor and floor.IsValid() and floor.HasAPI(UsdPhysics.MaterialAPI)
    )
    checks["floor_static_friction_readback"] = bool(
        floor
        and abs(float(floor.GetAttribute("physics:staticFriction").Get()) - 0.8) <= 1.0e-6
    )
    checks["floor_dynamic_friction_readback"] = bool(
        floor
        and abs(float(floor.GetAttribute("physics:dynamicFriction").Get()) - 0.6) <= 1.0e-6
    )
    checks["surface_is_render_only"] = bool(
        surface
        and surface.GetAttribute("kino:collisionAuthored").Get() is False
        and UsdGeom.Imageable(surface).GetPurposeAttr().Get() == UsdGeom.Tokens.render
    )
    checks["surface_has_opaque_ground_marker"] = bool(
        surface and surface.GetAttribute("kino:opaqueCompositedGround").Get() is True
    )
    checks["surface_point_count_matches_audit"] = bool(
        mesh and len(mesh.GetPointsAttr().Get() or []) == int(dense.get("vertex_count", -1))
    )

    retained: list[dict[str, Any]] = []
    for row in config["retained_failed_attempts"]:
        item = dict(row)
        if "manifest" in row:
            path = _resolve(row["manifest"]["path"])
            valid = path.is_file() and _sha256(path) == row["manifest"]["sha256"]
            payload = _json(path) if path.is_file() else {}
            checks[f"failed_attempt_{row['attempt']}_retained"] = valid and payload.get(
                "passed"
            ) is False
            item["observed_failed_checks"] = sorted(
                key for key, value in payload.get("checks", {}).items() if value is not True
            )
        else:
            path = _resolve(row["path"])
            checks[f"failed_attempt_{row['attempt']}_directory_retained"] = path.is_dir()
        retained.append(item)

    nominal_spec = config["calibration_result"]["nominal_manifest"]
    nominal_path = _resolve(nominal_spec["path"])
    nominal = _json(nominal_path) if nominal_path.is_file() else {}
    checks["nominal_manifest_hash"] = nominal_path.is_file() and _sha256(
        nominal_path
    ) == nominal_spec["sha256"]
    checks["nominal_all_checks_passed"] = nominal.get("passed") is True and all(
        nominal.get("checks", {}).values()
    )
    checks["nominal_route_completed_without_fall"] = (
        nominal.get("checks", {}).get("nominal_route_completed") is True
        and nominal.get("measurements", {}).get("fallen") is False
    )
    checks["nominal_operator_region_reached"] = nominal.get("checks", {}).get(
        "operator_region_reached"
    ) is True
    checks["bridge_manifest_not_formal"] = nominal.get(
        "collector_adapter_provenance", {}
    ).get("formal_corpus_eligible") is False
    checks["anomaly_not_yet_run_declared"] = config["calibration_result"].get(
        "operator_anomaly_not_run"
    ) is True

    result = {
        "schema_version": "kinofail.embodiedgen-terrain-adapter-calibration-audit.v1-development",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "retained_failed_attempts": retained,
        "nominal_result": {
            "manifest": str(nominal_path),
            "measurements": nominal.get("measurements", {}),
        },
        "frozen_inputs": frozen_inputs,
        "evidence_boundary": boundary,
        "interpretation": (
            "This closes a development integration preflight only. The successful nominal run "
            "was reached after four retained adapter failures and cannot serve as confirmatory, "
            "registry, corpus, or A0-A7 evidence."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "passed": result["passed"]}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
