#!/usr/bin/env python3
"""Audit the calibrated, phase-aware route-surface lighting-v5 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_spec(spec: dict[str, str]) -> bool:
    path = _resolve(spec["path"])
    return path.is_file() and _sha256(path) == spec["sha256"]


def _cell(scene: dict[str, Any], material_id: str, plan: dict[str, Any]) -> dict[str, Any]:
    corridor_path = _resolve(scene["corridor_v2_audit"])
    root = corridor_path.parent.parent / "route_surface_lighting_v5_development" / material_id
    compiled_path = root / "compiled_scene_audit.json"
    episode_path = root / "episode_v4.usda"
    rtx_path = root / "rtx_qa/rtx_scene_audit.json"
    compiled = _json(compiled_path) if compiled_path.is_file() else None
    rtx = _json(rtx_path) if rtx_path.is_file() else None
    checks: dict[str, bool] = {
        "compiled_present": compiled is not None,
        "episode_present": episode_path.is_file(),
        "rtx_present": rtx is not None,
    }
    architecture = plan["frozen_architecture"]
    if compiled is not None:
        appearance = compiled.get("appearance_contract", {})
        lighting = compiled.get("lighting_contract", {})
        geometry = appearance.get("geometry", {})
        panels = lighting.get("route_envelope_panels", {})
        checks.update(
            {
                "compiled_schema": compiled.get("schema_version")
                == "kinofail.embodiedgen-compiled-scene.v4-development",
                "compiled_scene_id": compiled.get("scene_id") == scene["scene_id"],
                "compiled_binds_corridor": compiled.get("corridor_v2_audit_sha256")
                == scene["sha256"],
                "compiled_material_id": appearance.get("material", {}).get("id")
                == material_id,
                "compiled_visual_only": appearance.get("visual_intervention_only")
                is True,
                "compiled_no_collision": appearance.get("collision_authored") is False,
                "compiled_render_purpose": appearance.get("purpose") == "render",
                "compiled_geometry_frozen": math.isclose(
                    float(geometry.get("surface_width_m", math.nan)),
                    float(architecture["route_surface_width_m"]),
                )
                and math.isclose(
                    float(geometry.get("endpoint_margin_m", math.nan)),
                    float(architecture["route_surface_endpoint_margin_m"]),
                )
                and math.isclose(
                    float(geometry.get("lift_above_floor_m", math.nan)),
                    float(architecture["route_surface_lift_m"]),
                ),
                "compiled_lighting_frozen": math.isclose(
                    float(lighting.get("neutral_dome_intensity", math.nan)),
                    float(architecture["dome_intensity"]),
                )
                and int(panels.get("count", -1)) == int(architecture["panel_count"])
                and math.isclose(
                    float(panels.get("intensity", math.nan)),
                    float(architecture["panel_intensity"]),
                )
                and math.isclose(
                    float(panels.get("exposure", math.nan)),
                    float(architecture["panel_exposure"]),
                ),
                "compiled_episode_hash": episode_path.is_file()
                and compiled.get("files", {}).get("episode_v4.usda")
                == _sha256(episode_path),
            }
        )
    if rtx is not None:
        structural = rtx.get("structural_checks", {})
        visual = rtx.get("visual_checks", {})
        clip = rtx.get("camera_intrinsics", {}).get(
            "calibrated_clipping_range_m", []
        )
        checks.update(
            {
                "rtx_schema": rtx.get("schema_version")
                == "kinofail.embodiedgen-rtx-scene-qa.v5-development",
                "rtx_view_contract": rtx.get("view_contract_schema")
                == "kinofail.embodiedgen-phase-aware-scene-views.v5",
                "rtx_binds_compiled": compiled_path.is_file()
                and rtx.get("compiled_audit_sha256") == _sha256(compiled_path),
                "rtx_binds_episode": episode_path.is_file()
                and rtx.get("episode_usd_sha256") == _sha256(episode_path),
                "rtx_structural_checks_complete": bool(structural)
                and all(structural.values()),
                "rtx_visual_checks_complete": len(visual) == 8,
                "rtx_clipping_frozen": len(clip) == 2
                and math.isclose(float(clip[0]), 0.01, abs_tol=1.0e-9)
                and math.isclose(float(clip[1]), 1000.0, abs_tol=1.0e-6),
                "rtx_result_boolean": isinstance(rtx.get("passed"), bool),
                "rtx_result_consistent": rtx.get("passed")
                == (bool(structural) and all(structural.values()) and len(visual) == 8 and all(visual.values())),
            }
        )
    return {
        "scene_id": scene["scene_id"],
        "material_id": material_id,
        "terminal": compiled is not None and rtx is not None,
        "passed": None if rtx is None else rtx.get("passed") is True,
        "integrity_passed": bool(checks) and all(checks.values()),
        "checks": checks,
        "compiled_audit": {
            "path": str(compiled_path),
            "sha256": _sha256(compiled_path) if compiled_path.is_file() else None,
        },
        "rtx_audit": {
            "path": str(rtx_path),
            "sha256": _sha256(rtx_path) if rtx_path.is_file() else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = _json(plan_path)
    code_checks = {name: _hash_spec(spec) for name, spec in plan["frozen_code"].items()}
    scene_checks = {
        row["scene_id"]: _hash_spec(
            {"path": row["corridor_v2_audit"], "sha256": row["sha256"]}
        )
        for row in plan["development_scenes"]
    }
    material_lock_check = _hash_spec(plan["material_policy"]["material_lock"])
    cells = [
        _cell(scene, material["material_id"], plan)
        for scene in plan["development_scenes"]
        for material in plan["materials"]
    ]
    contract = plan["execution_contract"]
    matrix_checks = {
        "scene_count": len(plan["development_scenes"]) == int(contract["scene_count"]),
        "material_count": len(plan["materials"]) == int(contract["material_count"]),
        "exact_cell_count": len(cells) == int(contract["exact_render_count"]),
        "unique_cells": len({(row["scene_id"], row["material_id"]) for row in cells})
        == len(cells),
    }
    sealed = all(row["terminal"] for row in cells)
    integrity = (
        all(code_checks.values())
        and all(scene_checks.values())
        and material_lock_check
        and all(matrix_checks.values())
        and sealed
        and all(row["integrity_passed"] for row in cells)
    )
    payload = {
        "schema_version": "kinofail.embodiedgen-route-surface-lighting-v5-matrix-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "sealed": sealed,
        "audit_integrity_passed": integrity,
        "matrix_passed": integrity and all(row["passed"] is True for row in cells),
        "passed_cells": sum(row["passed"] is True for row in cells),
        "cell_count": len(cells),
        "frozen_code_checks": code_checks,
        "scene_input_checks": scene_checks,
        "material_lock_check": material_lock_check,
        "matrix_checks": matrix_checks,
        "cells": cells,
        "interpretation": "Development-only phase-aware admission evidence; no cell is benchmark confirmation evidence."
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if sealed and not integrity:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
