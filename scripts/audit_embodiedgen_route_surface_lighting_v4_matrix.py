#!/usr/bin/env python3
"""Seal the preregistered route-surface lighting-v4 development matrix."""

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


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_spec(spec: dict[str, str]) -> bool:
    path = _resolve(spec["path"])
    return path.is_file() and _sha256(path) == spec["sha256"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = _json(plan_path)
    architecture = plan["frozen_architecture"]
    code_checks = {
        name: _hash_spec(spec) for name, spec in plan["frozen_code"].items()
    }
    material_lock_check = _hash_spec(plan["material_policy"]["material_lock"])
    scene_checks = {
        row["scene_id"]: _hash_spec(
            {"path": row["corridor_v2_audit"], "sha256": row["sha256"]}
        )
        for row in plan["development_scenes"]
    }

    cells = []
    for scene in plan["development_scenes"]:
        corridor_path = _resolve(scene["corridor_v2_audit"])
        scene_root = corridor_path.parent.parent
        for material in plan["materials"]:
            material_id = material["material_id"]
            cell_root = scene_root / "route_surface_lighting_v4_development" / material_id
            compiled_path = cell_root / "compiled_scene_audit.json"
            episode_path = cell_root / "episode_v4.usda"
            rtx_path = cell_root / "rtx_qa/rtx_scene_audit.json"
            compiled = _json(compiled_path) if compiled_path.is_file() else None
            rtx = _json(rtx_path) if rtx_path.is_file() else None
            checks: dict[str, bool] = {
                "compiled_present": compiled is not None,
                "episode_present": episode_path.is_file(),
                "rtx_present": rtx is not None,
            }
            if compiled is not None:
                surface = compiled.get("appearance_contract", {})
                lighting = compiled.get("lighting_contract", {})
                panels = lighting.get("route_envelope_panels", {})
                checks.update(
                    {
                        "compiled_schema": compiled.get("schema_version")
                        == "kinofail.embodiedgen-compiled-scene.v4-development",
                        "compiled_scene_id": compiled.get("scene_id")
                        == scene["scene_id"],
                        "compiled_binds_corridor": compiled.get(
                            "corridor_v2_audit_sha256"
                        )
                        == scene["sha256"],
                        "compiled_material_id": surface.get("material", {}).get("id")
                        == material_id,
                        "compiled_visual_only": surface.get("visual_intervention_only")
                        is True,
                        "compiled_no_collision": surface.get("collision_authored")
                        is False,
                        "compiled_render_purpose": surface.get("purpose") == "render",
                        "compiled_surface_width": math.isclose(
                            float(
                                surface.get("geometry", {}).get(
                                    "surface_width_m", math.nan
                                )
                            ),
                            float(architecture["route_surface_width_m"]),
                        ),
                        "compiled_surface_margin": math.isclose(
                            float(
                                surface.get("geometry", {}).get(
                                    "endpoint_margin_m", math.nan
                                )
                            ),
                            float(architecture["route_surface_endpoint_margin_m"]),
                        ),
                        "compiled_surface_lift": math.isclose(
                            float(
                                surface.get("geometry", {}).get(
                                    "lift_above_floor_m", math.nan
                                )
                            ),
                            float(architecture["route_surface_lift_m"]),
                        ),
                        "compiled_dome_intensity": math.isclose(
                            float(lighting.get("neutral_dome_intensity", math.nan)),
                            float(architecture["dome_intensity"]),
                        ),
                        "compiled_panel_count": int(panels.get("count", -1))
                        == int(architecture["panel_count"]),
                        "compiled_panel_intensity": math.isclose(
                            float(panels.get("intensity", math.nan)),
                            float(architecture["panel_intensity"]),
                        ),
                        "compiled_panel_exposure": math.isclose(
                            float(panels.get("exposure", math.nan)),
                            float(architecture["panel_exposure"]),
                        ),
                        "compiled_episode_hash": episode_path.is_file()
                        and compiled.get("files", {}).get("episode_v4.usda")
                        == _sha256(episode_path),
                    }
                )
            if rtx is not None:
                checks.update(
                    {
                        "rtx_schema": rtx.get("schema_version")
                        == "kinofail.embodiedgen-rtx-scene-qa.v4-development",
                        "rtx_binds_compiled": compiled_path.is_file()
                        and rtx.get("compiled_audit_sha256") == _sha256(compiled_path),
                        "rtx_binds_episode": episode_path.is_file()
                        and rtx.get("episode_usd_sha256") == _sha256(episode_path),
                        "rtx_structural_checks_complete": bool(
                            rtx.get("structural_checks")
                        )
                        and all(rtx["structural_checks"].values()),
                        "rtx_visual_checks_complete": bool(rtx.get("visual_checks"))
                        and len(rtx["visual_checks"]) == 8,
                        "rtx_result_boolean": isinstance(rtx.get("passed"), bool),
                        "rtx_result_consistent": rtx.get("passed")
                        == (
                            bool(rtx.get("structural_checks"))
                            and all(rtx["structural_checks"].values())
                            and bool(rtx.get("visual_checks"))
                            and all(rtx["visual_checks"].values())
                        ),
                    }
                )
            cells.append(
                {
                    "scene_id": scene["scene_id"],
                    "material_id": material_id,
                    "terminal": compiled is not None and rtx is not None,
                    "passed": None if rtx is None else rtx.get("passed") is True,
                    "integrity_passed": bool(checks) and all(checks.values()),
                    "checks": checks,
                    "compiled_audit": {
                        "path": str(compiled_path),
                        "sha256": _sha256(compiled_path)
                        if compiled_path.is_file()
                        else None,
                    },
                    "rtx_audit": {
                        "path": str(rtx_path),
                        "sha256": _sha256(rtx_path) if rtx_path.is_file() else None,
                    },
                }
            )

    contract = plan["execution_contract"]
    matrix_checks = {
        "scene_count": len(plan["development_scenes"])
        == int(contract["scene_count"]),
        "material_count": len(plan["materials"]) == int(contract["material_count"]),
        "exact_cell_count": len(cells) == int(contract["exact_render_count"]),
        "unique_cells": len(
            {(cell["scene_id"], cell["material_id"]) for cell in cells}
        )
        == len(cells),
    }
    sealed = all(cell["terminal"] for cell in cells)
    integrity = (
        all(code_checks.values())
        and material_lock_check
        and all(scene_checks.values())
        and all(matrix_checks.values())
        and sealed
        and all(cell["integrity_passed"] for cell in cells)
    )
    matrix_passed = integrity and all(cell["passed"] is True for cell in cells)
    payload = {
        "schema_version": "kinofail.embodiedgen-route-surface-lighting-v4-matrix-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "sealed": sealed,
        "audit_integrity_passed": integrity,
        "matrix_passed": matrix_passed,
        "passed_cells": sum(cell["passed"] is True for cell in cells),
        "cell_count": len(cells),
        "frozen_code_checks": code_checks,
        "material_lock_check": material_lock_check,
        "scene_input_checks": scene_checks,
        "matrix_checks": matrix_checks,
        "cells": cells,
        "interpretation": (
            "Development-only evidence. All preregistered scene-material cells remain in "
            "the denominator; no cell relabels a sealed confirmation result."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if sealed and not integrity:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
