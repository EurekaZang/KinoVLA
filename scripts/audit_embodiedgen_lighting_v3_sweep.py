#!/usr/bin/env python3
"""Seal the predeclared lighting-v3 development sweep, including failed cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    repo = Path(__file__).resolve().parents[1]
    frozen_code = {
        name: (repo / item["path"]).is_file()
        and _sha256(repo / item["path"]) == item["sha256"]
        for name, item in plan["frozen_code"].items()
    }
    cells = []
    for scene in plan["development_scenes"]:
        corridor = repo / scene["corridor_v2_audit"]
        corridor_ok = corridor.is_file() and _sha256(corridor) == scene["sha256"]
        scene_root = corridor.parent.parent
        for candidate in plan["candidates"]:
            root = scene_root / "lighting_v3_development" / candidate["candidate_id"]
            compiled_path = root / "compiled_scene_audit.json"
            audit_path = root / "rtx_qa/rtx_scene_audit.json"
            compiled = json.loads(compiled_path.read_text(encoding="utf-8"))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            views = audit.get("views", [])
            minimum_std = min(float(view["metrics"]["std_luminance"]) for view in views)
            minimum_range = min(
                float(view["metrics"]["p99_luminance"])
                - float(view["metrics"]["p01_luminance"])
                for view in views
            )
            checks = {
                "corridor_v2_hash": corridor_ok,
                "compiled_scene_id": compiled.get("scene_id") == scene["scene_id"],
                "compiled_candidate_id": compiled.get("lighting_contract", {}).get(
                    "development_candidate_id"
                )
                == candidate["candidate_id"],
                "compiled_dome_intensity": float(
                    compiled.get("lighting_contract", {}).get("neutral_dome_intensity", -1)
                )
                == float(candidate["dome_intensity"]),
                "compiled_sphere_intensity": float(
                    compiled.get("lighting_contract", {})
                    .get("ceiling_sphere_lights", {})
                    .get("intensity", -1)
                )
                == float(candidate["ceiling_sphere_intensity"]),
                "audit_binds_compiled": audit.get("compiled_audit_sha256")
                == _sha256(compiled_path),
                "audit_binds_episode": audit.get("episode_usd_sha256")
                == compiled.get("files", {}).get("episode_v3.usda"),
                "seven_views": len(views) == 7,
                "result_is_boolean": isinstance(audit.get("passed"), bool),
            }
            cells.append(
                {
                    "scene_id": scene["scene_id"],
                    "candidate_id": candidate["candidate_id"],
                    "passed": audit.get("passed") is True,
                    "minimum_std_luminance": minimum_std,
                    "minimum_dynamic_range": minimum_range,
                    "checks": checks,
                    "integrity_passed": all(checks.values()),
                    "compiled_audit": {
                        "path": str(compiled_path),
                        "sha256": _sha256(compiled_path),
                    },
                    "rtx_audit": {"path": str(audit_path), "sha256": _sha256(audit_path)},
                }
            )
    candidates = []
    for candidate in plan["candidates"]:
        subset = [cell for cell in cells if cell["candidate_id"] == candidate["candidate_id"]]
        candidates.append(
            {
                **candidate,
                "scene_passes": sum(cell["passed"] for cell in subset),
                "scene_count": len(subset),
                "eligible": len(subset) == len(plan["development_scenes"])
                and all(cell["passed"] for cell in subset),
                "minimum_std_luminance": min(cell["minimum_std_luminance"] for cell in subset),
                "minimum_dynamic_range": min(cell["minimum_dynamic_range"] for cell in subset),
            }
        )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = None
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                row["minimum_std_luminance"],
                row["minimum_dynamic_range"],
                -(row["dome_intensity"] + row["ceiling_sphere_intensity"]),
            ),
        )["candidate_id"]
    result = {
        "schema_version": "kinofail.embodiedgen-lighting-v3-development-sweep-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "sealed": len(cells) == int(plan["execution_contract"]["exact_render_count"]),
        "audit_integrity_passed": all(frozen_code.values())
        and all(cell["integrity_passed"] for cell in cells),
        "frozen_code_checks": frozen_code,
        "render_count": len(cells),
        "cells": cells,
        "candidates": candidates,
        "selected_candidate_id": selected,
        "sweep_passed": selected is not None,
        "decision": (
            "candidate_selected_for_new_heldout_confirmation"
            if selected is not None
            else "failed_no_candidate_selected_no_fourth_candidate_allowed"
        ),
        "interpretation": "Development-only lighting evidence; it does not relabel any sealed formal pair.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "sealed": result["sealed"],
                "audit_integrity_passed": result["audit_integrity_passed"],
                "render_count": result["render_count"],
                "selected_candidate_id": selected,
                "sweep_passed": result["sweep_passed"],
                "decision": result["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
