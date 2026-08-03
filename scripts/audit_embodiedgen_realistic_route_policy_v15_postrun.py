#!/usr/bin/env python3
"""Adjudicate the dependency-closed v15 unseen nominal-confirmation batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "kinofail.embodiedgen-realistic-route-policy-v15-postrun.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _artifact(path: Path) -> dict[str, Any]:
    record = _json(path)
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": _sha256(path) if path.is_file() else None,
        "passed": record.get("passed"),
        "schema_version": record.get("schema_version"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    preflight_path = args.preflight.resolve()
    config = _json(config_path)
    preflight = _json(preflight_path)
    source_root = _resolve(config["scene_pipeline"]["source_root"])
    runtime_root = _resolve(config["runtime_output_root"])

    rows: list[dict[str, Any]] = []
    for request in config["requested_scenes"]:
        source = source_root / request["output_directory"]
        route = (
            source
            / config["scene_pipeline"]["route_surface_output_name"]
            / request["material_id"]
        )
        terrain = route / config["scene_pipeline"]["terrain_output_name"]
        runtime = runtime_root / request["scene_id"] / "o2_nominal/lane_manifest.json"
        artifacts = {
            "source_preflight": _artifact(
                source / "source_geometry_preflight_v1/source_geometry_preflight_audit.json"
            ),
            "base_compile": _artifact(
                source / config["scene_pipeline"]["base_output_name"] / "compiled_scene_audit.json"
            ),
            "corridor_v2": _artifact(
                source / config["scene_pipeline"]["corridor_output_name"] / "compiled_scene_audit.json"
            ),
            "route_surface": _artifact(route / "compiled_scene_audit.json"),
            "rtx_v5": _artifact(route / "rtx_qa/rtx_scene_audit.json"),
            "terrain_compile": _artifact(terrain / "compiled_scene_audit.json"),
            "terrain_offline": _artifact(terrain / "offline_usd_audit.json"),
            "go2_qa": _artifact(terrain / "go2_qa/go2_scene_audit.json"),
            "go2_exception": _artifact(terrain / "go2_qa/go2_scene_exception_audit.json"),
            "stack_admission": _artifact(terrain / "realistic_stack_v15_admission.json"),
            "nominal_policy": _artifact(runtime),
        }
        admitted = artifacts["stack_admission"]["passed"] is True
        nominal_exists = artifacts["nominal_policy"]["exists"] is True
        terminal_failure = any(
            artifacts[name]["passed"] is False
            for name in (
                "source_preflight",
                "base_compile",
                "corridor_v2",
                "route_surface",
                "rtx_v5",
                "terrain_compile",
                "terrain_offline",
                "go2_qa",
                "go2_exception",
                "stack_admission",
            )
        )
        rows.append(
            {
                "scene_id": request["scene_id"],
                "scene_family": request["room_type"],
                "source_seed": request["source_scene_seed"],
                "runtime_seed": request["runtime_seed"],
                "material_id": request["material_id"],
                "artifacts": artifacts,
                "fully_admitted": admitted,
                "terminal_failure_observed": terminal_failure,
                "pipeline_terminal": admitted or terminal_failure,
                "nominal_policy_run_expected": admitted,
                "nominal_policy_run_present": nominal_exists,
                "nominal_policy_passed": artifacts["nominal_policy"]["passed"],
            }
        )

    anchor_path = _resolve(config["existing_anchor"]["planned_runtime_output"]) / "lane_manifest.json"
    anchor = _artifact(anchor_path)
    admitted = [row for row in rows if row["fully_admitted"]]
    selection = config["selection_policy"]
    checks = {
        "preflight_passed": preflight.get("passed") is True,
        "all_requests_accounted": len(rows) == int(selection["requested_scene_count"]),
        "all_pipelines_terminal": all(row["pipeline_terminal"] for row in rows),
        "runtime_only_for_admitted_scenes": all(
            row["nominal_policy_run_present"] == row["fully_admitted"] for row in rows
        ),
        "every_admitted_scene_nominal_passed": bool(admitted)
        and all(row["nominal_policy_passed"] is True for row in admitted),
        "minimum_admitted_scene_count": len(admitted)
        >= int(selection["minimum_new_fully_admitted_scenes_for_policy_confirmation"]),
        "minimum_admitted_family_count": len({row["scene_family"] for row in admitted})
        >= int(selection["minimum_new_scene_families_for_policy_confirmation"]),
        "existing_anchor_passed": anchor["passed"] is True,
    }
    confirmation_passed = all(checks.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "sealed": checks["all_requests_accounted"] and checks["all_pipelines_terminal"],
        "passed": confirmation_passed,
        "policy_confirmation_state": (
            "confirmed_on_new_dependency_closed_unseen_scenes"
            if confirmation_passed
            else "not_confirmed"
        ),
        "checks": checks,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "preflight": {"path": str(preflight_path), "sha256": _sha256(preflight_path)},
        "scene_admission": {
            "requested": len(rows),
            "admitted": len(admitted),
            "admission_rate": len(admitted) / len(rows) if rows else None,
            "admitted_scene_ids": [row["scene_id"] for row in admitted],
            "admitted_families": sorted({row["scene_family"] for row in admitted}),
        },
        "existing_anchor": anchor,
        "scenes": rows,
        "evidence_boundary": {
            "development_only": True,
            "counts_as_realistic_corpus_evidence": False,
            "counts_as_a0_a7_evidence": False,
            "realistic_a0_a7_readiness": "0/8",
            "old_a0_a7_results_are_reference_only": True,
            "a8_excluded": True,
        },
        "next_required_work": [
            "recalibrate_all_11_operators_on_admitted_realistic_scenes",
            "collect_registered_randomized_counterfactual_corpus",
            "train_and_infer_new_A0_on_realistic_corpus",
            "replicate_A1_through_A7_on_new_realistic_predictions",
        ],
    }
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite postrun: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "sealed": result["sealed"], "passed": confirmation_passed, "scene_admission": result["scene_admission"]}, indent=2))
    return 0 if confirmation_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
