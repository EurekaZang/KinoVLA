#!/usr/bin/env python3
"""Seal the v14 unseen-extension batch without converting protocol drift into a scene result."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "kinofail.embodiedgen-realistic-route-policy-v14-unseen-extension-postrun.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": _sha256(path) if path.is_file() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_embodiedgen_realistic_route_policy_v14_unseen_extension_batch_v1.json",
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_realistic_route_policy_v14_unseen_extension_batch_v1/preflight_audit.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_realistic_route_policy_v14_unseen_extension_batch_v1/postrun_audit.json",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    preflight_path = args.preflight.resolve()
    config = _json(config_path)
    preflight = _json(preflight_path)
    source_root = _resolve(config["scene_pipeline"]["source_root"])
    route_name = config["scene_pipeline"]["route_surface_output_name"]

    scene_rows: list[dict[str, Any]] = []
    friction_contract_absent_for_all = True
    rtx_pass_count = 0
    for request in config["requested_scenes"]:
        scene_dir = source_root / request["output_directory"]
        material_id = request["material_id"]
        route_dir = scene_dir / route_name / material_id
        collision_path = scene_dir / config["scene_pipeline"]["base_output_name"] / "collision.usda"
        collision_text = collision_path.read_text(encoding="utf-8") if collision_path.is_file() else ""
        friction_authored = (
            "physics:staticFriction" in collision_text
            and "physics:dynamicFriction" in collision_text
        )
        friction_contract_absent_for_all = friction_contract_absent_for_all and not friction_authored

        source_preflight_path = (
            scene_dir / "source_geometry_preflight_v1/source_geometry_preflight_audit.json"
        )
        base_path = scene_dir / config["scene_pipeline"]["base_output_name"] / "compiled_scene_audit.json"
        corridor_path = scene_dir / config["scene_pipeline"]["corridor_output_name"] / "compiled_scene_audit.json"
        route_path = route_dir / "compiled_scene_audit.json"
        rtx_path = route_dir / "rtx_qa/rtx_scene_audit.json"
        go2_path = route_dir / "go2_qa/go2_scene_audit.json"

        source_preflight = _json(source_preflight_path) if source_preflight_path.is_file() else {}
        base = _json(base_path) if base_path.is_file() else {}
        corridor = _json(corridor_path) if corridor_path.is_file() else {}
        route = _json(route_path) if route_path.is_file() else {}
        rtx = _json(rtx_path) if rtx_path.is_file() else {}
        if rtx.get("passed") is True:
            rtx_pass_count += 1

        scene_rows.append(
            {
                "scene_id": request["scene_id"],
                "scene_family": request["room_type"],
                "source_seed": request["source_scene_seed"],
                "runtime_seed": request["runtime_seed"],
                "material_id": material_id,
                "stages": {
                    "source_preflight": {**_record(source_preflight_path), "passed": source_preflight.get("passed")},
                    "base_compile": {**_record(base_path), "passed": base.get("passed")},
                    "corridor_v2": {**_record(corridor_path), "passed": corridor.get("passed")},
                    "route_surface_v4": {**_record(route_path), "passed": route.get("passed")},
                    "rtx_v5": {**_record(rtx_path), "passed": rtx.get("passed")},
                    "go2_qa": {
                        **_record(go2_path),
                        "passed": None,
                        "not_adjudicated": True,
                        "reason": "frozen QA/backend dependency closure cannot execute route-surface-only floor",
                    },
                },
                "route_surface_floor_has_authored_static_and_dynamic_friction": friction_authored,
                "fully_admitted": False,
                "nominal_policy_run_authorized": False,
            }
        )

    backend_path = ROOT / "kino_vla/sim/isaac_policy_backend.py"
    qa_path = ROOT / config["frozen_files"]["go2_qa"]["path"]
    frozen_backend_entry = config["frozen_files"].get("isaac_policy_backend")
    checks = {
        "preflight_passed": preflight.get("passed") is True,
        "all_four_requests_retained": len(scene_rows) == 4,
        "all_sources_passed": all(row["stages"]["source_preflight"]["passed"] is True for row in scene_rows),
        "all_base_compiles_passed": all(row["stages"]["base_compile"]["passed"] is True for row in scene_rows),
        "all_corridor_refinements_passed": all(row["stages"]["corridor_v2"]["passed"] is True for row in scene_rows),
        "all_route_surface_compiles_passed": all(row["stages"]["route_surface_v4"]["passed"] is True for row in scene_rows),
        "rtx_results_complete": all(row["stages"]["rtx_v5"]["passed"] in (True, False) for row in scene_rows),
        "route_surface_floor_friction_contract_absent_for_all": friction_contract_absent_for_all,
        "backend_missing_from_frozen_dependency_closure": frozen_backend_entry is None,
        "go2_audits_not_emitted": all(not row["stages"]["go2_qa"]["exists"] for row in scene_rows),
        "no_scene_fully_admitted": all(not row["fully_admitted"] for row in scene_rows),
        "no_nominal_policy_result_claimed": all(not row["nominal_policy_run_authorized"] for row in scene_rows),
    }
    sealed = all(checks.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "sealed": sealed,
        "batch_outcome": "invalidated_by_incomplete_frozen_dependency_closure_and_stage_order",
        "policy_conclusion": "not_tested",
        "scene_admission_conclusion": "not_estimable_from_this_batch",
        "failure_scope": "protocol_infrastructure_not_locomotion_policy_and_not_scene_geometry",
        "config": _record(config_path),
        "preflight": _record(preflight_path),
        "dependency_drift": {
            "go2_qa_script": _record(qa_path),
            "isaac_policy_backend_current": _record(backend_path),
            "backend_was_frozen_in_batch_config": frozen_backend_entry is not None,
            "incompatibility": (
                "backend.load_realistic_scene reads physics friction before the planned terrain-route "
                "layer is composed; route-surface v4 intentionally carries no physical material"
            ),
            "observed_attempts": [
                {
                    "scene_id": "indoor_office_34",
                    "exception": "RuntimeError: no physics material attributes found under /World/KinoIndoor/Collision/Floor",
                    "go2_audit_emitted": False,
                },
                {
                    "scene_id": "indoor_bedroom_35",
                    "exception": "RuntimeError: no physics material attributes found under /World/KinoIndoor/Collision/Floor",
                    "go2_audit_emitted": False,
                },
            ],
        },
        "stage_order_correction": [
            "source/base/corridor/route-surface offline checks",
            "RTX visual QA",
            "terrain-route v2 physical-material composition",
            "offline terrain USD audit",
            "articulated Go2 QA on episode_terrain_v2.usda",
            "combined visual-plus-physics admission",
            "nominal policy confirmation",
        ],
        "rtx_summary": {
            "passed": rtx_pass_count,
            "total": len(scene_rows),
            "failed_scene_ids": [
                row["scene_id"] for row in scene_rows if row["stages"]["rtx_v5"]["passed"] is False
            ],
        },
        "scenes": scene_rows,
        "checks": checks,
        "evidence_boundary": {
            "development_only": True,
            "counts_as_realistic_corpus_evidence": False,
            "counts_as_a0_a7_evidence": False,
            "realistic_a0_a7_readiness": "0/8",
            "old_a0_a7_results_are_reference_only": True,
            "a8_excluded": True,
        },
        "next_protocol_requirements": {
            "freeze_transitive_runtime_backend": True,
            "failure_safe_qa_must_emit_exception_audit": True,
            "terrain_layer_must_precede_go2_qa": True,
            "current_scenes_may_only_be_used_for_development_validation": True,
            "formal_confirmation_requires_new_untouched_scenes": True,
        },
    }
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite sealed postrun: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "sealed": sealed, "batch_outcome": result["batch_outcome"], "rtx_summary": result["rtx_summary"]}, indent=2))
    return 0 if sealed else 2


if __name__ == "__main__":
    raise SystemExit(main())
