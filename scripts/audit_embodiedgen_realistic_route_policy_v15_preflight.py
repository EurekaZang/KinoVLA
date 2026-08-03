#!/usr/bin/env python3
"""Fail-closed preflight for the dependency-closed v15 unseen scene batch."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/kinofail_embodiedgen_realistic_route_policy_v15_unseen_batch_v1.json"
OUT = ROOT / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_realistic_route_policy_v15_unseen_batch_v1/preflight_audit.json"


def _resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


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


def _bound(spec: dict[str, Any]) -> bool:
    path = _resolve(spec["path"])
    return path.is_file() and _sha256(path) == spec["sha256"]


def main() -> int:
    config = _json(CONFIG)
    boundary = config.get("evidence_boundary", {})
    selection = config.get("selection_policy", {})
    pipeline = config.get("scene_pipeline", {})
    scenes = config.get("requested_scenes", [])
    material_lock = _json(_resolve(config["frozen_files"]["material_lock"]["path"]))
    materials = {record["id"]: record for record in material_lock.get("materials", [])}
    checks: dict[str, bool] = {
        "supported_schema": config.get("schema_version")
        == "kinofail.embodiedgen-realistic-route-policy-v15-unseen-batch.v1",
        "frozen_before_generation": config.get("status")
        == "frozen_before_any_requested_scene_generation_or_rendering",
        "strict_evidence_boundary": boundary.get("development_only") is True
        and boundary.get("counts_as_realistic_corpus_evidence") is False
        and boundary.get("counts_as_a0_a7_evidence") is False
        and boundary.get("realistic_a0_a7_readiness") == "0/8"
        and boundary.get("old_a0_a7_results_are_reference_only") is True
        and boundary.get("a8_excluded") is True,
        "completion_boundary_retained": boundary.get(
            "completion_requires_new_realistic_corpus_training_inference_and_a0_a7"
        )
        is True,
        "six_fixed_requests": len(scenes)
        == int(selection.get("requested_scene_count", -1))
        == 6
        and [scene.get("scene_id") for scene in scenes] == selection.get("fixed_order"),
        "six_distinct_families": len({scene.get("room_type") for scene in scenes}) == 6,
        "one_attempt_no_replacement": selection.get("one_generation_attempt_per_scene")
        is True
        and selection.get("replacement_seed_allowed") is False
        and selection.get("all_requests_count_in_scene_admission_denominator") is True,
        "no_posthoc_scene_or_gate_tuning": selection.get(
            "post_generation_source_repair_allowed"
        )
        is False
        and selection.get(
            "post_generation_route_camera_lighting_physics_or_threshold_tuning_allowed"
        )
        is False,
        "all_eligible_stages_required": selection.get(
            "run_each_stage_on_every_scene_eligible_for_that_stage"
        )
        is True,
        "runtime_only_for_fully_admitted": selection.get(
            "run_nominal_policy_on_every_and_only_fully_admitted_scene"
        )
        is True,
        "no_anomaly_before_adjudication": selection.get(
            "no_anomaly_before_batch_adjudication"
        )
        is True,
        "minimum_confirmation_not_single_scene": int(
            selection.get("minimum_new_fully_admitted_scenes_for_policy_confirmation", 0)
        )
        >= 2
        and int(selection.get("minimum_new_scene_families_for_policy_confirmation", 0))
        >= 2,
        "corrected_stage_order": pipeline.get("stage_order")
        == [
            "source_generation",
            "source_geometry_preflight",
            "base_compile",
            "corridor_v2",
            "route_surface_v4",
            "rtx_v5",
            "terrain_route_v2_compile",
            "terrain_route_v2_offline_audit",
            "go2_qa_on_terrain_episode",
            "realistic_stack_v15_admission",
            "nominal_policy_confirmation",
        ],
        "terrain_precedes_go2": pipeline.get("stage_order", []).index(
            "terrain_route_v2_compile"
        )
        < pipeline.get("stage_order", []).index("go2_qa_on_terrain_episode"),
        "dependency_closure_includes_backend": "isaac_policy_backend"
        in config.get("frozen_files", {}),
        "dependency_closure_includes_configs": all(
            name in config.get("frozen_files", {})
            for name in ("benchmark_camera_config", "go2_sim_config", "route_protocol")
        ),
        "runtime_root_unused": not _resolve(config["runtime_output_root"]).exists(),
    }
    for name, spec in config.get("frozen_files", {}).items():
        checks[f"frozen_{name}_hash"] = _bound(spec)
    for name, spec in config.get("prior_evidence", {}).items():
        checks[f"prior_{name}_hash"] = _bound(spec)

    v14 = _json(_resolve(config["prior_evidence"]["v14_protocol_invalidation"]["path"]))
    checks["v14_failure_not_miscast_as_policy_result"] = (
        v14.get("batch_outcome")
        == "invalidated_by_incomplete_frozen_dependency_closure_and_stage_order"
        and v14.get("policy_conclusion") == "not_tested"
    )
    for name in (
        "livingroom36_dependency_closed_development_admission",
        "kitchen37_dependency_closed_development_admission",
    ):
        admission = _json(_resolve(config["prior_evidence"][name]["path"]))
        checks[f"prior_{name}_semantics"] = admission.get("passed") is True and admission.get(
            "counts_as_a0_a7_evidence"
        ) is False
    smoke = _json(_resolve(config["prior_evidence"]["failure_safe_qa_smoke"]["path"]))
    checks["failure_safe_exception_audit_proven"] = smoke.get("passed") is False and smoke.get(
        "admission_state"
    ) == "articulated_go2_scene_qa_exception"

    for name in ("terrain_compiled_audit", "terrain_episode", "offline_usd_audit"):
        checks[f"anchor_{name}_hash"] = _bound(config["existing_anchor"][name])
    checks["anchor_runtime_unused"] = not _resolve(
        config["existing_anchor"]["planned_runtime_output"]
    ).exists()

    source_root = _resolve(pipeline["source_root"])
    for index, scene in enumerate(scenes, start=1):
        slug = f"scene_{index}_{scene['room_type'].lower()}"
        output = source_root / scene["output_directory"]
        request = source_root / f"{scene['output_directory']}_request.json"
        checks[f"{slug}_source_output_unused"] = not output.exists()
        checks[f"{slug}_request_output_unused"] = not request.exists()
        material = materials.get(scene["material_id"], {})
        checks[f"{slug}_material_frozen"] = material.get("id") == scene[
            "material_id"
        ] and material.get("split") == scene["material_split"]

    runtime = config.get("runtime", {})
    checks["runtime_contract_frozen"] = (
        runtime.get("operator") == "o2"
        and runtime.get("lane") == "nominal"
        and runtime.get("camera_profile") == "go2_front_calib_b"
        and int(runtime.get("steps", 0)) == 650
        and abs(float(runtime.get("forward_speed_mps", 0.0)) - 0.18) <= 1.0e-12
        and abs(float(runtime.get("maximum_route_deviation_m", 0.0)) - 0.30)
        <= 1.0e-12
        and int(runtime.get("minimum_nominal_region_samples", 0)) == 40
    )
    checks["go2_qa_contract_frozen"] = (
        int(pipeline.get("go2_qa_steps", 0)) == 140
        and abs(float(pipeline.get("go2_qa_target_progress_m", 0.0)) - 0.50)
        <= 1.0e-12
        and pipeline.get("go2_qa_camera_profile") == "go2_front_calib_c"
        and abs(float(pipeline.get("terrain_static_friction", 0.0)) - 0.8) <= 1.0e-12
        and abs(float(pipeline.get("terrain_dynamic_friction", 0.0)) - 0.6)
        <= 1.0e-12
    )

    passed = all(checks.values())
    result = {
        "schema_version": "kinofail.embodiedgen-realistic-route-policy-v15-preflight.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": passed,
        "checks": checks,
        "config": {"path": str(CONFIG), "sha256": _sha256(CONFIG)},
        "requested_scene_count": len(scenes),
        "generation_authorized": passed,
        "runtime_collection_authorized": False,
        "counts_as_realistic_corpus_evidence": False,
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
    }
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {OUT}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "passed": passed}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
