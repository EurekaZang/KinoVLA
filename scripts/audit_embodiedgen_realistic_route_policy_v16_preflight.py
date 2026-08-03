#!/usr/bin/env python3
"""Fail-closed preflight for the v16 unseen realistic-route batch."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/kinofail_embodiedgen_realistic_route_policy_v16_unseen_batch_v1.json"
OUT = ROOT / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_realistic_route_policy_v16_unseen_batch_v1/preflight_audit.json"


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
        == "kinofail.embodiedgen-realistic-route-policy-v16-unseen-batch.v1",
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
            "realistic_stack_v16_admission",
            "nominal_policy_confirmation",
        ],
        "source_preflight_path_is_unambiguous": pipeline.get(
            "source_preflight_output_relative"
        )
        == "source_geometry_preflight_v1/source_geometry_preflight_audit.json",
        "dependency_closure_includes_continuous_o2": all(
            name in config.get("frozen_files", {})
            for name in (
                "isaac_policy_backend",
                "terramechanics",
                "o2_operator",
                "collector_engine",
                "collector_wrapper",
            )
        ),
        "dependency_closure_includes_failure_safe_stages": all(
            name in config.get("frozen_files", {})
            for name in (
                "base_compiler_wrapper",
                "corridor_wrapper",
                "route_surface_wrapper",
                "terrain_compiler_wrapper",
                "go2_qa",
            )
        ),
        "runtime_root_unused": not _resolve(config["runtime_output_root"]).exists(),
    }
    for name, spec in config.get("frozen_files", {}).items():
        checks[f"frozen_{name}_hash"] = _bound(spec)
    for name, spec in config.get("prior_evidence", {}).items():
        checks[f"prior_{name}_hash"] = _bound(spec)

    development = _json(_resolve(config["prior_evidence"]["v16_development_diagnosis"]["path"]))
    checks["development_diagnosis_passed"] = development.get("passed") is True
    checks["development_kept_out_of_a0_a7"] = development.get("evidence_boundary", {}).get(
        "counts_as_a0_a7_evidence"
    ) is False
    sentinels = config.get("development_sentinels", [])
    checks["two_distinct_sentinels"] = len(sentinels) == 2 and len(
        {item.get("scene_family") for item in sentinels}
    ) == 2
    for item in sentinels:
        spec = item["nominal_manifest"]
        manifest = _json(_resolve(spec["path"])) if _bound(spec) else {}
        checks[f"sentinel_{item['scene_id']}_hash"] = _bound(spec)
        checks[f"sentinel_{item['scene_id']}_passed"] = manifest.get("passed") is True
        checks[f"sentinel_{item['scene_id']}_not_a0_a7"] = (
            manifest.get("counts_as_a0_a7_evidence") is False
        )

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
        and abs(float(runtime.get("forward_speed_mps", 0.0)) - 0.32) <= 1.0e-12
        and abs(float(runtime.get("maximum_route_deviation_m", 0.0)) - 0.30)
        <= 1.0e-12
        and int(runtime.get("minimum_nominal_region_samples", 0)) == 40
        and abs(float(runtime.get("lateral_velocity_damping", -1.0))) <= 1.0e-12
        and abs(float(runtime.get("lateral_limit_mps", 0.0)) - 0.20) <= 1.0e-12
    )
    checks["go2_qa_contract_frozen"] = (
        int(pipeline.get("go2_qa_steps", 0)) == 140
        and abs(float(pipeline.get("go2_qa_target_progress_m", 0.0)) - 0.50)
        <= 1.0e-12
        and abs(float(pipeline.get("go2_qa_command_speed_mps", 0.0)) - 0.32)
        <= 1.0e-12
        and pipeline.get("go2_qa_camera_profile") == "go2_front_calib_c"
        and abs(float(pipeline.get("terrain_static_friction", 0.0)) - 0.8) <= 1.0e-12
        and abs(float(pipeline.get("terrain_dynamic_friction", 0.0)) - 0.6)
        <= 1.0e-12
    )

    passed = all(checks.values())
    result = {
        "schema_version": "kinofail.embodiedgen-realistic-route-policy-v16-preflight.v1",
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
