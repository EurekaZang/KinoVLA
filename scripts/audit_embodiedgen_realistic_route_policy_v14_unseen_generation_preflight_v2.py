#!/usr/bin/env python3
"""Apply amendment 1 and fail closed before unseen scene generation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/kinofail_embodiedgen_realistic_route_policy_v14_unseen_generation_freeze_v1.json"
AMENDMENT = ROOT / "configs/data/kinofail_embodiedgen_realistic_route_policy_v14_unseen_generation_freeze_v1_amendment1.json"
OUT = ROOT / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_realistic_route_policy_v14_unseen_generation_preflight_v2.json"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _bound(spec: dict[str, Any]) -> bool:
    path = _resolve(spec["path"])
    return path.is_file() and _sha256(path) == spec["sha256"]


def main() -> int:
    config = _json(CONFIG)
    amendment = _json(AMENDMENT)
    correction = amendment.get("correction", {})
    selection = config.get("selection_policy", {})
    existing = config.get("existing_untouched_scene", {})
    request = config.get("new_untouched_scene_request", {})
    boundary = config.get("evidence_boundary", {})
    runtime = config.get("runtime", {})
    house_path = _resolve(selection["house_exclusion_evidence"]["path"])
    checks: dict[str, bool] = {
        "supported_base_schema": config.get("schema_version")
        == "kinofail.embodiedgen-realistic-route-policy-v14-unseen-generation-freeze.v1",
        "supported_amendment_schema": amendment.get("schema_version")
        == "kinofail.embodiedgen-realistic-route-policy-v14-unseen-generation-freeze-amendment.v1",
        "base_freeze_bound": _bound(amendment["base_freeze"]),
        "failed_preflight_bound": _bound(amendment["failed_preflight_retained"]),
        "amendment_only_hash_correction": correction.get("json_pointer")
        == "/selection_policy/house_exclusion_evidence/sha256"
        and correction.get("incorrect_sha256")
        == selection["house_exclusion_evidence"]["sha256"]
        and amendment.get("scientific_fields_changed") is False
        and amendment.get("scene_selection_changed") is False
        and amendment.get("seed_changed") is False
        and amendment.get("material_assignment_changed") is False
        and amendment.get("runtime_or_threshold_changed") is False,
        "corrected_house_exclusion_bound": house_path.is_file()
        and _sha256(house_path) == correction.get("correct_sha256"),
        "not_a0_a7_or_corpus_evidence": boundary.get("counts_as_a0_a7_evidence") is False
        and boundary.get("counts_as_realistic_corpus_evidence") is False
        and boundary.get("realistic_a0_a7_readiness") == "0/8",
        "completion_boundary": boundary.get(
            "completion_requires_new_realistic_corpus_training_inference_and_a0_a7"
        )
        is True,
        "fixed_two_family_order": selection.get("fixed_order")
        == ["indoor_bedroom_28", "indoor_bathroom_33"],
        "single_attempt_no_replacement": selection.get(
            "one_generation_attempt_for_new_scene"
        )
        is True
        and selection.get("replacement_seed_allowed") is False
        and selection.get("one_runtime_attempt_per_scene") is True,
        "no_posthoc_tuning": selection.get("post_generation_source_repair_allowed")
        is False
        and selection.get(
            "post_generation_route_camera_lighting_or_threshold_tuning_allowed"
        )
        is False,
        "nominal_before_anomaly": selection.get("nominal_only") is True
        and selection.get("no_anomaly_before_both_nominals_pass") is True,
        "existing_scene_identity": existing.get("scene_id") == "indoor_bedroom_28"
        and existing.get("scene_family") == "Bedroom",
        "new_request_identity": request.get("scene_id") == "indoor_bathroom_33"
        and request.get("room_type") == "Bathroom"
        and request.get("complexity") == "simple"
        and int(request.get("source_scene_seed", -1)) == 20261213
        and request.get("material_id") == "test_pathway002",
        "new_source_path_unused": not _resolve(request["source_output"]).exists(),
        "new_route_surface_path_unused": not _resolve(request["route_surface_output"]).exists(),
        "existing_runtime_output_unused": not _resolve(
            existing["planned_runtime_output"]
        ).exists(),
        "new_runtime_output_unused": not _resolve(request["planned_runtime_output"]).exists(),
        "runtime_unchanged_from_v14": runtime.get("operator") == "o2"
        and runtime.get("lane") == "nominal"
        and abs(float(runtime.get("target_lateral_offset_m", -1.0))) <= 1.0e-9
        and abs(float(runtime.get("forward_speed_mps", -1.0)) - 0.18) <= 1.0e-9
        and abs(float(runtime.get("maximum_route_deviation_m", -1.0)) - 0.30)
        <= 1.0e-9,
    }
    for name in (
        "route_surface_admission",
        "terrain_compiled_audit",
        "terrain_episode",
        "offline_usd_audit",
    ):
        checks[f"existing_{name}_hash"] = _bound(existing[name])
    existing_admission = _json(_resolve(existing["route_surface_admission"]["path"]))
    existing_terrain = _json(_resolve(existing["terrain_compiled_audit"]["path"]))
    checks["existing_inputs_passed"] = (
        existing_admission.get("passed") is True
        and existing_terrain.get("passed") is True
        and existing_admission.get("scene_id") == existing.get("scene_id")
        and existing_terrain.get("scene_id") == existing.get("scene_id")
    )
    for name, spec in config.get("frozen_files", {}).items():
        checks[f"frozen_{name}_hash"] = _bound(spec)
    result = {
        "schema_version": "kinofail.embodiedgen-realistic-route-policy-v14-unseen-generation-preflight.v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "base_config": {"path": str(CONFIG), "sha256": _sha256(CONFIG)},
        "amendment": {"path": str(AMENDMENT), "sha256": _sha256(AMENDMENT)},
        "new_scene_generation_authorized": all(checks.values()),
        "runtime_collection_authorized": False,
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
    }
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {OUT}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "passed": result["passed"]}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
