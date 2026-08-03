#!/usr/bin/env python3
"""Fail-closed preflight for the v14 multi-scene unseen extension."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/kinofail_embodiedgen_realistic_route_policy_v14_unseen_extension_batch_v1.json"
OUT = ROOT / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_realistic_route_policy_v14_unseen_extension_batch_v1/preflight_audit.json"


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
    boundary = config.get("evidence_boundary", {})
    selection = config.get("selection_policy", {})
    scenes = config.get("requested_scenes", [])
    scene_ids = [scene.get("scene_id") for scene in scenes]
    material_lock = _json(_resolve(config["frozen_files"]["material_lock"]["path"]))
    materials = {record["id"]: record for record in material_lock.get("materials", [])}
    checks: dict[str, bool] = {
        "supported_schema": config.get("schema_version")
        == "kinofail.embodiedgen-realistic-route-policy-v14-unseen-extension-batch.v1",
        "frozen_status": config.get("status")
        == "frozen_before_any_requested_scene_generation_or_rendering",
        "not_corpus_or_a0_a7_evidence": boundary.get("counts_as_realistic_corpus_evidence")
        is False
        and boundary.get("counts_as_a0_a7_evidence") is False
        and boundary.get("realistic_a0_a7_readiness") == "0/8",
        "completion_boundary": boundary.get(
            "completion_requires_new_realistic_corpus_training_inference_and_a0_a7"
        )
        is True,
        "exact_fixed_order": scene_ids == selection.get("fixed_order")
        and len(scenes) == int(selection.get("requested_scene_count", -1)) == 4,
        "four_distinct_new_families": {scene.get("room_type") for scene in scenes}
        == {"Office", "Bedroom", "LivingRoom", "Kitchen"},
        "one_attempt_all_count": selection.get("one_generation_attempt_per_scene") is True
        and selection.get("replacement_seed_allowed") is False
        and selection.get("all_requests_count_in_scene_admission_denominator") is True,
        "no_posthoc_tuning": selection.get("post_generation_source_repair_allowed")
        is False
        and selection.get(
            "post_generation_route_camera_lighting_or_threshold_tuning_allowed"
        )
        is False,
        "all_admitted_scenes_must_run_and_pass": selection.get(
            "run_nominal_policy_on_every_fully_admitted_scene"
        )
        is True
        and selection.get("all_runtime_eligible_scenes_must_pass") is True,
        "no_anomaly": selection.get("no_anomaly_before_batch_adjudication") is True,
        "bathroom_failure_retained": selection.get(
            "bathroom33_failure_remains_in_cumulative_denominator"
        )
        is True,
    }
    for name, spec in config.get("prior_evidence", {}).items():
        checks[f"prior_{name}_hash"] = _bound(spec)
    prior_failure = _json(
        _resolve(config["prior_evidence"]["bathroom33_retained_rejection"]["path"])
    )
    checks["prior_bathroom_failure_semantics"] = (
        prior_failure.get("audit_passed") is True
        and prior_failure.get("counts_in_requested_scene_denominator") is True
        and prior_failure.get("replacement_seed_authorized") is False
    )
    for name, spec in config.get("frozen_files", {}).items():
        checks[f"frozen_{name}_hash"] = _bound(spec)
    anchor = config.get("existing_untouched_anchor", {})
    for name in ("terrain_compiled_audit", "terrain_episode", "offline_usd_audit"):
        checks[f"anchor_{name}_hash"] = _bound(anchor[name])
    checks["anchor_output_unused"] = not _resolve(anchor["planned_runtime_output"]).exists()
    source_root = _resolve(config["scene_pipeline"]["source_root"])
    for index, scene in enumerate(scenes):
        slug = f"scene_{index + 1}_{scene['room_type'].lower()}"
        output = source_root / scene["output_directory"]
        request_file = source_root / f"{scene['room_type']}_seed{scene['source_scene_seed']}_request.json"
        checks[f"{slug}_output_unused"] = not output.exists()
        checks[f"{slug}_request_unused"] = not request_file.exists()
        material = materials.get(scene["material_id"], {})
        checks[f"{slug}_material_frozen"] = (
            material.get("id") == scene["material_id"]
            and material.get("split") == scene["material_split"]
        )
    runtime = config.get("runtime", {})
    checks["runtime_unchanged_from_v14"] = (
        runtime.get("operator") == "o2"
        and runtime.get("lane") == "nominal"
        and abs(float(runtime.get("target_lateral_offset_m", -1.0))) <= 1.0e-9
        and abs(float(runtime.get("forward_speed_mps", -1.0)) - 0.18) <= 1.0e-9
        and abs(float(runtime.get("maximum_route_deviation_m", -1.0)) - 0.30)
        <= 1.0e-9
    )
    result = {
        "schema_version": "kinofail.embodiedgen-realistic-route-policy-v14-unseen-extension-preflight.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "config": {"path": str(CONFIG), "sha256": _sha256(CONFIG)},
        "requested_scene_count": len(scenes),
        "generation_authorized": all(checks.values()),
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
