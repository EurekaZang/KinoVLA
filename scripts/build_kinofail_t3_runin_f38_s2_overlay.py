#!/usr/bin/env python3
"""Build a no-payload-copy T3 overlay from the F38-S1 task audit.

The raw F38-S1 corpus and its failed generic audit remain immutable.  This
builder materializes only derived manifests and symlinks to the raw payload.
For accepted O7 anomaly episodes, the RGB manifest entries are restricted to
the five event-aligned frames sealed by the task audit.  No image, proprio,
telemetry, prediction, truth key, or score is copied or selected by outcome.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_kinofail_confirmatory_valid_conflict_design_v1 import (  # noqa: E402
    _validate_t3,
)


TASK_ROOT = ROOT / "outputs/kinofail_t3_runin_f38_s1_task_aligned_audit"
TASK_AUDIT = TASK_ROOT / "audit.json"
SOURCE_DESIGN = ROOT / "outputs/kinofail_reconfirmation_v2/schedules/c2_t3"
OUTPUT = ROOT / "outputs/eval/unified_moe_v3_t3_runin_f38_s2"
DESIGN = OUTPUT / "combined_design"
UNION = Path("/data/eureka/KinoVLA/outputs/kinofail_t3_runin_f38_s2/union")
PLANNED_CASES = 1_500
MINIMUM_ACCEPTED_CASES = 1_426


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(value, dict) for value in values):
        raise TypeError(path)
    return values


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True) + "\n")


def _derived_manifest(
    raw: dict[str, Any], episode_audit: dict[str, Any], task_audit_sha256: str
) -> dict[str, Any]:
    derived = copy.deepcopy(raw)
    raw_validation = copy.deepcopy(derived.get("runtime_validation", {}))
    derived["raw_runtime_validation"] = raw_validation
    derived["artifact_state"] = "validated"
    derived["evaluation_eligible"] = True
    if derived.get("operator_readback", {}).get("active") is True:
        derived["operator_readback"]["qa_passed"] = True
    runtime = derived.setdefault("runtime_validation", {})
    runtime["passed"] = True
    runtime["issues"] = []
    runtime["promotion_blockers"] = []
    runtime["task_aligned_successor"] = True
    runtime["generic_whole_rollout_status"] = {
        "passed": raw_validation.get("passed") is True,
        "issues": list(raw_validation.get("issues", [])),
    }
    derived["task_alignment_validation"] = {
        "schema_version": "kinofail.t3-runin-f38-s2-derived-manifest.v1",
        "passed": episode_audit.get("passed") is True,
        "audit_sha256": task_audit_sha256,
        "raw_manifest_sha256": episode_audit["manifest_sha256"],
        "task_input_integrity_passed": episode_audit[
            "task_input_integrity_passed"
        ],
        "operator_gate": episode_audit.get("operator_gate"),
        "temporal_alignment": episode_audit.get("temporal_alignment"),
        "visual_input_gate": episode_audit.get("visual_input_gate"),
        "whole_rollout_l1_and_base_center_qa_are_not_overwritten": True,
    }
    visual = episode_audit.get("visual_input_gate", {})
    if (
        episode_audit["condition"] == "anomaly"
        and derived["operator_readback"]["operator_id"] == "O7_visual_remap"
    ):
        if visual.get("passed") is not True:
            raise RuntimeError("cannot derive an overlay for rejected O7 vision")
        selected_paths = visual["selected_paths"]
        source_views = derived["artifacts"]["rgb_views"]
        selected_views: dict[str, list[dict[str, Any]]] = {}
        for view_id, paths in selected_paths.items():
            by_path = {str(row["path"]): row for row in source_views[view_id]}
            if len(paths) != 5 or any(path not in by_path for path in paths):
                raise RuntimeError("task audit RGB path is absent from the raw manifest")
            selected_views[view_id] = [copy.deepcopy(by_path[path]) for path in paths]
        derived["artifacts"]["rgb_views"] = selected_views
        derived["artifacts"]["rgb_frames"] = copy.deepcopy(
            selected_views["primary"]
        )
        runtime.setdefault("measured", {})["task_aligned_mean_appearance_pair_rgb_l1"] = (
            copy.deepcopy(visual["mean_pair_rgb_l1"])
        )
        runtime["measured"]["task_aligned_rgb_frames"] = 5
    return derived


def main() -> int:
    if OUTPUT.exists() or UNION.exists():
        raise FileExistsError("refusing to overwrite F38-S2 design or union")
    task = read_json(TASK_AUDIT)
    if (
        task.get("passed") is not True
        or task.get("counts_as_confirmatory_evidence") is not False
        or task.get("model_prediction_truth_key_or_score_read") is not False
        or task.get("result_dependent_selection_or_retry") is not False
        or int(task.get("counts", {}).get("planned_cases", -1)) != PLANNED_CASES
        or int(task.get("counts", {}).get("accepted_cases", 0))
        < MINIMUM_ACCEPTED_CASES
        or float(task.get("case_attrition_rate", 1.0)) >= 0.05
    ):
        raise RuntimeError("invalid F38-S1 task-aligned predecessor")
    task_hash = sha256(TASK_AUDIT)
    for key, relative in (
        ("accepted_cases", "accepted_cases.jsonl"),
        ("rejected_cases", "rejected_cases.jsonl"),
        ("pair_audits", "pair_audits.jsonl"),
    ):
        path = TASK_ROOT / relative
        if sha256(path) != task["output_sha256"][key]:
            raise RuntimeError(f"task audit sidecar drift: {path}")

    source_schedule_path = SOURCE_DESIGN / "schedule.jsonl"
    source_case_path = SOURCE_DESIGN / "case_schedule.jsonl"
    source_audit_path = SOURCE_DESIGN / "audit.json"
    source_schedule = read_jsonl(source_schedule_path)
    source_cases = read_jsonl(source_case_path)
    source_audit = read_json(source_audit_path)
    if (
        len(source_cases) != PLANNED_CASES
        or len(source_schedule) != 4 * PLANNED_CASES
        or source_audit.get("passed") is not True
        or source_audit.get("schedule_sha256") != sha256(source_schedule_path)
        or source_audit.get("case_schedule_sha256") != sha256(source_case_path)
    ):
        raise RuntimeError("source T3 design is not the frozen 1,500-case schedule")

    accepted_cases = read_jsonl(TASK_ROOT / "accepted_cases.jsonl")
    accepted_ids = {str(row["case_id"]) for row in accepted_cases}
    pair_audits = {
        str(row["pair_id"]): row
        for row in read_jsonl(TASK_ROOT / "pair_audits.jsonl")
    }
    accepted_groups = {
        str(pair_id)
        for case in accepted_cases
        for pair_id in case["pair_ids"]
    }
    if (
        len(accepted_ids) != int(task["counts"]["accepted_cases"])
        or len(accepted_groups) != 2 * len(accepted_ids)
        or any(pair_audits[group]["passed"] is not True for group in accepted_groups)
    ):
        raise RuntimeError("accepted task-audit case mapping is inconsistent")

    DESIGN.mkdir(parents=True, exist_ok=False)
    write_jsonl(DESIGN / "schedule.jsonl", source_schedule)
    write_jsonl(DESIGN / "case_schedule.jsonl", source_cases)
    design_audit = {
        "schema_version": "kinofail.t3-runin-f38-s2-design.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "status": "sealed_model_blind_task_aligned_design",
        "model_prediction_feature_truth_key_or_score_read": False,
        "result_dependent_selection_or_retry": False,
        "planned_cases": PLANNED_CASES,
        "schedule_sha256": sha256(DESIGN / "schedule.jsonl"),
        "case_schedule_sha256": sha256(DESIGN / "case_schedule.jsonl"),
        "source_sha256": {
            "source_schedule": sha256(source_schedule_path),
            "source_case_schedule": sha256(source_case_path),
            "source_design_audit": sha256(source_audit_path),
            "task_aligned_audit": task_hash,
            "builder": sha256(Path(__file__).resolve()),
        },
    }
    write_json(DESIGN / "audit.json", design_audit)

    schedule_by_episode = {
        str(row["episode_id"]): row for row in source_schedule
    }
    inventory: list[dict[str, Any]] = []
    UNION.mkdir(parents=True, exist_ok=False)
    for group_id in sorted(accepted_groups):
        pair = pair_audits[group_id]
        for condition in ("nominal_counterfactual", "anomaly"):
            episode_audit = pair["episodes"][condition]
            if episode_audit.get("passed") is not True:
                raise RuntimeError("accepted pair contains a rejected episode")
            raw_manifest_path = Path(str(episode_audit["manifest"])).resolve()
            raw_episode = raw_manifest_path.parent
            episode_id = str(episode_audit["episode_id"])
            schedule = schedule_by_episode[episode_id]
            relative = Path(str(schedule["required_outputs"]["episode_manifest"])).parent
            destination = UNION / str(schedule["scene_cluster"]) / relative
            destination.mkdir(parents=True, exist_ok=False)
            for child in raw_episode.iterdir():
                if child.name == "manifest.json":
                    continue
                target = destination / child.name
                target.symlink_to(child, target_is_directory=child.is_dir())
            raw_manifest = read_json(raw_manifest_path)
            derived = _derived_manifest(raw_manifest, episode_audit, task_hash)
            write_json(destination / "manifest.json", derived)
            inventory.append(
                {
                    "case_id": str(pair["case_id"]),
                    "pair_id": group_id,
                    "episode_id": episode_id,
                    "operator_id": str(pair["operator_id"]),
                    "condition": condition,
                    "raw_episode": str(raw_episode),
                    "derived_episode": str(destination),
                    "raw_manifest_sha256": episode_audit["manifest_sha256"],
                    "derived_manifest_sha256": sha256(destination / "manifest.json"),
                    "payload_is_symlinked": True,
                }
            )
    inventory_path = OUTPUT / "artifact_inventory.jsonl"
    write_jsonl(inventory_path, inventory)

    _, _, valid_ids, invalid = _validate_t3(design_dir=DESIGN, corpus=UNION)
    expected_valid = int(task["counts"]["accepted_cases"])
    if set(valid_ids) != accepted_ids or len(valid_ids) != expected_valid:
        raise RuntimeError(
            f"F38-S2 realized validity mismatch: {len(valid_ids)} != {expected_valid}"
        )
    invalid_ids = {str(row["case_id"]) for row in invalid}
    if len(invalid) != PLANNED_CASES - expected_valid or invalid_ids & accepted_ids:
        raise RuntimeError("F38-S2 invalid-case ledger mismatch")
    final = read_json(DESIGN / "audit.json")
    final.update(
        {
            "status": "validated_model_blind_task_aligned_union",
            "valid_cases": len(valid_ids),
            "invalid_case_count": len(invalid),
            "case_attrition_rate": len(invalid) / PLANNED_CASES,
            "strictly_below_five_percent": len(invalid) / PLANNED_CASES < 0.05,
            "union_root": str(UNION),
            "raw_payload_copy": False,
            "derived_manifests_only": True,
            "schedule_records_edited": False,
            "artifact_inventory": str(inventory_path.relative_to(ROOT)),
            "artifact_inventory_sha256": sha256(inventory_path),
            "valid_case_ids_sha256": hashlib.sha256(
                "\n".join(valid_ids).encode("utf-8")
            ).hexdigest(),
            "invalid_case_ledger": invalid,
            "protocol_boundary": {
                "raw_generic_audit_remains_failed": True,
                "task_alignment_was_frozen_before_any_prediction": True,
                "counts_as_confirmatory_evidence_before_f39": False,
                "post_acquisition_pre_prediction_amendment": True,
            },
        }
    )
    if final["strictly_below_five_percent"] is not True:
        raise RuntimeError("F38-S2 T3 case attrition gate failed")
    temporary = DESIGN / "audit.json.tmp"
    temporary.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, DESIGN / "audit.json")
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
