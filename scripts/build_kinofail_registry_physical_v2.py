#!/usr/bin/env python3
"""Freeze an independent O5/O11 remediation confirmation on unused checkpoints."""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW = Path("/data/eureka/kinofail_kino_v4_confirmation_v1")
RECORDS = ROOT / "outputs/eval/kino_v4_confirmation_v1_f5d/scale/records.jsonl"
SOURCE_PROTOCOL = ROOT / "outputs/kinofail_action_multiscene_v1_formal_v4/protocol.json"
SOURCE_REGISTRY = ROOT / "outputs/kinofail_action_multiscene_v1_formal_v4/scene_registry.json"
SOURCE_SCHEDULE = ROOT / "outputs/kinofail_action_multiscene_v1_formal_v4/schedule.jsonl"
OUTPUT = ROOT / "outputs/kinofail_registry_physical_v2"


PARAMETERS = {
    "O5_payload": {"mass_kg": 8.0, "com_offset_x_m": 0.08, "com_offset_y_m": 0.0},
    "O11_obs_bias": {"tilt_bias_rad": 0.32, "random_walk_rad_sqrt_s": 0.018, "latency_s": 0.18},
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def episode_dir(record: dict[str, Any]) -> Path:
    return (
        RAW
        / "corpus"
        / str(record["scene_cluster"])
        / "scale"
        / str(record["material_family"])
        / str(record["target_operator"])
        / str(record["physical_episode_id"])
    )


def first_event(operator: str, privileged: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in privileged:
        telemetry = dict(row.get("operator", {}))
        timestamp = float(row.get("timestamp_s", 0.0))
        if operator == "O5_payload" and timestamp >= 0.02 and float(telemetry.get("payload_kg", 0.0)) > 0.0:
            return row
        if (
            operator == "O11_obs_bias"
            and timestamp >= 0.40
            and telemetry.get("raw_pipeline_active") is True
            and telemetry.get("initialized") is True
        ):
            return row
    return None


def build_case(record: dict[str, Any], episode: Path, manifest: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    operator = str(record["target_operator"])
    nuisance = dict(manifest["collection"]["physical_nuisance"])
    source_record = {
        "target_operator": operator,
        "physics_parameters": copy.deepcopy(PARAMETERS[operator]),
        "condition": "anomaly",
        "physical_realization": str(manifest["geometry_readback"]["physical_realization"]),
        "physical_nuisance": nuisance,
        "operator_seed": int(nuisance["physics_seed"]),
    }
    actions = (
        ["recover_as_sensor_recalibrate", "recover_as_hold_request", "recover_as_payload_unload"]
        if operator == "O5_payload"
        else ["recover_as_payload_unload", "recover_as_hold_request", "recover_as_sensor_recalibrate"]
    )
    return {
        "schema_version": "kinofail.registry-physical-v2-schedule.v1",
        "case_id": (
            f"registryphysicalv2__{record['scene_cluster']}__{operator}__"
            f"{record['counterfactual_group_id']}"
        ),
        "scene_id": str(record["scene_cluster"]),
        "domain": str(record["domain"]),
        "operator": operator,
        "severity_id": "post_registry_change_capability_band",
        "source_counterfactual_group_id": str(record["counterfactual_group_id"]),
        "source_slot_episode_id": str(record["physical_episode_id"]),
        "source_physical_episode_id": str(record["physical_episode_id"]),
        "source_record": source_record,
        "source_is_f33_direct_o9": False,
        "reset_seed": int(nuisance["physics_seed"]),
        "source_physical_nuisance": nuisance,
        "source_f35_decision": {
            "sample_id": f"{record['physical_episode_id']}__action_boundary",
            "decision_time_s": float(event["timestamp_s"]),
            "event_time_s": float(event["timestamp_s"]),
            "certificate_mode": "same_run_action_boundary",
            "external_snapshot_replay_claimed": False,
            "source_manifest": str(episode / "manifest.json"),
        },
        "actions": actions,
        "operator_recovery": (
            "recover_as_payload_unload" if operator == "O5_payload" else "recover_as_sensor_recalibrate"
        ),
        "decision_mode": "first_recoverable_operator_event",
        "operator_engagement_dwell_steps": 1 if operator == "O5_payload" else 20,
        "pairing": "one unused physical checkpoint restored across three remediation arms",
        "source_observation_replay_required": False,
        "selection_used_model_predictions_or_action_outcomes": False,
        "decision_state_eligibility": {
            "last_base_height_m": float(event["base_height_m"]),
            "last_absolute_tilt_rad": abs(float(event["tilt_rad"])),
            "source_window_shape": None,
            "action_boundary_event_present": True,
            "source_admission": "runtime_validation_passed",
        },
        "strict_o9_semantic_certificate": None,
        "action_boundary_semantic_certificate": None,
        "development_only": False,
        "counts_as_publication_evidence": True,
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    used = {
        str(row["source_counterfactual_group_id"])
        for row in jsonl(SOURCE_SCHEDULE)
        if row["operator"] in {"O5_payload", "O11_obs_bias"}
    }
    candidates: dict[tuple[str, str], dict[str, tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any]]]] = defaultdict(dict)
    for record in jsonl(RECORDS):
        if (
            record.get("condition") != "anomaly"
            or record.get("appearance_intervention_id") != "primary"
            or record.get("target_operator") not in {"O5_payload", "O11_obs_bias"}
            or str(record["counterfactual_group_id"]) in used
        ):
            continue
        episode = episode_dir(record)
        manifest_path = episode / "manifest.json"
        privileged_path = episode / "privileged.jsonl"
        if not manifest_path.is_file() or not privileged_path.is_file():
            continue
        manifest = load(manifest_path)
        if (
            manifest.get("runtime_validation", {}).get("evaluation_eligible") is not True
            or manifest.get("operator_readback", {}).get("qa_passed") is not True
            or manifest.get("geometry_readback", {}).get("qa_passed") is not True
        ):
            continue
        event = first_event(str(record["target_operator"]), jsonl(privileged_path))
        if event is None or float(event["base_height_m"]) < 0.23 or abs(float(event["tilt_rad"])) >= 0.55:
            continue
        key = (str(record["scene_cluster"]), str(record["target_operator"]))
        candidates[key].setdefault(
            str(record["counterfactual_group_id"]), (record, episode, manifest, event)
        )

    selected: list[dict[str, Any]] = []
    cells: dict[str, int] = {}
    registry = load(SOURCE_REGISTRY)
    for scene_row in registry["scenes"]:
        scene = str(scene_row["scene_id"])
        for operator in ("O5_payload", "O11_obs_bias"):
            pool = candidates.get((scene, operator), {})
            ids = sorted(pool)[:2]
            cells[f"{scene}/{operator}"] = len(ids)
            if len(ids) != 2:
                raise RuntimeError(f"unused checkpoint coverage missing: {scene}/{operator}={len(ids)}")
            for group_id in ids:
                selected.append(build_case(*pool[group_id]))

    OUTPUT.mkdir(parents=True)
    schedule = OUTPUT / "schedule.jsonl"
    schedule.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected), encoding="utf-8"
    )
    (OUTPUT / "scene_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    protocol = load(SOURCE_PROTOCOL)
    protocol.update(
        {
            "schema_version": "kinofail.registry-physical-v2-protocol.v1",
            "protocol_id": "kinofail-registry-physical-v2-independent-confirmation-20260814",
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "formal_independent_confirmation_frozen",
            "development_only": False,
            "integrity_mode": "path_and_count",
            "schedule": str(schedule),
            "scene_registry": str(OUTPUT / "scene_registry.json"),
            "collector": "scripts/isaac_collect_kinofail_remediation_v2.py",
            "runner": "scripts/run_kinofail_experiment_v1.py",
            "horizon_steps": 1200,
            "counts": {
                "scenes": 12,
                "operators": 2,
                "physical_cases": len(selected),
                "action_arms": 3,
                "recovery_action_arms": 3,
                "unique_recovery_control_programs": 3,
                "physical_episodes": len(selected) * 3,
            },
            "operator_to_post_change_remediation": {
                "O5_payload": "recover_as_payload_unload",
                "O11_obs_bias": "recover_as_sensor_recalibrate",
            },
            "controller_program": "fault removal followed by 0.24 m/s nominal-posture controlled resume",
            "analysis_plan": {
                "unit": "unused physical checkpoint",
                "primary_endpoint": "operator_recovery_success",
                "primary_contrast": "cause-matched remediation versus cross-cause remediation",
                "reference_contrast": "cause-matched remediation versus hold/request",
                "cluster": "scene",
                "unfavorable_outcomes_retained": True,
            },
            "selection_contract": {
                "all_v1_source_ids_excluded": True,
                "selection_order": "first two lexicographic eligible unused physical group IDs per scene/operator",
                "reads_model_predictions": False,
                "reads_action_outcomes": False,
                "result_dependent_parameter_change_after_freeze": False,
            },
        }
    )
    for key in list(protocol):
        if key.endswith("_sha256"):
            protocol.pop(key)
    (OUTPUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "schema_version": "kinofail.registry-physical-v2-design-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": len(selected) == 48 and set(cells.values()) == {2},
        "counts": protocol["counts"],
        "unused_source_ids": True,
        "outcome_selection_used": False,
        "controller_frozen_before_v2_collection": True,
    }
    (OUTPUT / "design_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
