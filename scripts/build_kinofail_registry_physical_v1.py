#!/usr/bin/env python3
"""Freeze the post-registry-change O5/O11 physical action experiment."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/kinofail_action_multiscene_v1_formal_v4"
OUTPUT = ROOT / "outputs/kinofail_registry_physical_v1"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_protocol = load(SOURCE / "protocol.json")
    selected: list[dict[str, Any]] = []
    for source in rows(SOURCE / "schedule.jsonl"):
        operator = str(source["operator"])
        if operator not in {"O5_payload", "O11_obs_bias"}:
            continue
        case = copy.deepcopy(source)
        case["schema_version"] = "kinofail.registry-physical-v1-schedule.v1"
        case["case_id"] = str(case["case_id"]).replace("actionmsv1__formal__", "registryphysicalv1__")
        case["severity_id"] = "post_registry_change_capability_band"
        case["actions"] = (
            [
                "recover_as_sensor_recalibrate",
                "recover_as_hold_request",
                "recover_as_payload_unload",
            ]
            if operator == "O5_payload"
            else [
                "recover_as_payload_unload",
                "recover_as_hold_request",
                "recover_as_sensor_recalibrate",
            ]
        )
        case["pairing"] = "one captured physical checkpoint restored across three remediation arms"
        case["post_change_registry"] = {
            "O5_payload": "payload_unload",
            "O11_obs_bias": "sensor_recalibrate",
        }
        case["outcome_selection_used"] = False
        selected.append(case)

    scenes = sorted({str(row["scene_id"]) for row in selected})
    per_cell = {
        (scene, operator): sum(
            row["scene_id"] == scene and row["operator"] == operator for row in selected
        )
        for scene in scenes
        for operator in ("O5_payload", "O11_obs_bias")
    }
    if len(scenes) != 12 or set(per_cell.values()) != {2}:
        raise RuntimeError({"scenes": len(scenes), "per_cell": per_cell})

    OUTPUT.mkdir(parents=True)
    schedule = OUTPUT / "schedule.jsonl"
    schedule.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    registry = load(SOURCE / "scene_registry.json")
    (OUTPUT / "scene_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    protocol = copy.deepcopy(source_protocol)
    protocol.update(
        {
            "schema_version": "kinofail.registry-physical-v1-protocol.v1",
            "protocol_id": "kinofail-registry-physical-v1-post-freeze-20260814",
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "formal_post_freeze_confirmation",
            "development_only": False,
            "integrity_mode": "path_and_count",
            "schedule": str(schedule),
            "scene_registry": str(OUTPUT / "scene_registry.json"),
            "collector": "scripts/isaac_collect_kinofail_action_full_v1.py",
            "runner": "scripts/run_kinofail_experiment_v1.py",
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
            "action_arms": [
                "post-change cause-matched physical remediation",
                "pre-change shared hold/request program",
                "cross-cause mismatched physical remediation",
            ],
            "physical_interventions": {
                "payload_unload": "restore nominal trunk mass/CoM/inertia and remove the visible rigid payload before controlled resume",
                "sensor_recalibrate": "bypass the injected observation transform and resume from the raw synchronized state",
            },
            "analysis_plan": {
                "unit": "physical checkpoint",
                "primary_endpoint": "operator_recovery_success",
                "secondary_endpoints": ["fell", "terminal_cost", "safety_exposure_auc"],
                "primary_contrast": "post-change cause-matched remediation versus cross-cause remediation",
                "reference_contrast": "post-change cause-matched remediation versus pre-change hold/request",
                "cluster": "scene",
                "unfavorable_outcomes_retained": True,
            },
            "selection_contract": {
                "uses_all_formal O5/O11 source cases": True,
                "reads_model_predictions": False,
                "reads_action_outcomes": False,
                "result_dependent_parameter_change": False,
            },
        }
    )
    for key in list(protocol):
        if key.endswith("_sha256"):
            protocol.pop(key)
    (OUTPUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    design = {
        "schema_version": "kinofail.registry-physical-v1-design-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "counts": protocol["counts"],
        "scenes": scenes,
        "cases_per_scene_operator": 2,
        "outcome_selection_used": False,
        "post_change_registry_frozen_before_collection": True,
    }
    (OUTPUT / "design_audit.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(design, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
