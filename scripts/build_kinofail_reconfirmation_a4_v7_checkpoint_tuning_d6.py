#!/usr/bin/env python3
"""Freeze excluded O8-detour and O9-safe-stop development cases D6."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT_HINT = Path(__file__).resolve().parents[1]
if str(ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(ROOT_HINT))

from scripts.build_kinofail_reconfirmation_a4_v7_schedule import (
    ACTIONS,
    ROOT,
    SCALE,
    jsonl,
    load_f35_primary_decisions,
)


OUTPUT = ROOT / "outputs/kinofail_reconfirmation_a4_v7_checkpoint_tuning_d6"
P3 = ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p3/schedule.jsonl"
D5 = ROOT / "outputs/kinofail_reconfirmation_a4_v7_checkpoint_tuning_d5/design_audit.json"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_reconfirmation_a4_v7.py"
PRE = ROOT / "outputs/locomotion/policy.pt"
POLICY = ROOT / "outputs/locomotion/recovery_route_v1/policy.pt"
MANIFEST = ROOT / "outputs/locomotion/recovery_route_v1/training_manifest.json"
FREEZE = ROOT / "configs/locomotion/go2_recovery_route_ppo_v1_freeze.json"
SIM = ROOT / "configs/sim/go2_skeleton.yaml"
O8_GROUP = "cf_975fafa03ebbb740711a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path in (SCALE, P3, D5, COLLECTOR, PRE, POLICY, MANIFEST, FREEZE, SIM):
        if not path.is_file():
            raise FileNotFoundError(path)
    _, decisions = load_f35_primary_decisions()
    anomaly = {
        str(row["counterfactual_group_id"]): row
        for row in jsonl(SCALE)
        if row["condition"] == "anomaly"
    }
    source = anomaly[O8_GROUP]
    decision = decisions[O8_GROUP]
    values = decision["invariant_proprio_80"]
    if not (float(values[60]) >= 0.25 and abs(float(values[61])) < 0.55):
        raise RuntimeError("D6 O8 case is not recoverable at the frozen decision")
    o8 = {
        "schema_version": "kinofail.reconfirmation-a4-v7-schedule.v1",
        "case_id": "a4v7d6__confirm_v2_life_scene_02__O8_invisible_collider__moderate",
        "scene_id": source["scene_id"],
        "domain": source["domain"],
        "operator": "O8_invisible_collider",
        "severity_id": source["severity_id"],
        "source_counterfactual_group_id": O8_GROUP,
        "source_slot_episode_id": source["episode_id"],
        "source_physical_episode_id": source["episode_id"],
        "source_record": source,
        "source_is_f33_direct_o9": False,
        "source_is_f25_replacement": False,
        "reset_seed": int(source["physical_nuisance"]["physics_seed"]),
        "source_physical_nuisance": source["physical_nuisance"],
        "source_f35_decision": decision,
        "actions": list(ACTIONS),
        "operator_recovery": "backstep_detour_replan",
        "pairing": "outcome-exposed recoverable-at-decision O8 development D6",
        "source_observation_replay_required": True,
        "selection_used_model_predictions_or_outcomes": False,
        "decision_state_eligibility": {
            "last_base_height_m": float(values[60]),
            "last_absolute_tilt_rad": abs(float(values[61])),
            "minimum_base_height_m": 0.25,
            "maximum_absolute_tilt_rad": 0.55,
        },
        "development_only": True,
        "counts_as_a0_a7_evidence": False,
    }
    p3_o9 = next(row for row in jsonl(P3) if row["operator"] == "O9_high_centering")
    p3_o9.update(
        {
            "pairing": "outcome-exposed O9 safe-stop validation D6",
            "development_only": True,
            "counts_as_a0_a7_evidence": False,
            "f33_recorded_spawn_height_was_not_applied_by_source_collector_v2": True,
        }
    )
    rows = [o8, p3_o9]
    OUTPUT.mkdir(parents=True, exist_ok=False)
    schedule = OUTPUT / "schedule.jsonl"
    schedule.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    protocol = {
        "schema_version": "kinofail.reconfirmation-a4-v7-checkpoint-tuning-d6.v1",
        "protocol_id": "kinofail-reconfirmation-a4-v7-checkpoint-tuning-d6",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "outcome_exposed_development_only",
        "development_only": True,
        "outcome_exposed": True,
        "counts_as_a0_a7_evidence": False,
        "schedule": str(schedule.relative_to(ROOT)),
        "schedule_sha256": sha256(schedule),
        "collector": str(COLLECTOR.relative_to(ROOT)),
        "collector_sha256": sha256(COLLECTOR),
        "counts": {"scenes": 1, "physical_cases": 2, "physical_episodes": 14},
        "predecision_actor": {
            "policy": str(PRE.relative_to(ROOT)),
            "policy_sha256": sha256(PRE),
            "purpose": "replay_frozen_f35_predecision_state",
        },
        "low_level_actor": {
            "policy": str(POLICY.relative_to(ROOT)),
            "policy_sha256": sha256(POLICY),
            "training_manifest": str(MANIFEST.relative_to(ROOT)),
            "training_manifest_sha256": sha256(MANIFEST),
            "training_freeze": str(FREEZE.relative_to(ROOT)),
            "training_freeze_sha256": sha256(FREEZE),
            "sim_config": str(SIM.relative_to(ROOT)),
            "sim_config_sha256": sha256(SIM),
            "shared_across_all_action_arms": True,
            "receives_attribution_input": False,
        },
        "branching_contract": {
            "fresh_isaac_process_per_case": True,
            "shared_prefix_executions_per_case": 1,
            "all_seven_arms_restore_one_full_physical_checkpoint": True,
        },
        "d5_design_sha256": sha256(D5),
    }
    protocol_path = OUTPUT / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    audit = {
        "passed": True,
        "development_only": True,
        "outcome_exposed": True,
        "counts_as_a0_a7_evidence": False,
        "case_ids": [row["case_id"] for row in rows],
        "schedule_sha256": sha256(schedule),
        "protocol_sha256": sha256(protocol_path),
        "collector_sha256": sha256(COLLECTOR),
        "d5_design_sha256": sha256(D5),
    }
    (OUTPUT / "design_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
