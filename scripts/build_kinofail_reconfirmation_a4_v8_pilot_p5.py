#!/usr/bin/env python3
"""Freeze the final fresh-scene A4-v8 pilot after the O2 brace correction."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.build_kinofail_reconfirmation_a4_v7_schedule import (
    ACTIONS,
    F25_AUDIT,
    F25_SCHEDULE_ROOT,
    ROOT,
    SCALE,
    jsonl,
    load_f35_primary_decisions,
    sha256,
    stable_key,
)


OUTPUT = ROOT / "outputs/kinofail_reconfirmation_a4_v8_pilot_p5"
SCENE = "confirm_v2_production_scene_04"
CELLS = (
    ("O2_compliance", "hard", "deep_compliance_braced_safe_stop"),
    ("O4_tether", "hard", "adhesion_high_step_peel"),
    ("O5_payload", "moderate", "overload_braced_safe_stop"),
)
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_reconfirmation_a4_v7.py"
P4_AUDIT = ROOT / "outputs/kinofail_reconfirmation_a4_v8_pilot_p4/postrun_audit.json"
PRE = ROOT / "outputs/locomotion/policy.pt"
ACTOR = ROOT / "outputs/locomotion/recovery_route_v1/policy.pt"
ACTOR_MANIFEST = ROOT / "outputs/locomotion/recovery_route_v1/training_manifest.json"
ACTOR_FREEZE = ROOT / "configs/locomotion/go2_recovery_route_ppo_v1_freeze.json"
SIM = ROOT / "configs/sim/go2_skeleton.yaml"


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path in (COLLECTOR, P4_AUDIT, PRE, ACTOR, ACTOR_MANIFEST, ACTOR_FREEZE, SIM):
        if not path.is_file():
            raise FileNotFoundError(path)
    p4 = json.loads(P4_AUDIT.read_text())
    if p4.get("passed") is not True or p4.get("counts_as_a0_a7_evidence") is not False:
        raise RuntimeError("P5 requires a passed, excluded P4 engineering audit")

    valid, decisions = load_f35_primary_decisions()
    anomaly = {
        str(row["counterfactual_group_id"]): row
        for row in jsonl(SCALE)
        if row["condition"] == "anomaly"
    }
    f25 = json.loads(F25_AUDIT.read_text())
    replacements = {
        str(row["original_counterfactual_group_id"]): str(
            row["replacement_counterfactual_group_id"]
        )
        for row in f25["accepted"]
        if str(row["target_operator"]) != "O9_high_centering"
    }
    replacement_rows = {
        str(row["counterfactual_group_id"]): row
        for path in sorted(F25_SCHEDULE_ROOT.glob("*/scale/schedule.jsonl"))
        for row in jsonl(path)
        if row["condition"] == "anomaly"
    }
    schedule_rows = []
    for operator, severity, endpoint in CELLS:
        eligible = []
        for group_id in valid:
            original = anomaly[group_id]
            if (
                original["scene_id"] == SCENE
                and original["target_operator"] == operator
                and original["severity_id"] == severity
            ):
                values = decisions[group_id]["invariant_proprio_80"]
                height, tilt = float(values[60]), abs(float(values[61]))
                if height >= 0.25 and tilt < 0.55:
                    eligible.append((group_id, height, tilt))
        if not eligible:
            raise RuntimeError(f"no P5 eligible source: {SCENE}/{operator}/{severity}")
        group_id, height, tilt = min(
            eligible,
            key=lambda row: stable_key("a4-v8-pilot-p5", SCENE, operator, severity, row[0]),
        )
        original = anomaly[group_id]
        replacement_id = replacements.get(group_id)
        source = replacement_rows[replacement_id] if replacement_id else original
        nuisance = source["physical_nuisance"]
        schedule_rows.append(
            {
                "schema_version": "kinofail.reconfirmation-a4-v8-schedule.v1",
                "case_id": f"a4v8pilotp5__{SCENE}__{operator}__{severity}",
                "scene_id": SCENE,
                "domain": str(original["domain"]),
                "operator": operator,
                "severity_id": severity,
                "source_counterfactual_group_id": group_id,
                "source_slot_episode_id": str(original["episode_id"]),
                "source_physical_episode_id": str(source["episode_id"]),
                "source_record": source,
                "source_is_f33_direct_o9": False,
                "source_is_f25_replacement": replacement_id is not None,
                "reset_seed": int(nuisance["physics_seed"]),
                "source_physical_nuisance": nuisance,
                "source_f35_decision": decisions[group_id],
                "actions": list(ACTIONS),
                "operator_recovery": endpoint,
                "pairing": "fresh-scene final engineering pilot P5 after O2 brace correction",
                "source_observation_replay_required": True,
                "selection_used_model_predictions_or_outcomes": False,
                "decision_state_eligibility": {
                    "last_base_height_m": height,
                    "last_absolute_tilt_rad": tilt,
                    "minimum_base_height_m": 0.25,
                    "maximum_absolute_tilt_rad": 0.55,
                    "uses_only_frozen_f35_physical_observation": True,
                },
                "development_only": True,
                "counts_as_a0_a7_evidence": False,
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=False)
    schedule = OUTPUT / "schedule.jsonl"
    schedule.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in schedule_rows))
    protocol = {
        "schema_version": "kinofail.reconfirmation-a4-v8-pilot-p5.v1",
        "protocol_id": "kinofail-reconfirmation-a4-v8-excluded-pilot-p5",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "development_only_permanently_excluded",
        "development_only": True,
        "outcome_exposed": False,
        "counts_as_a0_a7_evidence": False,
        "scene": SCENE,
        "schedule": str(schedule.relative_to(ROOT)),
        "schedule_sha256": sha256(schedule),
        "collector": str(COLLECTOR.relative_to(ROOT)),
        "collector_sha256": sha256(COLLECTOR),
        "counts": {"scenes": 1, "physical_cases": 3, "physical_episodes": 21},
        "predecision_actor": {
            "policy": str(PRE.relative_to(ROOT)),
            "policy_sha256": sha256(PRE),
            "purpose": "replay_frozen_f35_predecision_state",
        },
        "low_level_actor": {
            "policy": str(ACTOR.relative_to(ROOT)),
            "policy_sha256": sha256(ACTOR),
            "training_manifest": str(ACTOR_MANIFEST.relative_to(ROOT)),
            "training_manifest_sha256": sha256(ACTOR_MANIFEST),
            "training_freeze": str(ACTOR_FREEZE.relative_to(ROOT)),
            "training_freeze_sha256": sha256(ACTOR_FREEZE),
            "sim_config": str(SIM.relative_to(ROOT)),
            "sim_config_sha256": sha256(SIM),
            "shared_across_all_action_arms": True,
            "receives_attribution_input": False,
        },
        "branching_contract": {
            "fresh_isaac_process_per_case": True,
            "serial_cases_required": True,
            "shared_prefix_executions_per_case": 1,
            "all_seven_arms_restore_one_full_physical_checkpoint": True,
        },
        "engineering_gate_only": {
            "exact_f35_replay": True,
            "identical_predecision_prefixes": True,
            "all_outcomes_retained": True,
            "favorable_recovery_outcome_required": False,
        },
        "selection_used_model_predictions_or_outcomes": False,
        "p4_postrun_audit_sha256": sha256(P4_AUDIT),
    }
    protocol_path = OUTPUT / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    audit = {
        "passed": True,
        "scene_id": SCENE,
        "source_counterfactual_group_ids": [
            row["source_counterfactual_group_id"] for row in schedule_rows
        ],
        "operators": [row["operator"] for row in schedule_rows],
        "schedule_sha256": sha256(schedule),
        "protocol_sha256": sha256(protocol_path),
        "collector_sha256": sha256(COLLECTOR),
        "development_only": True,
        "outcome_exposed": False,
        "counts_as_a0_a7_evidence": False,
        "selection_used_model_predictions_or_outcomes": False,
        "p4_postrun_audit_sha256": sha256(P4_AUDIT),
    }
    (OUTPUT / "design_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
