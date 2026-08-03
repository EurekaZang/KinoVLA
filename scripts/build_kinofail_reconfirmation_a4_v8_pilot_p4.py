#!/usr/bin/env python3
"""Freeze a fresh-scene, permanently excluded A4-v8 engineering pilot P4."""

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


OUTPUT = ROOT / "outputs/kinofail_reconfirmation_a4_v8_pilot_p4"
SCENE = "confirm_v2_production_scene_03"
CELLS = (
    ("O2_compliance", "hard", "deep_compliance_braced_safe_stop"),
    ("O4_tether", "hard", "adhesion_high_step_peel"),
    ("O5_payload", "moderate", "overload_braced_safe_stop"),
)
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_reconfirmation_a4_v7.py"
D7_AUDIT = ROOT / "outputs/kinofail_reconfirmation_a4_v7_checkpoint_tuning_d7/design_audit.json"
PREDECISION_POLICY = ROOT / "outputs/locomotion/policy.pt"
ACTOR_POLICY = ROOT / "outputs/locomotion/recovery_route_v1/policy.pt"
ACTOR_MANIFEST = ROOT / "outputs/locomotion/recovery_route_v1/training_manifest.json"
ACTOR_FREEZE = ROOT / "configs/locomotion/go2_recovery_route_ppo_v1_freeze.json"
SIM_CONFIG = ROOT / "configs/sim/go2_skeleton.yaml"


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path in (
        F25_AUDIT,
        COLLECTOR,
        D7_AUDIT,
        PREDECISION_POLICY,
        ACTOR_POLICY,
        ACTOR_MANIFEST,
        ACTOR_FREEZE,
        SIM_CONFIG,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    valid_groups, decisions = load_f35_primary_decisions()
    anomaly = {
        str(row["counterfactual_group_id"]): row
        for row in jsonl(SCALE)
        if row["condition"] == "anomaly"
    }
    f25_audit = json.loads(F25_AUDIT.read_text())
    replacement_by_original = {
        str(row["original_counterfactual_group_id"]): str(
            row["replacement_counterfactual_group_id"]
        )
        for row in f25_audit["accepted"]
        if str(row["target_operator"]) != "O9_high_centering"
    }
    f25_anomaly = {
        str(row["counterfactual_group_id"]): row
        for path in sorted(F25_SCHEDULE_ROOT.glob("*/scale/schedule.jsonl"))
        for row in jsonl(path)
        if row["condition"] == "anomaly"
    }

    rows = []
    for operator, severity, endpoint in CELLS:
        options = []
        for group_id in valid_groups:
            original = anomaly[group_id]
            if (
                original["scene_id"] != SCENE
                or original["target_operator"] != operator
                or original["severity_id"] != severity
            ):
                continue
            values = decisions[group_id]["invariant_proprio_80"]
            last_height = float(values[60])
            last_tilt = abs(float(values[61]))
            if last_height >= 0.25 and last_tilt < 0.55:
                options.append((group_id, last_height, last_tilt))
        if not options:
            raise RuntimeError(f"no P4 eligible source: {SCENE}/{operator}/{severity}")
        group_id, last_height, last_tilt = min(
            options,
            key=lambda row: stable_key(
                "a4-v8-pilot-p4", SCENE, operator, severity, row[0]
            ),
        )
        original = anomaly[group_id]
        replacement_id = replacement_by_original.get(group_id)
        source = f25_anomaly[replacement_id] if replacement_id is not None else original
        nuisance = source["physical_nuisance"]
        rows.append(
            {
                "schema_version": "kinofail.reconfirmation-a4-v8-schedule.v1",
                "case_id": f"a4v8pilotp4__{SCENE}__{operator}__{severity}",
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
                "pairing": "fresh-scene model-blind engineering pilot P4",
                "source_observation_replay_required": True,
                "selection_used_model_predictions_or_outcomes": False,
                "decision_state_eligibility": {
                    "last_base_height_m": last_height,
                    "last_absolute_tilt_rad": last_tilt,
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
    schedule.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    protocol = {
        "schema_version": "kinofail.reconfirmation-a4-v8-pilot-p4.v1",
        "protocol_id": "kinofail-reconfirmation-a4-v8-excluded-pilot-p4",
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
        "builder": str(Path(__file__).resolve().relative_to(ROOT)),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "counts": {"scenes": 1, "physical_cases": 3, "physical_episodes": 21},
        "predecision_actor": {
            "policy": str(PREDECISION_POLICY.relative_to(ROOT)),
            "policy_sha256": sha256(PREDECISION_POLICY),
            "purpose": "replay_frozen_f35_predecision_state",
        },
        "low_level_actor": {
            "policy": str(ACTOR_POLICY.relative_to(ROOT)),
            "policy_sha256": sha256(ACTOR_POLICY),
            "training_manifest": str(ACTOR_MANIFEST.relative_to(ROOT)),
            "training_manifest_sha256": sha256(ACTOR_MANIFEST),
            "training_freeze": str(ACTOR_FREEZE.relative_to(ROOT)),
            "training_freeze_sha256": sha256(ACTOR_FREEZE),
            "sim_config": str(SIM_CONFIG.relative_to(ROOT)),
            "sim_config_sha256": sha256(SIM_CONFIG),
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
        "d7_audit_sha256": sha256(D7_AUDIT),
    }
    protocol_path = OUTPUT / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    audit = {
        "passed": True,
        "scene_id": SCENE,
        "source_counterfactual_group_ids": [
            row["source_counterfactual_group_id"] for row in rows
        ],
        "operators": [row["operator"] for row in rows],
        "schedule_sha256": sha256(schedule),
        "protocol_sha256": sha256(protocol_path),
        "collector_sha256": sha256(COLLECTOR),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "development_only": True,
        "outcome_exposed": False,
        "counts_as_a0_a7_evidence": False,
        "selection_used_model_predictions_or_outcomes": False,
    }
    (OUTPUT / "design_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({**audit, "protocol": str(protocol_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
