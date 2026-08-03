#!/usr/bin/env python3
"""Build a fresh, permanently excluded A4-v7 integration pilot P2.

P2 uses a different scene and source groups from both the failed P1 pilot and
the untouched 300-case formal schedule.  It is an integration gate only and
can never contribute an A0--A7 result.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.build_kinofail_reconfirmation_a4_v7_schedule import (
    ACTIONS,
    F25_AUDIT,
    F25_SCHEDULE_ROOT,
    F33,
    OPERATORS,
    OUTPUT as FINAL_DESIGN,
    RECOVERY,
    ROOT,
    SCALE,
    jsonl,
    load_f35_primary_decisions,
    sha256,
    stable_key,
)


OUTPUT = ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p2"
P1_DESIGN = ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p1/design_audit.json"
P1_FAILURE_AUDIT = (
    ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p1/failure_audit.json"
)
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_reconfirmation_a4_v7.py"
ACTOR_POLICY = ROOT / "outputs/locomotion/recovery_route_v1/policy.pt"
ACTOR_MANIFEST = ROOT / "outputs/locomotion/recovery_route_v1/training_manifest.json"
ACTOR_FREEZE = ROOT / "configs/locomotion/go2_recovery_route_ppo_v1_freeze.json"
SIM_CONFIG = ROOT / "configs/sim/go2_skeleton.yaml"


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path in (
        P1_DESIGN,
        P1_FAILURE_AUDIT,
        ACTOR_POLICY,
        ACTOR_MANIFEST,
        ACTOR_FREEZE,
        SIM_CONFIG,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    p1 = json.loads(P1_DESIGN.read_text())
    p1_failure = json.loads(P1_FAILURE_AUDIT.read_text())
    if p1_failure.get("passed") is not True or p1_failure.get("pilot_passed") is not False:
        raise RuntimeError("P1 failure was not preserved before constructing P2")
    final_schedule = FINAL_DESIGN / "schedule.jsonl"
    excluded_ids = {
        str(row["source_counterfactual_group_id"]) for row in jsonl(final_schedule)
    }
    excluded_ids.update(map(str, p1["source_counterfactual_group_ids"]))
    excluded_scenes = {str(p1["scene_id"])}
    valid_groups, decision_meta = load_f35_primary_decisions()
    anomaly = {
        str(row["counterfactual_group_id"]): row
        for row in jsonl(SCALE)
        if row["condition"] == "anomaly"
    }
    f25_audit = json.loads(F25_AUDIT.read_text())
    f25_replacement_by_original = {
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
    f33_by_source = {
        str(row["o9_semantic_recollection"]["source_counterfactual_group_id"]): row
        for row in jsonl(F33)
        if row["condition"] == "anomaly"
    }
    by_cell: dict[tuple[str, str], list[str]] = {}
    for group_id in valid_groups - excluded_ids:
        original = anomaly[group_id]
        key = (str(original["scene_id"]), str(original["target_operator"]))
        if original["target_operator"] in OPERATORS and original["severity_id"] == "moderate":
            by_cell.setdefault(key, []).append(group_id)
    eligible_scenes = sorted(
        scene
        for scene in {key[0] for key in by_cell} - excluded_scenes
        if all(by_cell.get((scene, operator)) for operator in OPERATORS)
    )
    if not eligible_scenes:
        raise RuntimeError("no fresh scene has excluded moderate P2 cases for all operators")
    scene = eligible_scenes[0]
    rows = []
    for operator in OPERATORS:
        options = by_cell[(scene, operator)]
        group_id = min(
            options,
            key=lambda value: stable_key("a4-v7-pilot-p2", scene, operator, value),
        )
        original = anomaly[group_id]
        replacement_id = f25_replacement_by_original.get(group_id)
        if operator == "O9_high_centering":
            source = f33_by_source[group_id]
        elif replacement_id is not None:
            source = f25_anomaly[replacement_id]
        else:
            source = original
        rows.append(
            {
                "schema_version": "kinofail.reconfirmation-a4-v7-schedule.v1",
                "case_id": f"a4v7pilotp2__{scene}__{operator}__moderate",
                "scene_id": scene,
                "domain": str(original["domain"]),
                "operator": operator,
                "severity_id": "moderate",
                "source_counterfactual_group_id": group_id,
                "source_slot_episode_id": str(original["episode_id"]),
                "source_physical_episode_id": str(source["episode_id"]),
                "source_record": source,
                "source_is_f33_direct_o9": operator == "O9_high_centering",
                "source_is_f25_replacement": replacement_id is not None,
                "reset_seed": int(source["physical_nuisance"]["physics_seed"]),
                "source_physical_nuisance": source["physical_nuisance"],
                "source_f35_decision": decision_meta[group_id],
                "actions": list(ACTIONS),
                "operator_recovery": RECOVERY[operator],
                "pairing": "excluded fresh-scene integration pilot P2",
                "source_observation_replay_required": True,
                "selection_used_model_predictions_or_outcomes": False,
            }
        )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    schedule = OUTPUT / "schedule.jsonl"
    schedule.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    actor = {
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
    }
    protocol = {
        "schema_version": "kinofail.reconfirmation-a4-v7-pilot-p2.v1",
        "protocol_id": "kinofail-reconfirmation-a4-v7-excluded-pilot-p2",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "development_only_permanently_excluded",
        "development_only": True,
        "counts_as_a0_a7_evidence": False,
        "schedule": str(schedule.relative_to(ROOT)),
        "schedule_sha256": sha256(schedule),
        "collector": str(COLLECTOR.relative_to(ROOT)),
        "collector_sha256": sha256(COLLECTOR),
        "builder": str(Path(__file__).resolve().relative_to(ROOT)),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "counts": {"scenes": 1, "physical_cases": 5, "physical_episodes": 35},
        "low_level_actor": actor,
        "source_observation_replay": {
            "required": True,
            "absolute_tolerance": 0.005,
            "relative_tolerance": 0.005,
            "bitwise_hash_equality_across_resets_required": False,
        },
        "local_recovery_endpoint": {
            "progress_beyond_frozen_decision_state_m": 0.75,
            "maximum_absolute_route_lateral_offset_m": 0.25,
        },
        "excluded_from_final_source_group_ids": [
            row["source_counterfactual_group_id"] for row in rows
        ],
        "selection_used_model_predictions_or_outcomes": False,
        "p1_failure_audit_sha256": sha256(P1_FAILURE_AUDIT),
    }
    protocol_path = OUTPUT / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    audit = {
        "passed": True,
        "scene_id": scene,
        "source_counterfactual_group_ids": [
            row["source_counterfactual_group_id"] for row in rows
        ],
        "operators": list(OPERATORS),
        "schedule_sha256": sha256(schedule),
        "protocol_sha256": sha256(protocol_path),
        "collector_sha256": sha256(COLLECTOR),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "final_schedule_sha256": sha256(final_schedule),
        "p1_design_sha256": sha256(P1_DESIGN),
        "p1_failure_audit_sha256": sha256(P1_FAILURE_AUDIT),
        "low_level_actor": actor,
        "development_only": True,
        "counts_as_a0_a7_evidence": False,
        "disjoint_from_formal_and_p1": True,
    }
    (OUTPUT / "design_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({**audit, "protocol": str(protocol_path), "schedule": str(schedule)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
