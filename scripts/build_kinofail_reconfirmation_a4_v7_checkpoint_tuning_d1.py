#!/usr/bin/env python3
"""Build an outcome-exposed development set for checkpoint-branching recovery tuning."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p3/schedule.jsonl"
P3_FAILURE = ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p3/failure_audit.json"
OUTPUT = ROOT / "outputs/kinofail_reconfirmation_a4_v7_checkpoint_tuning_d1"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_reconfirmation_a4_v7.py"
PREDECISION_POLICY = ROOT / "outputs/locomotion/policy.pt"
ACTOR_POLICY = ROOT / "outputs/locomotion/recovery_route_v1/policy.pt"
ACTOR_MANIFEST = ROOT / "outputs/locomotion/recovery_route_v1/training_manifest.json"
ACTOR_FREEZE = ROOT / "configs/locomotion/go2_recovery_route_ppo_v1_freeze.json"
SIM_CONFIG = ROOT / "configs/sim/go2_skeleton.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path in (
        SOURCE,
        P3_FAILURE,
        COLLECTOR,
        PREDECISION_POLICY,
        ACTOR_POLICY,
        ACTOR_MANIFEST,
        ACTOR_FREEZE,
        SIM_CONFIG,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    failure = json.loads(P3_FAILURE.read_text())
    if failure.get("passed") is not True or failure.get("pilot_passed") is not False:
        raise RuntimeError("P3 failure audit is not valid")
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line]
    OUTPUT.mkdir(parents=True, exist_ok=False)
    schedule = OUTPUT / "schedule.jsonl"
    for row in rows:
        row["pairing"] = "outcome-exposed checkpoint-branching recovery tuning D1"
        row["development_only"] = True
        row["counts_as_a0_a7_evidence"] = False
    schedule.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    protocol = {
        "schema_version": "kinofail.reconfirmation-a4-v7-checkpoint-tuning-d1.v1",
        "protocol_id": "kinofail-reconfirmation-a4-v7-checkpoint-tuning-d1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "outcome_exposed_development_only",
        "development_only": True,
        "outcome_exposed": True,
        "counts_as_a0_a7_evidence": False,
        "schedule": str(schedule.relative_to(ROOT)),
        "schedule_sha256": sha256(schedule),
        "collector": str(COLLECTOR.relative_to(ROOT)),
        "collector_sha256": sha256(COLLECTOR),
        "counts": {"scenes": 1, "physical_cases": 5, "physical_episodes": 35},
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
            "shared_prefix_executions_per_case": 1,
            "full_physical_state_checkpoint_at_decision": True,
            "all_action_arms_restore_same_checkpoint": True,
        },
        "p3_failure_audit_sha256": sha256(P3_FAILURE),
    }
    protocol_path = OUTPUT / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    audit = {
        "passed": True,
        "development_only": True,
        "outcome_exposed": True,
        "counts_as_a0_a7_evidence": False,
        "scene_id": rows[0]["scene_id"],
        "schedule_sha256": sha256(schedule),
        "protocol_sha256": sha256(protocol_path),
        "collector_sha256": sha256(COLLECTOR),
        "p3_failure_audit_sha256": sha256(P3_FAILURE),
    }
    (OUTPUT / "design_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
