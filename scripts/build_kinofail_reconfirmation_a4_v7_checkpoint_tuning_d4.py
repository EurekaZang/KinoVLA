#!/usr/bin/env python3
"""Freeze an excluded O9 replay diagnostic after correcting F33 reset parity."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p3/schedule.jsonl"
D3 = ROOT / "outputs/kinofail_reconfirmation_a4_v7_checkpoint_tuning_d3/design_audit.json"
OUTPUT = ROOT / "outputs/kinofail_reconfirmation_a4_v7_checkpoint_tuning_d4"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_reconfirmation_a4_v7.py"
PRE = ROOT / "outputs/locomotion/policy.pt"
POLICY = ROOT / "outputs/locomotion/recovery_route_v1/policy.pt"
MANIFEST = ROOT / "outputs/locomotion/recovery_route_v1/training_manifest.json"
FREEZE = ROOT / "configs/locomotion/go2_recovery_route_ppo_v1_freeze.json"
SIM = ROOT / "configs/sim/go2_skeleton.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path in (SOURCE, D3, COLLECTOR, PRE, POLICY, MANIFEST, FREEZE, SIM):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = [
        row
        for line in SOURCE.read_text().splitlines()
        if line
        for row in [json.loads(line)]
        if row["operator"] == "O9_high_centering"
    ]
    if len(rows) != 1:
        raise RuntimeError(f"expected one O9 case, got {len(rows)}")
    row = rows[0]
    row["pairing"] = "outcome-exposed O9 effective-reset replay diagnostic D4"
    row["development_only"] = True
    row["counts_as_a0_a7_evidence"] = False
    row["f33_recorded_spawn_height_was_not_applied_by_source_collector_v2"] = True
    OUTPUT.mkdir(parents=True, exist_ok=False)
    schedule = OUTPUT / "schedule.jsonl"
    schedule.write_text(json.dumps(row, sort_keys=True) + "\n")
    protocol = {
        "schema_version": "kinofail.reconfirmation-a4-v7-checkpoint-tuning-d4.v1",
        "protocol_id": "kinofail-reconfirmation-a4-v7-checkpoint-tuning-d4",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "outcome_exposed_development_only",
        "development_only": True,
        "outcome_exposed": True,
        "counts_as_a0_a7_evidence": False,
        "schedule": str(schedule.relative_to(ROOT)),
        "schedule_sha256": sha256(schedule),
        "collector": str(COLLECTOR.relative_to(ROOT)),
        "collector_sha256": sha256(COLLECTOR),
        "counts": {"scenes": 1, "physical_cases": 1, "physical_episodes": 7},
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
        "source_effective_reset_contract": {
            "f33_recorded_spawn_base_height_m": 0.43,
            "f33_source_collector_v2_applied_recorded_spawn_height": False,
            "replay_uses_backend_default_spawn_height": True,
        },
        "d3_design_sha256": sha256(D3),
    }
    protocol_path = OUTPUT / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    audit = {
        "passed": True,
        "development_only": True,
        "outcome_exposed": True,
        "counts_as_a0_a7_evidence": False,
        "scene_id": row["scene_id"],
        "case_id": row["case_id"],
        "schedule_sha256": sha256(schedule),
        "protocol_sha256": sha256(protocol_path),
        "collector_sha256": sha256(COLLECTOR),
        "d3_design_sha256": sha256(D3),
    }
    (OUTPUT / "design_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
