#!/usr/bin/env python3
"""Seal supported-recovery A4-v8 before F36 prediction and formal outcomes."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "outputs/kinofail_reconfirmation_a4_v8/design_audit.json"
SCHEDULE = ROOT / "outputs/kinofail_reconfirmation_a4_v8/schedule.jsonl"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_reconfirmation_a4_v7.py"
RUNNER = ROOT / "scripts/run_kinofail_reconfirmation_a4_v8.py"
ANALYZER = ROOT / "scripts/analyze_kinofail_reconfirmation_a4_v8.py"
PRE = ROOT / "outputs/locomotion/policy.pt"
ACTOR = ROOT / "outputs/locomotion/recovery_route_v1/policy.pt"
ACTOR_MANIFEST = ROOT / "outputs/locomotion/recovery_route_v1/training_manifest.json"
ACTOR_FREEZE = ROOT / "configs/locomotion/go2_recovery_route_ppo_v1_freeze.json"
SIM = ROOT / "configs/sim/go2_skeleton.yaml"
PILOT_POSTRUN = (
    ROOT / "outputs/kinofail_reconfirmation_a4_v8_pilot_p4/postrun_audit.json",
    ROOT / "outputs/kinofail_reconfirmation_a4_v8_pilot_p5/postrun_audit.json",
)
PILOT_FAILURES = tuple(
    ROOT / f"outputs/kinofail_reconfirmation_a4_v7_pilot_p{i}/failure_audit.json"
    for i in (1, 2, 3)
)
PILOT_DESIGNS = tuple(
    [ROOT / f"outputs/kinofail_reconfirmation_a4_v7_pilot_p{i}/design_audit.json" for i in (1, 2, 3)]
    + [ROOT / f"outputs/kinofail_reconfirmation_a4_v8_pilot_p{i}/design_audit.json" for i in (4, 5)]
)
OUTPUT = ROOT / "outputs/freeze/kinofail_reconfirmation_a4_v8/seal_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    required = (
        DESIGN, SCHEDULE, COLLECTOR, RUNNER, ANALYZER, PRE, ACTOR,
        ACTOR_MANIFEST, ACTOR_FREEZE, SIM, *PILOT_POSTRUN, *PILOT_FAILURES,
        *PILOT_DESIGNS,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    design = json.loads(DESIGN.read_text())
    pilots = [json.loads(path.read_text()) for path in PILOT_POSTRUN]
    failures = [json.loads(path.read_text()) for path in PILOT_FAILURES]
    if (
        design.get("passed") is not True
        or design.get("model_prediction_or_outcome_read") is not False
        or design.get("selection_used_model_predictions_or_outcomes") is not False
        or design.get("schedule_sha256") != sha256(SCHEDULE)
        or int(design.get("counts", {}).get("formal_scenes", 0)) != 25
        or int(design.get("counts", {}).get("physical_cases", 0)) != 50
        or int(design.get("counts", {}).get("physical_episodes", 0)) != 350
        or not all(
            row.get("passed") is True
            and row.get("development_only") is True
            and row.get("counts_as_a0_a7_evidence") is False
            for row in pilots
        )
        or not all(
            row.get("passed") is True
            and row.get("pilot_passed") is False
            and row.get("counts_as_a0_a7_evidence") is False
            for row in failures
        )
    ):
        raise RuntimeError("invalid A4-v8 model-blind design or pilot boundary")
    seal = {
        "schema_version": "kinofail.reconfirmation-a4-v8-seal.v1",
        "protocol_id": "kinofail-reconfirmation-a4-v8-supported-recovery-20260801",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_f36_prediction_and_confirmatory_a4_outcomes",
        "passed": True,
        "model_prediction_or_outcome_read": False,
        "schedule": str(SCHEDULE.relative_to(ROOT)),
        "schedule_sha256": sha256(SCHEDULE),
        "collector": str(COLLECTOR.relative_to(ROOT)),
        "collector_sha256": sha256(COLLECTOR),
        "runner": str(RUNNER.relative_to(ROOT)),
        "runner_sha256": sha256(RUNNER),
        "analyzer": str(ANALYZER.relative_to(ROOT)),
        "analyzer_sha256": sha256(ANALYZER),
        "design_audit_sha256": sha256(DESIGN),
        "counts": design["counts"],
        "supported_recovery_cells": design["supported_recovery_cells"],
        "unsupported_actor_boundaries": design["unsupported_actor_boundaries"],
        "excluded_pilot_scenes": design["excluded_pilot_scenes"],
        "action_arms": design["action_arms"],
        "prediction_to_action": {
            "adhesion": "recover_as_O4_tether",
            "overload": "recover_as_O5_payload",
            "compliant_terrain": "recover_as_O2_compliance",
            "invisible_obstacle": "recover_as_O8_invisible_collider",
            "high_centering": "recover_as_O9_high_centering",
            "all_other_classes": "always_safe_halt",
        },
        "frozen_policy": {
            "method": "learned_router",
            "sample": "anomaly condition, primary appearance view",
            "checkpoint_aggregation": "majority vote over seed0--seed4 with SHA-256 tie break",
            "confidence_tuning": False,
            "oracle_true_label_used_to_choose_action": False,
        },
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
        "decision_contract": {
            "same_source_seed_nuisance_and_f35_observation": True,
            "proprio_window": "same 21 timestamps x 19 raw channels",
            "proprio_summary_dimensions": 80,
            "proprio_absolute_tolerance": 0.005,
            "proprio_relative_tolerance": 0.005,
            "maximum_decision_time_skew_s": 0.021,
            "maximum_absolute_predecision_difference": 0.001,
            "all_seven_arms_restore_one_checkpoint": True,
            "failure_policy": "terminal case failure; no result-dependent retry",
        },
        "execution": {
            "fresh_isaac_process_per_case": True,
            "maximum_concurrent_isaac_processes": 1,
            "physical_cases": 50,
            "physical_episodes": 350,
            "maximum_horizon_steps": 900,
            "control_hz": 50,
        },
        "primary_endpoints": {
            "correct_recovery_success": "one-sided scene-cluster 97.5% lower bound > 0.65",
            "correct_recovery_minus_mean_wrong_cost": "scene-cluster paired CI95 upper < 0",
            "selective_policy_minus_always_safe_cost": "scene-cluster paired CI95 upper < 0",
            "selective_fall_minus_always_safe": "one-sided 97.5% upper <= 0.05",
            "selective_policy_minus_continue_cost": "scene-cluster paired CI95 upper < 0",
            "selective_fall_minus_continue": "one-sided 97.5% upper <= 0.05",
            "attribution_accuracy": ">= 0.90 on the 50 action cases",
        },
        "statistics": {
            "physical_unit": "matched scene/operator case",
            "upper_cluster": "scene instance",
            "bootstrap_draws": 20000,
            "bootstrap_seed": 2027014801,
            "fall_noninferiority_margin": 0.05,
            "correct_recovery_success_lower_bound_floor": 0.65,
            "co_primary_multiplicity": "intersection-union: every global gate must pass",
            "operator_level_multiplicity": "two one-sided cost checks use Bonferroni alpha=0.025",
        },
        "outcome_policy": {
            "retain_all_unfavorable_actions": True,
            "result_dependent_retry_or_selection": False,
            "score_once": True,
        },
        "claim_boundary": (
            "Formal autonomous-recovery evidence is restricted to O4 adhesion and O5 overload. "
            "O2, O8, and O9 remain in the frozen attribution battery and are reported as actor-capability boundaries."
        ),
        "source_sha256": {
            "design": sha256(DESIGN),
            "schedule": sha256(SCHEDULE),
            "collector": sha256(COLLECTOR),
            "runner": sha256(RUNNER),
            "analyzer": sha256(ANALYZER),
            "pilot_designs": {str(path.relative_to(ROOT)): sha256(path) for path in PILOT_DESIGNS},
            "pilot_postrun": {str(path.relative_to(ROOT)): sha256(path) for path in PILOT_POSTRUN},
            "pilot_failures": {str(path.relative_to(ROOT)): sha256(path) for path in PILOT_FAILURES},
            "predecision_actor": sha256(PRE),
            "low_level_actor": sha256(ACTOR),
            "actor_manifest": sha256(ACTOR_MANIFEST),
            "actor_freeze": sha256(ACTOR_FREEZE),
            "sim_config": sha256(SIM),
            "sealer": sha256(Path(__file__).resolve()),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    OUTPUT.with_name("seal_manifest.sha256").write_text(f"{sha256(OUTPUT)}  {OUTPUT.name}\n")
    print(json.dumps(seal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
