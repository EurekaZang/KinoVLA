#!/usr/bin/env python3
"""Seal the expanded A4 action and analysis contract before F36 inference."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "outputs/kinofail_reconfirmation_a4_v7/design_audit.json"
SCHEDULE = ROOT / "outputs/kinofail_reconfirmation_a4_v7/schedule.jsonl"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_reconfirmation_a4_v7.py"
RUNNER = ROOT / "scripts/run_kinofail_reconfirmation_a4_v7.py"
ANALYZER = ROOT / "scripts/analyze_kinofail_reconfirmation_a4_v7.py"
PILOT_AUDIT = ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p3/postrun_audit.json"
P1_FAILURE_AUDIT = ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p1/failure_audit.json"
P2_FAILURE_AUDIT = ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p2/failure_audit.json"
ACTOR_POLICY = ROOT / "outputs/locomotion/recovery_route_v1/policy.pt"
PREDECISION_POLICY = ROOT / "outputs/locomotion/policy.pt"
ACTOR_MANIFEST = ROOT / "outputs/locomotion/recovery_route_v1/training_manifest.json"
ACTOR_FREEZE = ROOT / "configs/locomotion/go2_recovery_route_ppo_v1_freeze.json"
SIM_CONFIG = ROOT / "configs/sim/go2_skeleton.yaml"
OUTPUT = ROOT / "outputs/freeze/kinofail_reconfirmation_a4_v7/seal_manifest.json"


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
        DESIGN,
        SCHEDULE,
        COLLECTOR,
        RUNNER,
        ANALYZER,
        PILOT_AUDIT,
        P1_FAILURE_AUDIT,
        P2_FAILURE_AUDIT,
        PREDECISION_POLICY,
        ACTOR_POLICY,
        ACTOR_MANIFEST,
        ACTOR_FREEZE,
        SIM_CONFIG,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    design = json.loads(DESIGN.read_text())
    pilot = json.loads(PILOT_AUDIT.read_text())
    if (
        design.get("passed") is not True
        or design.get("model_prediction_or_outcome_read") is not False
        or design.get("selection_used_model_predictions_or_outcomes") is not False
        or design.get("schedule_sha256") != sha256(SCHEDULE)
        or int(design.get("counts", {}).get("physical_cases", 0)) != 300
        or int(design.get("counts", {}).get("physical_episodes", 0)) != 2100
        or pilot.get("passed") is not True
        or pilot.get("development_only") is not True
        or pilot.get("counts_as_a0_a7_evidence") is not False
        or pilot.get("checks", {}).get("prior_failures_preserved") is not True
        or pilot.get("checks", {}).get("fresh_from_formal_p1_p2") is not True
        or pilot.get("checks", {}).get("shared_recovery_actor_hash_bound") is not True
        or pilot.get("checks", {}).get("frozen_source_replayed_in_every_arm")
        is not True
        or pilot.get("checks", {}).get("all_recovery_programs_are_distinct")
        is not True
    ):
        raise RuntimeError("invalid A4-v7 model-blind design")
    seal = {
        "schema_version": "kinofail.reconfirmation-a4-v7-seal.v1",
        "protocol_id": "kinofail-reconfirmation-a4-v7-actual-action-20260801",
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
        "excluded_pilot_audit_sha256": sha256(PILOT_AUDIT),
        "excluded_failed_p1_audit_sha256": sha256(P1_FAILURE_AUDIT),
        "excluded_failed_p2_audit_sha256": sha256(P2_FAILURE_AUDIT),
        "counts": design["counts"],
        "action_arms": design["actions"]["common"],
        "prediction_to_action": {
            "adhesion": "recover_as_O4_tether",
            "compliant_terrain": "recover_as_O2_compliance",
            "invisible_obstacle": "recover_as_O8_invisible_collider",
            "high_centering": "recover_as_O9_high_centering",
            "overload": "recover_as_O5_payload",
            "all_other_classes": "always_safe_halt",
        },
        "frozen_policy": {
            "method": "learned_router",
            "sample": "anomaly condition, primary appearance view",
            "checkpoint_aggregation": "majority vote over seed0--seed4; SHA-256 class tie break",
            "confidence_tuning": False,
            "always_safe_policy": "always_safe_halt",
            "oracle_true_label_used_to_choose_action": False,
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
        "predecision_actor": {
            "policy": str(PREDECISION_POLICY.relative_to(ROOT)),
            "policy_sha256": sha256(PREDECISION_POLICY),
            "purpose": "replay_frozen_f35_predecision_state",
            "switch_boundary": "after the frozen F35 observation is certified",
        },
        "decision_contract": {
            "same_source_physics_seed_and_nuisance_as_f36_snapshot": True,
            "decision_time_source": "F35 anomaly-primary snapshot decision_time_s",
            "maximum_decision_time_skew_s": 0.021,
            "f35_policy_observation_replayed_within_frozen_tolerance": True,
            "bitwise_hash_equality_across_physics_resets_required": False,
            "proprio_window": "same 21 timestamps x 19 raw channels",
            "proprio_summary_dimensions": 80,
            "proprio_absolute_tolerance": 0.005,
            "proprio_relative_tolerance": 0.005,
            "branch_strictly_after_operator_specific_physical_onset": True,
            "maximum_absolute_predecision_difference": 0.001,
            "failure_policy": "exclude failing matched case; never recollect based on outcome",
        },
        "terminal_cost": {
            "formula": "100*fall + 25*not_recovered_past_hazard + 10*max(0,max_tilt-0.55) + remaining_to_local_checkpoint_m + 0.02*postdecision_time_s + 5*safety_exposure_auc",
            "safe_halt_has_task_delay_cost": True,
            "weights_frozen_before_outcomes": True,
            "safety_exposure_auc": {
                "tilt": "integral max(0,(abs(tilt_rad)-0.35)/0.20)",
                "slip": "integral max(0,(slip_ratio-0.25)/0.75)",
                "effort": "integral max(0,(effort_ratio-0.80)/0.20)",
                "support": "integral max(0,(0.75-support_ratio)/0.75)",
                "integration": "postdecision control-time seconds",
                "weight": 5.0,
            },
        },
        "local_recovery_endpoint": {
            "progress_beyond_decision_m": 0.75,
            "maximum_absolute_route_lateral_offset_m": 0.25,
            "terminates_on_first_success": True,
            "purpose": "isolate recovery from unrelated downstream scene geometry",
        },
        "primary_endpoints": {
            "correct_recovery_minus_continue_cost": "scene-cluster paired bootstrap CI95 upper < 0",
            "correct_recovery_minus_mean_wrong_recovery_cost": "scene-cluster paired bootstrap CI95 upper < 0",
            "correct_recovery_success": "one-sided scene-cluster 97.5% lower bound > 0.75",
            "selective_policy_minus_always_safe_cost": "scene-cluster paired bootstrap CI95 upper < 0",
            "selective_fall_rate_minus_always_safe": "one-sided 97.5% upper <= 0.02",
            "selective_policy_minus_continue_cost": "scene-cluster paired bootstrap CI95 upper < 0",
            "selective_fall_rate_minus_continue": "one-sided 97.5% upper <= 0.02",
        },
        "statistics": {
            "physical_unit": "matched scene/operator/severity case",
            "upper_cluster": "scene instance",
            "bootstrap_draws": 20000,
            "bootstrap_seed": 2027014401,
            "fall_noninferiority_margin": 0.02,
            "correct_recovery_success_lower_bound_floor": 0.75,
            "five_checkpoints_not_independent_units": True,
            "co_primary_multiplicity": (
                "intersection-union test: all seven preregistered co-primary gates "
                "must pass; no claim is made when any component fails"
            ),
            "operator_level_multiplicity": (
                "five one-sided correct-vs-mean-wrong recovery cost checks use "
                "Bonferroni alpha=0.05/5=0.01"
            ),
        },
        "outcome_policy": {
            "retain_all_unfavorable_actions": True,
            "result_dependent_retry_or_selection": False,
            "maximum_horizon_steps": 900,
            "control_hz": 50,
            "time_remains_penalized_in_terminal_cost": True,
            "score_once": True,
        },
        "source_sha256": {
            "design": sha256(DESIGN),
            "schedule": sha256(SCHEDULE),
            "collector": sha256(COLLECTOR),
            "runner": sha256(RUNNER),
            "analyzer": sha256(ANALYZER),
            "excluded_pilot_audit": sha256(PILOT_AUDIT),
            "excluded_failed_p1_audit": sha256(P1_FAILURE_AUDIT),
            "excluded_failed_p2_audit": sha256(P2_FAILURE_AUDIT),
            "low_level_actor_policy": sha256(ACTOR_POLICY),
            "predecision_actor_policy": sha256(PREDECISION_POLICY),
            "low_level_actor_training_manifest": sha256(ACTOR_MANIFEST),
            "low_level_actor_training_freeze": sha256(ACTOR_FREEZE),
            "sim_config": sha256(SIM_CONFIG),
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
