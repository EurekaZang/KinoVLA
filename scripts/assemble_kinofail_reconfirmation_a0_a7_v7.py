#!/usr/bin/env python3
"""Assemble a fail-closed A0--A7 ledger for the expanded confirmation set.

This script deliberately does not reinterpret the legacy ``realistic_a0_a7_v6``
files.  Every claim-bearing entry is rebound to the F35 Scale corpus, the F30
T3 union, and the single F36 blind prediction/scoring pass.  A4 and A6 are
separate, preregistered physical/stratified analyses and must bind the same F36
artifacts.  Merely having a file on disk never makes an experiment ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_F36 = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f37"
DEFAULT_OUT = ROOT / "outputs/eval/realistic_a0_a7_v7_expanded"
F30_AUDIT = ROOT / "outputs/eval/unified_moe_v3_t3_replenishment_f30/combined_design/audit.json"
F33_AUDIT = ROOT / "outputs/kinofail_confirmatory_o9_final_f33/final_audit.json"
F34_AUDIT = ROOT / "outputs/kinofail_confirmatory_o9_postprocess_f34/final_audit.json"
F35_AUDIT = ROOT / "outputs/eval/unified_moe_v3_scale_direct_o9_f35/overlay_audit.json"
F36_SEAL = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f37/seal_manifest.json"
F0 = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f0_transitive_amendment1/freeze_manifest.json"
F1 = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f1/seal_manifest.json"
FRESHNESS = ROOT / "outputs/kinofail_reconfirmation_v2/schedules/freshness_audit.json"
A4_SEAL = ROOT / "outputs/freeze/kinofail_reconfirmation_a4_v8/seal_manifest.json"
A4_COLLECTION = ROOT / "outputs/kinofail_reconfirmation_a4_v8/run/final_audit.json"
A4_ANALYZER = ROOT / "scripts/analyze_kinofail_reconfirmation_a4_v8.py"
A6_ANALYZER = ROOT / "scripts/analyze_kinofail_reconfirmation_a6_v7.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def all_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return bool(value) and all(all_true(item) for item in value.values())
    if isinstance(value, list):
        return bool(value) and all(all_true(item) for item in value)
    return False


def holm_registry(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = report.get("secondary_baseline_holm", [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("method")): row
        for row in rows
        if isinstance(row, dict) and row.get("method")
    }


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f36-root", type=Path, default=DEFAULT_F36)
    parser.add_argument(
        "--a4-report",
        type=Path,
        default=DEFAULT_F36 / "a4_actual_action_policy_report_v8.json",
    )
    parser.add_argument(
        "--a6-report",
        type=Path,
        default=DEFAULT_F36 / "a6_operator_boundary_report.json",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    f36_root = args.f36_root.resolve()
    out = args.out.resolve()
    report_path = f36_root / "confirmatory_report.json"
    audit_path = f36_root / "finalization_audit.json"
    predictions_path = f36_root / "blind_predictions/blind_predictions.jsonl"
    truth_path = f36_root / "blind_bundle/truth_key.jsonl"
    evaluation_inputs_audit_path = f36_root / "evaluation_inputs/audit.json"
    required = (
        F30_AUDIT,
        F33_AUDIT,
        F34_AUDIT,
        F35_AUDIT,
        F36_SEAL,
        F0,
        F1,
        FRESHNESS,
        A4_SEAL,
        A4_COLLECTION,
        A4_ANALYZER,
        A6_ANALYZER,
        report_path,
        audit_path,
        predictions_path,
        truth_path,
        evaluation_inputs_audit_path,
        args.a4_report.resolve(),
        args.a6_report.resolve(),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("A0--A7 v7 inputs are incomplete: " + ", ".join(missing))
    if out.exists():
        raise FileExistsError(f"refusing to overwrite expanded A0--A7 ledger: {out}")

    f30 = read_json(F30_AUDIT)
    f33 = read_json(F33_AUDIT)
    f34 = read_json(F34_AUDIT)
    f35 = read_json(F35_AUDIT)
    seal = read_json(F36_SEAL)
    f0 = read_json(F0)
    f1 = read_json(F1)
    freshness = read_json(FRESHNESS)
    report = read_json(report_path)
    finalization = read_json(audit_path)
    a4_seal = read_json(A4_SEAL)
    a4 = read_json(args.a4_report.resolve())
    a6 = read_json(args.a6_report.resolve())
    holm = holm_registry(report)

    common_hashes = {
        "f30_t3_union_audit": sha256(F30_AUDIT),
        "f33_direct_o9_audit": sha256(F33_AUDIT),
        "f34_direct_o9_postprocess_audit": sha256(F34_AUDIT),
        "f35_scale_overlay_audit": sha256(F35_AUDIT),
        "f36_pre_prediction_seal": sha256(F36_SEAL),
        "f36_finalization_audit": sha256(audit_path),
        "f36_confirmatory_report": sha256(report_path),
        "f36_blind_predictions": sha256(predictions_path),
        "f36_truth_key": sha256(truth_path),
        "f36_evaluation_inputs_audit": sha256(evaluation_inputs_audit_path),
        "f0_pre_generation_freeze": sha256(F0),
        "f1_pre_collection_seal": sha256(F1),
        "freshness_audit": sha256(FRESHNESS),
        "a4_pre_prediction_seal": sha256(A4_SEAL),
        "a4_collection_audit": sha256(A4_COLLECTION),
        "a4_analyzer": sha256(A4_ANALYZER),
        "a6_analyzer": sha256(A6_ANALYZER),
    }
    source_hashes = report.get("source_sha256", {})
    hash_checks = {
        "report_binds_blind_predictions": source_hashes.get("blind_predictions")
        == common_hashes["f36_blind_predictions"],
        "report_binds_truth_key": source_hashes.get("truth_key")
        == common_hashes["f36_truth_key"],
        "f36_seal_binds_f35": seal.get("source_sha256", {}).get("f35_scale_audit")
        == common_hashes["f35_scale_overlay_audit"],
        "f36_seal_binds_f30": seal.get("source_sha256", {}).get("f30_design_audit")
        == common_hashes["f30_t3_union_audit"],
        "f35_binds_f33_direct_o9": f35.get("source_sha256", {}).get(
            "f33_final_audit"
        )
        == common_hashes["f33_direct_o9_audit"],
        "f35_binds_f34_direct_event_postprocess": f35.get("source_sha256", {}).get(
            "f34_final_audit"
        )
        == common_hashes["f34_direct_o9_postprocess_audit"],
        "f36_seal_binds_f33": seal.get("source_sha256", {}).get("f33_final_audit")
        == common_hashes["f33_direct_o9_audit"],
        "f36_seal_binds_f34": seal.get("source_sha256", {}).get("f34_final_audit")
        == common_hashes["f34_direct_o9_postprocess_audit"],
        "f36_seal_binds_a4_pre_prediction_contract": seal.get(
            "source_sha256", {}
        ).get("a4_pre_prediction_seal")
        == common_hashes["a4_pre_prediction_seal"],
        "a4_report_binds_seal_and_collection": a4.get("source_sha256", {}).get(
            "a4_seal"
        )
        == common_hashes["a4_pre_prediction_seal"]
        and a4.get("source_sha256", {}).get("a4_collection_audit")
        == common_hashes["a4_collection_audit"],
        "a4_analyzer_is_frozen_and_bound": a4_seal.get("analyzer_sha256")
        == common_hashes["a4_analyzer"]
        and a4.get("source_sha256", {}).get("analysis_script")
        == common_hashes["a4_analyzer"],
        "a6_analyzer_is_frozen_and_bound": seal.get("source_sha256", {})
        .get("execution_scripts", {})
        .get("scripts/analyze_kinofail_reconfirmation_a6_v7.py")
        == common_hashes["a6_analyzer"]
        and a6.get("source_sha256", {}).get("analysis_script")
        == common_hashes["a6_analyzer"],
        "f36_finalization_binds_pre_prediction_seal": finalization.get(
            "f36_direct_o9_scale_t3_confirmation", {}
        ).get("seal_sha256")
        == common_hashes["f36_pre_prediction_seal"],
        "f0_was_frozen_before_new_scene_generation": f0.get("status")
        == "frozen_before_new_scene_generation",
        "f1_was_sealed_before_collection_and_inference": f1.get("status")
        == "sealed_before_anomaly_collection_and_model_inference"
        and f1.get("confirmatory") is True,
        "f1_binds_f0_and_freshness": f1.get("input_sha256", {}).get(
            "f0_manifest"
        )
        == common_hashes["f0_pre_generation_freeze"]
        and f1.get("input_sha256", {}).get("freshness_audit")
        == common_hashes["freshness_audit"],
        "freshness_audit_passes_every_frozen_check": freshness.get("passed")
        is True
        and all_true(freshness.get("checks", {})),
        "f36_finalization_binds_f0_f1_and_freshness": finalization.get(
            "source_sha256", {}
        ).get("f0")
        == common_hashes["f0_pre_generation_freeze"]
        and finalization.get("source_sha256", {}).get("f1")
        == common_hashes["f1_pre_collection_seal"]
        and finalization.get("source_sha256", {}).get("freshness")
        == common_hashes["freshness_audit"],
        "f36_finalizer_declares_frozen_model_and_statistics": finalization.get(
            "f36_direct_o9_scale_t3_confirmation", {}
        ).get("frozen_model_features_threshold_and_statistics_unchanged")
        is True,
        "selection_was_model_blind": finalization.get(
            "f36_direct_o9_scale_t3_confirmation", {}
        ).get("selection_used_model_predictions_or_outcome_strength")
        is False,
    }

    attrition = report.get("attrition", {})
    a0_checks = {
        "all_common_hashes_match": all(hash_checks.values()),
        "f30_t3_union_passed": f30.get("passed") is True,
        "f33_direct_o9_passed": f33.get("passed") is True,
        "f34_direct_event_postprocess_passed": f34.get("passed") is True,
        "f35_scale_overlay_passed": f35.get("passed") is True,
        "legacy_o9_excluded": f35.get("legacy_o9_features_excluded") is True,
        "all_30_scenes_present": int(f35.get("counts", {}).get("scenes", 0)) == 30,
        "all_11_operators_present": len(f35.get("pairs_by_operator", {})) == 11,
        "scale_attrition_below_five_percent": float(
            f35.get("effective_scale_attrition_rate", 1.0)
        )
        < 0.05,
        "t3_complete_case_attrition_below_five_percent": int(
            f30.get("invalid_case_count", 1500)
        )
        / 1500
        < 0.05,
        "combined_t2_t3_conflict_attrition_below_five_percent": float(
            attrition.get("conflict", {}).get("rate", 1.0)
        )
        < 0.05,
        "overall_attrition_below_five_percent": float(
            attrition.get("overall", {}).get("rate", 1.0)
        )
        < 0.05,
        "single_blind_score_protocol_valid": report.get("confirmatory_protocol_valid")
        is True,
    }

    route = report.get("route_fidelity_gates", {})
    a1_checks = {
        "direct_high_centering_contact_certified": int(
            f34.get("counts", {}).get("direct_event_snapshot_pairs", 0)
        )
        >= 750,
        "t2_vision_decisive_cell_present": "T2_vision_decisive" in route,
        "t3_proprio_decisive_cell_present": "T3_proprio_decisive" in route,
        "both_conflict_directions_have_decision_cases": all(
            int(route.get(cell, {}).get("decision_critical_cases", 0)) > 0
            for cell in ("T2_vision_decisive", "T3_proprio_decisive")
        ),
        "all_scale_operator_counts_positive": len(f35.get("pairs_by_operator", {})) == 11
        and min((int(value) for value in f35.get("pairs_by_operator", {}).values()), default=0)
        > 0,
    }

    primary = report.get("primary_cross_battery_gates", {})
    uni_methods = ("fixed_vision", "fixed_proprio")
    a2_checks = {
        "confirmatory_cross_battery_sequence_passed": report.get(
            "passed_all_preregistered_gates"
        )
        is True,
        "learned_router_beats_both_unimodal_worst_battery": all(
            method in holm
            and float(holm[method].get("difference_ours_minus_baseline", -1.0)) > 0.0
            and holm[method].get("holm_reject") is True
            for method in uni_methods
        ),
        "battery_noninferiority_passed": primary.get(
            "gate_1_battery_noninferiority", {}
        ).get("sequential_pass")
        is True,
        "equal_battery_macro_superiority_passed": primary.get(
            "gate_3_equal_battery_macro_superiority", {}
        ).get("sequential_pass")
        is True,
    }

    a3_checks = {
        "vision_before_contact_route_fidelity_passed": route.get(
            "T2_vision_decisive", {}
        ).get("passed")
        is True,
        "proprio_after_hidden_contact_route_fidelity_passed": route.get(
            "T3_proprio_decisive", {}
        ).get("passed")
        is True,
        "both_route_lower_bounds_above_0_85": all(
            float(route.get(cell, {}).get("one_sided_97_5_percent_lower_bound", -1.0))
            > 0.85
            for cell in ("T2_vision_decisive", "T3_proprio_decisive")
        ),
        "worst_battery_superiority_passed": primary.get(
            "gate_2_worst_battery_superiority", {}
        ).get("sequential_pass")
        is True,
    }

    a4_checks = {
        "report_is_confirmatory_and_hash_bound": a4.get("passed") is True
        and a4.get("source_sha256", {}).get("f36_blind_predictions")
        == common_hashes["f36_blind_predictions"]
        and a4.get("source_sha256", {}).get("f36_truth_key")
        == common_hashes["f36_truth_key"],
        "all_25_unexposed_scene_clusters": int(a4.get("scene_cluster_count", 0)) == 25,
        "all_50_cases_and_350_actual_action_episodes": int(
            a4.get("paired_case_count", 0)
        )
        == 50
        and int(a4.get("physical_episode_count", 0)) == 350,
        "all_matched_prefixes_and_f35_replays_passed": a4.get(
            "acceptance", {}
        ).get("all_matched_prefix_and_replay_audits_pass")
        is True,
        "all_five_recovery_arms_have_distinct_physical_trajectories": a4.get(
            "acceptance", {}
        ).get("all_five_recovery_trajectories_distinct")
        is True,
        "correct_attribution_recovery_beats_mean_wrong_recovery": a4.get(
            "acceptance", {}
        ).get("correct_recovery_beats_mean_wrong_cost")
        is True,
        "correct_recovery_success_is_high": a4.get("acceptance", {}).get(
            "correct_recovery_success_lower_bound_above_0_65"
        )
        is True,
        "selective_policy_beats_always_safe": a4.get("acceptance", {}).get(
            "selective_beats_always_safe_cost"
        )
        is True,
        "selective_policy_beats_continue": a4.get("acceptance", {}).get(
            "selective_beats_continue_cost"
        )
        is True,
        "no_selective_safety_regression": a4.get("acceptance", {}).get(
            "selective_fall_noninferior_to_always_safe"
        )
        is True
        and a4.get("acceptance", {}).get(
            "selective_fall_noninferior_to_continue"
        )
        is True,
    }

    risk = report.get("selective_risk", {}).get(
        "equal_battery_75_percent_risk_difference_ours_minus_late", {}
    )
    a5_checks = {
        "risk_at_75_percent_coverage_lower": float(risk.get("point", 1.0)) < 0.0,
        "risk_reduction_ci_excludes_zero": len(risk.get("ci95", [])) == 2
        and float(risk["ci95"][1]) < 0.0,
        "direct_policy_endpoint_present": a4_checks[
            "selective_policy_beats_always_safe"
        ]
        and a4_checks["selective_policy_beats_continue"],
        "direct_policy_safety_noninferior": a4_checks[
            "no_selective_safety_regression"
        ],
    }

    a6_checks = {
        "report_is_hash_bound": a6.get("passed") is True
        and a6.get("source_sha256", {}).get("f36_blind_predictions")
        == common_hashes["f36_blind_predictions"]
        and a6.get("source_sha256", {}).get("f36_truth_key")
        == common_hashes["f36_truth_key"],
        "all_11_operators_reported": int(a6.get("operator_count", 0)) == 11,
        "o9_has_all_16_frozen_parameter_points": int(
            a6.get("o9_direct_parameter_point_count", 0)
        )
        == 16,
        "minimum_scene_clusters_per_operator_30": int(
            a6.get("minimum_scene_clusters_per_operator", 0)
        )
        >= 30,
        "operator_and_boundary_gates_are_anomaly_only": a6.get(
            "boundary_analysis_condition"
        )
        == "anomaly_only"
        and a6.get(
            "nominal_counterfactuals_excluded_from_operator_and_boundary_gates"
        )
        is True,
        "boundary_analysis_passed": a6.get("acceptance", {}).get(
            "all_preregistered_boundary_checks_passed"
        )
        is True,
    }

    required_ablation_baselines = (
        "fixed_vision",
        "fixed_proprio",
        "fixed_joint",
        "legacy_class_rule",
        "random_route",
    )
    a7_checks = {
        "all_frozen_ablation_baselines_scored": all(
            method in holm for method in required_ablation_baselines
        ),
        "ours_positive_against_every_ablation": all(
            float(holm.get(method, {}).get("difference_ours_minus_baseline", -1.0)) > 0.0
            for method in required_ablation_baselines
        ),
        "holm_significant_against_every_ablation": all(
            holm.get(method, {}).get("holm_reject") is True
            for method in required_ablation_baselines
        ),
        "late_fusion_primary_comparator_beaten": primary.get(
            "gate_3_equal_battery_macro_superiority", {}
        ).get("sequential_pass")
        is True,
    }

    checks_by_experiment = {
        "A0": a0_checks,
        "A1": a1_checks,
        "A2": a2_checks,
        "A3": a3_checks,
        "A4": a4_checks,
        "A5": a5_checks,
        "A6": a6_checks,
        "A7": a7_checks,
    }
    now = datetime.now(UTC).isoformat()
    out.mkdir(parents=True, exist_ok=False)
    summaries: dict[str, dict[str, Any]] = {}
    for experiment, checks in checks_by_experiment.items():
        passed = all(checks.values())
        summary = {
            "schema_version": "kinofail.expanded-a0-a7-v7-experiment.v1",
            "created_utc": now,
            "experiment": experiment,
            "status": "confirmatory_passed" if passed else "confirmatory_gate_failed",
            "passed": passed,
            "checks": checks,
            "common_evidence_sha256": common_hashes,
            "legacy_v6_result_reused": False,
            "a8_in_scope": False,
        }
        write_json(out / f"{experiment.lower()}_report.json", summary)
        summaries[experiment] = summary

    passed_ids = [name for name, value in summaries.items() if value["passed"]]
    failed_ids = [name for name, value in summaries.items() if not value["passed"]]
    all_passed = len(passed_ids) == 8
    readiness = {
        "schema_version": "kinofail.expanded-a0-a7-v7-readiness.v1",
        "created_utc": now,
        "status": "strong_publication_evidence_ready" if all_passed else "not_ready",
        "passed": all_passed,
        "n_passed": len(passed_ids),
        "n_required": 8,
        "passed_experiments": passed_ids,
        "failed_experiments": failed_ids,
        "substantive_pass_required": True,
        "artifact_presence_alone_is_never_sufficient": True,
        "common_hash_checks": hash_checks,
        "common_evidence": {
            "f30": artifact(F30_AUDIT),
            "f0": artifact(F0),
            "f1": artifact(F1),
            "freshness": artifact(FRESHNESS),
            "f33": artifact(F33_AUDIT),
            "f34": artifact(F34_AUDIT),
            "f35": artifact(F35_AUDIT),
            "f36_seal": artifact(F36_SEAL),
            "f36_report": artifact(report_path),
            "f36_predictions": artifact(predictions_path),
            "f36_truth": artifact(truth_path),
            "a4_seal": artifact(A4_SEAL),
            "a4_collection": artifact(A4_COLLECTION),
            "a4": artifact(args.a4_report.resolve()),
            "a6": artifact(args.a6_report.resolve()),
        },
        "experiments": {
            name: {
                "passed": value["passed"],
                "status": value["status"],
                "failed_checks": [
                    check for check, passed in value["checks"].items() if not passed
                ],
            }
            for name, value in summaries.items()
        },
        "claim_boundary": {
            "allowed_if_passed": "strong expanded-simulation confirmation",
            "sim2real_validated": False,
            "real_go2_anchor_required_for_sim2real_validated": True,
        },
        "a8_in_scope": False,
    }
    write_json(out / "readiness_audit.json", readiness)
    write_json(
        out / "evidence_binding.json",
        {
            "schema_version": "kinofail.expanded-a0-a7-v7-binding.v1",
            "created_utc": now,
            "passed": all(hash_checks.values()),
            "checks": hash_checks,
            "sha256": common_hashes,
        },
    )
    print(json.dumps(readiness, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
