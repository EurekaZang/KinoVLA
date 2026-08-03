#!/usr/bin/env python3
"""Assemble the fail-closed F42/A4/A6 expanded A0--A7 ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
F42 = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f42"
F42_SEAL = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f42/seal_manifest.json"
F38_RAW = ROOT / "outputs/kinofail_t3_runin_f38_formal_s1/final_audit.json"
F38_TASK = ROOT / "outputs/kinofail_t3_runin_f38_s1_task_aligned_audit/audit.json"
F38_S2 = ROOT / "outputs/eval/unified_moe_v3_t3_runin_f38_s2/combined_design/audit.json"
F33 = ROOT / "outputs/kinofail_confirmatory_o9_final_f33/final_audit.json"
F34 = ROOT / "outputs/kinofail_confirmatory_o9_postprocess_f34/final_audit.json"
F35 = ROOT / "outputs/eval/unified_moe_v3_scale_direct_o9_f35/overlay_audit.json"
F0 = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f0_transitive_amendment1/freeze_manifest.json"
F1 = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f1/seal_manifest.json"
FRESHNESS = ROOT / "outputs/kinofail_reconfirmation_v2/schedules/freshness_audit.json"
F39_FAILURE = ROOT / "outputs/kinofail_reconfirmation_f39_failure_audit/audit.json"
F40_FAILURE = ROOT / "outputs/kinofail_reconfirmation_f40_failure_audit/audit.json"
F41_FAILURE = ROOT / "outputs/kinofail_reconfirmation_f41_failure_audit/audit.json"
A4_SEAL = ROOT / "outputs/freeze/kinofail_reconfirmation_a4_v8/seal_manifest.json"
A4_COLLECTION = ROOT / "outputs/kinofail_reconfirmation_a4_v8/run/final_audit.json"
A4_ANALYZER = ROOT / "scripts/analyze_kinofail_reconfirmation_a4_v8.py"
A6_ANALYZER = ROOT / "scripts/analyze_kinofail_reconfirmation_a6_v7.py"
DEFAULT_OUT = ROOT / "outputs/eval/realistic_a0_a7_v8_expanded_f42"


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
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def holm(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["method"]): row
        for row in report.get("secondary_baseline_holm", [])
        if isinstance(row, dict) and row.get("method")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f42-root", type=Path, default=F42)
    parser.add_argument(
        "--a4-report", type=Path, default=F42 / "a4_actual_action_policy_report_v8.json"
    )
    parser.add_argument(
        "--a6-report", type=Path, default=F42 / "a6_operator_boundary_report.json"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    root = args.f42_root.resolve()
    paths = {
        "report": root / "confirmatory_report.json",
        "finalization": root / "finalization_audit.json",
        "protocol": root / "scoring_protocol.json",
        "predictions": root / "blind_predictions/blind_predictions.jsonl",
        "prediction_manifest": root / "blind_predictions/prediction_manifest.json",
        "truth": root / "blind_bundle/truth_key.jsonl",
        "bundle": root / "blind_bundle/bundle_manifest.json",
        "inputs": root / "evaluation_inputs/audit.json",
        "a4": args.a4_report.resolve(),
        "a6": args.a6_report.resolve(),
    }
    fixed = (
        F42_SEAL,
        F38_RAW,
        F38_TASK,
        F38_S2,
        F33,
        F34,
        F35,
        F0,
        F1,
        FRESHNESS,
        F39_FAILURE,
        F40_FAILURE,
        F41_FAILURE,
        A4_SEAL,
        A4_COLLECTION,
        A4_ANALYZER,
        A6_ANALYZER,
    )
    missing = [str(path) for path in (*fixed, *paths.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError("A0--A7 v8 inputs incomplete: " + ", ".join(missing))
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)

    report = read_json(paths["report"])
    finalization = read_json(paths["finalization"])
    seal = read_json(F42_SEAL)
    raw = read_json(F38_RAW)
    task = read_json(F38_TASK)
    t3 = read_json(F38_S2)
    f33 = read_json(F33)
    f34 = read_json(F34)
    f35 = read_json(F35)
    f0 = read_json(F0)
    f1 = read_json(F1)
    freshness = read_json(FRESHNESS)
    a4_seal = read_json(A4_SEAL)
    a4 = read_json(paths["a4"])
    a6 = read_json(paths["a6"])
    methods = holm(report)
    source = report.get("source_sha256", {})
    final_source = finalization.get("source_sha256", {})
    prediction_manifest = read_json(paths["prediction_manifest"])
    bundle = read_json(paths["bundle"])

    binding = {
        "report_binds_predictions": source.get("blind_predictions")
        == sha256(paths["predictions"]),
        "report_binds_truth": source.get("truth_key") == sha256(paths["truth"]),
        "report_binds_protocol": source.get("protocol") == sha256(paths["protocol"]),
        "finalization_binds_report": final_source.get("confirmatory_report")
        == sha256(paths["report"]),
        "finalization_binds_prediction_manifest": final_source.get("blind_predictions")
        == sha256(paths["prediction_manifest"]),
        "finalization_binds_bundle": final_source.get("blind_bundle")
        == sha256(paths["bundle"]),
        "f42_finalization_binds_seal": finalization.get(
            "f42_result_blind_operational_recovery", {}
        ).get("seal_sha256")
        == sha256(F42_SEAL),
        "f42_seal_binds_f41_failure": seal.get("source_sha256", {}).get(
            "f41_failure_audit"
        )
        == sha256(F41_FAILURE),
        "all_result_blind_failures_preserved": all(
            read_json(path).get("passed") is True
            for path in (F39_FAILURE, F40_FAILURE, F41_FAILURE)
        ),
        "f38_s2_binds_task_audit": t3.get("source_sha256", {}).get(
            "task_aligned_audit"
        )
        == sha256(F38_TASK),
        "f35_binds_direct_o9": f35.get("source_sha256", {}).get("f33_final_audit")
        == sha256(F33)
        and f35.get("source_sha256", {}).get("f34_final_audit") == sha256(F34),
        "f1_binds_f0_and_freshness": f1.get("input_sha256", {}).get("f0_manifest")
        == sha256(F0)
        and f1.get("input_sha256", {}).get("freshness_audit") == sha256(FRESHNESS),
        "prediction_manifest_binds_frozen_checkpoints": len(
            prediction_manifest.get("source_sha256", {}).get("checkpoints", {})
        )
        == 5,
        "bundle_binds_observable_only_archive": bundle.get("blind_archive_keys")
        == ["sample_ids", "visual", "proprio"],
        "a6_binds_f42": a6.get("source_sha256", {}).get("f36_blind_predictions")
        == sha256(paths["predictions"])
        and a6.get("source_sha256", {}).get("f36_truth_key") == sha256(paths["truth"])
        and a6.get("source_sha256", {}).get("analysis_script") == sha256(A6_ANALYZER),
        "a4_binds_f42_and_sealed_collection": a4.get("source_sha256", {}).get(
            "f36_blind_predictions"
        )
        == sha256(paths["predictions"])
        and a4.get("source_sha256", {}).get("f36_truth_key") == sha256(paths["truth"])
        and a4.get("source_sha256", {}).get("a4_seal") == sha256(A4_SEAL)
        and a4.get("source_sha256", {}).get("a4_collection_audit")
        == sha256(A4_COLLECTION)
        and a4.get("source_sha256", {}).get("analysis_script") == sha256(A4_ANALYZER),
    }

    attrition = report["attrition"]
    primary = report["primary_cross_battery_gates"]
    route = report["route_fidelity_gates"]
    risk = report["selective_risk"][
        "equal_battery_75_percent_risk_difference_ours_minus_late"
    ]
    required_baselines = (
        "fixed_vision",
        "fixed_proprio",
        "fixed_joint",
        "legacy_class_rule",
        "random_route",
    )
    checks = {
        "A0": {
            "all_evidence_hashes_match": all(binding.values()),
            "valid_one_shot_scoring_protocol": report.get("confirmatory_protocol_valid")
            is True,
            "scale_t3_conflict_and_overall_attrition_below_0_05": all(
                float(attrition[name]["rate"]) < 0.05
                for name in ("scale", "conflict", "overall")
            )
            and float(t3["case_attrition_rate"]) < 0.05,
            "all_30_scenes_and_11_operators": int(f35["counts"]["scenes"]) == 30
            and len(f35["pairs_by_operator"]) == 11,
            "raw_failures_and_task_alignment_disclosed": raw.get("passed") is False
            and task.get("counts_as_confirmatory_evidence") is False,
        },
        "A1": {
            "direct_o9_contact_evidence_present": int(
                f34.get("counts", {}).get("direct_event_snapshot_pairs", 0)
            )
            >= 750,
            "all_11_scale_operators_present": len(f35["pairs_by_operator"]) == 11,
            "both_conflict_directions_present": all(
                int(route[cell]["decision_critical_cases"]) > 0
                for cell in ("T2_vision_decisive", "T3_proprio_decisive")
            ),
        },
        "A2": {
            "all_ordered_primary_gates_passed": report.get(
                "passed_all_preregistered_gates"
            )
            is True,
            "battery_noninferiority_passed": primary["gate_1_battery_noninferiority"][
                "sequential_pass"
            ]
            is True,
            "worst_battery_superiority_passed": primary[
                "gate_2_worst_battery_superiority"
            ]["sequential_pass"]
            is True,
            "equal_battery_macro_superiority_passed": primary[
                "gate_3_equal_battery_macro_superiority"
            ]["sequential_pass"]
            is True,
        },
        "A3": {
            "t2_vision_route_fidelity_passed": route["T2_vision_decisive"]["passed"]
            is True,
            "t3_proprio_route_fidelity_passed": route["T3_proprio_decisive"]["passed"]
            is True,
            "both_route_lower_bounds_above_0_85": all(
                float(route[cell]["one_sided_97_5_percent_lower_bound"]) > 0.85
                for cell in ("T2_vision_decisive", "T3_proprio_decisive")
            ),
        },
        "A4": {
            "confirmatory_action_report_passed": a4.get("passed") is True,
            "all_25_scenes_50_cases_350_episodes": int(a4.get("scene_cluster_count", 0))
            == 25
            and int(a4.get("paired_case_count", 0)) == 50
            and int(a4.get("physical_episode_count", 0)) == 350,
            "all_physical_acceptance_gates_passed": bool(a4.get("acceptance"))
            and all(a4.get("acceptance", {}).values()),
        },
        "A5": {
            "selective_risk_point_lower_than_late": float(risk["point"]) < 0.0,
            "selective_risk_ci_excludes_zero": float(risk["ci95"][1]) < 0.0,
            "action_policy_beats_safe_and_continue": a4.get("acceptance", {}).get(
                "selective_beats_always_safe_cost"
            )
            is True
            and a4.get("acceptance", {}).get("selective_beats_continue_cost") is True,
            "action_policy_fall_noninferior": a4.get("acceptance", {}).get(
                "selective_fall_noninferior_to_always_safe"
            )
            is True
            and a4.get("acceptance", {}).get("selective_fall_noninferior_to_continue")
            is True,
        },
        "A6": {
            "all_11_operators_30_scenes_16_o9_points": int(a6.get("operator_count", 0))
            == 11
            and int(a6.get("minimum_scene_clusters_per_operator", 0)) >= 30
            and int(a6.get("o9_direct_parameter_point_count", 0)) == 16,
            "all_preregistered_boundary_checks_passed": a6.get("passed") is True
            and a6.get("acceptance", {}).get("all_preregistered_boundary_checks_passed")
            is True,
            "appearance_randomization_checks_passed": a6.get("acceptance", {}).get(
                "appearance_view_gap_upper_bound_at_most_0_05"
            )
            is True
            and a6.get("acceptance", {}).get(
                "simultaneous_worst_operator_appearance_gap_upper_bound_at_most_0_05"
            )
            is True,
            "operator_and_o9_accuracy_boundaries_passed": a6.get("acceptance", {}).get(
                "worst_operator_point_accuracy_at_least_0_85"
            )
            is True
            and a6.get("acceptance", {}).get("all_16_direct_o9_points_at_least_0_80")
            is True,
        },
        "A7": {
            "all_frozen_ablation_baselines_scored": all(
                method in methods for method in required_baselines
            ),
            "ours_positive_against_every_ablation": all(
                float(methods.get(method, {}).get("difference_ours_minus_baseline", -1))
                > 0
                for method in required_baselines
            ),
            "holm_significant_against_every_ablation": all(
                methods.get(method, {}).get("holm_reject") is True
                for method in required_baselines
            ),
            "late_average_primary_comparator_beaten": primary[
                "gate_3_equal_battery_macro_superiority"
            ]["sequential_pass"]
            is True,
        },
    }
    now = datetime.now(UTC).isoformat()
    out.mkdir(parents=True, exist_ok=False)
    experiment_results = {}
    for name, experiment_checks in checks.items():
        passed = all(experiment_checks.values())
        value = {
            "schema_version": "kinofail.expanded-a0-a7-v8-experiment.v1",
            "created_utc": now,
            "experiment": name,
            "status": "confirmatory_passed" if passed else "confirmatory_gate_failed",
            "passed": passed,
            "checks": experiment_checks,
            "a8_in_scope": False,
        }
        write_json(out / f"{name.lower()}_report.json", value)
        experiment_results[name] = value
    strict_independence = task.get("counts_as_confirmatory_evidence") is True
    all_experiments_pass = all(value["passed"] for value in experiment_results.values())
    readiness = {
        "schema_version": "kinofail.expanded-a0-a7-v8-readiness.v1",
        "created_utc": now,
        "status": "strong_publication_evidence_ready"
        if all_experiments_pass and strict_independence
        else "not_ready",
        "passed": all_experiments_pass and strict_independence,
        "experiment_pass_count": sum(
            value["passed"] for value in experiment_results.values()
        ),
        "experiment_required_count": 8,
        "strict_independent_confirmation": strict_independence,
        "post_acquisition_pre_prediction_task_alignment_amendment": True,
        "raw_f38_failure_preserved": True,
        "f42_passed_all_preregistered_gates": report.get(
            "passed_all_preregistered_gates"
        ),
        "experiments": {
            name: {
                "passed": value["passed"],
                "failed_checks": [
                    key for key, passed in value["checks"].items() if not passed
                ],
            }
            for name, value in experiment_results.items()
        },
        "binding_checks": binding,
        "claim_boundary": {
            "strong_expanded_simulation_confirmation": False,
            "sim2real_validated": False,
            "real_go2_anchor_required": True,
            "a4_supported_recovery_operators_only": ["O4_tether", "O5_payload"],
        },
        "a8_in_scope": False,
    }
    write_json(out / "readiness_audit.json", readiness)
    write_json(
        out / "evidence_binding.json",
        {
            "schema_version": "kinofail.expanded-a0-a7-v8-binding.v1",
            "created_utc": now,
            "passed": all(binding.values()),
            "checks": binding,
            "artifacts": {
                "f42_seal": artifact(F42_SEAL),
                "f42_report": artifact(paths["report"]),
                "f42_finalization": artifact(paths["finalization"]),
                "f42_predictions": artifact(paths["predictions"]),
                "f42_truth": artifact(paths["truth"]),
                "f38_raw": artifact(F38_RAW),
                "f38_task": artifact(F38_TASK),
                "f38_s2": artifact(F38_S2),
                "f35": artifact(F35),
                "a4": artifact(paths["a4"]),
                "a6": artifact(paths["a6"]),
            },
        },
    )
    print(json.dumps(readiness, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if readiness["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
