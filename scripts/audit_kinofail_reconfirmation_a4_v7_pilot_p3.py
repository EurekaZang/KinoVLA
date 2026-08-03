#!/usr/bin/env python3
"""Audit the fresh-scene, permanently excluded A4-v7 P3 pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p3"
DESIGN = PILOT / "design_audit.json"
PROTOCOL = PILOT / "protocol.json"
CORPUS = Path(
    "/data/eureka/KinoVLA/outputs/kinofail_reconfirmation_a4_v7_pilot_p3/corpus"
)
P1_FAILURE = ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p1/failure_audit.json"
P2_FAILURE = ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p2/failure_audit.json"
OUTPUT = PILOT / "postrun_audit.json"
BUILDER = ROOT / "scripts/build_kinofail_reconfirmation_a4_v7_pilot_p3.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    design = load(DESIGN)
    protocol = load(PROTOCOL)
    scene_root = CORPUS / str(design["scene_id"])
    summary_path = scene_root / "summary.json"
    results_path = scene_root / "results.jsonl"
    case_audits_path = scene_root / "case_audits.jsonl"
    summary = load(summary_path)
    results = jsonl(results_path)
    case_audits = jsonl(case_audits_path)
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    phases: dict[tuple[str, str], list[str]] = {}
    trace_hashes_valid = True
    for row in results:
        case_id, action = str(row["case_id"]), str(row["action"])
        by_case.setdefault(case_id, {})[action] = row
        path = Path(str(row["trace"]))
        trace_hashes_valid &= path.is_file() and sha256(path) == row["trace_sha256"]
        with np.load(path, allow_pickle=False) as archive:
            phases[(case_id, action)] = archive["action_phases"].astype(str).tolist()
    expected_actions = {
        "continue",
        "always_safe_halt",
        "recover_as_O2_compliance",
        "recover_as_O4_tether",
        "recover_as_O5_payload",
        "recover_as_O8_invisible_collider",
        "recover_as_O9_high_centering",
    }
    first_phase = {
        "O2_compliance": "stabilize_slow_cross",
        "O4_tether": "tether_release_backstep",
        "O5_payload": "lower_stabilize_slow",
        "O8_invisible_collider": "obstacle_backstep",
        "O9_high_centering": "chassis_unload_backstep",
    }
    audit_by_case = {str(row["case_id"]): row for row in case_audits}
    per_case: dict[str, dict[str, Any]] = {}
    for case_id, arms in by_case.items():
        operator = str(next(iter(arms.values()))["true_operator"])
        correct_action = f"recover_as_{operator}"
        correct = arms.get(correct_action, {})
        safe = arms.get("always_safe_halt", {})
        audit = audit_by_case.get(case_id, {})
        recoveries = sorted(action for action in arms if action.startswith("recover_as_"))
        certificates = [
            row.get("source_proprio_replay_certificate", {}) for row in arms.values()
        ]
        per_case[case_id] = {
            "operator": operator,
            "complete": set(arms) == expected_actions,
            "prefix_within_tolerance": audit.get("passed") is True
            and float(audit.get("maximum_absolute_predecision_difference", float("inf")))
            <= float(audit.get("tolerance", 0.0)),
            "f35_replay_within_tolerance": all(
                certificate.get("passed") is True
                and float(
                    certificate.get(
                        "maximum_tolerance_normalized_difference", float("inf")
                    )
                )
                <= 1.0
                for certificate in certificates
            )
            and all(
                abs(float(row.get("decision_time_skew_s", 1.0))) <= 0.021
                for row in arms.values()
            ),
            "safe_halt_safe": safe.get("fell") is False
            and safe.get("safe_abort") is True
            and float(safe.get("final_speed_mps", 1.0)) < 0.08,
            "correct_phase_executed": first_phase[operator]
            in phases.get((case_id, correct_action), []),
            "five_programs_distinct": len(
                {
                    str(arms[action].get("postdecision_numeric_trace_sha256"))
                    for action in recoveries
                }
            )
            == 5,
            "correct_recovery_safe": correct.get("fell") is False,
            "correct_recovery_completes_endpoint": correct.get(
                "recovered_past_hazard"
            )
            is True,
        }
    actor = protocol["low_level_actor"]
    pre = protocol["predecision_actor"]
    actor_bindings = {
        "policy": "policy_sha256",
        "training_manifest": "training_manifest_sha256",
        "training_freeze": "training_freeze_sha256",
        "sim_config": "sim_config_sha256",
    }
    checks = {
        "development_only_and_excluded": design.get("development_only") is True
        and design.get("counts_as_a0_a7_evidence") is False,
        "fresh_from_formal_p1_p2": design.get("disjoint_from_formal_p1_p2") is True,
        "prior_failures_preserved": protocol.get("prior_failure_audit_sha256")
        == {"p1": sha256(P1_FAILURE), "p2": sha256(P2_FAILURE)},
        "builder_schedule_collector_hashes_valid": design.get("builder_sha256")
        == sha256(BUILDER)
        and protocol.get("builder_sha256") == sha256(BUILDER)
        and protocol.get("schedule_sha256") == design.get("schedule_sha256")
        and protocol.get("collector_sha256") == design.get("collector_sha256"),
        "predecision_actor_hash_bound": (ROOT / pre["policy"]).is_file()
        and pre["policy_sha256"] == sha256(ROOT / pre["policy"])
        and pre.get("purpose") == "replay_frozen_f35_predecision_state",
        "shared_recovery_actor_hash_bound": actor.get("shared_across_all_action_arms")
        is True
        and actor.get("receives_attribution_input") is False
        and all(
            (ROOT / actor[key]).is_file()
            and actor[hash_key] == sha256(ROOT / actor[key])
            for key, hash_key in actor_bindings.items()
        ),
        "scene_summary_passed": summary.get("passed") is True,
        "five_complete_seven_arm_cases": len(results) == 35
        and len(case_audits) == 5
        and len(by_case) == 5
        and all(row["complete"] for row in per_case.values()),
        "all_five_operators_present": {row["operator"] for row in per_case.values()}
        == set(first_phase),
        "matched_prefixes_passed": all(
            row["prefix_within_tolerance"] for row in per_case.values()
        ),
        "frozen_source_replayed_in_every_arm": all(
            row["f35_replay_within_tolerance"] for row in per_case.values()
        ),
        "all_trace_hashes_valid": trace_hashes_valid,
        "all_safe_halts_are_safe": all(
            row["safe_halt_safe"] for row in per_case.values()
        ),
        "all_correct_recovery_phases_execute": all(
            row["correct_phase_executed"] for row in per_case.values()
        ),
        "all_recovery_programs_are_distinct": all(
            row["five_programs_distinct"] for row in per_case.values()
        ),
        "all_correct_recoveries_are_safe": all(
            row["correct_recovery_safe"] for row in per_case.values()
        ),
        "every_correct_recovery_completes_local_endpoint": all(
            row["correct_recovery_completes_endpoint"] for row in per_case.values()
        ),
        "unfavorable_outcomes_retained": all(
            row.get("unfavorable_outcome_retained") is True for row in results
        ),
    }
    report = {
        "schema_version": "kinofail.reconfirmation-a4-v7-pilot-p3-postrun.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "per_case": per_case,
        "development_only": True,
        "counts_as_a0_a7_evidence": False,
        "source_sha256": {
            "design": sha256(DESIGN),
            "protocol": sha256(PROTOCOL),
            "summary": sha256(summary_path),
            "results": sha256(results_path),
            "case_audits": sha256(case_audits_path),
            "auditor": sha256(Path(__file__).resolve()),
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
