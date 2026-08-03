#!/usr/bin/env python3
"""Audit the fresh-scene, permanently excluded A4-v7 P2 pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p2"
DESIGN = PILOT / "design_audit.json"
PROTOCOL = PILOT / "protocol.json"
CORPUS = Path(
    "/data/eureka/KinoVLA/outputs/kinofail_reconfirmation_a4_v7_pilot_p2/corpus"
)
OUTPUT = PILOT / "postrun_audit.json"
BUILDER = ROOT / "scripts/build_kinofail_reconfirmation_a4_v7_pilot_p2.py"
P1_FAILURE = (
    ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p1/failure_audit.json"
)


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
    scene = str(design["scene_id"])
    root = CORPUS / scene
    summary_path = root / "summary.json"
    results_path = root / "results.jsonl"
    case_audits_path = root / "case_audits.jsonl"
    summary = load(summary_path)
    results = jsonl(results_path)
    case_audits = jsonl(case_audits_path)
    expected_actions = {
        "continue",
        "always_safe_halt",
        "recover_as_O2_compliance",
        "recover_as_O4_tether",
        "recover_as_O5_payload",
        "recover_as_O8_invisible_collider",
        "recover_as_O9_high_centering",
    }
    expected_phases = {
        "O2_compliance": "stabilize_slow_cross",
        "O4_tether": "tether_release_backstep",
        "O5_payload": "lower_stabilize_slow",
        "O8_invisible_collider": "obstacle_backstep",
        "O9_high_centering": "chassis_unload_backstep",
    }
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    phases: dict[tuple[str, str], list[str]] = {}
    trace_arrays: dict[tuple[str, str], np.ndarray] = {}
    for row in results:
        case_id = str(row["case_id"])
        action = str(row["action"])
        by_case.setdefault(case_id, {})[action] = row
        trace_path = Path(str(row["trace"]))
        if not trace_path.is_file() or sha256(trace_path) != row["trace_sha256"]:
            raise RuntimeError(f"P2 trace drift: {trace_path}")
        with np.load(trace_path, allow_pickle=False) as archive:
            trace_arrays[(case_id, action)] = np.asarray(
                archive["trace"], dtype=np.float64
            )
            phases[(case_id, action)] = archive["action_phases"].astype(str).tolist()
    audit_by_case = {str(row["case_id"]): row for row in case_audits}
    per_case: dict[str, dict[str, Any]] = {}
    for case_id, arms in by_case.items():
        operator = str(next(iter(arms.values()))["true_operator"])
        correct_action = f"recover_as_{operator}"
        correct = arms.get(correct_action, {})
        continued = arms.get("continue", {})
        safe = arms.get("always_safe_halt", {})
        recoveries = sorted(action for action in arms if action.startswith("recover_as_"))
        certificates = [
            row.get("source_proprio_replay_certificate", {}) for row in arms.values()
        ]
        case_audit = audit_by_case.get(case_id, {})
        per_case[case_id] = {
            "operator": operator,
            "arm_coverage": set(arms) == expected_actions,
            "matched_prefix_within_tolerance": case_audit.get("passed") is True
            and float(case_audit.get("maximum_absolute_predecision_difference", float("inf")))
            <= float(case_audit.get("tolerance", 0.0)),
            "decision_time_replayed": all(
                abs(float(row.get("decision_time_skew_s", 1.0))) <= 0.021
                for row in arms.values()
            ),
            "source_nuisance_shared": len(
                {
                    json.dumps(row.get("source_physical_nuisance"), sort_keys=True)
                    for row in arms.values()
                }
            )
            == 1,
            "source_proprio_within_frozen_tolerance": all(
                certificate.get("passed") is True
                and float(
                    certificate.get(
                        "maximum_tolerance_normalized_difference", float("inf")
                    )
                )
                <= 1.0
                for certificate in certificates
            ),
            "every_arm_branched": all(
                any(phase != "shared_prefix" for phase in phases[(case_id, action)])
                for action in arms
            ),
            "safe_halt_safe_and_stationary": safe.get("fell") is False
            and safe.get("safe_abort") is True
            and float(safe.get("final_speed_mps", 1.0)) < 0.08,
            "correct_recovery_phase_executed": expected_phases[operator]
            in phases.get((case_id, correct_action), []),
            "five_recovery_programs_distinct": len(
                {
                    tuple(
                        phase
                        for phase in phases[(case_id, action)]
                        if phase != "shared_prefix"
                    )
                    for action in recoveries
                }
            )
            == 5
            and len(
                {
                    str(arms[action].get("postdecision_numeric_trace_sha256"))
                    for action in recoveries
                }
            )
            == 5,
            "correct_recovery_not_catastrophic": correct.get("fell") is False
            and float(correct.get("terminal_cost", 1e9))
            <= float(continued.get("terminal_cost", 0.0)) + 10.0,
            "correct_recovery_completes_local_endpoint": correct.get(
                "recovered_past_hazard"
            )
            is True,
        }
    actor = protocol["low_level_actor"]
    actor_paths = {
        key: ROOT / str(actor[key])
        for key in ("policy", "training_manifest", "training_freeze", "sim_config")
    }
    actor_hash_keys = {
        "policy": "policy_sha256",
        "training_manifest": "training_manifest_sha256",
        "training_freeze": "training_freeze_sha256",
        "sim_config": "sim_config_sha256",
    }
    checks = {
        "development_only_and_excluded": design.get("development_only") is True
        and design.get("counts_as_a0_a7_evidence") is False,
        "fresh_from_p1_and_formal": design.get("disjoint_from_formal_and_p1") is True,
        "p1_failure_preserved": protocol.get("p1_failure_audit_sha256")
        == sha256(P1_FAILURE),
        "builder_and_protocol_hashes_valid": design.get("builder_sha256")
        == sha256(BUILDER)
        and protocol.get("builder_sha256") == sha256(BUILDER)
        and protocol.get("schedule_sha256") == design.get("schedule_sha256"),
        "shared_actor_is_hash_bound": actor.get("shared_across_all_action_arms") is True
        and actor.get("receives_attribution_input") is False
        and all(
            path.is_file() and actor[actor_hash_keys[key]] == sha256(path)
            for key, path in actor_paths.items()
        ),
        "scene_summary_passed": summary.get("passed") is True,
        "five_complete_seven_arm_cases": len(results) == 35
        and len(case_audits) == 5
        and len(by_case) == 5
        and all(row["arm_coverage"] for row in per_case.values()),
        "all_five_operators_present": {row["operator"] for row in per_case.values()}
        == set(expected_phases),
        "matched_prefixes_passed": all(
            row["matched_prefix_within_tolerance"] for row in per_case.values()
        ),
        "frozen_source_replayed_in_every_arm": all(
            row["decision_time_replayed"]
            and row["source_nuisance_shared"]
            and row["source_proprio_within_frozen_tolerance"]
            for row in per_case.values()
        ),
        "all_trace_arrays_present": len(trace_arrays) == 35
        and all(array.ndim == 2 and array.shape[1] == 15 for array in trace_arrays.values()),
        "every_arm_branches_after_decision": all(
            row["every_arm_branched"] for row in per_case.values()
        ),
        "all_always_safe_halts_are_safe_and_stationary": all(
            row["safe_halt_safe_and_stationary"] for row in per_case.values()
        ),
        "all_correct_recovery_phases_execute": all(
            row["correct_recovery_phase_executed"] for row in per_case.values()
        ),
        "all_recovery_programs_are_distinct": all(
            row["five_recovery_programs_distinct"] for row in per_case.values()
        ),
        "no_correct_recovery_is_catastrophic": all(
            row["correct_recovery_not_catastrophic"] for row in per_case.values()
        ),
        "every_correct_recovery_completes_local_endpoint": all(
            row["correct_recovery_completes_local_endpoint"]
            for row in per_case.values()
        ),
        "unfavorable_outcomes_retained": all(
            row.get("unfavorable_outcome_retained") is True for row in results
        ),
    }
    report = {
        "schema_version": "kinofail.reconfirmation-a4-v7-pilot-p2-postrun.v1",
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
