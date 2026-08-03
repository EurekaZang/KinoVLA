#!/usr/bin/env python3
"""Audit the permanently excluded five-operator A4-v7 integration pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p1/design_audit.json"
PILOT_ROOT = Path("/data/eureka/KinoVLA/outputs/kinofail_reconfirmation_a4_v7_pilot_p1/corpus")
OUTPUT = ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p1/postrun_audit.json"
BUILDER = ROOT / "scripts/build_kinofail_reconfirmation_a4_v7_pilot_p1.py"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    design = load(DESIGN)
    scene = str(design["scene_id"])
    root = PILOT_ROOT / scene
    summary_path = root / "summary.json"
    results_path = root / "results.jsonl"
    audits_path = root / "case_audits.jsonl"
    summary = load(summary_path)
    results = jsonl(results_path)
    case_audits = jsonl(audits_path)
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in results:
        by_case.setdefault(str(row["case_id"]), {})[str(row["action"])] = row
    expected = {
        "continue",
        "always_safe_halt",
        "recover_as_O2_compliance",
        "recover_as_O4_tether",
        "recover_as_O5_payload",
        "recover_as_O8_invisible_collider",
        "recover_as_O9_high_centering",
    }
    traces: dict[tuple[str, str], np.ndarray] = {}
    action_phases: dict[tuple[str, str], list[str]] = {}
    for case_id, arms in by_case.items():
        for action, row in arms.items():
            path = Path(str(row["trace"]))
            if not path.is_file() or sha256(path) != row["trace_sha256"]:
                raise RuntimeError(f"pilot trace drift: {path}")
            with np.load(path, allow_pickle=False) as archive:
                traces[(case_id, action)] = np.asarray(
                    archive["trace"], dtype=np.float64
                )
                action_phases[(case_id, action)] = archive[
                    "action_phases"
                ].astype(str).tolist()
    phase_by_operator = {
        "O2_compliance": "stabilize_slow_cross",
        "O4_tether": "tether_release_backstep",
        "O5_payload": "lower_stabilize_slow",
        "O8_invisible_collider": "obstacle_backstep",
        "O9_high_centering": "chassis_unload_backstep",
    }
    per_case = {}
    for case_id, arms in by_case.items():
        exemplar = next(iter(arms.values()))
        operator = str(exemplar["true_operator"])
        safe = arms.get("always_safe_halt", {})
        continued = arms.get("continue", {})
        correct_action = f"recover_as_{operator}"
        correct = arms.get(correct_action, {})
        recovery_actions = sorted(
            action for action in arms if action.startswith("recover_as_")
        )
        per_case[case_id] = {
            "operator": operator,
            "arm_coverage": set(arms) == expected,
            "shared_prefix_hash": len(
                {str(row["predecision_trace_sha256"]) for row in arms.values()}
            )
            == 1,
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
            "source_proprio_replay_passed": all(
                row.get("source_proprio_replay_certificate", {}).get("passed")
                is True
                for row in arms.values()
            )
            and len(
                {
                    row["source_proprio_replay_certificate"][
                        "replayed_invariant_proprio_80_sha256"
                    ]
                    for row in arms.values()
                }
            )
            == 1,
            "maximum_proprio_tolerance_normalized_difference": max(
                float(
                    row.get("source_proprio_replay_certificate", {}).get(
                        "maximum_tolerance_normalized_difference", float("inf")
                    )
                )
                for row in arms.values()
            ),
            "every_arm_branched": all(
                any(phase != "shared_prefix" for phase in action_phases[(case_id, action)])
                for action in arms
            ),
            "safe_halt_safe_and_stationary": safe.get("fell") is False
            and safe.get("safe_abort") is True
            and float(safe.get("final_speed_mps", 1.0)) < 0.08,
            "correct_recovery_phase_executed": phase_by_operator[operator]
            in action_phases.get((case_id, correct_action), []),
            "five_recovery_phase_programs_distinct": len(
                {
                    tuple(
                        phase
                        for phase in action_phases[(case_id, action)]
                        if phase != "shared_prefix"
                    )
                    for action in recovery_actions
                }
            )
            == 5,
            "five_recovery_numeric_trajectories_distinct": len(
                {
                    str(arms[action].get("postdecision_numeric_trace_sha256"))
                    for action in recovery_actions
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
            "terminal_costs": {
                action: float(row["terminal_cost"]) for action, row in arms.items()
            },
            "falls": {action: bool(row["fell"]) for action, row in arms.items()},
            "recovered_past_hazard": {
                action: bool(row["recovered_past_hazard"])
                for action, row in arms.items()
            },
        }
    audit_by_case = {str(row["case_id"]): row for row in case_audits}
    checks = {
        "development_only_and_excluded": design.get("development_only") is True
        and design.get("counts_as_a0_a7_evidence") is False,
        "pilot_design_builder_hash_valid": design.get("builder_sha256")
        == sha256(BUILDER),
        "scene_summary_passed": summary.get("passed") is True,
        "five_complete_seven_arm_cases": len(results) == 35
        and len(case_audits) == 5
        and len(by_case) == 5
        and all(row["arm_coverage"] for row in per_case.values()),
        "all_five_operators_present": {
            row["operator"] for row in per_case.values()
        }
        == set(phase_by_operator),
        "matched_prefix_certificates_passed": all(
            audit_by_case.get(case_id, {}).get("passed") is True
            and row["shared_prefix_hash"]
            for case_id, row in per_case.items()
        ),
        "frozen_source_decision_time_replayed_within_21ms": all(
            abs(float(row.get("decision_time_skew_s", 1.0))) <= 0.021
            for row in results
        ),
        "frozen_source_nuisance_identical_within_each_case": all(
            row["source_nuisance_shared"] for row in per_case.values()
        ),
        "frozen_f35_proprio_observation_replayed_in_every_arm": all(
            row["source_proprio_replay_passed"] for row in per_case.values()
        ),
        "all_trace_hashes_and_arrays_present": len(traces) == 35
        and all(array.ndim == 2 and array.shape[1] == 15 for array in traces.values()),
        "every_arm_branches_after_decision": all(
            row["every_arm_branched"] for row in per_case.values()
        ),
        "all_always_safe_halts_are_safe_and_stationary": all(
            row["safe_halt_safe_and_stationary"] for row in per_case.values()
        ),
        "all_correct_recovery_phases_are_physically_executed": all(
            row["correct_recovery_phase_executed"] for row in per_case.values()
        ),
        "all_five_recovery_programs_and_numeric_trajectories_are_distinct": all(
            row["five_recovery_phase_programs_distinct"]
            and row["five_recovery_numeric_trajectories_distinct"]
            for row in per_case.values()
        ),
        "no_correct_recovery_is_catastrophic": all(
            row["correct_recovery_not_catastrophic"] for row in per_case.values()
        ),
        "every_correct_recovery_completes_the_local_endpoint": all(
            row["correct_recovery_completes_local_endpoint"]
            for row in per_case.values()
        ),
        "wrong_action_consequences_are_retained": all(
            row.get("unfavorable_outcome_retained") is True for row in results
        ),
    }
    report = {
        "schema_version": "kinofail.reconfirmation-a4-v7-pilot-p1-postrun.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "per_case": per_case,
        "source_sha256": {
            "design": sha256(DESIGN),
            "summary": sha256(summary_path),
            "results": sha256(results_path),
            "case_audits": sha256(audits_path),
            "auditor": sha256(Path(__file__).resolve()),
        },
        "development_only": True,
        "counts_as_a0_a7_evidence": False,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
