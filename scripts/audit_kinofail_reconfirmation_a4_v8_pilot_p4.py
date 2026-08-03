#!/usr/bin/env python3
"""Audit the excluded A4-v8 P4 pilot without an efficacy gate."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT_ID = os.environ.get("KINO_A4_PILOT_ID", "p4")
if PILOT_ID not in {"p4", "p5"}:
    raise ValueError(f"unsupported pilot id: {PILOT_ID}")
DESIGN = ROOT / f"outputs/kinofail_reconfirmation_a4_v8_pilot_{PILOT_ID}"
CORPUS = Path(
    f"/data/eureka/KinoVLA/outputs/kinofail_reconfirmation_a4_v8_pilot_{PILOT_ID}/corpus"
)
OUTPUT = DESIGN / "postrun_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    schedule_path = DESIGN / "schedule.jsonl"
    protocol_path = DESIGN / "protocol.json"
    design_path = DESIGN / "design_audit.json"
    schedule = jsonl(schedule_path)
    protocol = json.loads(protocol_path.read_text())
    design = json.loads(design_path.read_text())
    issues: list[str] = []
    results: list[dict] = []
    case_audits: list[dict] = []
    summaries: list[dict] = []
    for case in schedule:
        case_dir = CORPUS / str(case["scene_id"]) / str(case["case_id"])
        for name in ("results.jsonl", "case_audits.jsonl", "summary.json"):
            if not (case_dir / name).is_file():
                issues.append(f"missing:{case['case_id']}:{name}")
        if issues:
            continue
        case_results = jsonl(case_dir / "results.jsonl")
        audits = jsonl(case_dir / "case_audits.jsonl")
        summary = json.loads((case_dir / "summary.json").read_text())
        results.extend(case_results)
        case_audits.extend(audits)
        summaries.append(summary)
        expected_actions = list(case["actions"])
        if len(case_results) != 7 or [row["action"] for row in case_results] != expected_actions:
            issues.append(f"action_matrix:{case['case_id']}")
        if len(audits) != 1 or audits[0].get("passed") is not True:
            issues.append(f"case_audit:{case['case_id']}")
        if summary.get("passed") is not True or summary.get("completed_episodes") != 7:
            issues.append(f"summary:{case['case_id']}")
    checks = {
        "design_passed": design.get("passed") is True,
        "protocol_is_permanently_excluded": (
            protocol.get("development_only") is True
            and protocol.get("counts_as_a0_a7_evidence") is False
        ),
        "three_cases_complete": len(summaries) == 3,
        "twenty_one_episodes_complete": len(results) == 21,
        "all_case_audits_pass": (
            len(case_audits) == 3
            and all(row.get("passed") is True for row in case_audits)
        ),
        "exact_f35_replay": (
            len(case_audits) == 3
            and all(
                row.get("source_proprio_replay_passed") is True
                and float(row.get("maximum_proprio_tolerance_normalized_difference", 1.0))
                == 0.0
                for row in case_audits
            )
        ),
        "identical_predecision_prefixes": (
            len(case_audits) == 3
            and all(
                float(row.get("maximum_absolute_predecision_difference", 1.0)) == 0.0
                for row in case_audits
            )
        ),
        "five_recovery_trajectories_distinct": (
            len(case_audits) == 3
            and all(row.get("five_recovery_numeric_trajectories_distinct") is True for row in case_audits)
        ),
        "all_outcomes_retained": all(
            row.get("unfavorable_outcome_retained") is True for row in results
        ),
        "no_efficacy_gate_applied": True,
    }
    issues.extend(key for key, value in checks.items() if not value)
    by_operator = {}
    for operator in sorted({str(row["true_operator"]) for row in results}):
        cells = [row for row in results if row["true_operator"] == operator]
        by_operator[operator] = {
            "episodes": len(cells),
            "falls": sum(bool(row["fell"]) for row in cells),
            "recovery_successes": sum(
                bool(row["operator_recovery_success"]) for row in cells
            ),
            "action_success": {
                str(row["action"]): bool(row["operator_recovery_success"])
                for row in cells
            },
        }
    report = {
        "schema_version": f"kinofail.reconfirmation-a4-v8-pilot-{PILOT_ID}-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": not issues,
        "development_only": True,
        "outcome_exposed": True,
        "counts_as_a0_a7_evidence": False,
        "engineering_gate_only": True,
        "checks": checks,
        "issues": issues,
        "counts": {
            "cases": len(summaries),
            "episodes": len(results),
            "falls": sum(bool(row["fell"]) for row in results),
            "recovery_successes": sum(
                bool(row["operator_recovery_success"]) for row in results
            ),
            "actions": dict(Counter(str(row["action"]) for row in results)),
        },
        "outcomes_reported_not_gated": by_operator,
        "sha256": {
            "schedule": sha256(schedule_path),
            "protocol": sha256(protocol_path),
            "design_audit": sha256(design_path),
            "collector": str(protocol["collector_sha256"]),
            "result_files": {
                str(path): sha256(path)
                for path in sorted(CORPUS.glob("*/*/results.jsonl"))
            },
            "case_audit_files": {
                str(path): sha256(path)
                for path in sorted(CORPUS.glob("*/*/case_audits.jsonl"))
            },
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
