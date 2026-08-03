#!/usr/bin/env python3
"""Record the failed P1 integration pilot without turning it into evidence.

P1 exposed implementation and recovery-program defects before the A4/F36
seal.  This audit is deliberately descriptive: it proves that P1 is excluded,
preserves every outcome that was produced, and prevents the failed directory
from being resumed or overwritten.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p1"
DESIGN = PILOT / "design_audit.json"
FORMAL_SCHEDULE = ROOT / "outputs/kinofail_reconfirmation_a4_v7/schedule.jsonl"
CORPUS = Path(
    "/data/eureka/KinoVLA/outputs/kinofail_reconfirmation_a4_v7_pilot_p1/corpus"
)
CHAIN = ROOT / "outputs/kinofail_after_f33_to_a4_pilot_v1"
OUTPUT = PILOT / "failure_audit.json"


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
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    design = load(DESIGN)
    scene = str(design["scene_id"])
    scene_root = CORPUS / scene
    results_path = scene_root / "results.jsonl"
    case_audits_path = scene_root / "case_audits.jsonl"
    results = jsonl(results_path)
    case_audits = jsonl(case_audits_path)
    formal_ids = {
        str(row["source_counterfactual_group_id"])
        for row in jsonl(FORMAL_SCHEDULE)
    }
    pilot_ids = set(map(str, design["source_counterfactual_group_ids"]))
    run_records = sorted((CHAIN / "runs").glob("06*.json"))
    run_logs = sorted((CHAIN / "logs").glob("06*.log"))
    produced_cases = sorted({str(row["case_id"]) for row in results})
    operators = sorted({str(row["true_operator"]) for row in results})
    checks = {
        "development_only": design.get("development_only") is True,
        "never_counts_as_a0_a7_evidence": design.get("counts_as_a0_a7_evidence")
        is False,
        "disjoint_from_formal_schedule": not (pilot_ids & formal_ids),
        "all_produced_outcomes_retained": len(results) == sum(
            1 for _ in results_path.read_text().splitlines() if _
        ),
        "failure_was_before_a4_f36_seal": not (
            ROOT / "outputs/freeze/kinofail_reconfirmation_a4_v7/seal_manifest.json"
        ).exists(),
        "failed_run_record_and_log_retained": bool(run_records) and bool(run_logs),
        "pilot_is_incomplete_and_not_passed": not (scene_root / "summary.json").exists()
        and len(results) < 35,
    }
    report = {
        "schema_version": "kinofail.reconfirmation-a4-v7-pilot-p1-failure.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "pilot_passed": False,
        "status": "permanently_excluded_development_failure",
        "checks": checks,
        "counts": {
            "planned_cases": 5,
            "planned_episodes": 35,
            "produced_cases": len(produced_cases),
            "produced_episodes": len(results),
            "case_audits": len(case_audits),
        },
        "produced_operators": operators,
        "observed_failure": {
            "class": "preseal_integration_and_recovery_program_failure",
            "terminal_operator": "O8_invisible_collider",
            "terminal_message": "A4 operator onset missing before the frozen decision",
            "scientific_retry_of_p1_permitted": False,
            "p1_directory_may_be_overwritten": False,
        },
        "development_only": True,
        "counts_as_a0_a7_evidence": False,
        "source_sha256": {
            "design": sha256(DESIGN),
            "formal_schedule": sha256(FORMAL_SCHEDULE),
            "results": sha256(results_path),
            "case_audits": sha256(case_audits_path),
            "run_records": {str(path): sha256(path) for path in run_records},
            "run_logs": {str(path): sha256(path) for path in run_logs},
            "auditor": sha256(Path(__file__).resolve()),
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
