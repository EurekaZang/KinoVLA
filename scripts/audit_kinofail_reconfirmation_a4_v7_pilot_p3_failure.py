#!/usr/bin/env python3
"""Preserve the partial P3 outcome set and its O5 replay failure."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p3"
DESIGN = PILOT / "design_audit.json"
PROTOCOL = PILOT / "protocol.json"
RUN = ROOT / "outputs/kinofail_after_f33_to_a4_pilot_v1/runs/10_collect_a4_excluded_pilot_p3.json"
LOG = ROOT / "outputs/kinofail_after_f33_to_a4_pilot_v1/logs/10_collect_a4_excluded_pilot_p3.log"
CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_reconfirmation_a4_v7_pilot_p3/corpus")
OUTPUT = PILOT / "failure_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    design = json.loads(DESIGN.read_text())
    run = json.loads(RUN.read_text())
    scene_root = CORPUS / str(design["scene_id"])
    results_path = scene_root / "results.jsonl"
    audits_path = scene_root / "case_audits.jsonl"
    results = jsonl(results_path)
    audits = jsonl(audits_path)
    log = LOG.read_text(errors="replace")
    checks = {
        "development_only": design.get("development_only") is True,
        "never_counts_as_a0_a7_evidence": design.get("counts_as_a0_a7_evidence") is False,
        "fresh_from_formal_p1_p2": design.get("disjoint_from_formal_p1_p2") is True,
        "all_saved_outcomes_retained": len(results) == 14 and len(audits) == 2,
        "exact_o5_replay_failure_recorded": (
            "A4 replayed proprio does not match the frozen F35 policy input" in log
            and "normalized residual=3.06457" in log
        ),
        "run_and_log_retained": run.get("state") == "terminal" and LOG.is_file(),
        "failure_was_before_a4_f36_seal": not (
            ROOT / "outputs/freeze/kinofail_reconfirmation_a4_v7/seal_manifest.json"
        ).exists(),
    }
    report = {
        "schema_version": "kinofail.reconfirmation-a4-v7-pilot-p3-failure.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "pilot_passed": False,
        "status": "permanently_excluded_partial_development_failure",
        "checks": checks,
        "counts": {"saved_cases": len(audits), "saved_episodes": len(results)},
        "failure_class": "independent_physics_reset_drift_under_o5_payload",
        "corrective_design": (
            "run the shared prefix once, capture the full decision-boundary state, "
            "and branch every action arm from that one checkpoint"
        ),
        "scientific_retry_of_p3_permitted": False,
        "counts_as_a0_a7_evidence": False,
        "source_sha256": {
            "design": sha256(DESIGN),
            "protocol": sha256(PROTOCOL),
            "results": sha256(results_path),
            "case_audits": sha256(audits_path),
            "run": sha256(RUN),
            "log": sha256(LOG),
            "auditor": sha256(Path(__file__).resolve()),
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
