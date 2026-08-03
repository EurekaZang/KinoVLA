#!/usr/bin/env python3
"""Preserve the zero-outcome P2 actor-binding integration failure."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p2"
DESIGN = PILOT / "design_audit.json"
PROTOCOL = PILOT / "protocol.json"
RUN = (
    ROOT
    / "outputs/kinofail_after_f33_to_a4_pilot_v1/runs/08_collect_a4_excluded_pilot_p2.json"
)
LOG = (
    ROOT
    / "outputs/kinofail_after_f33_to_a4_pilot_v1/logs/08_collect_a4_excluded_pilot_p2.log"
)
CORPUS = Path(
    "/data/eureka/KinoVLA/outputs/kinofail_reconfirmation_a4_v7_pilot_p2/corpus"
)
OUTPUT = PILOT / "failure_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    design = json.loads(DESIGN.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    run = json.loads(RUN.read_text())
    log = LOG.read_text(errors="replace")
    corpus_files = sorted(path for path in CORPUS.rglob("*") if path.is_file()) if CORPUS.exists() else []
    exact_failure = (
        "A4 replayed proprio does not match the frozen F35 policy input" in log
        and "normalized residual=834.797" in log
        and "loading policy /home/eureka/KinoVLA/outputs/locomotion/recovery_route_v1/policy.pt"
        in log
    )
    checks = {
        "development_only": design.get("development_only") is True,
        "never_counts_as_a0_a7_evidence": design.get("counts_as_a0_a7_evidence")
        is False,
        "fresh_from_p1_and_formal": design.get("disjoint_from_formal_and_p1") is True,
        "zero_saved_scientific_outcomes": not corpus_files,
        "exact_predecision_actor_mismatch_recorded": exact_failure,
        "original_protocol_and_run_retained": run.get("state") == "terminal"
        and int(run.get("returncode", -1)) == 0,
        "failure_was_before_a4_f36_seal": not (
            ROOT / "outputs/freeze/kinofail_reconfirmation_a4_v7/seal_manifest.json"
        ).exists(),
    }
    report = {
        "schema_version": "kinofail.reconfirmation-a4-v7-pilot-p2-failure.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "pilot_passed": False,
        "status": "permanently_excluded_zero-outcome_integration_failure",
        "checks": checks,
        "failure_class": "recovery_actor_used_before_frozen_f35_decision",
        "corrective_design": (
            "bind the original source actor before the decision and switch every arm "
            "to one shared attribution-blind recovery actor only after replay certification"
        ),
        "scientific_retry_of_p2_permitted": False,
        "counts_as_a0_a7_evidence": False,
        "source_sha256": {
            "design": sha256(DESIGN),
            "protocol": sha256(PROTOCOL),
            "run": sha256(RUN),
            "log": sha256(LOG),
            "auditor": sha256(Path(__file__).resolve()),
        },
        "original_collector_sha256": protocol["collector_sha256"],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
