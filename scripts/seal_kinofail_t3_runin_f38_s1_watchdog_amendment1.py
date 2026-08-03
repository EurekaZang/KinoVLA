#!/usr/bin/env python3
"""Seal the S1 process-only watchdog before successor collection."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "outputs/freeze/kinofail_t3_runin_f38_formal_s1/seal_manifest.json"
FAILURE = ROOT / "outputs/kinofail_t3_runin_f38_concurrency_failure/audit.json"
WATCHDOG = ROOT / "scripts/watch_kinofail_t3_runin_f38_formal_s1.py"
RUNNER = ROOT / "scripts/run_kinofail_t3_runin_f38_collection_v2.py"
OUTPUT = ROOT / "outputs/freeze/kinofail_t3_runin_f38_s1_watchdog_amendment1/amendment_manifest.json"
RUN = ROOT / "outputs/kinofail_t3_runin_f38_formal_s1"
CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_t3_runin_f38_formal_s1/corpus")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if OUTPUT.exists() or RUN.exists() or CORPUS.exists():
        raise FileExistsError("S1 watchdog amendment must precede collection")
    formal = json.loads(FORMAL.read_text())
    failure = json.loads(FAILURE.read_text())
    if (
        formal.get("status") != "sealed_before_three_process_result_blind_successor_collection"
        or formal.get("maximum_concurrent_isaac_processes") != 3
        or failure.get("status") != "model_blind_six_process_resource_failure_preserved"
    ):
        raise RuntimeError("invalid S1 predecessor boundary")
    value = {
        "schema_version": "kinofail.t3-runin-f38-s1-watchdog-amendment.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_formal_collection_supervisor_launch",
        "passed": True,
        "scientific_attempt_restart_authorized": False,
        "orchestration_supervisor_restart_only": True,
        "terminal_pair_attempts_never_retried": True,
        "simulation_sensor_feature_model_threshold_and_statistics_changed": False,
        "formal_s1_seal_sha256": sha256(FORMAL),
        "six_process_failure_audit_sha256": sha256(FAILURE),
        "watchdog_sha256": sha256(WATCHDOG),
        "runner_sha256": sha256(RUNNER),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    OUTPUT.with_name("amendment_manifest.sha256").write_text(f"{sha256(OUTPUT)}  {OUTPUT.name}\n")
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
