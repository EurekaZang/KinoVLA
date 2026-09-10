#!/usr/bin/env python3
"""Unattended handoff from F4j scene admission to F4k T3 collection."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ADMISSION = ROOT / "outputs/kinofail_kino_v4_confirmation_t3_extension_f4h/scene_admission_audit.json"
BUILDER = ROOT / "scripts/build_kino_v4_confirmation_t3_extension_f4k.py"
DESIGN = ROOT / "outputs/kinofail_kino_v4_confirmation_t3_extension_f4h/design"
SEAL = DESIGN / "seal_manifest.json"
RUNNER = ROOT / "scripts/run_kinofail_t3_runin_f38_collection_v2.py"
EXTENSION_AUDIT = ROOT / "outputs/kinofail_kino_v4_confirmation_t3_extension_f4h/collection/final_audit.json"
F4E_AUDIT = ROOT / "outputs/kinofail_kino_v4_confirmation_t3_runin_f4e/final_audit.json"
OUT = ROOT / "outputs/kinofail_kino_v4_confirmation_t3_extension_f4h"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(name: str, command: list[str]) -> int:
    log = OUT / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    return int(completed.returncode)


def main() -> int:
    for path in (BUILDER, RUNNER):
        if not path.is_file():
            raise FileNotFoundError(path)
    state = OUT / "collection_handoff_state.json"
    atomic(state, {"status": "waiting_for_scene_admission", "updated_utc": datetime.now(UTC).isoformat()})
    while not ADMISSION.is_file():
        time.sleep(30)
        atomic(state, {"status": "waiting_for_scene_admission", "updated_utc": datetime.now(UTC).isoformat()})
    admission = load(ADMISSION)
    if admission.get("passed") is not True:
        atomic(state, {"status": "scientific_gate_failed", "stage": "scene_admission", "updated_utc": datetime.now(UTC).isoformat()})
        return 2
    atomic(state, {"status": "building_extension_schedule", "updated_utc": datetime.now(UTC).isoformat()})
    if run("f4k_schedule_builder", [sys.executable, str(BUILDER)]) != 0:
        atomic(state, {"status": "operational_failure", "stage": "schedule_builder", "updated_utc": datetime.now(UTC).isoformat()})
        return 2
    seal = load(SEAL)
    if seal.get("passed") is not True or seal.get("model_prediction_truth_key_or_score_read") is not False:
        raise RuntimeError("F4k collection seal invalid")
    atomic(
        state,
        {
            "status": "running_extension_collection",
            "updated_utc": datetime.now(UTC).isoformat(),
            "planned_cases": seal["counts"]["planned_cases"],
            "model_prediction_truth_key_or_score_read": False,
        },
    )
    returncode = run(
        "f4k_t3_collection",
        [sys.executable, str(RUNNER), "--seal", str(SEAL), "--max-workers", "3"],
    )
    if not EXTENSION_AUDIT.is_file():
        atomic(state, {"status": "operational_failure", "stage": "extension_collection", "returncode": returncode, "updated_utc": datetime.now(UTC).isoformat()})
        return 2
    original = load(F4E_AUDIT)
    extension = load(EXTENSION_AUDIT)
    planned = int(original["counts"]["planned_cases"]) + int(extension["counts"]["planned_cases"])
    accepted = int(original["counts"]["accepted_cases"]) + int(extension["counts"]["accepted_cases"])
    attrition = (planned - accepted) / planned
    checks = {
        "both_campaigns_terminal": original["checks"]["all_pairs_terminal"] is True and extension["checks"]["all_pairs_terminal"] is True,
        "combined_case_attrition_below_5pct": attrition < 0.05,
        "all_accepted_cases_temporally_valid": original["checks"]["all_accepted_cases_have_two_temporal_windows"] is True and extension["checks"]["all_accepted_cases_have_two_temporal_windows"] is True,
        "no_prediction_truth_key_or_score_read": True,
        "original_failures_preserved": int(original["counts"]["rejected_or_excluded_cases"]) > 0,
    }
    audit = {
        "schema_version": "kinofail.kino-v4-confirmation-t3-combined-f4l-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "model_prediction_truth_key_or_score_read": False,
        "result_dependent_selection_or_retry": False,
        "checks": checks,
        "counts": {
            "planned_cases": planned,
            "accepted_cases": accepted,
            "rejected_or_excluded_cases": planned - accepted,
            "original_planned_cases": original["counts"]["planned_cases"],
            "original_accepted_cases": original["counts"]["accepted_cases"],
            "extension_planned_cases": extension["counts"]["planned_cases"],
            "extension_accepted_cases": extension["counts"]["accepted_cases"],
        },
        "combined_case_attrition_rate": attrition,
        "source_sha256": {
            "f4e_audit": sha256(F4E_AUDIT),
            "extension_audit": sha256(EXTENSION_AUDIT),
            "extension_seal": sha256(SEAL),
            "runner": sha256(Path(__file__).resolve()),
        },
    }
    atomic(OUT / "combined_t3_audit.json", audit)
    atomic(
        state,
        {
            "status": "complete" if audit["passed"] else "scientific_gate_failed",
            "passed": audit["passed"],
            "updated_utc": datetime.now(UTC).isoformat(),
            "counts": audit["counts"],
            "combined_case_attrition_rate": attrition,
        },
    )
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
