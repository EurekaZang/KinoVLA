#!/usr/bin/env python3
"""Unattended continuation from excluded A4 pilots to final A0--A7 v7.

The watcher performs no scientific retry.  Infrastructure/integrity failures
stop immediately; preregistered statistical gate failures are retained and the
remaining independent evidence is still collected before the final fail-closed
ledger is assembled.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.kinofail_unattended_stage import run_write_once_stage


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
STATE_ROOT = ROOT / "outputs/kinofail_after_a4_pilot_to_a0_a7_v3"
READINESS = ROOT / "outputs/eval/realistic_a0_a7_v7_expanded/readiness_audit.json"
PILOT_AUDIT = ROOT / "outputs/kinofail_reconfirmation_a4_v8_pilot_p5/postrun_audit.json"
A4_DESIGN = ROOT / "outputs/kinofail_reconfirmation_a4_v8/design_audit.json"
A4_SEAL = ROOT / "outputs/freeze/kinofail_reconfirmation_a4_v8/seal_manifest.json"
F36_SEAL = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f37/seal_manifest.json"
F36_ROOT = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f37"
A6_REPORT = F36_ROOT / "a6_operator_boundary_report.json"
A4_COLLECTION = ROOT / "outputs/kinofail_reconfirmation_a4_v8/run/final_audit.json"
A4_REPORT = F36_ROOT / "a4_actual_action_policy_report_v8.json"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_state(stage: str, state: str, **extra: Any) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": "kinofail.after-a4-pilot-to-a0-a7.v3",
        "updated_utc": datetime.now(UTC).isoformat(),
        "stage": stage,
        "state": state,
        **extra,
    }
    temporary = STATE_ROOT / "state.json.tmp"
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, STATE_ROOT / "state.json")


def run(
    name: str,
    command: list[str],
    artifacts: list[dict[str, Any]],
    *,
    scientific_gate_may_fail: bool = False,
    forbid_fresh_paths: tuple[Path, ...] = (),
) -> int:
    return run_write_once_stage(
        root=ROOT,
        state_root=STATE_ROOT,
        name=name,
        command=command,
        environment={**os.environ, "PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1"},
        write_state=write_state,
        artifacts=artifacts,
        allowed_returncodes=(0, 2) if scientific_gate_may_fail else (0,),
        forbid_fresh_paths=forbid_fresh_paths,
    )


def reconcile_json(
    name: str,
    path: Path,
    *,
    scientific_gate_may_fail: bool = False,
) -> bool:
    if not path.is_file():
        return False
    value = load(path)
    passed = value.get("passed") is True
    if not passed and not scientific_gate_may_fail:
        write_state(
            name,
            "terminal_failure",
            reason="existing write-once artifact did not pass",
            artifact=str(path),
        )
        raise RuntimeError(f"existing {name} artifact did not pass: {path}")
    write_state(
        name,
        "completed_reconciled" if passed else "completed_scientific_gate_failed_reconciled",
        artifact=str(path),
    )
    return True


def main() -> int:
    write_state("wait_a4_v8_pilot", "waiting")
    last = 0.0
    while True:
        if PILOT_AUDIT.is_file():
            pilot = load(PILOT_AUDIT)
            if (
                pilot.get("passed") is True
                and pilot.get("development_only") is True
                and pilot.get("counts_as_a0_a7_evidence") is False
            ):
                break
            raise RuntimeError(f"invalid excluded A4-v8 P5 audit: {pilot}")
        now = time.monotonic()
        if now - last >= 600:
            print(json.dumps({"stage": "wait_a4_v8_pilot", "utc": datetime.now(UTC).isoformat()}), flush=True)
            last = now
        time.sleep(20)

    if not reconcile_json("01_build_a4_v8_design", A4_DESIGN):
        run(
            "01_build_a4_v8_design",
            [str(PYTHON), "scripts/build_kinofail_reconfirmation_a4_v8_schedule.py"],
            [{"path": str(A4_DESIGN), "key": "passed", "allowed": [True]}],
        )
    if not reconcile_json("02_seal_a4", A4_SEAL):
        run(
            "02_seal_a4",
            [str(PYTHON), "scripts/seal_kinofail_reconfirmation_a4_v8.py"],
            [{"path": str(A4_SEAL), "key": "passed", "allowed": [True]}],
        )
    if not reconcile_json("03_seal_f37", F36_SEAL):
        run(
            "03_seal_f37",
            [str(PYTHON), "scripts/seal_kinofail_reconfirmation_f37.py"],
            [{"path": str(F36_SEAL), "key": "passed", "allowed": [True]}],
        )
    finalization_audit = F36_ROOT / "finalization_audit.json"
    confirmatory_report = F36_ROOT / "confirmatory_report.json"
    if finalization_audit.is_file() and confirmatory_report.is_file():
        if load(confirmatory_report).get("confirmatory_protocol_valid") is not True:
            raise RuntimeError("existing F36 one-shot report failed its integrity protocol")
        write_state(
            "04_finalize_f37_once",
            "completed_reconciled",
            artifact=str(finalization_audit),
        )
    else:
        run(
            "04_finalize_f37_once",
            [str(PYTHON), "scripts/finalize_kinofail_reconfirmation_f37.py", "--poll-seconds", "30"],
            [
                {"path": str(finalization_audit)},
                {
                    "path": str(confirmatory_report),
                    "key": "confirmatory_protocol_valid",
                    "allowed": [True],
                },
            ],
            scientific_gate_may_fail=True,
            forbid_fresh_paths=(F36_ROOT,),
        )
    if not reconcile_json("05_analyze_a6", A6_REPORT, scientific_gate_may_fail=True):
        run(
            "05_analyze_a6",
            [
                str(PYTHON),
                "scripts/analyze_kinofail_reconfirmation_a6_v7.py",
                "--truth",
                str(F36_ROOT / "blind_bundle/truth_key.jsonl"),
                "--predictions",
                str(F36_ROOT / "blind_predictions/blind_predictions.jsonl"),
                "--out",
                str(A6_REPORT),
            ],
            [{"path": str(A6_REPORT), "key": "passed", "allowed": [True, False]}],
            scientific_gate_may_fail=True,
        )
    if not reconcile_json("06_collect_a4_actual_actions", A4_COLLECTION):
        run(
            "06_collect_a4_actual_actions",
            [
                str(PYTHON),
                "scripts/run_kinofail_reconfirmation_a4_v8.py",
                "--max-workers",
                "1",
                "--timeout-s",
                "1800",
            ],
            [{"path": str(A4_COLLECTION), "key": "passed", "allowed": [True]}],
        )
    if not reconcile_json("07_analyze_a4", A4_REPORT, scientific_gate_may_fail=True):
        run(
            "07_analyze_a4",
            [
                str(PYTHON),
                "scripts/analyze_kinofail_reconfirmation_a4_v8.py",
                "--truth",
                str(F36_ROOT / "blind_bundle/truth_key.jsonl"),
                "--predictions",
                str(F36_ROOT / "blind_predictions/blind_predictions.jsonl"),
                "--out",
                str(A4_REPORT),
            ],
            [{"path": str(A4_REPORT), "key": "passed", "allowed": [True, False]}],
            scientific_gate_may_fail=True,
        )
    if not reconcile_json("08_assemble_a0_a7", READINESS, scientific_gate_may_fail=True):
        run(
            "08_assemble_a0_a7",
            [str(PYTHON), "scripts/assemble_kinofail_reconfirmation_a0_a7_v7.py"],
            [{"path": str(READINESS), "key": "passed", "allowed": [True, False]}],
            scientific_gate_may_fail=True,
        )
    readiness = load(READINESS)
    write_state(
        "a0_a7_terminal",
        "terminal_success" if readiness.get("passed") is True else "terminal_scientific_gate_failed",
        readiness=readiness,
    )
    print(
        json.dumps(
            {
                "passed": readiness.get("passed"),
                "status": readiness.get("status"),
                "passed_experiments": readiness.get("passed_experiments"),
                "failed_experiments": readiness.get("failed_experiments"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if readiness.get("passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
