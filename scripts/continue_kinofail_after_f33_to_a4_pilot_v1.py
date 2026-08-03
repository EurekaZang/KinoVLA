#!/usr/bin/env python3
"""Unattended continuation from F33 completion through the excluded A4 pilot.

Every stage is write-once and stops on the first failure.  It deliberately
stops before the A4/F36 confirmatory seals so the excluded action pilot can be
inspected without exposing a final prediction or outcome.
"""

from __future__ import annotations

import json
import hashlib
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.kinofail_unattended_stage import run_write_once_stage


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
CONDA_ENV = PYTHON.parent.parent
STATE_ROOT = ROOT / "outputs/kinofail_after_f33_to_a4_pilot_v1"
F33_AUDIT = ROOT / "outputs/kinofail_confirmatory_o9_final_f33/final_audit.json"
REGISTRY = ROOT / "outputs/kinofail_reconfirmation_v2/scene_registry.json"
F34_ROOT = ROOT / "outputs/kinofail_confirmatory_o9_postprocess_f34"
F35_AUDIT = ROOT / "outputs/eval/unified_moe_v3_scale_direct_o9_f35/overlay_audit.json"
A4_DESIGN = ROOT / "outputs/kinofail_reconfirmation_a4_v7/design_audit.json"
PILOT_DESIGN = ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p1/design_audit.json"
P1_FAILURE_AUDIT = (
    ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p1/failure_audit.json"
)
P2_ROOT = ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p2"
P2_DESIGN = P2_ROOT / "design_audit.json"
P2_FAILURE_AUDIT = P2_ROOT / "failure_audit.json"
P3_ROOT = ROOT / "outputs/kinofail_reconfirmation_a4_v7_pilot_p3"
P3_DESIGN = P3_ROOT / "design_audit.json"
P3_AUDIT = P3_ROOT / "postrun_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_environment() -> dict[str, str]:
    """Return the explicit Isaac runtime environment for detached watchers."""

    environment = os.environ.copy()
    environment.update(
        {
            "CONDA_PREFIX": str(CONDA_ENV),
            "PATH": f"{CONDA_ENV / 'bin'}:{environment.get('PATH', '')}",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "TERM": "xterm-256color",
            "PYTHONPATH": str(ROOT),
            "HF_HUB_OFFLINE": "1",
        }
    )
    return environment


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_state(stage: str, state: str, **extra: Any) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": "kinofail.after-f33-to-a4-pilot.v1",
        "updated_utc": datetime.now(UTC).isoformat(),
        "stage": stage,
        "state": state,
        **extra,
    }
    temporary = STATE_ROOT / "state.json.tmp"
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, STATE_ROOT / "state.json")


def run(name: str, command: list[str], artifact: Path) -> None:
    run_write_once_stage(
        root=ROOT,
        state_root=STATE_ROOT,
        name=name,
        command=command,
        environment=runtime_environment(),
        write_state=write_state,
        artifacts=[{"path": str(artifact), "key": "passed", "allowed": [True]}],
    )


def reconcile_passed(name: str, path: Path) -> bool:
    if not path.is_file():
        return False
    value = load(path)
    if value.get("passed") is not True:
        write_state(
            name,
            "terminal_failure",
            reason="existing write-once artifact did not pass",
            artifact=str(path),
        )
        raise RuntimeError(f"existing {name} artifact did not pass: {path}")
    write_state(name, "completed_reconciled", artifact=str(path))
    return True


def authorize_zero_outcome_infrastructure_retry(
    *, failed_name: str, retry_name: str, command: list[str], corpus: Path
) -> bool:
    """Authorize one retry only for the observed pre-simulator launch failure.

    The original write-once record and log remain untouched.  This path is
    deliberately narrow: the failed child must have exited before creating any
    corpus file, and its log must contain the missing-Conda-runtime diagnostic.
    """

    retry_record = STATE_ROOT / "runs" / f"{retry_name}.json"
    if retry_record.is_file():
        return True
    failed_record_path = STATE_ROOT / "runs" / f"{failed_name}.json"
    failed_log_path = STATE_ROOT / "logs" / f"{failed_name}.log"
    if not failed_record_path.is_file() or not failed_log_path.is_file():
        return False
    failed_record = load(failed_record_path)
    log_text = failed_log_path.read_text(errors="replace")
    corpus_files = sorted(str(path) for path in corpus.rglob("*") if path.is_file()) if corpus.exists() else []
    exact_infrastructure_failure = (
        failed_record.get("state") == "terminal"
        and int(failed_record.get("returncode", -1)) == 1
        and list(failed_record.get("command", [])) == command
        and "Unable to find any Python executable" in log_text
        and "_isaac_sim/python.sh" in log_text
        and not corpus_files
    )
    if not exact_infrastructure_failure:
        return False
    audit = {
        "schema_version": "kinofail.a4-pilot-zero-outcome-infra-retry.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "failed_stage": failed_name,
        "retry_stage": retry_name,
        "failure_class": "missing_conda_runtime_before_simulator_start",
        "scientific_outcomes_created_before_retry": False,
        "result_dependent_retry": False,
        "scientific_design_or_command_changed": False,
        "failed_record": str(failed_record_path),
        "failed_record_sha256": sha256(failed_record_path),
        "failed_log": str(failed_log_path),
        "failed_log_sha256": sha256(failed_log_path),
        "command": command,
        "operational_change": "set CONDA_PREFIX, PATH, OMNI_KIT_ACCEPT_EULA, and TERM",
    }
    audit_path = STATE_ROOT / "06b_zero_outcome_infrastructure_retry_audit.json"
    temporary = audit_path.with_suffix(audit_path.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, audit_path)
    return True


def main() -> int:
    write_state("wait_f33", "waiting")
    last = 0.0
    while not F33_AUDIT.is_file():
        now = time.monotonic()
        if now - last >= 600:
            print(json.dumps({"stage": "wait_f33", "utc": datetime.now(UTC).isoformat()}), flush=True)
            last = now
        time.sleep(20)
    f33 = load(F33_AUDIT)
    if f33.get("passed") is not True:
        write_state("wait_f33", "terminal_failure", f33_passed=False)
        raise RuntimeError("F33 terminal audit did not pass; continuation stopped")
    if not reconcile_passed("01_seal_f34", F34_ROOT / "seal_manifest.json"):
        run(
            "01_seal_f34",
            [str(PYTHON), "scripts/seal_kinofail_confirmatory_o9_postprocess_f34.py"],
            F34_ROOT / "seal_manifest.json",
        )
    if not reconcile_passed("02_run_f34", F34_ROOT / "final_audit.json"):
        run(
            "02_run_f34",
            [str(PYTHON), "scripts/run_kinofail_confirmatory_o9_postprocess_f34.py"],
            F34_ROOT / "final_audit.json",
        )
    if not reconcile_passed("03_build_f35", F35_AUDIT):
        run(
            "03_build_f35",
            [str(PYTHON), "scripts/build_kinofail_scale_direct_o9_overlay_f35.py"],
            F35_AUDIT,
        )
    if not reconcile_passed("04_build_a4_final_design", A4_DESIGN):
        run(
            "04_build_a4_final_design",
            [str(PYTHON), "scripts/build_kinofail_reconfirmation_a4_v7_schedule.py"],
            A4_DESIGN,
        )
    if not reconcile_passed("05_build_a4_excluded_pilot", PILOT_DESIGN):
        run(
            "05_build_a4_excluded_pilot",
            [str(PYTHON), "scripts/build_kinofail_reconfirmation_a4_v7_pilot_p1.py"],
            PILOT_DESIGN,
        )
    # P1 is a permanently excluded development failure.  Preserve it exactly
    # and move to a fresh-scene P2; never resume or overwrite P1 outcomes.
    if not reconcile_passed("06c_audit_excluded_p1_failure", P1_FAILURE_AUDIT):
        run(
            "06c_audit_excluded_p1_failure",
            [
                str(PYTHON),
                "scripts/audit_kinofail_reconfirmation_a4_v7_pilot_p1_failure.py",
            ],
            P1_FAILURE_AUDIT,
        )
    if not reconcile_passed("07_build_a4_excluded_pilot_p2", P2_DESIGN):
        run(
            "07_build_a4_excluded_pilot_p2",
            [str(PYTHON), "scripts/build_kinofail_reconfirmation_a4_v7_pilot_p2.py"],
            P2_DESIGN,
        )
    if not reconcile_passed("08b_audit_excluded_p2_failure", P2_FAILURE_AUDIT):
        run(
            "08b_audit_excluded_p2_failure",
            [
                str(PYTHON),
                "scripts/audit_kinofail_reconfirmation_a4_v7_pilot_p2_failure.py",
            ],
            P2_FAILURE_AUDIT,
        )
    if not reconcile_passed("09_build_a4_excluded_pilot_p3", P3_DESIGN):
        run(
            "09_build_a4_excluded_pilot_p3",
            [str(PYTHON), "scripts/build_kinofail_reconfirmation_a4_v7_pilot_p3.py"],
            P3_DESIGN,
        )
    protocol = load(P3_ROOT / "protocol.json")
    scene = str(load(P3_DESIGN)["scene_id"])
    pilot_corpus = Path(
        "/data/eureka/KinoVLA/outputs/kinofail_reconfirmation_a4_v7_pilot_p3/corpus"
    )
    pilot_summary_path = pilot_corpus / scene / "summary.json"
    if not reconcile_passed("10_collect_a4_excluded_pilot_p3", pilot_summary_path):
        command = [
            str(PYTHON),
            str(ROOT / protocol["collector"]),
            "--schedule",
            str(ROOT / protocol["schedule"]),
            "--protocol",
            str(P3_ROOT / "protocol.json"),
            "--scene-registry",
            str(REGISTRY),
            "--scene",
            scene,
            "--out",
            str(pilot_corpus),
            "--resume",
            "--headless",
        ]
        run("10_collect_a4_excluded_pilot_p3", command, pilot_summary_path)
    summary = load(pilot_corpus / scene / "summary.json")
    if summary.get("passed") is not True:
        write_state("10_collect_a4_excluded_pilot_p3", "terminal_failure", summary=summary)
        raise RuntimeError("excluded A4 pilot did not pass its integration gates")
    if not reconcile_passed("11_audit_a4_excluded_pilot_p3", P3_AUDIT):
        run(
            "11_audit_a4_excluded_pilot_p3",
            [str(PYTHON), "scripts/audit_kinofail_reconfirmation_a4_v7_pilot_p3.py"],
            P3_AUDIT,
        )
    write_state(
        "ready_for_a4_f36_seals",
        "terminal_success",
        pilot_summary=summary,
        pilot_postrun_audit=str(P3_AUDIT),
        confirmatory_prediction_or_outcome_read=False,
    )
    print(json.dumps({"passed": True, "stage": "ready_for_a4_f36_seals", "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
