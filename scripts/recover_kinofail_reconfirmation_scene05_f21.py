#!/usr/bin/env python3
"""Recover scene05 T2, finish the scene, and hand off to F21 unattended mode."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_kinofail_reconfirmation_t2_atomic_f21 as runner
from scripts import seal_kinofail_reconfirmation_f21 as sealer
from scripts.recover_kinofail_reconfirmation_t2_f14 import (
    _process_state,
    terminate_exact_processes,
)


SCENE = sealer.SCENE
OUT = sealer.OUT
AUDIT_PATH = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/orchestration"
    / SCENE
    / "f21_scene05_recovery.json"
)
RECEIPT = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/prune_receipts"
    / SCENE
    / "completed.json"
)
RECOVERY_LOG = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/orchestration"
    / SCENE
    / "logs/t2_f21_atomic_recovery.log"
)
F21_ALL_SCENES_LOG = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/orchestration_logs/f21/"
    "remaining_scenes_pipeline_v5.log"
)
F21_FINALIZER_LOG = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/orchestration_logs/f21/"
    "finalizer_v7.log"
)


def _write(value: dict[str, Any]) -> None:
    runner.write_json(AUDIT_PATH, value)


def _pause(processes: list[dict[str, Any]]) -> list[int]:
    pids = []
    for row in processes:
        pid = int(row["pid"])
        if _process_state(pid) not in {None, "Z"}:
            os.kill(pid, signal.SIGSTOP)
            pids.append(pid)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if all(_process_state(pid) == "T" for pid in pids):
            return pids
        time.sleep(0.1)
    raise RuntimeError("F21 failed to pause orchestration processes")


def _resume(pids: list[int]) -> list[int]:
    resumed = []
    for pid in pids:
        if _process_state(pid) not in {None, "Z"}:
            os.kill(pid, signal.SIGCONT)
            resumed.append(pid)
    return resumed


def _exact_token_processes(
    script_name: str,
    *required: str,
) -> list[dict[str, Any]]:
    matches = sealer.matching_processes(script_name, *required)
    return [
        row
        for row in matches
        if any(
            token == script_name or Path(token).name == script_name
            for token in row["command"]
        )
    ]


def _wait_absent(
    script_name: str,
    *required: str,
    timeout: int = 45,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _exact_token_processes(script_name, *required):
            return
        time.sleep(0.5)
    raise RuntimeError(
        f"processes did not exit: {(script_name, *required)}"
    )


def _terminate_stopped(pids: list[int]) -> dict[str, list[int]]:
    term = []
    kill = []
    for pid in pids:
        if _process_state(pid) not in {None, "Z"}:
            os.kill(pid, signal.SIGTERM)
            os.kill(pid, signal.SIGCONT)
            term.append(pid)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if all(_process_state(pid) in {None, "Z"} for pid in pids):
            return {"sigterm": term, "sigkill": kill}
        time.sleep(0.5)
    for pid in pids:
        if _process_state(pid) not in {None, "Z"}:
            os.kill(pid, signal.SIGKILL)
            kill.append(pid)
    return {"sigterm": term, "sigkill": kill}


def _launch_detached(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.close()
    return int(process.pid)


def _start_finalizer_v7() -> dict[str, Any]:
    old = _exact_token_processes(
        "finalize_kinofail_reconfirmation_v6.py",
        "--poll-seconds",
        "30",
    )
    termination = terminate_exact_processes(old) if old else {}
    _wait_absent(
        "finalize_kinofail_reconfirmation_v6.py",
        "--poll-seconds",
        "30",
    )
    command = [
        sys.executable,
        str(ROOT / "scripts/finalize_kinofail_reconfirmation_v7.py"),
        "--poll-seconds",
        "30",
    ]
    return {
        "old_finalizer_termination": termination,
        "command": command,
        "pid": _launch_detached(command, F21_FINALIZER_LOG),
        "log": str(F21_FINALIZER_LOG),
    }


def main() -> int:
    amendment = runner._validate_f21()
    if AUDIT_PATH.exists() or RECOVERY_LOG.exists():
        raise FileExistsError(
            AUDIT_PATH if AUDIT_PATH.exists() else RECOVERY_LOG
        )
    state = sealer.incident_case_state()
    inventory, inventory_sha = sealer.authenticated_inventory(state)
    expected = amendment["incident"]
    if (
        state["sealed_manifests"] != expected["sealed_manifests"]
        or state["empty_zero_observation_case_ids"]
        != expected["empty_zero_observation_case_ids"]
        or state["never_attempted_case_ids"]
        != expected["never_attempted_case_ids"]
        or inventory_sha
        != amendment["evidence"][
            "authenticated_relative_size_content_inventory_sha256"
        ]
        or runner.sha256(sealer.SOURCE_AUDIT)
        != expected["source_launcher_audit_sha256"]
        or runner.sha256(sealer.SOURCE_LOG)
        != expected["source_runtime_log_sha256"]
    ):
        raise RuntimeError("scene05 state differs from sealed F21 evidence")
    all_scenes = _exact_token_processes(
        "run_kinofail_reconfirmation_all_scene_pipelines_v4.py"
    )
    scene_pipelines = _exact_token_processes(
        "run_kinofail_reconfirmation_scene_pipeline_v4.py",
        "--scene",
        SCENE,
    )
    t2_processes = _exact_token_processes(
        "isaac_collect_kinofail_confirmatory_t2_v1.py",
        "--scene",
        SCENE,
    )
    if len(all_scenes) != 1 or len(scene_pipelines) != 2 or not t2_processes:
        raise RuntimeError("unexpected F21 live process topology")
    audit: dict[str, Any] = {
        "schema_version": "kinofail.reconfirmation-f21-scene05-recovery.v1",
        "state": "started",
        "started_utc": datetime.now(UTC).isoformat(),
        "scene_id": SCENE,
        "operational_amendment": str(runner.F21),
        "operational_amendment_sha256": runner.sha256(runner.F21),
        "sealed_case_state": state,
        "authenticated_case_inventory_sha256": inventory_sha,
        "existing_completed_case_reexecuted": False,
        "model_feature_prediction_score_or_label_read": False,
        "scientific_content_changed": False,
        "result_dependent_retry_or_selection": False,
        "t2_recovery_state": "started",
        "t2_recovery_passed": False,
    }
    _write(audit)
    paused_all_scenes = _pause(all_scenes)
    paused_scene = _pause(scene_pipelines)
    audit["paused_all_scenes_pids"] = paused_all_scenes
    audit["paused_scene_pipeline_pids"] = paused_scene
    audit["source_t2_termination"] = terminate_exact_processes(t2_processes)
    _wait_absent(
        "isaac_collect_kinofail_confirmatory_t2_v1.py",
        "--scene",
        SCENE,
    )
    _wait_absent(
        "run_kinofail_reconfirmation_slotted_t2_v1.py",
        "--scene",
        SCENE,
    )
    command = [
        sys.executable,
        str(Path(runner.__file__).resolve()),
        "--schedule",
        str(sealer.SCHEDULE),
        "--scene-registry",
        str(
            ROOT / "outputs/kinofail_reconfirmation_v2/scene_registry.json"
        ),
        "--asset-lock",
        str(
            ROOT
            / "outputs/assets/terrain_pbr_confirmatory_v2/"
            "terrain_assets.lock.json"
        ),
        "--protocol",
        str(
            ROOT
            / "outputs/kinofail_reconfirmation_v2/schedules/c2_t2/"
            "collection_protocol.json"
        ),
        "--out",
        str(OUT),
        "--scene",
        SCENE,
        "--standalone-recovery",
    ]
    RECOVERY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RECOVERY_LOG.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    atomic_audit_path = (
        OUT / "launcher_audits" / f"{SCENE}_f21_atomic.json"
    )
    atomic_audit = (
        runner.read_json(atomic_audit_path)
        if atomic_audit_path.is_file()
        else {}
    )
    if (
        completed.returncode != 0
        or atomic_audit.get("state") != "terminal"
        or atomic_audit.get("passed") is not True
        or atomic_audit.get("initial_manifest_hashes_preserved") is not True
    ):
        audit["state"] = "terminal_failure"
        audit["failed_utc"] = datetime.now(UTC).isoformat()
        audit["t2_recovery_returncode"] = int(completed.returncode)
        audit["error"] = "F21 atomic T2 recovery did not pass"
        _write(audit)
        return 2
    audit.update(
        {
            "t2_recovery_state": "terminal",
            "t2_recovery_passed": True,
            "t2_recovery_returncode": int(completed.returncode),
            "atomic_t2_audit": str(atomic_audit_path),
            "atomic_t2_audit_sha256": runner.sha256(atomic_audit_path),
            "recovery_log": str(RECOVERY_LOG),
            "recovery_log_sha256": runner.sha256(RECOVERY_LOG),
            "scene_pipeline_pids_resumed": _resume(paused_scene),
            "finalizer_v7": _start_finalizer_v7(),
            "state": "waiting_for_scene05_completion",
            "heartbeat_utc": datetime.now(UTC).isoformat(),
        }
    )
    _write(audit)

    while True:
        if RECEIPT.is_file():
            receipt = runner.read_json(RECEIPT)
            if (
                receipt.get("state") == "completed"
                and receipt.get("passed") is True
            ):
                break
        if all(_process_state(pid) in {None, "Z"} for pid in paused_scene):
            audit["state"] = "terminal_failure"
            audit["failed_utc"] = datetime.now(UTC).isoformat()
            audit["error"] = "scene05 pipeline exited before a passed receipt"
            _write(audit)
            return 2
        audit["heartbeat_utc"] = datetime.now(UTC).isoformat()
        audit["receipt_exists"] = RECEIPT.is_file()
        _write(audit)
        time.sleep(30)

    deadline = time.monotonic() + 60
    while (
        time.monotonic() < deadline
        and any(_process_state(pid) not in {None, "Z"} for pid in paused_scene)
    ):
        time.sleep(1)
    old_all_scenes_termination = _terminate_stopped(paused_all_scenes)
    command_v5 = [
        sys.executable,
        str(
            ROOT
            / "scripts/run_kinofail_reconfirmation_all_scene_pipelines_v5.py"
        ),
    ]
    v5_pid = _launch_detached(command_v5, F21_ALL_SCENES_LOG)
    audit.update(
        {
            "state": "handoff_complete",
            "completed_utc": datetime.now(UTC).isoformat(),
            "passed": True,
            "scene05_receipt": str(RECEIPT),
            "scene05_receipt_sha256": runner.sha256(RECEIPT),
            "old_all_scenes_termination": old_all_scenes_termination,
            "all_scenes_v5": {
                "command": command_v5,
                "pid": v5_pid,
                "log": str(F21_ALL_SCENES_LOG),
            },
            "heartbeat_utc": datetime.now(UTC).isoformat(),
        }
    )
    _write(audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
