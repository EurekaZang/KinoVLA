#!/usr/bin/env python3
"""Resume the sealed F21 scene05 recovery after removing its orphan wrapper."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import recover_kinofail_reconfirmation_scene05_f21 as recovery
from scripts import run_kinofail_reconfirmation_t2_atomic_f21 as runner
from scripts import seal_kinofail_reconfirmation_f21 as f21_sealer
from scripts import seal_kinofail_reconfirmation_f22 as f22_sealer
from scripts.recover_kinofail_reconfirmation_t2_f14 import (
    _process_state,
    terminate_exact_processes,
)


def _validate_f22() -> dict[str, Any]:
    amendment = runner.read_json(f22_sealer.F22)
    correction = amendment.get("correction", {})
    scientific = amendment.get("scientific_contract", {})
    if (
        amendment.get("schema_version")
        != "kinofail.reconfirmation-f22-orphan-wrapper-amendment.v1"
        or amendment.get("status")
        != "sealed_after_f21_source_termination_before_f22_resume"
        or amendment.get("passed") is not True
        or amendment.get("predecessor_f21_sha256")
        != runner.sha256(runner.F21)
        or correction.get("resumer_f22_sha256")
        != runner.sha256(Path(__file__).resolve())
        or correction.get("sealer_f22_sha256")
        != runner.sha256(Path(f22_sealer.__file__).resolve())
        or correction.get("terminate_orphan_wrapper_before_atomic_resume")
        is not True
        or correction.get("f21_atomic_runner_unchanged") is not True
        or correction.get("f21_atomic_runner_sha256")
        != runner.sha256(Path(runner.__file__).resolve())
        or scientific.get("schedule_protocol_or_case_membership_changed")
        is not False
        or scientific.get("physics_sensor_render_or_seed_changed") is not False
        or scientific.get("existing_completed_case_reexecuted") is not False
        or scientific.get("feature_model_route_threshold_or_analysis_changed")
        is not False
        or scientific.get("result_dependent_retry_enabled") is not False
    ):
        raise RuntimeError("F22 orphan-wrapper amendment is invalid")
    runner._validate_f21()
    return amendment


def _write(audit: dict[str, Any]) -> None:
    runner.write_json(recovery.AUDIT_PATH, audit)


def _fail(audit: dict[str, Any], message: str, returncode: int = 2) -> int:
    audit.update(
        {
            "state": "terminal_failure",
            "failed_utc": datetime.now(UTC).isoformat(),
            "passed": False,
            "error": message,
            "heartbeat_utc": datetime.now(UTC).isoformat(),
        }
    )
    _write(audit)
    return returncode


def main() -> int:
    amendment = _validate_f22()
    audit = runner.read_json(recovery.AUDIT_PATH)
    incident = amendment["incident"]
    if (
        audit.get("state") != "started"
        or runner.sha256(recovery.AUDIT_PATH)
        != incident["f21_recovery_audit_sha256"]
        or recovery.RECOVERY_LOG.exists()
    ):
        raise RuntimeError("F22 recovery prestate is not sealed")

    all_scenes = recovery._exact_token_processes(
        "run_kinofail_reconfirmation_all_scene_pipelines_v4.py"
    )
    scene_pipelines = recovery._exact_token_processes(
        "run_kinofail_reconfirmation_scene_pipeline_v4.py",
        "--scene",
        recovery.SCENE,
    )
    orphan = recovery._exact_token_processes(
        "run_kinofail_reconfirmation_slotted_t2_v1.py",
        "--scene",
        recovery.SCENE,
    )
    if (
        len(all_scenes) != 1
        or len(scene_pipelines) != 2
        or len(orphan) != 1
        or any(
            row["state"] != "T"
            for row in all_scenes + scene_pipelines
        )
    ):
        raise RuntimeError("F22 live process topology differs from seal")

    audit["f22_orphan_wrapper_amendment"] = {
        "path": str(f22_sealer.F22),
        "sha256": runner.sha256(f22_sealer.F22),
        "status": amendment["status"],
    }
    audit["orphan_slotted_t2_termination"] = terminate_exact_processes(orphan)
    recovery._wait_absent(
        "run_kinofail_reconfirmation_slotted_t2_v1.py",
        "--scene",
        recovery.SCENE,
    )
    audit["state"] = "f22_atomic_t2_recovery"
    audit["heartbeat_utc"] = datetime.now(UTC).isoformat()
    _write(audit)

    command = [
        sys.executable,
        str(Path(runner.__file__).resolve()),
        "--schedule",
        str(f21_sealer.SCHEDULE),
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
        str(recovery.OUT),
        "--scene",
        recovery.SCENE,
        "--standalone-recovery",
    ]
    recovery.RECOVERY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with recovery.RECOVERY_LOG.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    atomic_audit_path = (
        recovery.OUT
        / "launcher_audits"
        / f"{recovery.SCENE}_f21_atomic.json"
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
        audit["t2_recovery_returncode"] = int(completed.returncode)
        return _fail(audit, "F22 atomic T2 recovery did not pass")

    paused_scene = [int(row["pid"]) for row in scene_pipelines]
    audit.update(
        {
            "t2_recovery_state": "terminal",
            "t2_recovery_passed": True,
            "t2_recovery_returncode": int(completed.returncode),
            "atomic_t2_audit": str(atomic_audit_path),
            "atomic_t2_audit_sha256": runner.sha256(atomic_audit_path),
            "recovery_log": str(recovery.RECOVERY_LOG),
            "recovery_log_sha256": runner.sha256(recovery.RECOVERY_LOG),
            "scene_pipeline_pids_resumed": recovery._resume(paused_scene),
            "finalizer_v7": recovery._start_finalizer_v7(),
            "state": "waiting_for_scene05_completion",
            "heartbeat_utc": datetime.now(UTC).isoformat(),
        }
    )
    _write(audit)

    while True:
        if recovery.RECEIPT.is_file():
            receipt = runner.read_json(recovery.RECEIPT)
            if (
                receipt.get("state") == "completed"
                and receipt.get("passed") is True
            ):
                break
        if all(
            _process_state(pid) in {None, "Z"} for pid in paused_scene
        ):
            return _fail(
                audit,
                "scene05 pipeline exited before a passed receipt",
            )
        audit["heartbeat_utc"] = datetime.now(UTC).isoformat()
        audit["receipt_exists"] = recovery.RECEIPT.is_file()
        _write(audit)
        time.sleep(30)

    deadline = time.monotonic() + 60
    while (
        time.monotonic() < deadline
        and any(
            _process_state(pid) not in {None, "Z"} for pid in paused_scene
        )
    ):
        time.sleep(1)
    old_all_scenes_termination = recovery._terminate_stopped(
        [int(row["pid"]) for row in all_scenes]
    )
    command_v5 = [
        sys.executable,
        str(
            ROOT
            / "scripts/run_kinofail_reconfirmation_all_scene_pipelines_v5.py"
        ),
    ]
    v5_pid = recovery._launch_detached(
        command_v5, recovery.F21_ALL_SCENES_LOG
    )
    audit.update(
        {
            "state": "handoff_complete",
            "completed_utc": datetime.now(UTC).isoformat(),
            "passed": True,
            "scene05_receipt": str(recovery.RECEIPT),
            "scene05_receipt_sha256": runner.sha256(recovery.RECEIPT),
            "old_all_scenes_termination": old_all_scenes_termination,
            "all_scenes_v5": {
                "command": command_v5,
                "pid": v5_pid,
                "log": str(recovery.F21_ALL_SCENES_LOG),
            },
            "heartbeat_utc": datetime.now(UTC).isoformat(),
        }
    )
    _write(audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
