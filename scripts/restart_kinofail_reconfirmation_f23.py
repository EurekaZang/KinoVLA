#!/usr/bin/env python3
"""Restart unattended collection after the sealed F23 preflight correction."""

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

from scripts import recover_kinofail_reconfirmation_scene05_f21 as process
from scripts import run_kinofail_reconfirmation_scene_pipeline_v6 as pipeline
from scripts import run_kinofail_reconfirmation_t2_atomic_f21 as t2
from scripts.recover_kinofail_reconfirmation_t2_f14 import (
    terminate_exact_processes,
)


AUDIT_PATH = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/orchestration/"
    "f23_unattended_restart.json"
)
ALL_SCENES_LOG = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/orchestration_logs/f23/"
    "remaining_scenes_pipeline_v6.log"
)
FINALIZER_LOG = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/orchestration_logs/f23/"
    "finalizer_v8.log"
)


def _launch(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    child = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.close()
    return int(child.pid)


def main() -> int:
    amendment = pipeline._validate_f23()
    if AUDIT_PATH.exists():
        raise FileExistsError(AUDIT_PATH)
    if t2._prediction_files():
        raise RuntimeError("prediction or score exists before F23 restart")
    if process._exact_token_processes(
        "run_kinofail_reconfirmation_all_scene_pipelines_v5.py"
    ):
        raise RuntimeError("failed v5 all-scenes process is unexpectedly live")
    if process._exact_token_processes(
        "run_kinofail_reconfirmation_scene_pipeline_v5.py"
    ):
        raise RuntimeError("failed v5 scene process is unexpectedly live")

    old_finalizer = process._exact_token_processes(
        "finalize_kinofail_reconfirmation_v7.py",
        "--poll-seconds",
        "30",
    )
    termination = (
        terminate_exact_processes(old_finalizer) if old_finalizer else {}
    )
    process._wait_absent(
        "finalize_kinofail_reconfirmation_v7.py",
        "--poll-seconds",
        "30",
    )

    all_command = [
        sys.executable,
        str(
            ROOT
            / "scripts/run_kinofail_reconfirmation_all_scene_pipelines_v6.py"
        ),
    ]
    finalizer_command = [
        sys.executable,
        str(ROOT / "scripts/finalize_kinofail_reconfirmation_v8.py"),
        "--poll-seconds",
        "30",
    ]
    all_pid = _launch(all_command, ALL_SCENES_LOG)
    finalizer_pid = _launch(finalizer_command, FINALIZER_LOG)
    time.sleep(2)
    if (
        not (Path("/proc") / str(all_pid)).exists()
        or not (Path("/proc") / str(finalizer_pid)).exists()
    ):
        raise RuntimeError("F23 successor failed immediately after launch")
    audit: dict[str, Any] = {
        "schema_version": "kinofail.reconfirmation-f23-restart.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "state": "successors_running",
        "passed": True,
        "amendment": str(pipeline.F23),
        "amendment_sha256": t2.sha256(pipeline.F23),
        "amendment_status": amendment["status"],
        "old_finalizer_termination": termination,
        "all_scenes_v6": {
            "command": all_command,
            "pid": all_pid,
            "log": str(ALL_SCENES_LOG),
        },
        "finalizer_v8": {
            "command": finalizer_command,
            "pid": finalizer_pid,
            "log": str(FINALIZER_LOG),
        },
        "scene06_reexecution": False,
        "model_feature_prediction_score_or_label_read": False,
        "scientific_content_changed": False,
    }
    t2.write_json(AUDIT_PATH, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
