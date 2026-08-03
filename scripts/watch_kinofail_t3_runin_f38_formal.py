#!/usr/bin/env python3
"""Restart only the resumable F38 supervisor, never a terminal pair attempt."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
RUNNER = ROOT / "scripts/run_kinofail_t3_runin_f38_collection_v2.py"
SEAL = ROOT / "outputs/freeze/kinofail_t3_runin_f38_formal/seal_manifest.json"
AMENDMENT = ROOT / "outputs/freeze/kinofail_t3_runin_f38_watchdog_amendment1/amendment_manifest.json"
OUTPUT = ROOT / "outputs/kinofail_t3_runin_f38_watchdog"
FINAL = ROOT / "outputs/kinofail_t3_runin_f38_formal/final_audit.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) else {}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def matching_pids() -> list[int]:
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, PermissionError):
            continue
        if RUNNER.name in command and str(SEAL.relative_to(ROOT)) in command:
            pids.append(int(entry.name))
    return sorted(pids)


def launch() -> int:
    log = OUTPUT / "runner.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("ab", buffering=0)
    process = subprocess.Popen(
        [str(PYTHON), str(RUNNER), "--seal", str(SEAL)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1"},
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    return int(process.pid)


def main() -> int:
    amendment = read_json(AMENDMENT)
    if (
        amendment.get("status") != "sealed_before_formal_collection_supervisor_launch"
        or amendment.get("passed") is not True
        or amendment.get("scientific_attempt_restart_authorized") is not False
    ):
        raise RuntimeError("invalid F38 watchdog amendment")
    while True:
        final = read_json(FINAL)
        terminal = bool(final)
        pids = matching_pids()
        launched = None
        if not terminal and not pids:
            launched = launch()
            pids = [launched]
        atomic_json(
            OUTPUT / "state.json",
            {
                "schema_version": "kinofail.t3-runin-f38-watchdog-state.v1",
                "updated_utc": datetime.now(UTC).isoformat(),
                "state": "terminal" if terminal else "watching",
                "runner_pids": pids,
                "launched_pid": launched,
                "final_passed": final.get("passed") if terminal else None,
                "scientific_attempt_restart_authorized": False,
                "orchestration_supervisor_restart_only": True,
            },
        )
        if terminal:
            return 0
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
