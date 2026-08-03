#!/usr/bin/env python3
"""Restart only interrupted orchestration watchers, never scientific attempts."""

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
OUTPUT = ROOT / "outputs/kinofail_a0_a7_unattended_watchdog_v1"
WATCHERS = (
    {
        "name": "after_f33_to_a4_pilot",
        "script": ROOT / "scripts/continue_kinofail_after_f33_to_a4_pilot_v1.py",
        "state": ROOT / "outputs/kinofail_after_f33_to_a4_pilot_v1/state.json",
        "terminal": {"terminal_success", "terminal_failure"},
    },
    {
        "name": "after_a4_pilot_to_a0_a7",
        "script": ROOT / "scripts/continue_kinofail_after_a4_pilot_to_a0_a7_v1.py",
        "state": ROOT / "outputs/kinofail_after_a4_pilot_to_a0_a7_v3/state.json",
        "terminal": {
            "terminal_success",
            "terminal_scientific_gate_failed",
            "terminal_failure",
        },
    },
)


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) else {}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def matching_pids(script: Path) -> list[int]:
    target = script.resolve()
    pids: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = [
                item.decode(errors="surrogateescape")
                for item in (entry / "cmdline").read_bytes().split(b"\0")
                if item
            ]
            process_cwd = (entry / "cwd").resolve()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for argument in command[1:]:
            candidate = Path(argument)
            if candidate.suffix != ".py":
                continue
            if not candidate.is_absolute():
                candidate = process_cwd / candidate
            try:
                if candidate.resolve() == target:
                    pids.append(int(entry.name))
                    break
            except (OSError, RuntimeError):
                continue
    return sorted(pids)


def launch(row: dict[str, Any]) -> int:
    log = OUTPUT / f"{row['name']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("ab", buffering=0)
    process = subprocess.Popen(
        [str(PYTHON), str(row["script"])],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1"},
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    return int(process.pid)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    while True:
        status = []
        all_terminal = True
        for row in WATCHERS:
            state = load(row["state"])
            terminal = str(state.get("state", "")) in row["terminal"]
            pids = matching_pids(row["script"])
            launched_pid = None
            if not terminal:
                all_terminal = False
                if not pids:
                    launched_pid = launch(row)
                    pids = [launched_pid]
            status.append(
                {
                    "name": row["name"],
                    "stage": state.get("stage"),
                    "state": state.get("state"),
                    "terminal": terminal,
                    "pids": pids,
                    "launched_pid": launched_pid,
                }
            )
        atomic_json(
            OUTPUT / "state.json",
            {
                "schema_version": "kinofail.a0-a7-unattended-watchdog.v1",
                "updated_utc": datetime.now(UTC).isoformat(),
                "state": "terminal" if all_terminal else "watching",
                "watchers": status,
                "scientific_attempt_restart_authorized": False,
                "orchestration_process_restart_only": True,
            },
        )
        if all_terminal:
            return 0
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
