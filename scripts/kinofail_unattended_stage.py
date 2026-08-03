#!/usr/bin/env python3
"""Crash-safe write-once subprocess stages for unattended confirmations.

The persistent stage record lets a restarted orchestration process wait for an
already-running child instead of launching a duplicate.  A child whose exit
status can no longer be reaped is accepted only when every predeclared output
artifact is present and passes its structural check.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


StateWriter = Callable[..., None]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _process_start_ticks(pid: int) -> int | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    tail = raw.rpartition(") ")[2].split()
    if len(tail) <= 19:
        return None
    return int(tail[19])


def _process_command(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return [item.decode(errors="surrogateescape") for item in raw.split(b"\0") if item]


def _same_process(record: Mapping[str, Any]) -> bool:
    pid = int(record.get("pid", -1))
    if pid <= 0:
        return False
    return (
        _process_start_ticks(pid) == int(record.get("pid_start_ticks", -1))
        and _process_command(pid) == list(record.get("command", []))
    )


def _lookup(value: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = value
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(dotted_key)
        current = current[part]
    return current


def artifacts_pass(specifications: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    """Check predeclared JSON artifacts without interpreting experiment scores."""

    issues: list[str] = []
    for specification in specifications:
        path = Path(str(specification["path"]))
        if not path.is_file():
            issues.append(f"missing:{path}")
            continue
        try:
            value = _read_json(path)
        except (OSError, ValueError, TypeError) as exc:
            issues.append(f"invalid_json:{path}:{type(exc).__name__}")
            continue
        key = specification.get("key")
        if key is None:
            continue
        try:
            observed = _lookup(value, str(key))
        except KeyError:
            issues.append(f"missing_key:{path}:{key}")
            continue
        allowed = list(specification.get("allowed", []))
        if observed not in allowed:
            issues.append(f"unexpected_value:{path}:{key}:{observed!r}")
    return not issues, issues


def run_write_once_stage(
    *,
    root: Path,
    state_root: Path,
    name: str,
    command: Sequence[str],
    environment: Mapping[str, str],
    write_state: StateWriter,
    artifacts: Sequence[Mapping[str, Any]],
    allowed_returncodes: Iterable[int] = (0,),
    forbid_fresh_paths: Sequence[Path] = (),
    poll_seconds: float = 10.0,
) -> int:
    """Run or reattach to a stage, never launching the same stage twice."""

    command = [str(item) for item in command]
    allowed = {int(value) for value in allowed_returncodes}
    record_path = state_root / "runs" / f"{name}.json"
    log_path = state_root / "logs" / f"{name}.log"
    record: dict[str, Any] | None = None
    if record_path.is_file():
        record = _read_json(record_path)
        if list(record.get("command", [])) != command:
            raise RuntimeError(f"write-once stage command drift: {name}")
        was_running = record.get("state") == "running"
        if was_running and _same_process(record):
            write_state(
                name,
                "reattached_running_child",
                pid=record["pid"],
                command=command,
                log=str(log_path),
            )
            while _same_process(record):
                time.sleep(poll_seconds)
        known_returncode = record.get("returncode")
        if not was_running and known_returncode is not None and int(known_returncode) not in allowed:
            write_state(
                name,
                "terminal_failure",
                reason="recorded stage has a disallowed reaped exit status",
                returncode=int(known_returncode),
                record=str(record_path),
            )
            raise RuntimeError(
                f"recorded write-once stage failed: {name}: returncode={known_returncode}"
            )
        passed, issues = artifacts_pass(artifacts)
        if not passed:
            write_state(
                name,
                "terminal_failure",
                reason="recorded stage exited without complete valid artifacts",
                artifact_issues=issues,
                record=str(record_path),
            )
            raise RuntimeError(f"interrupted write-once stage is incomplete: {name}: {issues}")
        reconciled = {
            **record,
            "state": "completed_reconciled_from_artifacts",
            "reconciled_utc": datetime.now(UTC).isoformat(),
            "exit_status_reaped_by_current_orchestrator": False,
        }
        _atomic_json(record_path, reconciled)
        write_state(name, "completed_reconciled", artifacts=[str(row["path"]) for row in artifacts])
        return int(record.get("returncode", 0) or 0)

    if log_path.exists():
        raise RuntimeError(f"stage log exists without a persistent stage record: {log_path}")
    existing = [str(path) for path in forbid_fresh_paths if path.exists()]
    if existing:
        raise RuntimeError(f"partial write-once output exists before a recorded launch: {existing}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=dict(environment),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        start_ticks = _process_start_ticks(process.pid)
        if start_ticks is None:
            process.terminate()
            process.wait()
            raise RuntimeError(f"could not certify child process identity: {name}")
        record = {
            "schema_version": "kinofail.unattended-write-once-stage.v1",
            "name": name,
            "state": "running",
            "started_utc": datetime.now(UTC).isoformat(),
            "pid": int(process.pid),
            "pid_start_ticks": int(start_ticks),
            "command": command,
            "log": str(log_path),
            "artifacts": [dict(row) for row in artifacts],
            "allowed_returncodes": sorted(allowed),
            "result_dependent_retry": False,
        }
        _atomic_json(record_path, record)
        write_state(name, "running", pid=process.pid, command=command, log=str(log_path))
        returncode = int(process.wait())

    terminal = {
        **record,
        "state": "terminal",
        "completed_utc": datetime.now(UTC).isoformat(),
        "returncode": returncode,
        "exit_status_reaped_by_current_orchestrator": True,
    }
    _atomic_json(record_path, terminal)
    passed, issues = artifacts_pass(artifacts)
    if returncode not in allowed or not passed:
        write_state(
            name,
            "terminal_failure",
            returncode=returncode,
            artifact_issues=issues,
            log=str(log_path),
        )
        raise RuntimeError(
            f"{name} failed: returncode={returncode}, artifact_issues={issues}: {log_path}"
        )
    state = "completed" if returncode == 0 else "completed_scientific_gate_failed"
    write_state(name, state, returncode=returncode, log=str(log_path))
    return returncode
