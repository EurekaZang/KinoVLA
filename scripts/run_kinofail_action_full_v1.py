#!/usr/bin/env python3
"""Run a hash-bound KiNO-Fail full-action schedule with monitored subprocesses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 4:
        raise ValueError("max-workers must be in [1,4]")

    protocol_path = args.protocol.resolve()
    protocol = load(protocol_path)
    schedule_path = Path(str(protocol["schedule"])).resolve()
    registry_path = Path(str(protocol["scene_registry"])).resolve()
    collector = ROOT / str(protocol["collector"])
    if (
        protocol.get("schedule_sha256") != sha256(schedule_path)
        or protocol.get("scene_registry_sha256") != sha256(registry_path)
        or protocol.get("collector_sha256") != sha256(collector)
        or protocol.get("base_collector_sha256")
        != sha256(ROOT / str(protocol["base_collector"]))
    ):
        raise RuntimeError("full-action protocol hash binding failed")
    cases = {str(row["case_id"]): row for row in jsonl(schedule_path)}
    if len(cases) != int(protocol["counts"]["physical_cases"]):
        raise RuntimeError("full-action schedule count mismatch")

    state_root = args.state_root.resolve()
    corpus_root = args.corpus_root.resolve()
    attempts = state_root / "attempts"
    logs = state_root / "logs"
    attempts.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    corpus_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": str(ROOT),
            "TERM": "xterm-256color",
        }
    )

    def summary_path(case_id: str, row: dict[str, Any]) -> Path:
        return corpus_root / str(row["scene_id"]) / case_id / "summary.json"

    completed = {
        case_id
        for case_id, row in cases.items()
        if summary_path(case_id, row).is_file()
        and load(summary_path(case_id, row)).get("passed") is True
    }
    pending = [
        (case_id, row)
        for case_id, row in sorted(cases.items())
        if case_id not in completed
    ]
    for case_id, _row in pending:
        if (attempts / f"{case_id}.json").exists():
            raise RuntimeError(f"recorded attempt forbids silent retry: {case_id}")

    started = time.monotonic()

    def run_case(case_id: str, row: dict[str, Any]) -> dict[str, Any]:
        scene = str(row["scene_id"])
        summary = summary_path(case_id, row)
        attempt_path = attempts / f"{case_id}.json"
        log_path = logs / f"{case_id}.log"
        command = [
            str(PYTHON),
            str(collector),
            "--schedule",
            str(schedule_path),
            "--protocol",
            str(protocol_path),
            "--scene-registry",
            str(registry_path),
            "--scene",
            scene,
            "--case-id",
            case_id,
            "--out",
            str(corpus_root),
            "--headless",
        ]
        record = {
            "schema_version": "kinofail.action-full-v1-attempt.v1",
            "case_id": case_id,
            "scene_id": scene,
            "operator": row["operator"],
            "state": "started",
            "started_utc": datetime.now(UTC).isoformat(),
            "command": command,
            "result_dependent_retry_permitted": False,
        }
        write(attempt_path, record)
        case_started = time.monotonic()
        with log_path.open("ab", buffering=0) as log:
            try:
                completed_process = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout_s,
                    check=False,
                )
                returncode = int(completed_process.returncode)
            except subprocess.TimeoutExpired:
                returncode = 124
        accepted = summary.is_file() and load(summary).get("passed") is True
        record.update(
            {
                "state": "terminal_accepted" if accepted else "terminal_failure",
                "returncode": returncode,
                "elapsed_s": time.monotonic() - case_started,
                "finished_utc": datetime.now(UTC).isoformat(),
                "summary": str(summary),
                "summary_sha256": sha256(summary) if summary.is_file() else None,
            }
        )
        write(attempt_path, record)
        return record

    terminal: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_case, case_id, row): case_id
            for case_id, row in pending
        }
        for future in as_completed(futures):
            record = future.result()
            terminal.append(record)
            accepted_count = len(completed) + sum(
                row["state"] == "terminal_accepted" for row in terminal
            )
            failure_count = sum(
                row["state"] == "terminal_failure" for row in terminal
            )
            write(
                state_root / "supervisor_state.json",
                {
                    "schema_version": "kinofail.action-full-v1-supervisor.v1",
                    "state": "running",
                    "planned_cases": len(cases),
                    "accepted_cases": accepted_count,
                    "failed_cases": failure_count,
                    "terminal_cases": len(completed) + len(terminal),
                    "pending_cases": len(cases) - len(completed) - len(terminal),
                    "maximum_concurrent_isaac_processes": args.max_workers,
                    "last_case_id": record["case_id"],
                    "elapsed_s": time.monotonic() - started,
                    "updated_utc": datetime.now(UTC).isoformat(),
                },
            )
            print(json.dumps(record, sort_keys=True), flush=True)

    summaries = [
        load(summary_path(case_id, row))
        for case_id, row in cases.items()
        if summary_path(case_id, row).is_file()
    ]
    failed = [row for row in terminal if row["state"] == "terminal_failure"]
    audit = {
        "schema_version": "kinofail.action-full-v1-collection-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "terminal",
        "passed": len(summaries) == len(cases)
        and not failed
        and all(row.get("passed") is True for row in summaries),
        "development_only": bool(protocol.get("development_only")),
        "counts": {
            "planned_cases": len(cases),
            "terminal_summaries": len(summaries),
            "accepted_cases": sum(row.get("passed") is True for row in summaries),
            "failed_cases": len(failed),
            "physical_episodes": sum(
                int(row.get("completed_episodes", 0)) for row in summaries
            ),
        },
        "failed_case_ids": [row["case_id"] for row in failed],
        "maximum_concurrent_isaac_processes": args.max_workers,
        "unfavorable_outcomes_retained": True,
        "result_dependent_retry_or_selection": False,
        "source_sha256": {
            "protocol": sha256(protocol_path),
            "schedule": sha256(schedule_path),
            "scene_registry": sha256(registry_path),
            "collector": sha256(collector),
            "runner": sha256(Path(__file__).resolve()),
        },
        "corpus_root": str(corpus_root),
    }
    write(state_root / "final_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
