#!/usr/bin/env python3
"""Resumable, count-bound Isaac experiment supervisor."""

from __future__ import annotations

import argparse
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


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--infrastructure-retries", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 4:
        raise ValueError("max-workers must be in [1,4]")

    protocol_path = args.protocol.resolve()
    protocol = load(protocol_path)
    schedule_path = Path(str(protocol["schedule"])).resolve()
    registry_path = Path(str(protocol["scene_registry"])).resolve()
    collector = ROOT / str(protocol["collector"])
    required = (schedule_path, registry_path, collector, PYTHON)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError([str(path) for path in required if not path.is_file()])

    cases = {str(row["case_id"]): row for row in jsonl(schedule_path)}
    if len(cases) != int(protocol["counts"]["physical_cases"]):
        raise RuntimeError("schedule count mismatch")

    state_root = args.state_root.resolve()
    corpus_root = args.corpus_root.resolve()
    logs = state_root / "logs"
    attempts_root = state_root / "attempts"
    logs.mkdir(parents=True, exist_ok=True)
    attempts_root.mkdir(parents=True, exist_ok=True)
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
    pending = [(case_id, row) for case_id, row in sorted(cases.items()) if case_id not in completed]
    started = time.monotonic()

    def run_case(case_id: str, row: dict[str, Any]) -> dict[str, Any]:
        scene = str(row["scene_id"])
        summary = summary_path(case_id, row)
        case_started = time.monotonic()
        history: list[dict[str, Any]] = []
        for attempt in range(1, args.infrastructure_retries + 2):
            existing_partial = (summary.parent / "results.jsonl").is_file()
            command = [
                str(PYTHON),
                str(collector),
                "--schedule", str(schedule_path),
                "--protocol", str(protocol_path),
                "--scene-registry", str(registry_path),
                "--scene", scene,
                "--case-id", case_id,
                "--out", str(corpus_root),
                "--headless",
            ]
            if "horizon_steps" in protocol:
                command.extend(["--horizon-steps", str(int(protocol["horizon_steps"]))])
            if existing_partial:
                command.append("--resume")
            log_path = logs / f"{case_id}__attempt_{attempt}.log"
            with log_path.open("ab", buffering=0) as log:
                try:
                    process = subprocess.run(
                        command,
                        cwd=ROOT,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout_s,
                        check=False,
                    )
                    returncode = int(process.returncode)
                except subprocess.TimeoutExpired:
                    returncode = 124
            scientific_terminal = summary.is_file()
            accepted = scientific_terminal and load(summary).get("passed") is True
            history.append(
                {
                    "attempt": attempt,
                    "returncode": returncode,
                    "scientific_terminal": scientific_terminal,
                    "accepted": accepted,
                    "log": str(log_path),
                }
            )
            if scientific_terminal:
                break
        record = {
            "schema_version": "kinofail.experiment-supervisor-attempt.v1",
            "case_id": case_id,
            "scene_id": scene,
            "operator": row["operator"],
            "state": (
                "terminal_accepted"
                if summary.is_file() and load(summary).get("passed") is True
                else "terminal_scientific_attrition"
                if summary.is_file()
                else "terminal_infrastructure_failure"
            ),
            "attempts": history,
            "elapsed_s": time.monotonic() - case_started,
            "finished_utc": datetime.now(UTC).isoformat(),
            "summary": str(summary),
        }
        write(attempts_root / f"{case_id}.json", record)
        return record

    terminal: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_case, case_id, row): case_id for case_id, row in pending}
        for future in as_completed(futures):
            record = future.result()
            terminal.append(record)
            states = [row["state"] for row in terminal]
            supervisor = {
                "schema_version": "kinofail.experiment-supervisor.v1",
                "state": "running",
                "planned_cases": len(cases),
                "accepted_cases": len(completed) + states.count("terminal_accepted"),
                "scientific_attrition_cases": states.count("terminal_scientific_attrition"),
                "infrastructure_failure_cases": states.count("terminal_infrastructure_failure"),
                "terminal_cases": len(completed) + len(terminal),
                "pending_cases": len(cases) - len(completed) - len(terminal),
                "maximum_concurrent_isaac_processes": args.max_workers,
                "last_case_id": record["case_id"],
                "elapsed_s": time.monotonic() - started,
                "updated_utc": datetime.now(UTC).isoformat(),
            }
            write(state_root / "supervisor_state.json", supervisor)
            print(json.dumps(supervisor, sort_keys=True), flush=True)

    summaries = [
        load(summary_path(case_id, row))
        for case_id, row in cases.items()
        if summary_path(case_id, row).is_file()
    ]
    infrastructure_failures = [
        row["case_id"] for row in terminal if row["state"] == "terminal_infrastructure_failure"
    ]
    final = {
        "schema_version": "kinofail.experiment-collection-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "terminal",
        "passed": not infrastructure_failures and len(summaries) == len(cases),
        "counts": {
            "planned_cases": len(cases),
            "terminal_summaries": len(summaries),
            "accepted_cases": sum(row.get("passed") is True for row in summaries),
            "scientific_attrition_cases": sum(row.get("passed") is not True for row in summaries),
            "infrastructure_failure_cases": len(infrastructure_failures),
            "physical_episodes": sum(int(row.get("completed_episodes", 0)) for row in summaries),
        },
        "infrastructure_failure_case_ids": infrastructure_failures,
        "unfavorable_outcomes_retained": True,
        "result_dependent_parameter_change": False,
        "corpus_root": str(corpus_root),
    }
    write(state_root / "final_audit.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0 if final["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
