#!/usr/bin/env python3
"""Keep all three F24 workers alive and write an auditable live status."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
WORKER = ROOT / "scripts/run_kinofail_replenishment_f24_worker.py"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _counts(corpus_root: Path, planned: int) -> dict[str, Any]:
    attempts = list((corpus_root / "attempts").glob("*.json"))
    states = [_json(path) for path in attempts]
    terminal = [row for row in states if row.get("state") == "terminal"]
    passed = [row for row in terminal if row.get("passed") is True]
    started = [row for row in states if row.get("state") == "started"]
    return {
        "planned_pairs": planned,
        "attempt_records": len(attempts),
        "started_pairs": len(started),
        "terminal_pairs": len(terminal),
        "passed_pairs": len(passed),
        "failed_pairs": len(terminal) - len(passed),
        "unstarted_pairs": planned - len(attempts),
        "successful_replacements_needed_for_strict_below_5_percent": 890,
        "successes_remaining_to_target": max(0, 890 - len(passed)),
        "projected_remaining_scale_attrition": 1_417 - len(passed),
        "projected_scale_attrition_rate": (1_417 - len(passed)) / 10_560,
        "projected_overall_attrition_rate": (1_531 - len(passed)) / 13_560,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--timeout-s", type=int, default=600)
    args = parser.parse_args()
    freeze = args.freeze.resolve()
    corpus_root = args.corpus_root.resolve()
    state_path = args.state.resolve()
    corpus_root.mkdir(parents=True, exist_ok=True)
    log_root = corpus_root / "worker_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    planned = int(_json(freeze)["scheduled_pairs"])

    processes: dict[int, subprocess.Popen[str]] = {}
    logs: dict[int, Any] = {}
    restarts = {index: 0 for index in range(3)}

    def launch(index: int) -> None:
        log = (log_root / f"worker_{index}.log").open(
            "a", encoding="utf-8", buffering=1
        )
        process = subprocess.Popen(
            [
                str(PYTHON),
                str(WORKER),
                "--freeze",
                str(freeze),
                "--corpus-root",
                str(corpus_root),
                "--worker-index",
                str(index),
                "--worker-count",
                "3",
                "--timeout-s",
                str(args.timeout_s),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        processes[index] = process
        logs[index] = log

    for index in range(3):
        launch(index)

    while True:
        worker_status: dict[str, Any] = {}
        all_complete = True
        for index in range(3):
            process = processes[index]
            returncode = process.poll()
            worker_status[str(index)] = {
                "pid": process.pid,
                "returncode": returncode,
                "restart_count": restarts[index],
                "log": str(log_root / f"worker_{index}.log"),
            }
            if returncode is None:
                all_complete = False
                continue
            logs[index].close()
            if returncode != 0:
                restarts[index] += 1
                launch(index)
                worker_status[str(index)]["relaunched"] = True
                all_complete = False
        status = {
            "schema_version": "kinofail.f24-supervisor-state.v1",
            "updated_utc": datetime.now(UTC).isoformat(),
            "supervisor_pid": os.getpid(),
            "freeze": str(freeze),
            "corpus_root": str(corpus_root),
            "workers": worker_status,
            "counts": _counts(corpus_root, planned),
            "state": "completed" if all_complete else "running",
        }
        _atomic_json(state_path, status)
        print(json.dumps(status, sort_keys=True), flush=True)
        if all_complete:
            break
        time.sleep(20.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
