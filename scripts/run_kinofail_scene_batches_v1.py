#!/usr/bin/env python3
"""Run a count-bound KiNO-Fail schedule in one Isaac process per scene."""

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
    parser.add_argument("--timeout-s", type=int, default=7200)
    parser.add_argument("--infrastructure-retries", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 4:
        raise ValueError("max-workers must be in [1,4]")

    protocol_path = args.protocol.resolve()
    protocol = load(protocol_path)
    schedule = Path(str(protocol["schedule"])).resolve()
    registry = Path(str(protocol["scene_registry"])).resolve()
    collector = ROOT / str(protocol["collector"])
    cases = jsonl(schedule)
    if len(cases) != int(protocol["counts"]["physical_cases"]):
        raise RuntimeError("schedule count mismatch")
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_scene.setdefault(str(case["scene_id"]), []).append(case)
    if len(by_scene) != int(protocol["counts"]["scenes"]):
        raise RuntimeError("scene count mismatch")
    if len({len(rows) for rows in by_scene.values()}) != 1:
        raise RuntimeError("scene batches are unbalanced")

    state_root = args.state_root.resolve()
    corpus_root = args.corpus_root.resolve()
    logs = state_root / "logs"
    attempts = state_root / "attempts"
    logs.mkdir(parents=True, exist_ok=True)
    attempts.mkdir(parents=True, exist_ok=True)
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

    def summary_path(scene: str) -> Path:
        return corpus_root / scene / "summary.json"

    completed = {
        scene
        for scene in by_scene
        if summary_path(scene).is_file() and load(summary_path(scene)).get("passed") is True
    }
    pending = [scene for scene in sorted(by_scene) if scene not in completed]
    started = time.monotonic()

    def run_scene(scene: str) -> dict[str, Any]:
        summary = summary_path(scene)
        history: list[dict[str, Any]] = []
        scene_started = time.monotonic()
        for attempt in range(1, args.infrastructure_retries + 2):
            command = [
                str(PYTHON), str(collector),
                "--schedule", str(schedule),
                "--protocol", str(protocol_path),
                "--scene-registry", str(registry),
                "--scene", scene,
                "--out", str(corpus_root),
                "--headless",
            ]
            if "horizon_steps" in protocol:
                command.extend(["--horizon-steps", str(int(protocol["horizon_steps"]))])
            if (corpus_root / scene / "results.jsonl").is_file():
                command.append("--resume")
            log_path = logs / f"{scene}__attempt_{attempt}.log"
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
            "schema_version": "kinofail.scene-batch-attempt.v1",
            "scene_id": scene,
            "state": (
                "terminal_accepted"
                if summary.is_file() and load(summary).get("passed") is True
                else "terminal_scientific_attrition"
                if summary.is_file()
                else "terminal_infrastructure_failure"
            ),
            "attempts": history,
            "elapsed_s": time.monotonic() - scene_started,
            "finished_utc": datetime.now(UTC).isoformat(),
            "summary": str(summary),
        }
        write(attempts / f"{scene}.json", record)
        return record

    terminal: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_scene, scene): scene for scene in pending}
        for future in as_completed(futures):
            record = future.result()
            terminal.append(record)
            states = [row["state"] for row in terminal]
            supervisor = {
                "schema_version": "kinofail.scene-batch-supervisor.v1",
                "state": "running",
                "planned_scenes": len(by_scene),
                "accepted_scenes": len(completed) + states.count("terminal_accepted"),
                "scientific_attrition_scenes": states.count("terminal_scientific_attrition"),
                "infrastructure_failure_scenes": states.count("terminal_infrastructure_failure"),
                "terminal_scenes": len(completed) + len(terminal),
                "pending_scenes": len(by_scene) - len(completed) - len(terminal),
                "maximum_concurrent_isaac_processes": args.max_workers,
                "last_scene_id": record["scene_id"],
                "elapsed_s": time.monotonic() - started,
                "updated_utc": datetime.now(UTC).isoformat(),
            }
            write(state_root / "supervisor_state.json", supervisor)
            print(json.dumps(supervisor, sort_keys=True), flush=True)

    summaries = [load(summary_path(scene)) for scene in by_scene if summary_path(scene).is_file()]
    infrastructure_failures = [
        row["scene_id"] for row in terminal if row["state"] == "terminal_infrastructure_failure"
    ]
    final = {
        "schema_version": "kinofail.scene-batch-collection-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "terminal",
        "passed": not infrastructure_failures and len(summaries) == len(by_scene),
        "counts": {
            "planned_scenes": len(by_scene),
            "terminal_summaries": len(summaries),
            "accepted_scenes": sum(row.get("passed") is True for row in summaries),
            "scientific_attrition_scenes": sum(row.get("passed") is not True for row in summaries),
            "infrastructure_failure_scenes": len(infrastructure_failures),
            "physical_cases": len(cases),
            "physical_episodes": sum(int(row.get("completed_episodes", 0)) for row in summaries),
        },
        "infrastructure_failure_scene_ids": infrastructure_failures,
        "unfavorable_outcomes_retained": True,
        "result_dependent_parameter_change": False,
        "corpus_root": str(corpus_root),
    }
    write(state_root / "final_audit.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0 if final["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
