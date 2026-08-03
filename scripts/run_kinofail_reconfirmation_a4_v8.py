#!/usr/bin/env python3
"""Run the sealed A4-v8 cases serially with one attempt per physical case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
SEAL = ROOT / "outputs/freeze/kinofail_reconfirmation_a4_v8/seal_manifest.json"
REGISTRY = ROOT / "outputs/kinofail_reconfirmation_v2/scene_registry.json"
STATE_ROOT = ROOT / "outputs/kinofail_reconfirmation_a4_v8/run"
CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_reconfirmation_a4_v8/corpus")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=1800)
    args = parser.parse_args()
    if args.max_workers != 1:
        raise RuntimeError("A4-v8 is frozen to exactly one Isaac process")
    seal = load(SEAL)
    sidecar = SEAL.with_name("seal_manifest.sha256")
    schedule = ROOT / str(seal["schedule"])
    collector = ROOT / str(seal["collector"])
    if (
        not sidecar.is_file()
        or sidecar.read_text().split()[0] != sha256(SEAL)
        or seal.get("status") != "sealed_before_f36_prediction_and_confirmatory_a4_outcomes"
        or seal.get("passed") is not True
        or seal.get("model_prediction_or_outcome_read") is not False
        or seal.get("schedule_sha256") != sha256(schedule)
        or seal.get("collector_sha256") != sha256(collector)
        or seal.get("runner_sha256") != sha256(Path(__file__).resolve())
        or seal.get("execution", {}).get("maximum_concurrent_isaac_processes") != 1
    ):
        raise RuntimeError("invalid A4-v8 seal")
    cases = {
        str(row["case_id"]): row
        for line in schedule.read_text().splitlines()
        if line
        for row in [json.loads(line)]
    }
    if len(cases) != 50 or len({row["scene_id"] for row in cases.values()}) != 25:
        raise RuntimeError("A4-v8 schedule must contain 50 cases over 25 scenes")

    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    CORPUS.mkdir(parents=True, exist_ok=True)
    attempts = STATE_ROOT / "attempts"
    logs = STATE_ROOT / "logs"
    attempts.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": str(ROOT),
            "TERM": "xterm-256color",
        }
    )
    started = time.monotonic()
    for index, (case_id, row) in enumerate(sorted(cases.items()), start=1):
        scene = str(row["scene_id"])
        summary = CORPUS / scene / case_id / "summary.json"
        attempt_path = attempts / f"{case_id}.json"
        if summary.is_file() and load(summary).get("passed") is True:
            continue
        if attempt_path.is_file():
            raise RuntimeError(f"scientific retry forbidden after recorded attempt: {case_id}")
        log_path = logs / f"{case_id}.log"
        command = [
            str(PYTHON),
            str(collector),
            "--schedule",
            str(schedule),
            "--protocol",
            str(SEAL),
            "--scene-registry",
            str(REGISTRY),
            "--scene",
            scene,
            "--case-id",
            case_id,
            "--out",
            str(CORPUS),
            "--headless",
        ]
        record = {
            "schema_version": "kinofail.reconfirmation-a4-v8-attempt.v1",
            "case_id": case_id,
            "scene_id": scene,
            "state": "started",
            "started_utc": datetime.now(UTC).isoformat(),
            "command": command,
            "result_dependent_retry_permitted": False,
        }
        write(attempt_path, record)
        case_started = time.monotonic()
        with log_path.open("ab", buffering=0) as log:
            try:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout_s,
                    check=False,
                )
                returncode = int(completed.returncode)
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
        accepted_count = sum(
            (CORPUS / str(item["scene_id"]) / item_id / "summary.json").is_file()
            and load(CORPUS / str(item["scene_id"]) / item_id / "summary.json").get("passed") is True
            for item_id, item in cases.items()
        )
        write(
            STATE_ROOT / "supervisor_state.json",
            {
                "schema_version": "kinofail.reconfirmation-a4-v8-supervisor.v1",
                "state": "running" if accepted_count < 50 and accepted else "terminal_failure" if not accepted else "terminal",
                "planned_cases": 50,
                "accepted_cases": accepted_count,
                "active_cases": [],
                "pending_cases": 50 - accepted_count,
                "maximum_concurrent_isaac_processes": 1,
                "last_case_index": index,
                "last_case_id": case_id,
                "elapsed_s": time.monotonic() - started,
                "updated_utc": datetime.now(UTC).isoformat(),
            },
        )
        print(json.dumps(record, sort_keys=True), flush=True)
        if not accepted:
            raise RuntimeError(f"A4-v8 case failed without retry: {case_id}")

    summaries = [
        load(CORPUS / str(row["scene_id"]) / case_id / "summary.json")
        for case_id, row in sorted(cases.items())
    ]
    audit = {
        "schema_version": "kinofail.reconfirmation-a4-v8-collection-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "terminal",
        "passed": len(summaries) == 50 and all(row.get("passed") is True for row in summaries),
        "counts": {
            "scenes": 25,
            "case_processes": 50,
            "physical_cases": sum(int(row["terminal_cases"]) for row in summaries),
            "physical_episodes": sum(int(row["completed_episodes"]) for row in summaries),
        },
        "maximum_concurrent_isaac_processes": 1,
        "unfavorable_outcomes_retained": True,
        "result_dependent_retry_or_selection": False,
        "source_sha256": {
            "seal": sha256(SEAL),
            "schedule": sha256(schedule),
            "collector": sha256(collector),
            "runner": sha256(Path(__file__).resolve()),
        },
        "corpus_root": str(CORPUS),
    }
    write(STATE_ROOT / "final_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
