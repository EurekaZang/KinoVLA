#!/usr/bin/env python3
"""Unattended three-process launcher for case-isolated expanded A4-v7."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONDA_ENV = Path("/home/eureka/miniconda3/envs/kinovla")
PYTHON = CONDA_ENV / "bin/python"
SEAL = ROOT / "outputs/freeze/kinofail_reconfirmation_a4_v7/seal_manifest.json"
REGISTRY = ROOT / "outputs/kinofail_reconfirmation_v2/scene_registry.json"
STATE_ROOT = ROOT / "outputs/kinofail_reconfirmation_a4_v7/run"
CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_reconfirmation_a4_v7/corpus")


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


def validate() -> tuple[dict[str, Any], Path, Path]:
    seal = load(SEAL)
    sidecar = SEAL.with_name("seal_manifest.sha256")
    schedule = ROOT / str(seal["schedule"])
    collector = ROOT / str(seal["collector"])
    actor = seal.get("low_level_actor", {})
    predecision_actor = seal.get("predecision_actor", {})
    predecision_policy = ROOT / str(predecision_actor.get("policy", ""))
    actor_paths = {
        key: ROOT / str(actor.get(key, ""))
        for key in ("policy", "training_manifest", "training_freeze", "sim_config")
    }
    actor_hash_keys = {
        "policy": "policy_sha256",
        "training_manifest": "training_manifest_sha256",
        "training_freeze": "training_freeze_sha256",
        "sim_config": "sim_config_sha256",
    }
    if (
        not sidecar.is_file()
        or sidecar.read_text().split()[0] != sha256(SEAL)
        or seal.get("status")
        != "sealed_before_f36_prediction_and_confirmatory_a4_outcomes"
        or seal.get("passed") is not True
        or seal.get("model_prediction_or_outcome_read") is not False
        or seal.get("schedule_sha256") != sha256(schedule)
        or seal.get("collector_sha256") != sha256(collector)
        or seal.get("runner_sha256") != sha256(Path(__file__).resolve())
        or actor.get("shared_across_all_action_arms") is not True
        or actor.get("receives_attribution_input") is not False
        or not predecision_policy.is_file()
        or predecision_actor.get("policy_sha256") != sha256(predecision_policy)
        or predecision_actor.get("purpose")
        != "replay_frozen_f35_predecision_state"
        or not all(
            path.is_file() and actor.get(actor_hash_keys[key]) == sha256(path)
            for key, path in actor_paths.items()
        )
    ):
        raise RuntimeError("invalid A4-v7 seal")
    return seal, schedule, collector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--timeout-s", type=int, default=7200)
    args = parser.parse_args()
    if args.max_workers != 3:
        raise RuntimeError("A4-v7 concurrency is frozen at exactly three Isaac processes")
    seal, schedule, collector = validate()
    rows = [json.loads(line) for line in schedule.read_text().splitlines() if line]
    scenes = sorted({str(row["scene_id"]) for row in rows})
    case_by_id = {str(row["case_id"]): row for row in rows}
    if len(scenes) != 30 or len(case_by_id) != 300:
        raise RuntimeError("A4-v7 schedule does not have 30 scenes and 300 cases")
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    CORPUS.mkdir(parents=True, exist_ok=True)
    attempts_dir = STATE_ROOT / "attempts"
    logs_dir = STATE_ROOT / "logs"
    attempts_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    terminal_failures = []
    pending: deque[str] = deque()
    for case_id, row in sorted(case_by_id.items()):
        scene = str(row["scene_id"])
        summary = CORPUS / scene / case_id / "summary.json"
        attempt = attempts_dir / f"{case_id}.json"
        if summary.is_file() and load(summary).get("passed") is True:
            continue
        if attempt.is_file() and load(attempt).get("state") == "terminal_failure":
            terminal_failures.append(case_id)
        else:
            pending.append(case_id)
    if terminal_failures:
        raise RuntimeError(
            "A4-v7 has terminal failures; scientific retries are forbidden: "
            + ", ".join(terminal_failures[:10])
        )

    active: dict[str, tuple[subprocess.Popen[bytes], Any, float]] = {}
    environment = os.environ.copy()
    environment.update(
        {
            "CONDA_PREFIX": str(CONDA_ENV),
            "PATH": f"{CONDA_ENV / 'bin'}:{environment.get('PATH', '')}",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "TERM": "xterm-256color",
            "PYTHONPATH": str(ROOT),
            "HF_HUB_OFFLINE": "1",
        }
    )
    started = time.monotonic()
    while pending or active:
        while pending and len(active) < args.max_workers:
            case_id = pending.popleft()
            scene = str(case_by_id[case_id]["scene_id"])
            log_path = logs_dir / f"{case_id}.log"
            log = log_path.open("ab", buffering=0)
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
                "--resume",
                "--headless",
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active[case_id] = (process, log, time.monotonic())
            write(
                attempts_dir / f"{case_id}.json",
                {
                    "schema_version": "kinofail.reconfirmation-a4-v7-attempt.v1",
                    "scene_id": scene,
                    "case_id": case_id,
                    "state": "started",
                    "pid": process.pid,
                    "started_utc": datetime.now(UTC).isoformat(),
                    "command": command,
                    "result_dependent_retry_permitted": False,
                },
            )
        finished = []
        for case_id, (process, log, scene_started) in active.items():
            returncode = process.poll()
            elapsed = time.monotonic() - scene_started
            if returncode is None and elapsed <= args.timeout_s:
                continue
            if returncode is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                returncode = 124
            log.close()
            scene = str(case_by_id[case_id]["scene_id"])
            summary_path = CORPUS / scene / case_id / "summary.json"
            accepted = summary_path.is_file() and load(summary_path).get("passed") is True
            record = {
                "schema_version": "kinofail.reconfirmation-a4-v7-attempt.v1",
                "scene_id": scene,
                "case_id": case_id,
                "state": "terminal_accepted" if accepted else "terminal_failure",
                "returncode": returncode,
                "elapsed_s": elapsed,
                "finished_utc": datetime.now(UTC).isoformat(),
                "summary": str(summary_path),
                "summary_sha256": sha256(summary_path) if summary_path.is_file() else None,
                "result_dependent_retry_permitted": False,
            }
            write(attempts_dir / f"{case_id}.json", record)
            print(json.dumps(record, sort_keys=True), flush=True)
            finished.append(case_id)
            if not accepted:
                for other, (other_process, other_log, _) in active.items():
                    if other != case_id and other_process.poll() is None:
                        other_process.terminate()
                        other_log.close()
                raise RuntimeError(f"A4-v7 case failed without retry: {case_id}")
        for case_id in finished:
            active.pop(case_id)
        write(
            STATE_ROOT / "supervisor_state.json",
            {
                "schema_version": "kinofail.reconfirmation-a4-v7-supervisor.v1",
                "state": "running" if pending or active else "terminal",
                "planned_cases": 300,
                "accepted_cases": sum(
                    (CORPUS / str(row["scene_id"]) / case_id / "summary.json").is_file()
                    and load(
                        CORPUS / str(row["scene_id"]) / case_id / "summary.json"
                    ).get("passed")
                    is True
                    for case_id, row in case_by_id.items()
                ),
                "active_cases": sorted(active),
                "pending_cases": len(pending),
                "maximum_concurrent_isaac_processes": 3,
                "elapsed_s": time.monotonic() - started,
                "updated_utc": datetime.now(UTC).isoformat(),
            },
        )
        if pending or active:
            time.sleep(2.0)

    summaries = [
        load(CORPUS / str(row["scene_id"]) / case_id / "summary.json")
        for case_id, row in sorted(case_by_id.items())
    ]
    audit = {
        "schema_version": "kinofail.reconfirmation-a4-v7-collection-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "terminal",
        "passed": all(row.get("passed") is True for row in summaries),
        "counts": {
            "scenes": len(scenes),
            "case_processes": len(summaries),
            "physical_cases": sum(int(row["terminal_cases"]) for row in summaries),
            "physical_episodes": sum(int(row["completed_episodes"]) for row in summaries),
        },
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
