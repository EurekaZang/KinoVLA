#!/usr/bin/env python3
"""Collect untouched F27 T3 replacements with a three-process health gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.run_kinofail_t3_replenishment_f28 import (
    CONDA_PREFIX,
    EXPERIENCE,
    ISAACLAB,
    PYTHONPATH,
    _pid_matches,
    _terminate_group,
    atomic_json,
    read_json,
    read_jsonl,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/kinofail_t3_replenishment_f29"


def validate_seal(path: Path) -> dict[str, Any]:
    seal = read_json(path)
    sidecar = path.with_name("seal_manifest.sha256")
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").split()[0] != sha256(path)
        or seal.get("status") != "sealed_before_collection"
        or seal.get("passed") is not True
        or seal.get("selection_depends_on_pass_fail_outcome") is not False
        or seal.get("result_dependent_retry") is not False
        or seal.get("model_prediction_feature_or_score_read") is not False
        or seal.get("counterfactual_pairs") != 160
        or seal.get("maximum_concurrent_isaac_processes") != 3
    ):
        raise RuntimeError("invalid F29 seal")
    for artifact in seal["dependencies"]:
        artifact_path = Path(str(artifact["path"]))
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise RuntimeError(f"F29 dependency drift: {artifact_path}")
    for section in ("inherited_schedules", "inherited_protocols"):
        for artifact in seal[section]:
            artifact_path = ROOT / str(artifact["path"])
            if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
                raise RuntimeError(f"F29 inherited artifact drift: {artifact_path}")
    for key in (
        "eligible_case_mapping", "excluded_case_mapping", "eligible_pair_ids",
        "f27_freeze", "f28_interruption_audit", "scene_registry", "material_lock",
    ):
        artifact_path = ROOT / str(seal[key])
        if not artifact_path.is_file() or sha256(artifact_path) != seal[f"{key}_sha256"]:
            raise RuntimeError(f"F29 sealed artifact drift: {artifact_path}")
    return seal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", type=Path, default=OUTPUT / "seal_manifest.json")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--timeout-s", type=int, default=900)
    args = parser.parse_args()
    if args.max_workers != 3:
        raise ValueError("F29 is frozen to exactly three concurrent Isaac processes")
    seal_path = args.seal.resolve()
    seal = validate_seal(seal_path)
    corpus_root = Path(str(seal["corpus_root"]))
    corpus_root.mkdir(parents=True, exist_ok=True)
    attempts_root = OUTPUT / "attempts"
    logs_root = OUTPUT / "logs"
    attempts_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    eligible = {
        str(row["counterfactual_group_id"])
        for row in read_jsonl(ROOT / str(seal["eligible_pair_ids"]))
    }
    protocols = {
        Path(str(row["path"])).parts[-2]: ROOT / str(row["path"])
        for row in seal["inherited_protocols"]
    }
    tasks: list[dict[str, Any]] = []
    for artifact in seal["inherited_schedules"]:
        schedule = ROOT / str(artifact["path"])
        rows = read_jsonl(schedule)
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row["counterfactual_group_id"]), []).append(row)
        scene_id = str(rows[0]["scene_id"])
        for pair_id, pair_rows in sorted(groups.items()):
            if pair_id not in eligible:
                continue
            if len(pair_rows) != 2:
                raise RuntimeError(f"incomplete F29 pair: {pair_id}")
            tasks.append(
                {"pair_id": pair_id, "scene_id": scene_id, "schedule": schedule, "protocol": protocols[scene_id]}
            )
    if len(tasks) != 160 or {str(row["pair_id"]) for row in tasks} != eligible:
        raise RuntimeError("F29 task selection mismatch")

    registry = ROOT / str(seal["scene_registry"])
    asset_lock = ROOT / str(seal["material_lock"])
    collector = ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v9.py"
    environment = os.environ.copy()
    environment.update(
        {
            "CONDA_PREFIX": str(CONDA_PREFIX),
            "PATH": f"{CONDA_PREFIX / 'bin'}:{environment.get('PATH', '')}",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONPATH": PYTHONPATH,
            "TERM": "xterm-256color",
            "HF_HUB_OFFLINE": "1",
        }
    )
    output_lock = threading.Lock()

    def run_one(task: dict[str, Any]) -> dict[str, Any]:
        pair_id = str(task["pair_id"])
        scene_id = str(task["scene_id"])
        attempt_path = attempts_root / f"{pair_id}.json"
        summary_path = corpus_root / scene_id / "pair_summaries" / f"{pair_id}.json"
        if attempt_path.exists():
            prior = read_json(attempt_path)
            if prior.get("state") == "terminal":
                return prior
            pid = int(prior.get("pid", -1))
            if pid > 0 and _pid_matches(pid, pair_id):
                while _pid_matches(pid, pair_id):
                    time.sleep(5.0)
            summary = read_json(summary_path) if summary_path.is_file() else {}
            terminal = {
                **prior, "state": "terminal", "completed_utc": datetime.now(UTC).isoformat(),
                "returncode": None, "reconciled_after_supervisor_restart": True,
                "summary_exists": summary_path.is_file(), "summary_passed": summary.get("passed") is True,
                "passed": summary.get("passed") is True,
                "summary_sha256": sha256(summary_path) if summary_path.is_file() else None,
            }
            atomic_json(attempt_path, terminal)
            return terminal
        log_path = logs_root / f"{pair_id}.log"
        started = {
            "schema_version": "kinofail.f29-attempt.v1", "state": "started",
            "started_utc": datetime.now(UTC).isoformat(), "pair_id": pair_id, "scene_id": scene_id,
            "f28_pair_was_attempted": False, "scientific_collection_attempt_index": 1,
            "retry_authorized": False, "result_dependent_retry": False,
            "model_prediction_feature_or_score_read": False,
            "schedule_sha256": sha256(Path(task["schedule"])),
            "protocol_sha256": sha256(Path(task["protocol"])), "log": str(log_path),
        }
        atomic_json(attempt_path, started)
        command = [
            str(ISAACLAB), "-p", str(collector), "--schedule", str(task["schedule"]),
            "--scene-registry", str(registry), "--protocol", str(task["protocol"]),
            "--asset-lock", str(asset_lock), "--corpus-root", str(corpus_root / scene_id),
            "--counterfactual-group-id", pair_id, "--headless", "--enable_cameras",
            "--experience", str(EXPERIENCE),
        ]
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            started["pid"] = process.pid
            atomic_json(attempt_path, started)
            timed_out = False
            try:
                returncode = process.wait(timeout=args.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_group(process.pid)
                returncode = process.wait()
        summary = read_json(summary_path) if summary_path.is_file() else {}
        terminal = {
            **started, "state": "terminal", "completed_utc": datetime.now(UTC).isoformat(),
            "returncode": int(returncode), "timed_out": timed_out,
            "summary_exists": summary_path.is_file(), "summary_passed": summary.get("passed") is True,
            "passed": returncode == 0 and summary.get("passed") is True,
            "summary_sha256": sha256(summary_path) if summary_path.is_file() else None,
        }
        atomic_json(attempt_path, terminal)
        with output_lock:
            print(json.dumps({"pair": pair_id, "scene": scene_id, "returncode": returncode, "passed": terminal["passed"], "timed_out": timed_out}, sort_keys=True), flush=True)
        return terminal

    started_at = time.monotonic()

    def update_state(completed: int, state: str) -> None:
        records = [read_json(path) for path in attempts_root.glob("*.json")]
        atomic_json(
            OUTPUT / "supervisor_state.json",
            {
                "schema_version": "kinofail.f29-supervisor-state.v1",
                "updated_utc": datetime.now(UTC).isoformat(), "state": state,
                "completed_tasks_this_run": completed, "total_tasks": len(tasks),
                "attempt_records": len(records),
                "terminal_pairs": sum(row.get("state") == "terminal" for row in records),
                "passed_pairs": sum(row.get("passed") is True for row in records),
                "active_pairs": sum(row.get("state") == "started" for row in records),
                "max_workers": 3, "elapsed_s_this_run": time.monotonic() - started_at,
            },
        )

    pilot_results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pilot_futures = []
        for index, task in enumerate(tasks[:3]):
            pilot_futures.append(pool.submit(run_one, task))
            if index < 2:
                time.sleep(8.0)
        for future in concurrent.futures.as_completed(pilot_futures):
            pilot_results.append(future.result())
            update_state(len(pilot_results), "pilot_health_gate")
    pilot_infrastructure_failure = all(
        row.get("returncode") not in (0, None) and row.get("summary_exists") is False
        for row in pilot_results
    )
    pilot_path = OUTPUT / "pilot_health_audit.json"
    atomic_json(
        pilot_path,
        {
            "schema_version": "kinofail.f29-pilot-health-audit.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "passed": not pilot_infrastructure_failure,
            "pilot_pairs": [row.get("pair_id") for row in pilot_results],
            "all_three_nonzero_without_summary": pilot_infrastructure_failure,
            "results": pilot_results,
        },
    )
    if pilot_infrastructure_failure:
        update_state(3, "pilot_infrastructure_gate_failed")
        print(json.dumps({"pilot_health_gate": "failed", "audit": str(pilot_path)}, sort_keys=True), flush=True)
        return 3

    completed = 3
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_one, task) for task in tasks[3:]]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            update_state(completed, "running" if completed < len(tasks) else "finalizing")

    attempts = {path.stem: read_json(path) for path in sorted(attempts_root.glob("*.json"))}
    mappings = read_jsonl(ROOT / str(seal["eligible_case_mapping"]))
    accepted_cases = []
    rejected_cases = []
    for mapping in mappings:
        pair_ids = [str(mapping["replacement_o7_group_id"]), str(mapping["replacement_o8_group_id"])]
        record = {**mapping, "passed": all(attempts.get(pair_id, {}).get("passed") is True for pair_id in pair_ids)}
        (accepted_cases if record["passed"] else rejected_cases).append(record)
    remaining_invalid = 91 - len(accepted_cases)
    effective_rate = remaining_invalid / 1_500
    all_terminal = len(attempts) == 160 and all(row.get("state") == "terminal" for row in attempts.values())
    audit = {
        "schema_version": "kinofail.t3-replenishment-f29-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(), "status": "final",
        "passed": all_terminal and effective_rate < 0.05,
        "model_prediction_feature_or_score_read": False,
        "result_dependent_selection_or_retry": False,
        "f28_touched_pairs_excluded": True, "fresh_isaac_process_per_pair": True,
        "counts": {
            "scheduled_cases": 80, "scheduled_pairs": 160,
            "terminal_pairs": sum(row.get("state") == "terminal" for row in attempts.values()),
            "passed_pairs": sum(row.get("passed") is True for row in attempts.values()),
            "accepted_complete_cases": len(accepted_cases),
            "rejected_incomplete_cases": len(rejected_cases),
            "excluded_f28_touched_cases": 11, "original_valid_t3_cases": 1_409,
            "effective_valid_t3_cases": 1_409 + len(accepted_cases),
            "remaining_invalid_t3_cases": remaining_invalid,
        },
        "effective_t3_case_attrition_rate": effective_rate,
        "gates": {
            "pilot_health_gate": True, "all_frozen_pairs_terminal": all_terminal,
            "at_least_17_complete_cases": len(accepted_cases) >= 17,
            "t3_case_attrition_strictly_below_five_percent": effective_rate < 0.05,
        },
        "passed_pairs_by_scene": dict(sorted(Counter(row["scene_id"] for row in attempts.values() if row.get("passed") is True).items())),
        "accepted_cases": accepted_cases, "rejected_cases": rejected_cases,
        "seal_sha256": sha256(seal_path), "pilot_health_audit_sha256": sha256(pilot_path),
    }
    atomic_json(OUTPUT / "final_audit.json", audit)
    update_state(len(tasks), "completed" if audit["passed"] else "gate_failed")
    print(json.dumps({"final_audit": str(OUTPUT / "final_audit.json"), "passed": audit["passed"], "counts": audit["counts"]}, sort_keys=True), flush=True)
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
