#!/usr/bin/env python3
"""Run the sealed F27 T3 case replenishment with fresh Isaac processes."""

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

from scripts.prepare_kinofail_t3_replenishment_f27 import (
    OUTPUT,
    ROOT,
    read_json,
    read_jsonl,
    sha256,
    write_json,
)


ISAACLAB = Path("/home/eureka/IsaacLab-v2.3.0/isaaclab.sh")
EXPERIENCE = Path(
    "/home/eureka/IsaacLab-v2.3.0/apps/isaaclab.python.headless.rendering.kit"
)
PYTHONPATH = (
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab:"
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_tasks:"
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_assets:"
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_rl:"
    "/home/eureka/KinoVLA"
)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def validate_freeze(path: Path) -> dict[str, Any]:
    freeze = read_json(path)
    sidecar = path.with_name("freeze_manifest.sha256")
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").split()[0] != sha256(path)
        or freeze.get("status") != "sealed_before_collection"
        or freeze.get("passed") is not True
        or freeze.get("model_prediction_feature_or_score_read") is not False
        or freeze.get("counterfactual_pairs") != 182
    ):
        raise RuntimeError("invalid F27 freeze")
    for section in ("schedules", "protocols", "dependencies"):
        for artifact in freeze[section]:
            artifact_path = ROOT / str(artifact["path"])
            if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
                raise RuntimeError(f"F27 frozen artifact mismatch: {artifact_path}")
    for key in ("mapping", "case_schedule", "scene_registry", "material_lock"):
        artifact_path = ROOT / str(freeze[key])
        if not artifact_path.is_file() or sha256(artifact_path) != freeze[f"{key}_sha256"]:
            raise RuntimeError(f"F27 frozen artifact mismatch: {artifact_path}")
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=OUTPUT / "freeze_manifest.json")
    parser.add_argument("--max-workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 8:
        raise ValueError("max-workers must be in [1, 8]")
    freeze_path = args.freeze.resolve()
    freeze = validate_freeze(freeze_path)
    corpus_root = Path(str(freeze["corpus_root"]))
    corpus_root.mkdir(parents=True, exist_ok=True)
    attempts_root = OUTPUT / "attempts"
    logs_root = OUTPUT / "logs"
    attempts_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    protocols = {
        Path(str(row["path"])).parts[-2]: ROOT / str(row["path"])
        for row in freeze["protocols"]
    }
    tasks: list[dict[str, Any]] = []
    for artifact in freeze["schedules"]:
        schedule_path = ROOT / str(artifact["path"])
        rows = read_jsonl(schedule_path)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["counterfactual_group_id"]), []).append(row)
        scene_id = str(rows[0]["scene_id"])
        for pair_id, pair_rows in sorted(grouped.items()):
            if len(pair_rows) != 2:
                raise RuntimeError(f"incomplete F27 scheduled pair: {pair_id}")
            tasks.append(
                {
                    "pair_id": pair_id,
                    "scene_id": scene_id,
                    "schedule": schedule_path,
                    "protocol": protocols[scene_id],
                }
            )
    if len(tasks) != 182:
        raise RuntimeError(f"unexpected F27 task count: {len(tasks)}")

    registry = ROOT / str(freeze["scene_registry"])
    asset_lock = ROOT / str(freeze["material_lock"])
    collector = ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v9.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = PYTHONPATH
    environment["HF_HUB_OFFLINE"] = "1"
    lock = threading.Lock()

    def run_one(task: dict[str, Any]) -> dict[str, Any]:
        pair_id = str(task["pair_id"])
        scene_id = str(task["scene_id"])
        attempt_path = attempts_root / f"{pair_id}.json"
        if attempt_path.exists():
            return read_json(attempt_path)
        scene_corpus = corpus_root / scene_id
        summary_path = scene_corpus / "pair_summaries" / f"{pair_id}.json"
        log_path = logs_root / f"{pair_id}.log"
        started = {
            "schema_version": "kinofail.f27-attempt.v1",
            "state": "started",
            "started_utc": datetime.now(UTC).isoformat(),
            "pair_id": pair_id,
            "scene_id": scene_id,
            "schedule_sha256": sha256(Path(task["schedule"])),
            "protocol_sha256": sha256(Path(task["protocol"])),
            "retry_authorized": False,
            "model_prediction_feature_or_score_read": False,
            "log": str(log_path),
        }
        atomic_json(attempt_path, started)
        command = [
            str(ISAACLAB),
            "-p",
            str(collector),
            "--schedule",
            str(task["schedule"]),
            "--scene-registry",
            str(registry),
            "--protocol",
            str(task["protocol"]),
            "--asset-lock",
            str(asset_lock),
            "--corpus-root",
            str(scene_corpus),
            "--counterfactual-group-id",
            pair_id,
            "--headless",
            "--enable_cameras",
            "--experience",
            str(EXPERIENCE),
        ]
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            started["pid"] = process.pid
            atomic_json(attempt_path, started)
            returncode = process.wait()
        summary = read_json(summary_path) if summary_path.is_file() else {}
        terminal = {
            **started,
            "state": "terminal",
            "completed_utc": datetime.now(UTC).isoformat(),
            "returncode": int(returncode),
            "summary_exists": summary_path.is_file(),
            "summary_passed": summary.get("passed") is True,
            "passed": returncode == 0 and summary.get("passed") is True,
            "summary_sha256": sha256(summary_path) if summary_path.is_file() else None,
        }
        atomic_json(attempt_path, terminal)
        with lock:
            print(json.dumps(terminal, sort_keys=True), flush=True)
        return terminal

    started_at = time.monotonic()
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        future_map = {pool.submit(run_one, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_map):
            future.result()
            completed += 1
            state = {
                "schema_version": "kinofail.f27-supervisor-state.v1",
                "updated_utc": datetime.now(UTC).isoformat(),
                "state": "running" if completed < len(tasks) else "finalizing",
                "completed_tasks": completed,
                "total_tasks": len(tasks),
                "max_workers": args.max_workers,
                "elapsed_s": time.monotonic() - started_at,
            }
            atomic_json(OUTPUT / "supervisor_state.json", state)

    attempts = {
        path.stem: read_json(path) for path in sorted(attempts_root.glob("*.json"))
    }
    mappings = read_jsonl(ROOT / str(freeze["mapping"]))
    accepted_cases = []
    rejected_cases = []
    for mapping in mappings:
        pair_ids = [
            str(mapping["replacement_o7_group_id"]),
            str(mapping["replacement_o8_group_id"]),
        ]
        passed = all(attempts.get(pair_id, {}).get("passed") is True for pair_id in pair_ids)
        record = {**mapping, "passed": passed}
        (accepted_cases if passed else rejected_cases).append(record)
    remaining_invalid = 91 - len(accepted_cases)
    effective_rate = remaining_invalid / 1_500
    all_terminal = len(attempts) == 182 and all(
        row.get("state") == "terminal" for row in attempts.values()
    )
    audit = {
        "schema_version": "kinofail.t3-replenishment-f27-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "final",
        "passed": all_terminal and effective_rate < 0.05,
        "model_prediction_feature_or_score_read": False,
        "result_dependent_retry_used": False,
        "fresh_isaac_process_per_pair": True,
        "counts": {
            "scheduled_cases": 91,
            "scheduled_pairs": 182,
            "terminal_pairs": sum(
                row.get("state") == "terminal" for row in attempts.values()
            ),
            "passed_pairs": sum(row.get("passed") is True for row in attempts.values()),
            "accepted_complete_cases": len(accepted_cases),
            "rejected_incomplete_cases": len(rejected_cases),
            "original_valid_t3_cases": 1_409,
            "effective_valid_t3_cases": 1_409 + len(accepted_cases),
            "remaining_invalid_t3_cases": remaining_invalid,
        },
        "effective_t3_case_attrition_rate": effective_rate,
        "gates": {
            "all_frozen_pairs_terminal": all_terminal,
            "at_least_17_complete_cases": len(accepted_cases) >= 17,
            "t3_case_attrition_strictly_below_five_percent": effective_rate < 0.05,
        },
        "passed_pairs_by_scene": dict(
            sorted(
                Counter(
                    row["scene_id"]
                    for row in attempts.values()
                    if row.get("passed") is True
                ).items()
            )
        ),
        "accepted_cases": accepted_cases,
        "rejected_cases": rejected_cases,
        "freeze_sha256": sha256(freeze_path),
    }
    write_json(OUTPUT / "final_audit.json", audit)
    atomic_json(
        OUTPUT / "supervisor_state.json",
        {
            "schema_version": "kinofail.f27-supervisor-state.v1",
            "updated_utc": datetime.now(UTC).isoformat(),
            "state": "completed" if audit["passed"] else "gate_failed",
            "completed_tasks": len(attempts),
            "total_tasks": len(tasks),
            "max_workers": args.max_workers,
            "elapsed_s": time.monotonic() - started_at,
            "audit_passed": audit["passed"],
        },
    )
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
