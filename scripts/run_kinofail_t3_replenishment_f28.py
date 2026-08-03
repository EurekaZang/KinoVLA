#!/usr/bin/env python3
"""Run the F28 environment-corrected relaunch of frozen F27 pairs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/kinofail_t3_replenishment_f28"
ISAACLAB = Path("/home/eureka/IsaacLab-v2.3.0/isaaclab.sh")
CONDA_PREFIX = Path("/home/eureka/miniconda3/envs/kinovla")
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(path)
    return rows


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _pid_matches(pid: int, pair_id: str) -> bool:
    path = Path(f"/proc/{pid}/cmdline")
    if not path.is_file():
        return False
    try:
        command = path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return pair_id in command and "isaac_collect_kinofail_confirmatory_pair_v9" in command


def _terminate_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.25)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def validate_seal(path: Path) -> dict[str, Any]:
    seal = read_json(path)
    sidecar = path.with_name("seal_manifest.sha256")
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").split()[0] != sha256(path)
        or seal.get("status") != "sealed_before_operational_relaunch"
        or seal.get("passed") is not True
        or seal.get("scientific_design_changed_from_f27") is not False
        or seal.get("model_prediction_feature_or_score_read") is not False
        or seal.get("counterfactual_pairs") != 182
    ):
        raise RuntimeError("invalid F28 operational seal")
    for artifact in seal["dependencies"]:
        artifact_path = Path(str(artifact["path"]))
        if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
            raise RuntimeError(f"F28 dependency drift: {artifact_path}")
    for section in ("inherited_schedules", "inherited_protocols"):
        for artifact in seal[section]:
            artifact_path = ROOT / str(artifact["path"])
            if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
                raise RuntimeError(f"F28 inherited artifact drift: {artifact_path}")
    for key in ("f27_freeze", "f27_failed_audit", "scene_registry", "material_lock", "mapping"):
        artifact_path = ROOT / str(seal[key])
        if not artifact_path.is_file() or sha256(artifact_path) != seal[f"{key}_sha256"]:
            raise RuntimeError(f"F28 sealed artifact drift: {artifact_path}")
    return seal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", type=Path, default=OUTPUT / "seal_manifest.json")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--timeout-s", type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 8:
        raise ValueError("max-workers must be in [1, 8]")
    seal_path = args.seal.resolve()
    seal = validate_seal(seal_path)
    corpus_root = Path(str(seal["corpus_root"]))
    corpus_root.mkdir(parents=True, exist_ok=True)
    attempts_root = OUTPUT / "attempts"
    logs_root = OUTPUT / "logs"
    attempts_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    protocols = {
        Path(str(row["path"])).parts[-2]: ROOT / str(row["path"])
        for row in seal["inherited_protocols"]
    }
    tasks: list[dict[str, Any]] = []
    for artifact in seal["inherited_schedules"]:
        schedule_path = ROOT / str(artifact["path"])
        rows = read_jsonl(schedule_path)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["counterfactual_group_id"]), []).append(row)
        scene_id = str(rows[0]["scene_id"])
        for pair_id, pair_rows in sorted(grouped.items()):
            if len(pair_rows) != 2:
                raise RuntimeError(f"incomplete F28 inherited pair: {pair_id}")
            tasks.append(
                {
                    "pair_id": pair_id,
                    "scene_id": scene_id,
                    "schedule": schedule_path,
                    "protocol": protocols[scene_id],
                }
            )
    if len(tasks) != 182:
        raise RuntimeError(f"unexpected F28 task count: {len(tasks)}")

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

    def terminalize_started(
        attempt_path: Path,
        started: dict[str, Any],
        summary_path: Path,
    ) -> dict[str, Any]:
        pid = int(started.get("pid", -1))
        if pid > 0 and _pid_matches(pid, str(started["pair_id"])):
            while _pid_matches(pid, str(started["pair_id"])):
                time.sleep(5.0)
        summary = read_json(summary_path) if summary_path.is_file() else {}
        terminal = {
            **started,
            "state": "terminal",
            "completed_utc": datetime.now(UTC).isoformat(),
            "returncode": None,
            "reconciled_after_supervisor_restart": True,
            "summary_exists": summary_path.is_file(),
            "summary_passed": summary.get("passed") is True,
            "passed": summary.get("passed") is True,
            "summary_sha256": sha256(summary_path) if summary_path.is_file() else None,
        }
        atomic_json(attempt_path, terminal)
        return terminal

    def run_one(task: dict[str, Any]) -> dict[str, Any]:
        pair_id = str(task["pair_id"])
        scene_id = str(task["scene_id"])
        attempt_path = attempts_root / f"{pair_id}.json"
        scene_corpus = corpus_root / scene_id
        summary_path = scene_corpus / "pair_summaries" / f"{pair_id}.json"
        if attempt_path.exists():
            prior = read_json(attempt_path)
            if prior.get("state") == "terminal":
                return prior
            return terminalize_started(attempt_path, prior, summary_path)
        log_path = logs_root / f"{pair_id}.log"
        started = {
            "schema_version": "kinofail.f28-attempt.v1",
            "state": "started",
            "started_utc": datetime.now(UTC).isoformat(),
            "pair_id": pair_id,
            "scene_id": scene_id,
            "launcher_attempt_index": 2,
            "scientific_collection_attempt_index": 1,
            "retry_authorized": False,
            "result_dependent_relaunch": False,
            "model_prediction_feature_or_score_read": False,
            "schedule_sha256": sha256(Path(task["schedule"])),
            "protocol_sha256": sha256(Path(task["protocol"])),
            "log": str(log_path),
        }
        atomic_json(attempt_path, started)
        command = [
            str(ISAACLAB), "-p", str(collector),
            "--schedule", str(task["schedule"]),
            "--scene-registry", str(registry),
            "--protocol", str(task["protocol"]),
            "--asset-lock", str(asset_lock),
            "--corpus-root", str(scene_corpus),
            "--counterfactual-group-id", pair_id,
            "--headless", "--enable_cameras",
            "--experience", str(EXPERIENCE),
        ]
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
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
            **started,
            "state": "terminal",
            "completed_utc": datetime.now(UTC).isoformat(),
            "returncode": int(returncode),
            "timed_out": timed_out,
            "summary_exists": summary_path.is_file(),
            "summary_passed": summary.get("passed") is True,
            "passed": returncode == 0 and summary.get("passed") is True,
            "summary_sha256": sha256(summary_path) if summary_path.is_file() else None,
        }
        atomic_json(attempt_path, terminal)
        with output_lock:
            print(
                json.dumps(
                    {
                        "pair": pair_id,
                        "scene": scene_id,
                        "returncode": returncode,
                        "passed": terminal["passed"],
                        "timed_out": timed_out,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return terminal

    started_at = time.monotonic()
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        future_map = {pool.submit(run_one, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_map):
            future.result()
            completed += 1
            attempts = [read_json(path) for path in attempts_root.glob("*.json")]
            atomic_json(
                OUTPUT / "supervisor_state.json",
                {
                    "schema_version": "kinofail.f28-supervisor-state.v1",
                    "updated_utc": datetime.now(UTC).isoformat(),
                    "state": "running" if completed < len(tasks) else "finalizing",
                    "completed_tasks_this_run": completed,
                    "total_tasks": len(tasks),
                    "attempt_records": len(attempts),
                    "terminal_pairs": sum(row.get("state") == "terminal" for row in attempts),
                    "passed_pairs": sum(row.get("passed") is True for row in attempts),
                    "active_pairs": sum(row.get("state") == "started" for row in attempts),
                    "max_workers": args.max_workers,
                    "elapsed_s_this_run": time.monotonic() - started_at,
                },
            )

    attempts = {
        path.stem: read_json(path) for path in sorted(attempts_root.glob("*.json"))
    }
    mappings = read_jsonl(ROOT / str(seal["mapping"]))
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
        "schema_version": "kinofail.t3-replenishment-f28-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "final",
        "passed": all_terminal and effective_rate < 0.05,
        "scientific_design_changed_from_f27": False,
        "model_prediction_feature_or_score_read": False,
        "result_dependent_selection_or_retry": False,
        "f27_prelaunch_failure_retained": True,
        "fresh_isaac_process_per_pair": True,
        "counts": {
            "scheduled_cases": 91,
            "scheduled_pairs": 182,
            "terminal_pairs": sum(row.get("state") == "terminal" for row in attempts.values()),
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
            sorted(Counter(row["scene_id"] for row in attempts.values() if row.get("passed") is True).items())
        ),
        "accepted_cases": accepted_cases,
        "rejected_cases": rejected_cases,
        "seal_sha256": sha256(seal_path),
    }
    atomic_json(OUTPUT / "final_audit.json", audit)
    atomic_json(
        OUTPUT / "supervisor_state.json",
        {
            "schema_version": "kinofail.f28-supervisor-state.v1",
            "updated_utc": datetime.now(UTC).isoformat(),
            "state": "completed" if audit["passed"] else "gate_failed",
            "completed_tasks_this_run": len(tasks),
            "total_tasks": len(tasks),
            "attempt_records": len(attempts),
            "terminal_pairs": audit["counts"]["terminal_pairs"],
            "passed_pairs": audit["counts"]["passed_pairs"],
            "active_pairs": 0,
            "max_workers": args.max_workers,
            "elapsed_s_this_run": time.monotonic() - started_at,
            "audit_passed": audit["passed"],
        },
    )
    print(json.dumps({"final_audit": str(OUTPUT / "final_audit.json"), "passed": audit["passed"], "counts": audit["counts"]}, sort_keys=True), flush=True)
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
