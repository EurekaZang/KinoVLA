#!/usr/bin/env python3
"""Run one durable worker of the sealed F24 replenishment cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ISAACLAB = Path("/home/eureka/IsaacLab-v2.3.0/isaaclab.sh")
EXPERIENCE = Path(
    "/home/eureka/IsaacLab-v2.3.0/apps/isaaclab.python.headless.rendering.kit"
)
CONDA_PREFIX = Path("/home/eureka/miniconda3/envs/kinovla")
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_replenishment_f24_v1.py"
REGISTRY = ROOT / "outputs/kinofail_reconfirmation_v2/scene_registry.json"
ASSET_LOCK = ROOT / "outputs/assets/terrain_pbr_confirmatory_v2/terrain_assets.lock.json"
PYTHONPATH = (
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab:"
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_tasks:"
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_assets:"
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_rl:"
    "/home/eureka/KinoVLA"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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
    return pair_id in command and "isaac_collect_kinofail_replenishment_f24" in command


def _terminate_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and _pid_matches(pid, "cf_f24_"):
        time.sleep(0.25)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _validate_freeze(freeze_path: Path) -> dict[str, Any]:
    freeze = _json(freeze_path)
    if (
        freeze.get("passed") is not True
        or freeze.get("status") != "sealed_before_collection"
        or freeze.get("model_prediction_or_score_read") is not False
    ):
        raise RuntimeError("F24 manifest is not a valid pre-collection freeze")
    for section in ("schedules", "protocols"):
        for artifact in freeze[section]:
            path = ROOT / artifact["path"]
            if not path.is_file() or _sha256(path) != artifact["sha256"]:
                raise RuntimeError(f"F24 frozen artifact mismatch: {path}")
    for key in ("scene_registry", "material_lock", "design", "mapping"):
        path = ROOT / freeze[key]
        if not path.is_file() or _sha256(path) != freeze[f"{key}_sha256"]:
            raise RuntimeError(f"F24 frozen artifact mismatch: {path}")
    scripts = freeze.get("execution_artifacts", [])
    for artifact in scripts:
        path = ROOT / artifact["path"]
        if not path.is_file() or _sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"F24 execution artifact mismatch: {path}")
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, default=3)
    parser.add_argument("--timeout-s", type=int, default=600)
    args = parser.parse_args()
    if args.worker_count != 3 or args.worker_index not in range(args.worker_count):
        raise ValueError("F24 uses exactly three workers with indices 0, 1, 2")

    freeze_path = args.freeze.resolve()
    freeze = _validate_freeze(freeze_path)
    corpus_root = args.corpus_root.resolve()
    corpus_root.mkdir(parents=True, exist_ok=True)
    attempts_root = corpus_root / "attempts"
    logs_root = corpus_root / "launcher_logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    work: list[tuple[Path, Path, str, str]] = []
    for schedule_artifact in freeze["schedules"]:
        schedule = ROOT / schedule_artifact["path"]
        protocol = schedule.parent / "collection_protocol.json"
        records = _jsonl(schedule)
        groups = sorted({str(row["counterfactual_group_id"]) for row in records})
        scene_id = str(records[0]["scene_family"])
        for pair_id in groups:
            work.append((schedule, protocol, scene_id, pair_id))
    work.sort(key=lambda item: (item[2], item[3]))
    selected = [
        item
        for index, item in enumerate(work)
        if index % args.worker_count == args.worker_index
    ]

    environment = os.environ.copy()
    environment.update(
        {
            "CONDA_PREFIX": str(CONDA_PREFIX),
            "PATH": f"{CONDA_PREFIX / 'bin'}:{environment.get('PATH', '')}",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONPATH": PYTHONPATH,
            # IsaacLab's shell calls `tabs`; nohup supplies TERM=dumb, which
            # makes that harmless formatting command abort before Python starts.
            "TERM": "xterm-256color",
        }
    )
    for progress, (schedule, protocol, scene_id, pair_id) in enumerate(
        selected, start=1
    ):
        summary_path = corpus_root / scene_id / "pair_summaries" / f"{pair_id}.json"
        attempt_path = attempts_root / f"{pair_id}.json"
        if summary_path.is_file() and _json(summary_path).get("passed") is True:
            print(json.dumps({"pair": pair_id, "status": "skip_sealed_pass"}), flush=True)
            continue
        if attempt_path.is_file():
            prior = _json(attempt_path)
            if prior.get("state") == "terminal":
                print(json.dumps({"pair": pair_id, "status": "skip_terminal"}), flush=True)
                continue
            pid = int(prior.get("pid", -1))
            if pid > 0 and _pid_matches(pid, pair_id):
                while _pid_matches(pid, pair_id):
                    time.sleep(5.0)
                passed = summary_path.is_file() and _json(summary_path).get("passed") is True
                _atomic_json(
                    attempt_path,
                    {
                        **prior,
                        "state": "terminal",
                        "completed_utc": datetime.now(UTC).isoformat(),
                        "returncode": None,
                        "reconciled_after_launcher_restart": True,
                        "summary_exists": summary_path.is_file(),
                        "passed": passed,
                    },
                )
                continue
            _atomic_json(
                attempt_path,
                {
                    **prior,
                    "state": "terminal",
                    "completed_utc": datetime.now(UTC).isoformat(),
                    "returncode": None,
                    "unrecoverable_interrupted_attempt": True,
                    "summary_exists": summary_path.is_file(),
                    "passed": False,
                },
            )
            continue

        log_path = logs_root / scene_id / f"{pair_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(ISAACLAB),
            "-p",
            str(COLLECTOR),
            "--schedule",
            str(schedule),
            "--scene-registry",
            str(REGISTRY),
            "--protocol",
            str(protocol),
            "--asset-lock",
            str(ASSET_LOCK),
            "--corpus-root",
            str(corpus_root / scene_id),
            "--counterfactual-group-id",
            pair_id,
            "--headless",
            "--enable_cameras",
            "--experience",
            str(EXPERIENCE),
        ]
        started = {
            "schema_version": "kinofail.f24-attempt.v1",
            "state": "started",
            "started_utc": datetime.now(UTC).isoformat(),
            "pair": pair_id,
            "scene_id": scene_id,
            "worker_index": args.worker_index,
            "retry_authorized": False,
            "replacement_of_replacement_permitted": False,
            "schedule": str(schedule),
            "schedule_sha256": _sha256(schedule),
            "protocol": str(protocol),
            "protocol_sha256": _sha256(protocol),
            "collector": str(COLLECTOR),
            "collector_sha256": _sha256(COLLECTOR),
            "log": str(log_path),
        }
        _atomic_json(attempt_path, started)
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
            _atomic_json(attempt_path, started)
            timed_out = False
            try:
                returncode = process.wait(timeout=args.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_group(process.pid)
                returncode = process.wait()
        summary = _json(summary_path) if summary_path.is_file() else {}
        terminal = {
            **started,
            "state": "terminal",
            "completed_utc": datetime.now(UTC).isoformat(),
            "returncode": int(returncode),
            "timed_out": timed_out,
            "summary_exists": summary_path.is_file(),
            "passed": summary.get("passed") is True,
        }
        _atomic_json(attempt_path, terminal)
        print(
            json.dumps(
                {
                    "progress": f"{progress}/{len(selected)}",
                    "pair": pair_id,
                    "scene": scene_id,
                    "returncode": returncode,
                    "passed": terminal["passed"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
