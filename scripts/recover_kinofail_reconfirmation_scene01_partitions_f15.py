#!/usr/bin/env python3
"""Launch scene-01 scale partitions that failed before collector startup.

The original in-memory scene pipeline started partitions 0/1 before F13 was
sealed.  Partitions 2/3 were dequeued later and rejected by the older F12
runner hash check before creating a launcher audit or starting an Isaac
collector.  F15 supersedes only that transitive hash check.  This supervisor
keeps the scene parent stopped, launches the two never-started partitions,
enforces the F8 three-runner concurrency ceiling, validates the complete
four-way schedule accounting, and only then resumes the parent.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCENE = "confirm_v2_life_scene_01"
F12 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f12_process_local_rtx_texture_amendment1/"
    "amendment_manifest.json"
)
F15 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f15_runner_supersession_amendment1/"
    "amendment_manifest.json"
)
RUNNER = (
    ROOT / "scripts/run_kinofail_reconfirmation_pair_partition_v2.py"
)
SCHEDULE = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/schedules/scenes"
    / SCENE
    / "scale/schedule.jsonl"
)
PROTOCOL = SCHEDULE.parent / "collection_protocol.json"
REGISTRY = ROOT / "outputs/kinofail_reconfirmation_v2/scene_registry.json"
ASSET_LOCK = (
    ROOT
    / "outputs/assets/terrain_pbr_confirmatory_v2/terrain_assets.lock.json"
)
CORPUS = (
    ROOT / "outputs/kinofail_reconfirmation_v2/corpus_ext4" / SCENE
)
AUDIT_DIR = CORPUS / "launcher_audits"
LOG_DIR = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/orchestration"
    / SCENE
    / "logs"
)
OUTPUT_AUDIT = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/orchestration"
    / SCENE
    / "f15_partition_recovery.json"
)
RUNNER_TOKEN = "run_kinofail_reconfirmation_pair_partition_v2.py"
PIPELINE_TOKEN = "run_kinofail_reconfirmation_scene_pipeline_v2.py"


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
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _processes_with(token: str) -> list[dict[str, Any]]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            tokens = [
                value.decode("utf-8", errors="replace")
                for value in (entry / "cmdline").read_bytes().split(b"\0")
                if value
            ]
            state = (entry / "stat").read_text().split()[2]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        joined = " ".join(tokens)
        if (
            token not in joined
            or SCENE not in joined
            or state == "Z"
        ):
            continue
        found.append(
            {
                "pid": int(entry.name),
                "state": state,
                "command": tokens,
            }
        )
    return sorted(found, key=lambda row: int(row["pid"]))


def active_pair_runners() -> list[dict[str, Any]]:
    return _processes_with(RUNNER_TOKEN)


def scene_parent() -> dict[str, Any]:
    parents = _processes_with(PIPELINE_TOKEN)
    if len(parents) != 1:
        raise RuntimeError(
            f"expected one scene-01 parent, found {len(parents)}"
        )
    return parents[0]


def partition_pair_ids(index: int) -> list[str]:
    pairs = sorted(
        {
            str(row["counterfactual_group_id"])
            for row in read_jsonl(SCHEDULE)
        }
    )
    selected = [
        pair_id
        for pair_index, pair_id in enumerate(pairs)
        if pair_index % 4 == index
    ]
    if len(selected) != 88:
        raise RuntimeError("scene-01 scale partition is not 88 pairs")
    return selected


def partition_audit(index: int) -> Path:
    return (
        AUDIT_DIR
        / f"scale_f12v9_partition_{index}_of_4.json"
    )


def _partition_complete(index: int) -> bool:
    path = partition_audit(index)
    if not path.is_file():
        return False
    audit = read_json(path)
    attempts = audit.get("attempts", [])
    return (
        len(attempts) == 88
        and all(row.get("state") == "terminal" for row in attempts)
        and {
            str(row["counterfactual_group_id"]) for row in attempts
        }
        == set(partition_pair_ids(index))
    )


def validate_all_partitions() -> dict[str, Any]:
    rows = {}
    all_attempts = []
    for index in range(4):
        path = partition_audit(index)
        if not _partition_complete(index):
            raise RuntimeError(f"scale partition {index} is incomplete")
        audit = read_json(path)
        attempts = audit["attempts"]
        all_attempts.extend(attempts)
        rows[str(index)] = {
            "audit": str(path),
            "audit_sha256": sha256(path),
            "attempts": len(attempts),
            "returncode_zero": sum(
                row.get("returncode") == 0 for row in attempts
            ),
            "terminal_attrition": sum(
                row.get("returncode") != 0 for row in attempts
            ),
        }
    pair_ids = [
        str(row["counterfactual_group_id"]) for row in all_attempts
    ]
    scheduled = {
        str(row["counterfactual_group_id"])
        for row in read_jsonl(SCHEDULE)
    }
    if (
        len(pair_ids) != 352
        or len(set(pair_ids)) != 352
        or set(pair_ids) != scheduled
    ):
        raise RuntimeError("F15 four-partition accounting is invalid")
    return {
        "partitions": rows,
        "planned_pairs": 352,
        "terminal_attempts": 352,
        "unique_terminal_pair_ids": 352,
        "returncode_zero": sum(
            row.get("returncode") == 0 for row in all_attempts
        ),
        "terminal_attrition": sum(
            row.get("returncode") != 0 for row in all_attempts
        ),
        "schedule_partitioned_exactly_once": True,
    }


def _command(index: int) -> list[str]:
    return [
        sys.executable,
        str(RUNNER),
        "--schedule",
        str(SCHEDULE),
        "--scene-registry",
        str(REGISTRY),
        "--protocol",
        str(PROTOCOL),
        "--asset-lock",
        str(ASSET_LOCK),
        "--corpus-root",
        str(CORPUS),
        "--operational-amendment",
        str(F12),
        "--partition-index",
        str(index),
        "--partition-count",
        "4",
    ]


def _launch(index: int) -> tuple[subprocess.Popen[Any], Any, Path]:
    log_path = LOG_DIR / f"scale_p{index}_f15.log"
    if log_path.exists():
        raise FileExistsError(log_path)
    stream = log_path.open("x", encoding="utf-8")
    process = subprocess.Popen(
        _command(index),
        cwd=ROOT,
        stdout=stream,
        stderr=subprocess.STDOUT,
    )
    return process, stream, log_path


def main() -> int:
    if OUTPUT_AUDIT.exists():
        raise FileExistsError(OUTPUT_AUDIT)
    amendment = read_json(F15)
    correction = amendment.get("correction", {})
    if (
        amendment.get("passed") is not True
        or amendment.get("schema_version")
        != (
            "kinofail.reconfirmation-f15-runner-"
            "supersession-amendment.v1"
        )
        or amendment.get("status")
        != "sealed_before_scene01_unstarted_partition_launch"
        or correction.get("recovery_script_sha256")
        != sha256(Path(__file__).resolve())
        or correction.get("runner_sha256") != sha256(RUNNER)
    ):
        raise RuntimeError("invalid F15 amendment")
    for index in (2, 3):
        if partition_audit(index).exists():
            raise RuntimeError(
                f"F15 partition {index} already has a launcher attempt"
            )
    parent = scene_parent()
    if parent["state"] != "T":
        raise RuntimeError("scene-01 parent is not stopped before F15")
    initial_runners = active_pair_runners()
    if len(initial_runners) != 2:
        raise RuntimeError("F15 expected only scale partitions 0/1 active")

    state = {
        "schema_version": (
            "kinofail.reconfirmation-f15-scene01-partition-recovery.v1"
        ),
        "state": "started",
        "started_utc": datetime.now(UTC).isoformat(),
        "scene_id": SCENE,
        "operational_amendment": str(F15),
        "operational_amendment_sha256": sha256(F15),
        "scene_parent": parent,
        "initial_active_pair_runners": initial_runners,
        "partitions_2_and_3_had_launcher_attempts_before_f15": False,
        "existing_pair_reexecution_permitted": False,
        "never_attempted_pairs_only": True,
        "result_dependent_retry_or_selection": False,
        "model_feature_label_outcome_prediction_or_score_read": False,
        "maximum_concurrent_pair_runners_allowed": 3,
        "maximum_concurrent_pair_runners_observed": len(initial_runners),
        "launches": {},
    }
    write_json(OUTPUT_AUDIT, state)

    processes: dict[int, subprocess.Popen[Any]] = {}
    streams: dict[int, Any] = {}
    logs: dict[int, Path] = {}
    try:
        process, stream, log = _launch(2)
        processes[2], streams[2], logs[2] = process, stream, log
        state["launches"]["2"] = {
            "started_utc": datetime.now(UTC).isoformat(),
            "pid": process.pid,
            "command": _command(2),
            "log": str(log),
        }
        write_json(OUTPUT_AUDIT, state)

        # Wait until one of the three active partition runners completes.
        # Counting runners, rather than short-lived Isaac children, prevents
        # oversubscription during the gap between two pair launches.
        while True:
            runners = active_pair_runners()
            state["maximum_concurrent_pair_runners_observed"] = max(
                int(state["maximum_concurrent_pair_runners_observed"]),
                len(runners),
            )
            if len(runners) > 3:
                raise RuntimeError("F15 concurrency ceiling exceeded")
            if len(runners) < 3:
                break
            time.sleep(10)

        process, stream, log = _launch(3)
        processes[3], streams[3], logs[3] = process, stream, log
        state["launches"]["3"] = {
            "started_utc": datetime.now(UTC).isoformat(),
            "pid": process.pid,
            "command": _command(3),
            "log": str(log),
        }
        write_json(OUTPUT_AUDIT, state)

        while not all(_partition_complete(index) for index in range(4)):
            runners = active_pair_runners()
            state["maximum_concurrent_pair_runners_observed"] = max(
                int(state["maximum_concurrent_pair_runners_observed"]),
                len(runners),
            )
            if len(runners) > 3:
                raise RuntimeError("F15 concurrency ceiling exceeded")
            time.sleep(20)

        for index, process in processes.items():
            returncode = process.wait()
            streams[index].close()
            state["launches"][str(index)].update(
                {
                    "completed_utc": datetime.now(UTC).isoformat(),
                    "returncode": int(returncode),
                    "log_sha256": sha256(logs[index]),
                }
            )
        accounting = validate_all_partitions()
        os.kill(int(parent["pid"]), signal.SIGCONT)
        state.update(
            {
                "state": "terminal",
                "completed_utc": datetime.now(UTC).isoformat(),
                "passed": True,
                "scene_parent_resumed": True,
                "accounting": accounting,
            }
        )
        write_json(OUTPUT_AUDIT, state)
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        for stream in streams.values():
            if not stream.closed:
                stream.close()
        state.update(
            {
                "state": "terminal_failure",
                "failed_utc": datetime.now(UTC).isoformat(),
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        write_json(OUTPUT_AUDIT, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
