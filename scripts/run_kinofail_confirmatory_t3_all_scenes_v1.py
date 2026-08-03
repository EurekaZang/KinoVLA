#!/usr/bin/env python3
"""Execute one deterministic T3 partition across all sealed scene shards.

This is an operational scheduler only.  It never edits schedules, retries a
terminal attempt, or branches on model outcomes.  The per-scene launcher owns
the append-only attempt audit and enforces the sealed schedule hashes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEDULE_ROOT = (
    ROOT
    / "outputs/kinofail_confirmatory_v1/"
    "schedules_f2_scene_source_amendment"
)
DEFAULT_REGISTRY = ROOT / "outputs/kinofail_confirmatory_v1/scene_registry.json"
DEFAULT_LOCK = (
    ROOT / "outputs/assets/terrain_pbr_confirmatory_v1/terrain_assets.lock.json"
)
DEFAULT_CORPUS = ROOT / "outputs/kinofail_confirmatory_v1/corpus"
SCENE_LAUNCHER = ROOT / "scripts/run_kinofail_confirmatory_scale_shard_v1.py"
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _process_is_expected(pid: int, marker: str) -> bool:
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.is_file():
        return False
    try:
        command = cmdline.read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except (FileNotFoundError, ProcessLookupError):
        return False
    return marker in command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--start-scene-index", type=int, default=0)
    parser.add_argument("--end-scene-index", type=int)
    parser.add_argument("--orchestration-id", required=True)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument(
        "--wait-command-marker",
        default="run_kinofail_confirmatory_scale_shard_v1.py",
    )
    parser.add_argument("--schedule-root", type=Path, default=DEFAULT_SCHEDULE_ROOT)
    parser.add_argument("--scene-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--asset-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()

    if args.wait_pid is not None:
        while _process_is_expected(args.wait_pid, args.wait_command_marker):
            time.sleep(5)

    schedule_root = args.schedule_root.resolve()
    registry_path = args.scene_registry.resolve()
    asset_lock = args.asset_lock.resolve()
    corpus_root = args.corpus_root.resolve()
    registry = _json(registry_path)
    scenes = [str(row["scene_id"]) for row in registry["scenes"]]
    if len(scenes) != 30 or len(set(scenes)) != 30:
        raise RuntimeError("confirmatory registry must contain 30 unique scenes")
    end_scene_index = (
        len(scenes)
        if args.end_scene_index is None
        else args.end_scene_index
    )
    if not 0 <= args.start_scene_index < end_scene_index <= len(scenes):
        raise ValueError("invalid frozen scene-registry slice")
    selected_scenes = scenes[args.start_scene_index : end_scene_index]

    log_path = (
        corpus_root
        / "orchestration_audits"
        / (
            f"t3_{args.orchestration_id}_partition_"
            f"{args.partition_index}_of_2.json"
        )
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    if log_path.is_file():
        prior = _json(log_path)
        if (
            prior.get("partition_index") != args.partition_index
            or prior.get("orchestration_id") != args.orchestration_id
            or prior.get("scenes") != selected_scenes
        ):
            raise RuntimeError("existing T3 orchestration audit mismatch")
        attempts = list(prior.get("attempts", []))
    attempted_scenes = {str(row["scene_id"]) for row in attempts}

    for scene_id in selected_scenes:
        if scene_id in attempted_scenes:
            continue
        shard = schedule_root / "scenes" / scene_id / "c2_t3"
        command = [
            str(PYTHON),
            str(SCENE_LAUNCHER),
            "--schedule",
            str(shard / "schedule.jsonl"),
            "--scene-registry",
            str(registry_path),
            "--asset-lock",
            str(asset_lock),
            "--protocol",
            str(shard / "collection_protocol.json"),
            "--corpus-root",
            str(corpus_root / scene_id),
            "--partition-index",
            str(args.partition_index),
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        record = {
            "scene_id": scene_id,
            "returncode": int(completed.returncode),
            "completed_utc": datetime.now(UTC).isoformat(),
            "retry_authorized": False,
        }
        attempts.append(record)
        attempted_scenes.add(scene_id)
        log_path.write_text(
            json.dumps(
                {
                    "schema_version":
                    "kinofail.confirmatory-t3-orchestration.v1",
                    "partition_index": args.partition_index,
                    "partition_count": 2,
                    "orchestration_id": args.orchestration_id,
                    "start_scene_index": args.start_scene_index,
                    "end_scene_index": end_scene_index,
                    "scenes": selected_scenes,
                    "schedule_root": str(schedule_root),
                    "scene_registry": str(registry_path),
                    "asset_lock": str(asset_lock),
                    "launcher": str(SCENE_LAUNCHER),
                    "attempts": attempts,
                    "orchestrator_pid": os.getpid(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(record, sort_keys=True), flush=True)

    return 0 if all(row["returncode"] == 0 for row in attempts) else 2


if __name__ == "__main__":
    raise SystemExit(main())
