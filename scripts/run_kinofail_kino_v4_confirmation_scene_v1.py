#!/usr/bin/env python3
"""Collect one frozen KiNO-v4 confirmation scene with bounded parallelism."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEDULES = ROOT / "outputs/kinofail_kino_v4_confirmation_v1/schedules"
REGISTRY = ROOT / "outputs/kinofail_kino_v4_confirmation_v1/scene_registry.json"
LOCK = Path("/data/eureka/kinofail_kino_v4_confirmation_v1/assets/terrain_pbr/terrain_assets.lock.json")
CORPUS = Path("/data/eureka/kinofail_kino_v4_confirmation_v1/corpus")
ORCHESTRATION = ROOT / "outputs/kinofail_kino_v4_confirmation_v1/orchestration"
PAIR_RUNNER = ROOT / "scripts/run_kinofail_reconfirmation_pair_partition_v2.py"
T2_RUNNER = ROOT / "scripts/run_kinofail_reconfirmation_t2_scene_v2.py"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_kino_v4_confirmation_pair_v1.py"
AMENDMENT = (
    ROOT
    / "outputs/freeze/kino_v4_confirmation_collector_f3/amendment_manifest.json"
)


def _run(name: str, command: list[str], log_dir: Path) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{name}.log"
    started = datetime.now(UTC).isoformat()
    with log.open("a") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    return {
        "name": name,
        "returncode": int(result.returncode),
        "started_utc": started,
        "completed_utc": datetime.now(UTC).isoformat(),
        "log": str(log),
    }


def _pair_command(scene: str, battery: str, partition: int, partitions: int) -> list[str]:
    root = SCHEDULES / "scenes" / scene / battery
    schedule = root / "schedule_f3_remaining.jsonl"
    if not schedule.is_file():
        schedule = root / "schedule.jsonl"
    return [
        sys.executable,
        str(PAIR_RUNNER),
        "--schedule",
        str(schedule),
        "--scene-registry",
        str(REGISTRY),
        "--protocol",
        str(root / "collection_protocol_f3.json"),
        "--asset-lock",
        str(LOCK),
        "--corpus-root",
        str(CORPUS / scene),
        "--collector",
        str(COLLECTOR),
        "--operational-amendment",
        str(AMENDMENT),
        "--partition-index",
        str(partition),
        "--partition-count",
        str(partitions),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--partitions", type=int, default=6)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.partitions <= 8 or not 1 <= args.workers <= 3:
        raise ValueError("partitions must be 1..8 and workers 1..3")
    registry = json.loads(REGISTRY.read_text())
    if args.scene not in {row["scene_id"] for row in registry["scenes"]}:
        raise ValueError(args.scene)
    state_path = ORCHESTRATION / f"{args.scene}.json"
    if state_path.is_file():
        prior = json.loads(state_path.read_text())
        if prior.get("status") == "terminal":
            return 0 if prior.get("passed") is True else 2
        raise RuntimeError("non-terminal orchestration record already exists")
    ORCHESTRATION.mkdir(parents=True, exist_ok=True)
    log_dir = ORCHESTRATION / "logs" / args.scene
    started = {
        "schema_version": "kinofail.kino-v4-confirmation-scene-run.v1",
        "scene_id": args.scene,
        "status": "started",
        "started_utc": datetime.now(UTC).isoformat(),
        "partitions": args.partitions,
        "workers": args.workers,
        "result_dependent_retry_permitted": False,
    }
    state_path.write_text(json.dumps(started, indent=2, sort_keys=True) + "\n")
    t2_root = SCHEDULES / "scenes" / args.scene / "c2_t2_f3"
    if t2_root.is_dir():
        t2_schedule = t2_root / "schedule.jsonl"
        t2_protocol = t2_root / "collection_protocol_f3.json"
    else:
        t2_schedule = SCHEDULES / "c2_t2/schedule.jsonl"
        t2_protocol = SCHEDULES / "c2_t2/collection_protocol.json"
    t2 = [
        sys.executable,
        str(T2_RUNNER),
        "--schedule",
        str(t2_schedule),
        "--scene-registry",
        str(REGISTRY),
        "--asset-lock",
        str(LOCK),
        "--protocol",
        str(t2_protocol),
        "--out",
        str(CORPUS / args.scene / "c2_t2"),
        "--scene",
        args.scene,
    ]
    first = [("t2", t2)] + [
        (f"scale_p{part}", _pair_command(args.scene, "scale", part, args.partitions))
        for part in range(args.partitions)
    ]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run, name, command, log_dir): name for name, command in first}
        for future in as_completed(futures):
            value = future.result()
            results.append(value)
            print(json.dumps(value), flush=True)
    second = [
        (f"t3_p{part}", _pair_command(args.scene, "c2_t3", part, args.partitions))
        for part in range(args.partitions)
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run, name, command, log_dir): name for name, command in second}
        for future in as_completed(futures):
            value = future.result()
            results.append(value)
            print(json.dumps(value), flush=True)
    passed = all(row["returncode"] == 0 for row in results)
    terminal = {
        **started,
        "status": "terminal",
        "completed_utc": datetime.now(UTC).isoformat(),
        "passed": passed,
        "jobs": sorted(results, key=lambda row: row["name"]),
    }
    state_path.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"scene": args.scene, "passed": passed}), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
