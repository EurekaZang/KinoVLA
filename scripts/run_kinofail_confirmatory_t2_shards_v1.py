#!/usr/bin/env python3
"""Run disjoint T2 scene partitions once, retaining every terminal attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_confirmatory_t2_v1.py"
ISAAC_SETUP = Path("/home/eureka/nvidia/isaacsim/setup_conda_env.sh")
ISAACLAB_PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--scene-registry", type=Path, required=True)
    parser.add_argument("--asset-lock", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partition-index", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()

    schedule = args.schedule.resolve()
    registry = args.scene_registry.resolve()
    lock = args.asset_lock.resolve()
    protocol_path = args.protocol.resolve()
    out = args.out.resolve()
    protocol = _json(protocol_path)
    for runtime_path in (ISAAC_SETUP, ISAACLAB_PYTHON):
        if not runtime_path.is_file():
            raise FileNotFoundError(runtime_path)
    for key, actual in (
        ("schedule_sha256", _sha256(schedule)),
        ("scene_registry_sha256", _sha256(registry)),
        ("asset_lock_sha256", _sha256(lock)),
        ("collector_sha256", _sha256(COLLECTOR)),
    ):
        if protocol.get(key) != actual:
            raise RuntimeError(f"T2 protocol mismatch: {key}")
    scenes = sorted({str(row["scene_cluster"]) for row in _jsonl(schedule)})
    if len(scenes) != 30:
        raise RuntimeError("T2 schedule must span exactly 30 scenes")
    selected = [
        scene
        for index, scene in enumerate(scenes)
        if index % 2 == args.partition_index
    ]
    audit_path = (
        out
        / "launcher_audits"
        / f"t2_partition_{args.partition_index}_of_2.json"
    )
    if audit_path.exists():
        prior = _json(audit_path)
        if (
            prior.get("selected_scenes") != selected
            or prior.get("schedule_sha256") != _sha256(schedule)
            or prior.get("collector_sha256") != _sha256(COLLECTOR)
        ):
            raise RuntimeError("existing T2 launcher audit belongs to another design")
        attempts = list(prior.get("attempts", []))
    else:
        attempts = []
    attempted = {str(row["scene"]) for row in attempts}
    for index, scene in enumerate(selected, start=1):
        if scene in attempted:
            print(
                json.dumps(
                    {
                        "scene": scene,
                        "progress": f"{index}/{len(selected)}",
                        "status": "skip_recorded_terminal_attempt",
                    }
                ),
                flush=True,
            )
            continue
        scene_out = out / scene / "c2_t2"
        command = [
            "/bin/bash",
            "-lc",
            'source "$1" >/dev/null 2>&1 && exec "$2" "${@:3}"',
            "kinofail-confirmatory-t2",
            str(ISAAC_SETUP),
            str(ISAACLAB_PYTHON),
            str(COLLECTOR),
            "--schedule",
            str(schedule),
            "--scene-registry",
            str(registry),
            "--asset-lock",
            str(lock),
            "--protocol",
            str(protocol_path),
            "--out",
            str(scene_out),
            "--scene",
            scene,
            "--headless",
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        summary_path = scene_out / "scene_summaries" / f"{scene}.json"
        summary = _json(summary_path) if summary_path.is_file() else {}
        result = {
            "scene": scene,
            "returncode": int(completed.returncode),
            "summary_exists": summary_path.is_file(),
            "passed": summary.get("passed") is True,
            "retry_authorized": False,
            "completed_utc": datetime.now(UTC).isoformat(),
        }
        attempts.append(result)
        attempted.add(scene)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(
                {
                    "schema_version": "kinofail.confirmatory-t2-launcher.v1",
                    "partition_index": args.partition_index,
                    "partition_count": 2,
                    "selected_scenes": selected,
                    "schedule_sha256": _sha256(schedule),
                    "protocol_sha256": _sha256(protocol_path),
                    "collector_sha256": _sha256(COLLECTOR),
                    "attempts": attempts,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps({**result, "progress": f"{index}/{len(selected)}"}),
            flush=True,
        )
    return 0 if all(row["passed"] for row in attempts) else 2


if __name__ == "__main__":
    raise SystemExit(main())
