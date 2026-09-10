#!/usr/bin/env python3
"""Advance the sealed KiNO-v4 confirmation scenes without unattended gaps."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/kinofail_kino_v4_confirmation_v1"
REGISTRY = BASE / "scene_registry.json"
ORCHESTRATION = BASE / "orchestration"
CAMPAIGN = ORCHESTRATION / "campaign_v1.json"
LOGS = ORCHESTRATION / "campaign_logs"
SCENE_RUNNER = ROOT / "scripts/run_kinofail_kino_v4_confirmation_scene_v1.py"
F3 = ROOT / "outputs/freeze/kino_v4_confirmation_collector_f3/freeze_manifest.json"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _write_state(value: dict[str, Any]) -> None:
    ORCHESTRATION.mkdir(parents=True, exist_ok=True)
    temporary = CAMPAIGN.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, CAMPAIGN)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partitions", type=int, default=6)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if not F3.is_file() or _json(F3).get("passed") is not True:
        raise RuntimeError("F3 confirmation freeze is absent")
    scenes = [str(row["scene_id"]) for row in _json(REGISTRY)["scenes"]]
    started = datetime.now(UTC).isoformat()
    completed: list[str] = []
    LOGS.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        state_path = ORCHESTRATION / f"{scene}.json"
        while state_path.is_file() and _json(state_path).get("status") == "started":
            _write_state(
                {
                    "schema_version": "kinofail.kino-v4-confirmation-campaign.v1",
                    "status": "waiting_for_active_scene",
                    "started_utc": started,
                    "updated_utc": datetime.now(UTC).isoformat(),
                    "active_scene": scene,
                    "completed_scenes": completed,
                    "result_dependent_retry_permitted": False,
                }
            )
            time.sleep(args.poll_seconds)
        if state_path.is_file():
            scene_state = _json(state_path)
            if scene_state.get("status") != "terminal" or scene_state.get("passed") is not True:
                _write_state(
                    {
                        "schema_version": "kinofail.kino-v4-confirmation-campaign.v1",
                        "status": "stopped_on_failed_scene",
                        "started_utc": started,
                        "completed_utc": datetime.now(UTC).isoformat(),
                        "failed_scene": scene,
                        "completed_scenes": completed,
                        "result_dependent_retry_permitted": False,
                    }
                )
                return 2
            completed.append(scene)
            continue
        command = [
            sys.executable,
            str(SCENE_RUNNER),
            "--scene",
            scene,
            "--partitions",
            str(args.partitions),
            "--workers",
            str(args.workers),
        ]
        log_path = LOGS / f"{scene}.log"
        _write_state(
            {
                "schema_version": "kinofail.kino-v4-confirmation-campaign.v1",
                "status": "running_scene",
                "started_utc": started,
                "updated_utc": datetime.now(UTC).isoformat(),
                "active_scene": scene,
                "completed_scenes": completed,
                "scene_log": str(log_path),
                "result_dependent_retry_permitted": False,
            }
        )
        with log_path.open("x") as stream:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            _write_state(
                {
                    "schema_version": "kinofail.kino-v4-confirmation-campaign.v1",
                    "status": "stopped_on_failed_scene",
                    "started_utc": started,
                    "completed_utc": datetime.now(UTC).isoformat(),
                    "failed_scene": scene,
                    "returncode": int(result.returncode),
                    "completed_scenes": completed,
                    "result_dependent_retry_permitted": False,
                }
            )
            return int(result.returncode)
        completed.append(scene)
    _write_state(
        {
            "schema_version": "kinofail.kino-v4-confirmation-campaign.v1",
            "status": "terminal",
            "passed": True,
            "started_utc": started,
            "completed_utc": datetime.now(UTC).isoformat(),
            "completed_scenes": completed,
            "result_dependent_retry_permitted": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
