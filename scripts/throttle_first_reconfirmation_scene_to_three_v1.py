#!/usr/bin/env python3
"""Reduce the active first-scene collectors from four to three.

The deterministic four-way partition is preserved.  Partition 3 is paused,
its currently incomplete collector is failed closed without retry, and the
partition runner is resumed only after one of partitions 0--2 has finished.
No result, label, feature, prediction, or score is read.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.watch_kinofail_reconfirmation_collectors_v2 import (
    _pair_processes,
    _terminate_exact_processes,
)


SCENE = "confirm_v2_life_scene_00"
CORPUS = Path(
    "/media/eureka/FC28565528560ED0/tmp/"
    f"KinoVLA_reconfirmation_v2/corpus/{SCENE}"
)
AUDIT = (
    CORPUS
    / "launcher_audits/scale_f3v7_partition_3_of_4.json"
)
STATUS = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/operational_throttle/"
    "first_scene_three_process_transition.json"
)
RUNNER_TOKEN = "run_kinofail_reconfirmation_pair_partition_v2.py"


def _write(value: dict[str, Any]) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, STATUS)


def _argument(tokens: list[str], name: str) -> str | None:
    try:
        index = tokens.index(name)
    except ValueError:
        return None
    return tokens[index + 1] if index + 1 < len(tokens) else None


def _runners() -> dict[int, int]:
    runners: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            tokens = [
                token.decode("utf-8", errors="replace")
                for token in (entry / "cmdline").read_bytes().split(b"\0")
                if token
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (
            not tokens
            or RUNNER_TOKEN not in " ".join(tokens)
            or str(CORPUS) not in tokens
        ):
            continue
        raw_partition = _argument(tokens, "--partition-index")
        if raw_partition is None:
            continue
        runners[int(raw_partition)] = int(entry.name)
    return runners


def _current_partition_three_pair() -> str:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    active = [
        str(row["counterfactual_group_id"])
        for row in audit["attempts"]
        if row.get("state") != "terminal"
    ]
    if len(active) != 1:
        raise RuntimeError(
            f"expected one active partition-3 pair, found {active}"
        )
    return active[0]


def main() -> int:
    runners = _runners()
    if set(runners) != {0, 1, 2, 3}:
        raise RuntimeError(f"unexpected active runner inventory: {runners}")
    paused_pid = runners[3]
    pair_id = _current_partition_three_pair()
    os.kill(paused_pid, signal.SIGSTOP)
    grouped = _pair_processes()
    processes = grouped.get((pair_id, CORPUS.resolve()), [])
    if not processes:
        os.kill(paused_pid, signal.SIGCONT)
        raise RuntimeError("active partition-3 collector processes not found")
    action = _terminate_exact_processes(processes, grace_seconds=15)
    state = {
        "schema_version": (
            "kinofail.reconfirmation-first-scene-three-process-throttle.v1"
        ),
        "status": "partition_3_paused",
        "created_utc": datetime.now(UTC).isoformat(),
        "scene_id": SCENE,
        "paused_partition": 3,
        "paused_runner_pid": paused_pid,
        "interrupted_pair_id": pair_id,
        "interrupted_pair_retried": False,
        "initial_other_runner_pids": {
            str(index): runners[index] for index in (0, 1, 2)
        },
        "collector_termination": action,
        "labels_features_predictions_or_scores_read": False,
        "scientific_content_changed": False,
    }
    _write(state)
    while True:
        live = _runners()
        finished = [
            index
            for index in (0, 1, 2)
            if live.get(index) != runners[index]
        ]
        if finished:
            break
        if live.get(3) != paused_pid:
            raise RuntimeError("paused partition-3 runner disappeared")
        time.sleep(20)
    os.kill(paused_pid, signal.SIGCONT)
    state.update(
        {
            "status": "partition_3_resumed_after_capacity_available",
            "resumed_utc": datetime.now(UTC).isoformat(),
            "finished_partition_that_released_capacity": min(finished),
        }
    )
    _write(state)
    print(json.dumps(state, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
