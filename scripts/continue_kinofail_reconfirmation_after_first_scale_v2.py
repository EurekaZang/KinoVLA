#!/usr/bin/env python3
"""Finish the active first shard, then hand off to the all-scene pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENE = "confirm_v2_life_scene_00"
CORPUS = Path(
    "/media/eureka/FC28565528560ED0/tmp/"
    f"KinoVLA_reconfirmation_v2/corpus/{SCENE}"
)
AUDIT_ROOT = CORPUS / "launcher_audits"
STATUS = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/orchestration/"
    f"{SCENE}/continuation_status.json"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _scale_complete() -> bool:
    for partition in range(4):
        path = (
            AUDIT_ROOT
            / f"scale_f3v7_partition_{partition}_of_4.json"
        )
        if not path.is_file():
            return False
        audit = _json(path)
        selected = {str(value) for value in audit["selected_pair_ids"]}
        attempts = {
            str(row["counterfactual_group_id"]): row
            for row in audit["attempts"]
        }
        for pair_id in selected:
            summary = CORPUS / "pair_summaries" / f"{pair_id}.json"
            if summary.is_file():
                continue
            if attempts.get(pair_id, {}).get("state") != "terminal":
                return False
    return True


def _write(state: str, **extra: object) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(
        json.dumps(
            {
                "updated_utc": datetime.now(UTC).isoformat(),
                "state": state,
                **extra,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    _write("waiting_for_first_scale")
    while not _scale_complete():
        time.sleep(20)
    _write("running_first_scene_remainder")
    first = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_kinofail_reconfirmation_scene_pipeline_v2.py"),
            "--scene",
            SCENE,
            "--partition-count",
            "4",
        ],
        cwd=ROOT,
        check=False,
    )
    if first.returncode != 0:
        _write("first_scene_terminal_failure", returncode=int(first.returncode))
        return int(first.returncode)
    _write("running_remaining_scenes")
    remaining = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "scripts/run_kinofail_reconfirmation_all_scene_pipelines_v2.py"
            ),
            "--exclude-scene",
            SCENE,
        ],
        cwd=ROOT,
        check=False,
    )
    state = "complete" if remaining.returncode == 0 else "remaining_terminal_failure"
    _write(state, returncode=int(remaining.returncode))
    return int(remaining.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
