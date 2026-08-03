#!/usr/bin/env python3
"""Run every incomplete scene through the F21 unattended v5 pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENE_ROOT = ROOT / "outputs/kinofail_reconfirmation_v2/schedules/scenes"
RECEIPT_ROOT = ROOT / "outputs/kinofail_reconfirmation_v2/prune_receipts"
PIPELINE = ROOT / "scripts/run_kinofail_reconfirmation_scene_pipeline_v5.py"
AUDIT_PATH = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/orchestration/"
    "all_scenes_pipeline_f21.jsonl"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-scene", action="append", default=[])
    args = parser.parse_args()
    inventory = {path.name for path in SCENE_ROOT.iterdir() if path.is_dir()}
    if len(inventory) != 30:
        raise RuntimeError("unexpected reconfirmation scene inventory")
    scenes = sorted(inventory - set(args.exclude_scene))
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        receipt = RECEIPT_ROOT / scene / "completed.json"
        if receipt.is_file():
            status = "skip_completed_receipt"
            returncode = 0
        else:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PIPELINE),
                    "--scene",
                    scene,
                    "--partition-count",
                    "4",
                ],
                cwd=ROOT,
                check=False,
            )
            returncode = int(completed.returncode)
            status = "terminal"
        row = {
            "created_utc": datetime.now(UTC).isoformat(),
            "scene_id": scene,
            "status": status,
            "returncode": returncode,
            "pipeline": str(PIPELINE),
        }
        with AUDIT_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        if returncode != 0:
            return returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
