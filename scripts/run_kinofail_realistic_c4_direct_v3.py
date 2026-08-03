#!/usr/bin/env python3
"""Run each frozen C4-v3 case pair in a fresh Isaac application."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schedule",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/design_c4_direct_v3/schedule.jsonl",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_realistic_c4_direct_formal_v3.json",
    )
    parser.add_argument(
        "--scene-registry",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_realistic_scale_scene_registry_v1.json",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "outputs/locomotion/recovery_route_v1/policy.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/kinofail_realistic/corpus_c4_direct_v3",
    )
    parser.add_argument("--scene")
    args = parser.parse_args()
    schedule = [
        json.loads(line)
        for line in args.schedule.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if args.scene:
        schedule = [
            row for row in schedule if row["scene_cluster"] == args.scene
        ]
    # Avoid launching a full Isaac application merely to discover that --resume
    # already has both frozen branches.  This is only a launcher-side efficiency
    # guard: incomplete case pairs still go through the hash-bound collector.
    completed: set[str] = set()
    result_files = (
        [args.out / args.scene / "results.jsonl"]
        if args.scene
        else sorted(args.out.glob("*/results.jsonl"))
    )
    branches_by_case: dict[str, set[str]] = {}
    for result_file in result_files:
        if not result_file.exists():
            continue
        for line in result_file.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            result = json.loads(line)
            branches_by_case.setdefault(str(result["case_id"]), set()).add(
                str(result["branch"])
            )
    completed = {
        case_id
        for case_id, branches in branches_by_case.items()
        if branches == {"selective", "always_safe"}
    }
    schedule = [row for row in schedule if str(row["case_id"]) not in completed]
    environment = os.environ.copy()
    environment.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    environment.setdefault("HF_HUB_OFFLINE", "1")
    launcher = ROOT / "scripts/isaac_collect_kinofail_realistic_c4_direct_v3.py"
    for index, row in enumerate(schedule, start=1):
        print(
            json.dumps(
                {
                    "case": index,
                    "total": len(schedule),
                    "scene_cluster": row["scene_cluster"],
                    "case_id": row["case_id"],
                    "branch_order": row["branch_order"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        command = [
            sys.executable,
            str(launcher),
            "--schedule",
            str(args.schedule.relative_to(ROOT)),
            "--protocol",
            str(args.protocol.relative_to(ROOT)),
            "--scene-registry",
            str(args.scene_registry.relative_to(ROOT)),
            "--scene",
            str(row["scene_cluster"]),
            "--case-id",
            str(row["case_id"]),
            "--policy",
            str(args.policy.relative_to(ROOT)),
            "--out",
            str(args.out.relative_to(ROOT)),
            "--resume",
            "--headless",
        ]
        completed = subprocess.run(
            command, cwd=ROOT, env=environment, check=False
        )
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
