#!/usr/bin/env python3
"""Launch one isolated Isaac application per C2 v3 T3 scene."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--scene-registry", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--scene")
    args = parser.parse_args()
    schedule = args.schedule.resolve()
    rows = _jsonl(schedule)
    if args.scene:
        rows = [
            row
            for row in rows
            if row["scene_family"] == args.scene
        ]
    scenes = sorted({str(row["scene_family"]) for row in rows})
    if not scenes:
        raise RuntimeError("C2 v3 T3 selection contains no scenes")
    environment = os.environ.copy()
    environment.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    environment.setdefault("HF_HUB_OFFLINE", "1")
    collector = (
        ROOT
        / "scripts"
        / "isaac_collect_kinofail_realistic_c2_v3_t3_scene.py"
    )
    for index, scene in enumerate(scenes, start=1):
        summary = (
            args.corpus_root.resolve()
            / "scene_summaries"
            / f"{scene}.json"
        )
        if summary.exists():
            value = json.loads(summary.read_text(encoding="utf-8"))
            if value.get("passed") is True:
                print(
                    json.dumps(
                        {
                            "scene": scene,
                            "index": index,
                            "total": len(scenes),
                            "resumed": True,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
        print(
            json.dumps(
                {
                    "scene": scene,
                    "index": index,
                    "total": len(scenes),
                    "resumed": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        command = [
            sys.executable,
            str(collector),
            "--schedule",
            str(schedule),
            "--scene-registry",
            str(args.scene_registry.resolve()),
            "--protocol",
            str(args.protocol.resolve()),
            "--corpus-root",
            str(args.corpus_root.resolve()),
            "--scene",
            scene,
            "--resume",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
