#!/usr/bin/env python3
"""Collect C2 v4 T3 with one v3-collector Isaac process per pair."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


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
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(
            str(row["counterfactual_group_id"]), []
        ).append(row)
    if not groups or not all(
        len(values) == 2
        and {
            str(row["condition"]) for row in values
        }
        == {"nominal_counterfactual", "anomaly"}
        for values in groups.values()
    ):
        raise RuntimeError("C2 v4 isolated T3 selection is invalid")
    ordered = sorted(
        groups,
        key=lambda group_id: (
            str(groups[group_id][0]["scene_family"]),
            str(groups[group_id][0]["target_operator"]),
            group_id,
        ),
    )
    environment = os.environ.copy()
    environment.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    environment.setdefault("HF_HUB_OFFLINE", "1")
    collector = (
        ROOT
        / "scripts/isaac_collect_kinofail_realistic_pair_v3.py"
    )
    corpus = args.corpus_root.resolve()
    for index, group_id in enumerate(ordered, start=1):
        summary_path = (
            corpus / "pair_summaries" / f"{group_id}.json"
        )
        representative = groups[group_id][0]
        if summary_path.exists():
            summary = _json(summary_path)
            if summary.get("passed") is True:
                print(
                    json.dumps(
                        {
                            "progress": f"{index}/{len(ordered)}",
                            "counterfactual_group_id": group_id,
                            "scene_cluster": representative[
                                "scene_family"
                            ],
                            "operator": representative[
                                "target_operator"
                            ],
                            "resumed": True,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
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
            str(corpus),
            "--counterfactual-group-id",
            group_id,
            "--headless",
        ]
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(ordered)}",
                    "counterfactual_group_id": group_id,
                    "scene_cluster": representative[
                        "scene_family"
                    ],
                    "operator": representative[
                        "target_operator"
                    ],
                    "resumed": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=False,
        )
        if (
            completed.returncode != 0
            or not summary_path.exists()
            or _json(summary_path).get("passed") is not True
        ):
            return (
                int(completed.returncode)
                if completed.returncode != 0
                else 2
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
