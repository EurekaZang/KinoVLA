#!/usr/bin/env python3
"""Build a development-only two-episode smoke input for the frozen collector.

This uses a previously exposed scene/material/pair and therefore can never be
mistaken for confirmatory evidence.  Its sole purpose is to execute every
collector adapter before F0 is signed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "outputs/kinofail_realistic/design_scale_v8_replication_v1/full_schedule.jsonl"
)
GROUP_ID = "cf_0112c0b60e40f34a56a1"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v1.py"
RUNTIME = ROOT / "kino_vla/data/runtime_manifest.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = [
        json.loads(line)
        for line in SOURCE.read_text(encoding="utf-8").splitlines()
        if line
    ]
    pair = [
        dict(row)
        for row in rows
        if str(row["counterfactual_group_id"]) == GROUP_ID
    ]
    if len(pair) != 2:
        raise RuntimeError("development smoke pair is unavailable")
    for row in pair:
        row["physical_nuisance"] = {
            "profile_index": 0,
            "start_progress_m": 0.02,
            "start_lateral_offset_m": -0.05,
            "start_heading_offset_rad": -0.035,
            "forward_speed_mps": 0.21,
            "controller_target_lateral_offset_m": 0.0,
            "physics_seed": int(row["operator_seed"]),
            "pair_shared": True,
        }
    output.mkdir(parents=True, exist_ok=False)
    schedule_path = output / "schedule.jsonl"
    schedule_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in pair),
        encoding="utf-8",
    )
    representative = pair[0]
    protocol = {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": "kinofail-confirmatory-collector-prefreeze-smoke-v1",
        "status": "frozen",
        "benchmark_id": str(representative["benchmark_id"]),
        "development_only": True,
        "confirmatory_evidence_eligible": False,
        "schedule_sha256": _sha256(schedule_path),
        "collector_sha256": _sha256(COLLECTOR),
        "runtime_manifest_sha256": _sha256(RUNTIME),
        "allowed": {
            "counterfactual_group_ids": [GROUP_ID],
            "target_operators": [str(representative["target_operator"])],
            "scene_families": [str(representative["scene_family"])],
            "physical_realizations": [
                str(representative["physical_realization"])
            ],
            "geometry_profiles": [str(representative["geometry_profile"])],
        },
    }
    _write_json(output / "protocol.json", protocol)
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "kinofail.confirmatory-collector-smoke-input.v1",
            "development_only": True,
            "source_schedule_sha256": _sha256(SOURCE),
            "schedule_sha256": _sha256(schedule_path),
            "protocol_sha256": _sha256(output / "protocol.json"),
            "collector_sha256": _sha256(COLLECTOR),
            "runtime_manifest_sha256": _sha256(RUNTIME),
            "counterfactual_group_id": GROUP_ID,
        },
    )
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
