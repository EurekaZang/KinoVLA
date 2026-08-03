#!/usr/bin/env python3
"""Freeze the final nonblocking raw-prefix audit A4 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--base-collector", type=Path, required=True)
    parser.add_argument("--o5-amendment", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    schedule, base = args.schedule.resolve(), args.base_collector.resolve()
    o5, launcher = args.o5_amendment.resolve(), args.launcher.resolve()
    rows = [json.loads(line) for line in schedule.read_text(encoding="utf-8").splitlines() if line]
    payload = {
        "schema_version": "kinofail.realistic-a4-actual-action-protocol.v4",
        "protocol_id": "kinofail-realistic-a4-actual-action-v4",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "final_nonblocking_scale_freeze_before_any_v4_outcome",
        "schedule": str(schedule.relative_to(ROOT)),
        "schedule_sha256": _sha(schedule),
        "collector": str(base.relative_to(ROOT)),
        "collector_sha256": _sha(base),
        "o5_amendment": str(o5.relative_to(ROOT)),
        "o5_amendment_sha256": _sha(o5),
        "launcher": str(launcher.relative_to(ROOT)),
        "launcher_sha256": _sha(launcher),
        "supersedes": "configs/data/kinofail_realistic_a4_actual_action_formal_v3.json",
        "excluded_preflight_roots": [
            "outputs/kinofail_realistic/corpus_a4_actual_action_v1",
            "outputs/kinofail_realistic/corpus_a4_actual_action_v2",
            "outputs/kinofail_realistic/corpus_a4_actual_action_v3",
        ],
        "runtime_pair_gate": "same operator-independent decision step",
        "postrun_pair_gate": {
            "source": "all unquantized raw telemetry rows through decision step",
            "maximum_absolute_state_or_command_difference": 0.001,
            "action_onset": "strictly after decision step",
            "failure_policy": "exclude only the failing pair; do not tune or recollect",
        },
        "o5_trigger": "0.5 s with payload readback active",
        "unchanged": [
            "scene clusters", "reset seeds", "operator severity", "action definitions",
            "outcome rules", "statistics",
        ],
        "case_count": len(rows),
        "physical_episode_count": sum(len(row["actions"]) for row in rows),
        "minimum_episode_seeds_per_action_cell": 10,
        "scene_clusters": sorted({row["scene_cluster"] for row in rows}),
        "operators": sorted({row["operator"] for row in rows}),
        "outcome_policy": "retain weak, null, and unfavorable cells; no further tuning",
        "statistics": "scene-cluster paired bootstrap with case-level paired differences",
    }
    if payload["case_count"] != 50 or payload["physical_episode_count"] != 100:
        raise RuntimeError("invalid A4 schedule size")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
