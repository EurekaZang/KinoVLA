#!/usr/bin/env python3
"""Hash-freeze the realistic A4 actual-action protocol before collection."""

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
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    schedule = args.schedule.resolve()
    collector = args.collector.resolve()
    rows = [json.loads(line) for line in schedule.read_text(encoding="utf-8").splitlines() if line]
    payload = {
        "schema_version": "kinofail.realistic-a4-actual-action-protocol.v1",
        "protocol_id": "kinofail-realistic-a4-actual-action-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_before_any_a4_v1_outcome",
        "schedule": str(schedule.relative_to(ROOT)),
        "schedule_sha256": _sha(schedule),
        "collector": str(collector.relative_to(ROOT)),
        "collector_sha256": _sha(collector),
        "case_count": len(rows),
        "physical_episode_count": sum(len(row["actions"]) for row in rows),
        "minimum_episode_seeds_per_action_cell": 10,
        "scene_clusters": sorted({row["scene_cluster"] for row in rows}),
        "operators": sorted({row["operator"] for row in rows}),
        "actions": {
            "O2_compliance": ["continue", "slow_high_step"],
            "O4_tether": ["continue", "backstep_release"],
            "O5_payload": ["continue", "hold_and_request"],
            "O8_invisible_collider": ["continue", "backstep_detour_replan"],
            "O9_high_centering": ["continue", "raise_body_slow_cross"],
        },
        "decision_rule": "fixed operator-specific physical-onset dwell, independent of action arm",
        "paired_prefix_gate": "exact decision step and canonical telemetry-row SHA256 equality",
        "outcome_policy": "retain all weak, null, and unfavorable cells; never tune after outcome",
        "statistics": "scene-cluster paired bootstrap with case-level paired differences",
    }
    if payload["case_count"] != 50 or payload["physical_episode_count"] != 100:
        raise RuntimeError("schedule does not satisfy frozen A4 size")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
