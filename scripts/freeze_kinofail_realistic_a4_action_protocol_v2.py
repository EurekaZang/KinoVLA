#!/usr/bin/env python3
"""Freeze the sole A4 preflight amendment while preserving the v1 implementation hash."""

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
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    schedule = args.schedule.resolve()
    base = args.base_collector.resolve()
    launcher = args.launcher.resolve()
    rows = [json.loads(line) for line in schedule.read_text(encoding="utf-8").splitlines() if line]
    payload = {
        "schema_version": "kinofail.realistic-a4-actual-action-protocol.v2",
        "protocol_id": "kinofail-realistic-a4-actual-action-v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_after_v1_runtime_preflight_failure_before_any_v2_outcome",
        "schedule": str(schedule.relative_to(ROOT)),
        "schedule_sha256": _sha(schedule),
        "collector": str(base.relative_to(ROOT)),
        "collector_sha256": _sha(base),
        "launcher": str(launcher.relative_to(ROOT)),
        "launcher_sha256": _sha(launcher),
        "supersedes": "configs/data/kinofail_realistic_a4_actual_action_formal_v1.json",
        "excluded_preflight_root": "outputs/kinofail_realistic/corpus_a4_actual_action_v1",
        "amendment": {
            "scope": "O5 decision time only",
            "before": "2.2 s; severe payload fell before either action could branch",
            "after": "0.5 s after reset with payload readback active",
            "scientific_reason": "ensure both actual-action arms are physically executed",
            "unchanged": [
                "scene clusters", "reset seeds", "operator severity", "action definitions",
                "paired-prefix gate", "outcome rules", "statistics",
            ],
        },
        "case_count": len(rows),
        "physical_episode_count": sum(len(row["actions"]) for row in rows),
        "minimum_episode_seeds_per_action_cell": 10,
        "scene_clusters": sorted({row["scene_cluster"] for row in rows}),
        "operators": sorted({row["operator"] for row in rows}),
        "paired_prefix_gate": "exact decision step and canonical telemetry-row SHA256 equality",
        "outcome_policy": "retain all weak, null, and unfavorable cells; no further tuning",
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
