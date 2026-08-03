#!/usr/bin/env python3
"""Freeze the final numerical-equivalence amendment for scale A4."""

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
    schedule = args.schedule.resolve()
    base = args.base_collector.resolve()
    o5 = args.o5_amendment.resolve()
    launcher = args.launcher.resolve()
    rows = [json.loads(line) for line in schedule.read_text(encoding="utf-8").splitlines() if line]
    payload = {
        "schema_version": "kinofail.realistic-a4-actual-action-protocol.v3",
        "protocol_id": "kinofail-realistic-a4-actual-action-v3",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "final_scale_freeze_after_runtime_preflight_before_any_v3_outcome",
        "schedule": str(schedule.relative_to(ROOT)),
        "schedule_sha256": _sha(schedule),
        "collector": str(base.relative_to(ROOT)),
        "collector_sha256": _sha(base),
        "o5_amendment": str(o5.relative_to(ROOT)),
        "o5_amendment_sha256": _sha(o5),
        "launcher": str(launcher.relative_to(ROOT)),
        "launcher_sha256": _sha(launcher),
        "supersedes": "configs/data/kinofail_realistic_a4_actual_action_formal_v2.json",
        "excluded_preflight_roots": [
            "outputs/kinofail_realistic/corpus_a4_actual_action_v1",
            "outputs/kinofail_realistic/corpus_a4_actual_action_v2",
        ],
        "preflight_findings": {
            "v1": "O5 severe payload fell before the 2.2 s action branch",
            "v2": {
                "finding": "O5 raw prefix SHA mismatch was GPU floating-point noise",
                "maximum_position_difference_m": 2.384185791015625e-06,
                "maximum_velocity_difference_mps": 7.860362529754639e-06,
                "maximum_tilt_difference_rad": 5.175458581085923e-06,
                "maximum_slip_ratio_difference": 6.760470569133759e-05,
            },
        },
        "final_amendments": [
            "O5 branches at 0.5 s with payload readback active",
            "paired physical prefixes use 1e-3 numerical quantization before SHA256",
            "unquantized raw telemetry remains stored for post-run maximum-difference audit",
        ],
        "unchanged": [
            "scene clusters", "reset seeds", "operator severity", "action definitions",
            "outcome rules", "statistics",
        ],
        "case_count": len(rows),
        "physical_episode_count": sum(len(row["actions"]) for row in rows),
        "minimum_episode_seeds_per_action_cell": 10,
        "scene_clusters": sorted({row["scene_cluster"] for row in rows}),
        "operators": sorted({row["operator"] for row in rows}),
        "paired_prefix_gate": "same decision step and 1e-3-quantized physical-prefix SHA256",
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
