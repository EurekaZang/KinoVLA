#!/usr/bin/env python3
"""Create the one-time hash-locked protocol authorizing all 198 realistic pilot pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, default=ROOT / "outputs/kinofail_realistic/design_scale_v1/pilot_schedule.jsonl")
    parser.add_argument("--scene-registry", type=Path, default=ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json")
    parser.add_argument("--collector", type=Path, default=ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v1.py")
    parser.add_argument("--out", type=Path, default=ROOT / "configs/data/kinofail_realistic_scale_formal_pilot_v1.json")
    parser.add_argument("--protocol-id", default="kinofail_realistic_scale_396_v1")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol: {args.out}")
    rows = _jsonl(args.schedule.resolve())
    pair_ids = sorted({row["counterfactual_group_id"] for row in rows})
    benchmark_ids = {str(row["benchmark_id"]) for row in rows}
    if len(benchmark_ids) != 1:
        raise ValueError(f"schedule must contain one benchmark id: {sorted(benchmark_ids)}")
    protocol = {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": args.protocol_id,
        "status": "frozen",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "benchmark_id": next(iter(benchmark_ids)),
        "scope": "All 396 physical episodes / 198 pairs on nine registry-bound realistic scenes and O1-O11; A8 excluded.",
        "schedule_path": str(args.schedule.resolve().relative_to(ROOT)),
        "schedule_sha256": _sha256(args.schedule.resolve()),
        "scene_registry_path": str(args.scene_registry.resolve().relative_to(ROOT)),
        "scene_registry_sha256": _sha256(args.scene_registry.resolve()),
        "collector_path": str(args.collector.resolve().relative_to(ROOT)),
        "collector_sha256": _sha256(args.collector.resolve()),
        "runtime_manifest_path": "kino_vla/data/runtime_manifest.py",
        "runtime_manifest_sha256": _sha256(ROOT / "kino_vla/data/runtime_manifest.py"),
        "allowed": {
            "counterfactual_group_ids": pair_ids,
            "target_operators": sorted({row["target_operator"] for row in rows}),
            "scene_families": sorted({row["scene_family"] for row in rows}),
            "physical_realizations": sorted({row["physical_realization"] for row in rows}),
            "geometry_profiles": sorted({row["geometry_profile"] for row in rows}),
            "severity_ids": sorted({row["severity_id"] for row in rows}),
            "conditions": ["anomaly", "nominal_counterfactual"],
        },
        "collection_contract": {
            "physical_episodes": 396,
            "counterfactual_pairs": 198,
            "appearance_views_per_episode": 3,
            "scene_families": 9,
            "domains": 3,
            "operators": 11,
            "formal_collection_status": "formal_pilot",
            "runtime_schema": "kinofail.realistic-runtime.v5",
            "outcome_strength_is_reported_not_gated": True,
            "per_scene_parameter_tuning_forbidden": True,
            "blocking_bug_policy": "Only evidence corruption, inactive anomaly mechanisms, or counterfactual leakage blocks collection.",
            "a8_in_scope": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out), "sha256": _sha256(args.out),
        "pairs": len(pair_ids), "episodes": len(rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
