#!/usr/bin/env python3
"""Freeze the targeted realistic A6 boundary collection before simulation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    schedule = ROOT / "outputs/kinofail_realistic/design_a6_boundary_v2/schedule.jsonl"
    audit_path = ROOT / "outputs/kinofail_realistic/design_a6_boundary_v2/audit.json"
    registry = ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json"
    collector = ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v6.py"
    output = ROOT / "configs/data/kinofail_realistic_a6_boundary_formal_v2.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    rows = [json.loads(line) for line in schedule.read_text(encoding="utf-8").splitlines() if line]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True or len(rows) != 150:
        raise RuntimeError("A6 design audit did not pass")
    pair_ids = sorted({row["counterfactual_group_id"] for row in rows})
    protocol = {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": "kinofail_realistic_a6_boundary_150_v2",
        "status": "frozen",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "benchmark_id": "kinofail_realistic_a6_boundary_v2",
        "scope": "A6 targeted boundary extension: five measured operator families, nominal plus three anomaly levels, five seeds; A8 excluded.",
        "supersedes_unrun_protocol": "configs/data/kinofail_realistic_a6_boundary_formal_v1.json",
        "amendment_reason": "Pair boundary_seed_index across levels by the already-frozen scene_seed rather than per-level pair-hash order; no v1 simulation was launched.",
        "schedule_path": str(schedule.relative_to(ROOT)),
        "schedule_sha256": _sha(schedule),
        "design_audit_path": str(audit_path.relative_to(ROOT)),
        "design_audit_sha256": _sha(audit_path),
        "scene_registry_path": str(registry.relative_to(ROOT)),
        "scene_registry_sha256": _sha(registry),
        "collector_path": str(collector.relative_to(ROOT)),
        "collector_sha256": _sha(collector),
        "runtime_manifest_path": "kino_vla/data/runtime_manifest.py",
        "runtime_manifest_sha256": _sha(ROOT / "kino_vla/data/runtime_manifest.py"),
        "allowed": {
            "counterfactual_group_ids": pair_ids,
            "target_operators": sorted({row["target_operator"] for row in rows}),
            "scene_families": sorted({row["scene_family"] for row in rows}),
            "physical_realizations": sorted({row["physical_realization"] for row in rows}),
            "severity_ids": ["mild", "moderate", "severe"],
            "conditions": ["anomaly", "nominal_counterfactual"],
        },
        "collection_contract": {
            "physical_episodes": 150,
            "counterfactual_pairs": 75,
            "appearance_views_per_episode": 3,
            "operator_parameter_families": 5,
            "levels_per_family_including_nominal": 4,
            "independent_seeds_per_level": 5,
            "outcome_strength_is_reported_not_gated": True,
            "per_scene_parameter_tuning_forbidden": True,
            "a8_in_scope": False,
        },
    }
    output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": _sha(output), "pairs": len(pair_ids)}, indent=2))


if __name__ == "__main__":
    main()
