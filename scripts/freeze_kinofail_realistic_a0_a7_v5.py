#!/usr/bin/env python3
"""Freeze the A0--A7 replication contract for the five-seed scale-v8 corpus."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/eval/kinofail_realistic_a0_a7_v4.json"
OUT = ROOT / "configs/eval/kinofail_realistic_a0_a7_v5.json"
SCHEDULE = ROOT / "outputs/kinofail_realistic/design_scale_v8_replication_v1/full_schedule.jsonl"
PROTOCOL = ROOT / "configs/data/kinofail_realistic_scale_v8_replication_formal_v2.json"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v7.py"
STATIC_AUDIT = ROOT / "outputs/kinofail_realistic/design_scale_v8_replication_v1/scale_v8_replication_audit.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    value = json.loads(BASE.read_text(encoding="utf-8"))
    value["schema_version"] = "kinofail.realistic-a0-a7-replication-contract.v5"
    value["contract_id"] = "kinofail-realistic-a0-a7-scale-v8-five-seed-20260723"
    value["status"] = "frozen_before_scale_v8_outcomes"
    value["scope"] = (
        "A0-A7 on the hash-frozen five-seed realistic scale-v8 corpus; independently frozen "
        "A1/A4/A6 side experiments are re-reduced and rebound, never replaced by A8."
    )
    value["benchmark"] = {
        "physical_episodes": 1980,
        "counterfactual_groups": 990,
        "appearance_views_per_episode": 3,
        "appearance_view_sequences": 5940,
        "scene_families": 9,
        "scene_families_per_domain": 3,
        "domains": 3,
        "operators": 11,
        "camera_profiles": 3,
        "severity_levels": 2,
        "physical_seeds_per_scene_operator_severity": 5,
    }
    value["claim_policy"].update({
        "corpus_protocol_id": "kinofail_realistic_scale_1980_v8_replication_v2",
        "current_allowed_term_before_real_anchor": "sim2real-oriented synthetic causal benchmark",
        "a8_in_scope": False,
    })
    value["evidence_roots"] = {
        "scene_registry": "outputs/kinofail_realistic/runtime_audit/formal_scale_v8/scene_registry_audit.json",
        "runtime_audit": "outputs/kinofail_realistic/runtime_audit/formal_scale_v8/runtime_audit.json",
        "runtime_coverage": "outputs/kinofail_realistic/runtime_audit/formal_scale_v8/coverage.json",
        "snapshot_development_audit": "outputs/eval/realistic_a0_a7_v5/snapshots/extraction_audit.json",
        "training_manifest": "outputs/eval/realistic_a0_a7_v5/training_manifest.json",
        "inference_manifest": "outputs/eval/realistic_a0_a7_v5/inference_manifest.json",
        "predictions": "outputs/eval/realistic_a0_a7_v5/predictions.jsonl",
    }
    for experiment_id, spec in value["experiments"].items():
        filename = Path(spec["output"]).name
        spec["output"] = f"outputs/eval/realistic_a0_a7_v5/{filename}"
    value["experiments"]["A0"]["acceptance"].update({
        "evaluation_eligible_records": 1980,
        "complete_counterfactual_pairs": 990,
        "publication_freeze_ready": True,
    })
    value["freeze_provenance"] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "base_contract": str(BASE.relative_to(ROOT)),
        "base_contract_sha256": _sha(BASE),
        "schedule": str(SCHEDULE.relative_to(ROOT)),
        "schedule_sha256": _sha(SCHEDULE),
        "collection_protocol": str(PROTOCOL.relative_to(ROOT)),
        "collection_protocol_sha256": _sha(PROTOCOL),
        "collector": str(COLLECTOR.relative_to(ROOT)),
        "collector_sha256": _sha(COLLECTOR),
        "static_design_audit": str(STATIC_AUDIT.relative_to(ROOT)),
        "static_design_audit_sha256": _sha(STATIC_AUDIT),
        "model_outcomes_available_at_freeze": False,
        "physical_collection_started_at_freeze": False,
        "a8_in_scope": False,
    }
    OUT.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(OUT), "sha256": _sha(OUT),
        "episodes": 1980, "pairs": 990, "appearance_sequences": 5940,
        "a8_in_scope": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
