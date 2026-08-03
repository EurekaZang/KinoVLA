#!/usr/bin/env python3
"""Freeze the final A1 bounded-horizon throughput amendment before simulation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    schedule = ROOT / "outputs/kinofail_realistic/design_a1_matched_v1/schedule.jsonl"
    audit_path = ROOT / "outputs/kinofail_realistic/design_a1_matched_v1/audit.json"
    registry = ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json"
    collector = ROOT / "scripts/isaac_collect_kinofail_realistic_a1_matched_v3.py"
    prior = ROOT / "configs/data/kinofail_realistic_a1_matched_formal_v2.json"
    output = ROOT / "configs/data/kinofail_realistic_a1_matched_formal_v3.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    rows = [json.loads(line) for line in schedule.read_text(encoding="utf-8").splitlines() if line]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True or len(rows) != 36:
        raise RuntimeError("A1 matched design audit did not pass")
    allowed = {
        "counterfactual_group_ids": sorted({row["counterfactual_group_id"] for row in rows}),
        "target_operators": ["O2_compliance", "O4_tether"],
        "scene_families": sorted({row["scene_family"] for row in rows}),
        "physical_realizations": sorted({row["physical_realization"] for row in rows}),
        "geometry_profiles": sorted({row["geometry_profile"] for row in rows}),
        "severity_ids": sorted({row["severity_id"] for row in rows}),
        "conditions": ["anomaly", "nominal_counterfactual"],
    }
    protocol = {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": "kinofail_realistic_a1_matched_o2_o4_36_v3",
        "status": "frozen",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "benchmark_id": "kinofail_realistic_a1_matched_v1",
        "scope": "A1 fresh-app/deep-reset O2/O4 pre-contact matched construct across nine realistic scene clusters; A8 excluded.",
        "supersedes_uncollected_protocol": str(prior.relative_to(ROOT)),
        "superseded_protocol_sha256": _sha(prior),
        "amendment_reason": "Throughput only: stop A1 after 220 steps/4.4 s because the frozen construct ends at first contact with a region starting at 0.70 m. Full pre-contact proprio/telemetry and the first mechanism readback remain; no v1/v2 manifest exists.",
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
        "allowed": allowed,
        "collection_contract": {
            "physical_episodes": 36,
            "counterfactual_pairs": 18,
            "cross_operator_match_groups": 9,
            "independent_scene_clusters": 9,
            "domains": 3,
            "appearance_views_per_episode": 3,
            "same_scene_seed_camera_and_reset_seed_required": True,
            "snapshot_alignment": "end exactly at first measured operator contact; no positive post-contact delay",
            "fresh_isaac_app_per_counterfactual_pair": True,
            "deep_reset_before_each_episode": True,
            "construct_horizon_control_steps": 220,
            "construct_horizon_s": 4.4,
            "rgb_capture_start_route_progress_m": 0.45,
            "rgb_capture_max_frames_per_view": 12,
            "operator_region_start_route_progress_m": 0.70,
            "operator_first_contact_and_local_qa_required": True,
            "post_contact_outcome_not_an_a1_endpoint": True,
            "per_scene_parameter_tuning_forbidden": True,
            "a8_in_scope": False,
        },
    }
    output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": _sha(output), "pairs": 18}, indent=2))


if __name__ == "__main__":
    main()
