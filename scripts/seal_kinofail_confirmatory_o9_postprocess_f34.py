#!/usr/bin/env python3
"""Seal direct-event snapshots and frozen features for accepted F33 O9 pairs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.build_kinofail_confirmatory_o9_pilot_f32 import (
    ASSET_LOCK,
    REGISTRY,
    ROOT,
    RUNTIME,
    SEMANTICS,
    read_jsonl,
    sha256,
    write_json,
    write_jsonl,
)


F33 = ROOT / "outputs/kinofail_confirmatory_o9_final_f33"
OUTPUT = ROOT / "outputs/kinofail_confirmatory_o9_postprocess_f34"
EVAL = ROOT / "outputs/eval/unified_moe_v3_o9_direct_f34"
SNAPSHOT_SCRIPT = ROOT / "scripts/build_kinofail_o9_direct_snapshots_f34.py"
VISUAL_SCRIPT = ROOT / "scripts/extract_kinofail_realistic_features.py"
UNIFIED_SCRIPT = ROOT / "scripts/build_kinofail_unified_invariant_features_v1.py"
RUNNER = ROOT / "scripts/run_kinofail_confirmatory_o9_postprocess_f34.py"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    if OUTPUT.exists() or EVAL.exists():
        raise FileExistsError("refusing to overwrite F34 postprocess seal or evaluation")
    f33_seal_path = F33 / "seal_manifest.json"
    f33_audit_path = F33 / "final_audit.json"
    f33_schedule_path = F33 / "schedule.jsonl"
    f33_seal = read_json(f33_seal_path)
    f33_audit = read_json(f33_audit_path)
    if (
        f33_seal.get("passed") is not True
        or f33_audit.get("passed") is not True
        or f33_audit.get("model_prediction_feature_label_outcome_or_score_read") is not False
        or int(f33_audit.get("counts", {}).get("strictly_accepted_o9_pairs", 0)) < 750
    ):
        raise RuntimeError("F33 did not pass the frozen physical confirmation gates")
    forbidden = [
        path
        for path in EVAL.rglob("*")
        if path.is_file() and any(token in path.name.lower() for token in ("prediction", "score"))
    ] if EVAL.exists() else []
    if forbidden:
        raise RuntimeError("model prediction or score artifact exists before F34 seal")

    accepted_ids = {str(row["pair_id"]) for row in f33_audit["accepted"]}
    rows = [
        row
        for row in read_jsonl(f33_schedule_path)
        if str(row["counterfactual_group_id"]) in accepted_ids
    ]
    if len(rows) != 2 * len(accepted_ids) or {
        str(row["counterfactual_group_id"]) for row in rows
    } != accepted_ids:
        raise RuntimeError("F34 accepted schedule does not exactly match F33 audit")
    schedule_path = OUTPUT / "accepted_schedule.jsonl"
    write_jsonl(schedule_path, rows)

    protocol_path = OUTPUT / "snapshot_protocol.json"
    protocol = {
        "schema_version": "kinofail.realistic-snapshot-protocol.v2",
        "protocol_id": "kinofail-confirmatory-o9-direct-postprocess-f34",
        "status": "frozen_before_model_blind_feature_extraction",
        "development_only": False,
        "a8_in_scope": False,
        "source_corpus_root": str(Path(str(f33_seal["corpus_root"]))),
        "source_schedule": str(schedule_path.relative_to(ROOT)),
        "source_schedule_sha256": sha256(schedule_path),
        "output_dir": str((EVAL / "snapshots").relative_to(ROOT)),
        "selection": {
            "operators_with_admitted_event_adapter": ["O9_high_centering"],
            "require_complete_counterfactual_pair": True,
            "require_evaluation_eligible": False,
            "require_runtime_validation_passed": True,
            "allow_nonblocking_runtime_issue_suffixes": [
                "appearance_effect_too_small",
                "rgb_spatial_contrast_too_low",
            ],
            "required_collection_protocol_id": "kinofail-confirmatory-o9-direct-final-f33",
            "on_temporal_alignment_failure": "exclude_complete_pair_and_audit",
        },
        "operator_event_adapters": {
            "coverage": ["O9_high_centering"],
            "selection_rule": "first frame whose cumulative telemetry passes the frozen strict direct O9 semantic contract",
        },
        "temporal_alignment": {
            "anchor": "first_strict_direct_o9_semantic_event_in_anomaly_only",
            "decision_delay_s": 0.0,
            "decision_delay_s_by_operator": {},
            "minimum_decision_time_s": 0.42,
            "nominal_alignment": "reuse_anomaly_event_and_decision_times_exactly",
            "rgb_offsets_from_decision_s": [-0.4, -0.3, -0.2, -0.1, 0.0],
            "max_rgb_target_skew_s": 0.041,
            "proprio_window_s": 0.5,
            "proprio_samples": 21,
            "max_proprio_end_skew_s": 0.021,
        },
        "appearance_interventions": {
            "include_all_manifest_views": True,
            "require_timestamp_identity_across_views": True,
            "share_proprio_across_views": True,
            "count_views_as_independent_physical_samples": False,
        },
        "freeze_provenance": {
            "collector": f33_seal["protocol"],
            "collector_seal": str(f33_seal_path.relative_to(ROOT)),
            "collector_seal_sha256": sha256(f33_seal_path),
            "physical_audit": str(f33_audit_path.relative_to(ROOT)),
            "physical_audit_sha256": sha256(f33_audit_path),
            "runtime_manifest": str(RUNTIME.relative_to(ROOT)),
            "runtime_manifest_sha256": sha256(RUNTIME),
            "semantic_module": str(SEMANTICS.relative_to(ROOT)),
            "semantic_module_sha256": sha256(SEMANTICS),
            "scene_registry": str(REGISTRY.relative_to(ROOT)),
            "scene_registry_sha256": sha256(REGISTRY),
            "material_lock": str(ASSET_LOCK.relative_to(ROOT)),
            "material_lock_sha256": sha256(ASSET_LOCK),
            "model_predictions_available_at_freeze": False,
        },
        "publication_guard": {
            "may_satisfy_realistic_a0_a7": True,
            "reason": "Prospective F33 direct-contact O9 confirmation; pilot episodes excluded.",
        },
    }
    write_json(protocol_path, protocol)
    dependencies = [
        SNAPSHOT_SCRIPT,
        VISUAL_SCRIPT,
        UNIFIED_SCRIPT,
        RUNNER,
        ROOT / "kino_vla/data/realistic_snapshots.py",
        SEMANTICS,
    ]
    seal = {
        "schema_version": "kinofail.confirmatory-o9-postprocess-f34-seal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_model_blind_feature_extraction",
        "passed": True,
        "model_prediction_feature_label_outcome_or_score_read": False,
        "selection_basis": "F33 strict physical audit accepted=true only",
        "accepted_physical_pairs": len(accepted_ids),
        "minimum_snapshot_pairs": 750,
        "f33_seal": str(f33_seal_path.relative_to(ROOT)),
        "f33_seal_sha256": sha256(f33_seal_path),
        "f33_audit": str(f33_audit_path.relative_to(ROOT)),
        "f33_audit_sha256": sha256(f33_audit_path),
        "accepted_schedule": str(schedule_path.relative_to(ROOT)),
        "accepted_schedule_sha256": sha256(schedule_path),
        "snapshot_protocol": str(protocol_path.relative_to(ROOT)),
        "snapshot_protocol_sha256": sha256(protocol_path),
        "eval_root": str(EVAL.relative_to(ROOT)),
        "dependencies": [{"path": str(path), "sha256": sha256(path)} for path in dependencies],
    }
    seal_path = OUTPUT / "seal_manifest.json"
    write_json(seal_path, seal)
    (OUTPUT / "seal_manifest.sha256").write_text(f"{sha256(seal_path)}  {seal_path.name}\n")
    print(json.dumps(seal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
