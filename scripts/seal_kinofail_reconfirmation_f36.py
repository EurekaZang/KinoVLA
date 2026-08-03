#!/usr/bin/env python3
"""Seal the F35 Scale + F30 T3 one-shot blind evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
F29_AUDIT = ROOT / "outputs/kinofail_t3_replenishment_f29/final_audit.json"
F30_ROOT = ROOT / "outputs/eval/unified_moe_v3_t3_replenishment_f30"
F30_AUDIT = F30_ROOT / "combined_design/audit.json"
F33_AUDIT = ROOT / "outputs/kinofail_confirmatory_o9_final_f33/final_audit.json"
F34_AUDIT = ROOT / "outputs/kinofail_confirmatory_o9_postprocess_f34/final_audit.json"
F35_ROOT = ROOT / "outputs/eval/unified_moe_v3_scale_direct_o9_f35"
F35_AUDIT = F35_ROOT / "overlay_audit.json"
A4_SEAL = ROOT / "outputs/freeze/kinofail_reconfirmation_a4_v8/seal_manifest.json"
TARGET = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f36"
OUTPUT = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f36/seal_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    if OUTPUT.exists() or TARGET.exists():
        raise FileExistsError("refusing to overwrite F36 seal or evaluation")
    f29 = read_json(F29_AUDIT)
    f30 = read_json(F30_AUDIT)
    f33 = read_json(F33_AUDIT)
    f34 = read_json(F34_AUDIT)
    f35 = read_json(F35_AUDIT)
    a4_seal = read_json(A4_SEAL)
    if (
        f29.get("passed") is not True
        or f29.get("model_prediction_feature_or_score_read") is not False
        or f30.get("passed") is not True
        or f30.get("status") != "validated_model_blind_union"
        or f30.get("strictly_below_five_percent") is not True
        or int(f30.get("valid_cases", 0)) < 1425
        or f30.get("model_prediction_feature_or_score_read") is not False
        or f33.get("passed") is not True
        or f33.get("model_prediction_feature_label_outcome_or_score_read") is not False
        or f33.get("result_dependent_retry_or_selection") is not False
        or f34.get("passed") is not True
        or f34.get("model_prediction_feature_label_outcome_or_score_read") is not False
        or f35.get("passed") is not True
        or f35.get("model_prediction_or_score_read") is not False
        or f35.get("selection_uses_model_outcome_label_or_score") is not False
        or f35.get("legacy_o9_features_excluded") is not True
        or f35.get("f33_allowed_visual_runtime_warnings", {}).get(
            "all_warning_pairs_retained"
        )
        is not True
        or f35.get("source_sha256", {}).get("f33_final_audit")
        != sha256(F33_AUDIT)
        or f35.get("source_sha256", {}).get("f34_final_audit")
        != sha256(F34_AUDIT)
        or f35.get("source_sha256", {}).get("builder")
        != sha256(ROOT / "scripts/build_kinofail_scale_direct_o9_overlay_f35.py")
        or int(f35.get("counts", {}).get("direct_event_f34_o9_pairs", 0)) < 750
        or float(f35.get("effective_scale_attrition_rate", 1.0)) >= 0.05
        or a4_seal.get("passed") is not True
        or a4_seal.get("status")
        != "sealed_before_f36_prediction_and_confirmatory_a4_outcomes"
        or a4_seal.get("model_prediction_or_outcome_read") is not False
        or not A4_SEAL.with_name("seal_manifest.sha256").is_file()
        or A4_SEAL.with_name("seal_manifest.sha256").read_text().split()[0]
        != sha256(A4_SEAL)
    ):
        raise RuntimeError("invalid F29/F30/F33/F34/F35 model-blind evidence")
    conflict_schedule = ROOT / str(f30["combined_conflict_schedule"])
    if sha256(conflict_schedule) != f30["combined_conflict_schedule_sha256"]:
        raise RuntimeError("F30 combined conflict schedule drift")
    scripts = [
        ROOT / "scripts/finalize_kinofail_reconfirmation_f36.py",
        ROOT / "scripts/build_kinofail_t3_replenishment_f30_overlay.py",
        ROOT / "scripts/build_kinofail_scale_direct_o9_overlay_f35.py",
        ROOT / "scripts/finalize_kinofail_reconfirmation_v8.py",
        ROOT / "scripts/prepare_kinofail_confirmatory_valid_conflict_design_v1.py",
        ROOT / "scripts/build_kinofail_confirmatory_c2_base_features_v1.py",
        ROOT / "scripts/build_kinofail_realistic_c2_v5_features.py",
        ROOT / "scripts/build_kinofail_confirmatory_evaluation_bundle_inputs_v1.py",
        ROOT / "scripts/assemble_kinofail_confirmatory_blind_bundle_v1.py",
        ROOT / "scripts/predict_kinofail_unified_moe_v3_blind.py",
        ROOT / "scripts/build_kinofail_confirmatory_scoring_protocol_v1.py",
        ROOT / "scripts/score_kinofail_unified_moe_v3_confirmatory.py",
        ROOT / "scripts/analyze_kinofail_reconfirmation_a6_v7.py",
        ROOT / "scripts/assemble_kinofail_reconfirmation_a0_a7_v7.py",
        ROOT / "scripts/build_kinofail_reconfirmation_a4_v8_schedule.py",
        ROOT / "scripts/seal_kinofail_reconfirmation_a4_v8.py",
        ROOT / "scripts/run_kinofail_reconfirmation_a4_v8.py",
        ROOT / "scripts/isaac_collect_kinofail_reconfirmation_a4_v7.py",
        ROOT / "scripts/analyze_kinofail_reconfirmation_a4_v8.py",
        ROOT / "scripts/audit_kinofail_reconfirmation_a4_v7_pilot_p1.py",
    ]
    if any(not path.is_file() for path in scripts):
        raise FileNotFoundError("F36 execution dependency missing")
    scale_valid = int(f35["counts"]["effective_valid_scale_pairs"])
    t3_valid = int(f30["valid_cases"])
    manifest = {
        "schema_version": "kinofail.reconfirmation-f36-seal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_direct_o9_scale_t3_blind_prediction",
        "passed": True,
        "model_prediction_or_score_read": False,
        "original_confirmation_attrition_must_be_reported": True,
        "scientific_contract": {
            "frozen_f0_models_unchanged": True,
            "frozen_f1_analysis_unchanged": True,
            "router_features_threshold_eta_and_statistics_unchanged": True,
            "legacy_o9_features_removed": True,
            "o9_change": "insert only F33 strict-contact pairs with F34 direct-event snapshots",
            "scale_non_o9_change": "retain original valid groups plus F25 accepted non-O9 one-to-one replacements",
            "t3_change": "complete-case substitution using only F29 accepted untouched cases",
            "rejected_or_interrupted_replacements_excluded": True,
            "selection_uses_model_predictions_or_outcome_strength": False,
            "score_once": True,
        },
        "counts": {
            "scale_planned_pairs": 10560,
            "scale_valid_pairs": scale_valid,
            "scale_remaining_attrition": 10560 - scale_valid,
            "scale_direct_o9_pairs": int(f35["counts"]["direct_event_f34_o9_pairs"]),
            "t3_planned_cases": 1500,
            "t3_valid_cases": t3_valid,
            "t3_remaining_attrition": int(f30["invalid_case_count"]),
            "t3_accepted_replacement_cases": int(f30["accepted_replacement_cases"]),
            "t2_planned_and_valid_cases": 1500,
            "combined_conflict_planned_cases": 3000,
            "combined_conflict_valid_cases": 1500 + t3_valid,
            "combined_conflict_remaining_attrition": int(f30["invalid_case_count"]),
        },
        "source_sha256": {
            "f29_final_audit": sha256(F29_AUDIT),
            "f30_design_audit": sha256(F30_AUDIT),
            "f33_final_audit": sha256(F33_AUDIT),
            "f34_final_audit": sha256(F34_AUDIT),
            "f35_scale_audit": sha256(F35_AUDIT),
            "a4_pre_prediction_seal": sha256(A4_SEAL),
            "f30_combined_schedule": sha256(F30_ROOT / "combined_design/schedule.jsonl"),
            "f30_combined_case_schedule": sha256(F30_ROOT / "combined_design/case_schedule.jsonl"),
            "f30_combined_conflict_schedule": sha256(conflict_schedule),
            "f30_artifact_inventory": sha256(F30_ROOT / "artifact_inventory.jsonl"),
            "execution_scripts": {str(path.relative_to(ROOT)): sha256(path) for path in scripts},
        },
        "target_eval_root": str(TARGET.relative_to(ROOT)),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    OUTPUT.with_name("seal_manifest.sha256").write_text(f"{sha256(OUTPUT)}  {OUTPUT.name}\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
