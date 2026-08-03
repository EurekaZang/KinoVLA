#!/usr/bin/env python3
"""Seal the Scale+T3 replenished evaluation before blind prediction."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCALE_OVERLAY = ROOT / "outputs/eval/unified_moe_v3_replenishment_f25/accepted_overlay/overlay_audit.json"
F25_AUDIT = ROOT / "outputs/kinofail_replenishment_f25/final_audit.json"
F29_AUDIT = ROOT / "outputs/kinofail_t3_replenishment_f29/final_audit.json"
F30_ROOT = ROOT / "outputs/eval/unified_moe_v3_t3_replenishment_f30"
F30_AUDIT = F30_ROOT / "combined_design/audit.json"
TARGET = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_replenished_f31"
OUTPUT = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f31_scale_t3_overlay/seal_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    if OUTPUT.exists() or TARGET.exists():
        raise FileExistsError("refusing to overwrite F31 seal or evaluation")
    scale = read_json(SCALE_OVERLAY)
    f25 = read_json(F25_AUDIT)
    f29 = read_json(F29_AUDIT)
    f30 = read_json(F30_AUDIT)
    if (
        scale.get("passed") is not True
        or scale.get("model_prediction_or_score_read") is not False
        or scale.get("counts", {}).get("accepted_replacements") != 1_005
        or f25.get("passed") is not True
        or f29.get("passed") is not True
        or f29.get("model_prediction_feature_or_score_read") is not False
        or f30.get("passed") is not True
        or f30.get("status") != "validated_model_blind_union"
        or f30.get("strictly_below_five_percent") is not True
        or f30.get("valid_cases", 0) < 1_425
        or f30.get("model_prediction_feature_or_score_read") is not False
    ):
        raise RuntimeError("invalid F25/F29/F30 model-blind overlay evidence")
    conflict_schedule = ROOT / str(f30["combined_conflict_schedule"])
    if sha256(conflict_schedule) != f30["combined_conflict_schedule_sha256"]:
        raise RuntimeError("F30 combined conflict schedule drift")

    scripts = [
        ROOT / "scripts/finalize_kinofail_reconfirmation_f31.py",
        ROOT / "scripts/build_kinofail_t3_replenishment_f30_overlay.py",
        ROOT / "scripts/build_kinofail_replenishment_f26_overlay.py",
        ROOT / "scripts/finalize_kinofail_reconfirmation_v8.py",
        ROOT / "scripts/prepare_kinofail_confirmatory_valid_conflict_design_v1.py",
        ROOT / "scripts/build_kinofail_confirmatory_c2_base_features_v1.py",
        ROOT / "scripts/build_kinofail_realistic_c2_v5_features.py",
        ROOT / "scripts/build_kinofail_confirmatory_evaluation_bundle_inputs_v1.py",
        ROOT / "scripts/assemble_kinofail_confirmatory_blind_bundle_v1.py",
        ROOT / "scripts/predict_kinofail_unified_moe_v3_blind.py",
        ROOT / "scripts/build_kinofail_confirmatory_scoring_protocol_v1.py",
        ROOT / "scripts/score_kinofail_unified_moe_v3_confirmatory.py",
    ]
    if any(not path.is_file() for path in scripts):
        raise FileNotFoundError("F31 execution dependency missing")
    manifest = {
        "schema_version": "kinofail.reconfirmation-f31-seal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_scale_t3_replenished_blind_prediction",
        "passed": True,
        "posthoc_supplementary_replenishment": True,
        "original_confirmation_attrition_must_be_reported": True,
        "model_prediction_or_score_read": False,
        "scientific_contract": {
            "frozen_f0_models_unchanged": True,
            "frozen_f1_analysis_unchanged": True,
            "router_features_threshold_and_statistics_unchanged": True,
            "scale_change": "add only F25 audit-accepted one-to-one replacements",
            "t3_change": "complete-case substitution using only F29 audit-accepted untouched cases",
            "rejected_or_interrupted_replacements_excluded": True,
            "selection_uses_model_predictions_or_outcome_strength": False,
            "score_once": True,
        },
        "counts": {
            "scale_planned_pairs": 10_560,
            "scale_valid_pairs": 10_148,
            "scale_remaining_attrition": 412,
            "t3_planned_cases": 1_500,
            "t3_valid_cases": int(f30["valid_cases"]),
            "t3_remaining_attrition": int(f30["invalid_case_count"]),
            "t3_accepted_replacement_cases": int(f30["accepted_replacement_cases"]),
        },
        "source_sha256": {
            "scale_overlay_audit": sha256(SCALE_OVERLAY),
            "f25_final_audit": sha256(F25_AUDIT),
            "f29_final_audit": sha256(F29_AUDIT),
            "f30_design_audit": sha256(F30_AUDIT),
            "f30_combined_schedule": sha256(F30_ROOT / "combined_design/schedule.jsonl"),
            "f30_combined_case_schedule": sha256(F30_ROOT / "combined_design/case_schedule.jsonl"),
            "f30_combined_conflict_schedule": sha256(conflict_schedule),
            "f30_artifact_inventory": sha256(F30_ROOT / "artifact_inventory.jsonl"),
            "execution_scripts": {
                str(path.relative_to(ROOT)): sha256(path) for path in scripts
            },
        },
        "target_eval_root": str(TARGET.relative_to(ROOT)),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    OUTPUT.with_name("seal_manifest.sha256").write_text(
        f"{sha256(OUTPUT)}  {OUTPUT.name}\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
