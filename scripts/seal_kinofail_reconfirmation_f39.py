#!/usr/bin/env python3
"""Seal the F38-S2 T3 + F35 Scale one-shot blind evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
F33_AUDIT = ROOT / "outputs/kinofail_confirmatory_o9_final_f33/final_audit.json"
F34_AUDIT = ROOT / "outputs/kinofail_confirmatory_o9_postprocess_f34/final_audit.json"
F35_ROOT = ROOT / "outputs/eval/unified_moe_v3_scale_direct_o9_f35"
F35_AUDIT = F35_ROOT / "overlay_audit.json"
F36_SEAL = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f36/seal_manifest.json"
F36_TARGET = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f36"
F37_AUDIT = ROOT / "outputs/kinofail_reconfirmation_f37_failure_audit_v1/audit.json"
F38_RAW_AUDIT = ROOT / "outputs/kinofail_t3_runin_f38_formal_s1/final_audit.json"
F38_TASK_ROOT = ROOT / "outputs/kinofail_t3_runin_f38_s1_task_aligned_audit"
F38_TASK_AUDIT = F38_TASK_ROOT / "audit.json"
F38_S2_ROOT = ROOT / "outputs/eval/unified_moe_v3_t3_runin_f38_s2"
F38_S2_AUDIT = F38_S2_ROOT / "combined_design/audit.json"
F38_S2_UNION = Path("/data/eureka/KinoVLA/outputs/kinofail_t3_runin_f38_s2/union")
A4_SEAL = ROOT / "outputs/freeze/kinofail_reconfirmation_a4_v8/seal_manifest.json"
TARGET = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f39"
OUTPUT = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f39/seal_manifest.json"


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
        raise FileExistsError("refusing to overwrite F39 seal or evaluation")
    f33 = read_json(F33_AUDIT)
    f34 = read_json(F34_AUDIT)
    f35 = read_json(F35_AUDIT)
    f36 = read_json(F36_SEAL)
    f37 = read_json(F37_AUDIT)
    f38_raw = read_json(F38_RAW_AUDIT)
    f38_task = read_json(F38_TASK_AUDIT)
    f38_s2 = read_json(F38_S2_AUDIT)
    a4 = read_json(A4_SEAL)
    forbidden_f36 = [
        path
        for path in F36_TARGET.rglob("*")
        if path.is_file()
        and any(
            token in path.name.lower()
            for token in ("prediction", "truth_key", "score", "confirmatory_report")
        )
    ]
    if (
        f33.get("passed") is not True
        or f34.get("passed") is not True
        or f35.get("passed") is not True
        or f35.get("model_prediction_or_score_read") is not False
        or f35.get("legacy_o9_features_excluded") is not True
        or int(f35.get("counts", {}).get("effective_valid_scale_pairs", 0)) < 10_032
        or float(f35.get("effective_scale_attrition_rate", 1.0)) >= 0.05
        or f36.get("passed") is not True
        or f36.get("model_prediction_or_score_read") is not False
        or forbidden_f36
        or f37.get("passed") is not True
        or f37.get("status")
        != "result_blind_systematic_temporal_contract_failure_preserved"
        or f37.get("model_prediction_truth_key_or_score_read") is not False
        or f38_raw.get("passed") is not False
        or f38_raw.get("model_prediction_truth_key_or_score_read") is not False
        or f38_raw.get("result_dependent_selection_or_retry") is not False
        or f38_task.get("passed") is not True
        or f38_task.get("counts_as_confirmatory_evidence") is not False
        or f38_task.get("model_prediction_truth_key_or_score_read") is not False
        or f38_task.get("result_dependent_selection_or_retry") is not False
        or int(f38_task.get("counts", {}).get("accepted_cases", 0)) != 1_430
        or float(f38_task.get("case_attrition_rate", 1.0)) >= 0.05
        or f38_s2.get("passed") is not True
        or f38_s2.get("status") != "validated_model_blind_task_aligned_union"
        or f38_s2.get("model_prediction_feature_truth_key_or_score_read") is not False
        or f38_s2.get("result_dependent_selection_or_retry") is not False
        or int(f38_s2.get("valid_cases", 0)) != 1_430
        or f38_s2.get("strictly_below_five_percent") is not True
        or f38_s2.get("source_sha256", {}).get("task_aligned_audit")
        != sha256(F38_TASK_AUDIT)
        or not F38_S2_UNION.is_dir()
        or a4.get("passed") is not True
        or a4.get("status")
        != "sealed_before_f36_prediction_and_confirmatory_a4_outcomes"
        or a4.get("model_prediction_or_outcome_read") is not False
        or not A4_SEAL.with_name("seal_manifest.sha256").is_file()
        or A4_SEAL.with_name("seal_manifest.sha256").read_text().split()[0]
        != sha256(A4_SEAL)
    ):
        raise RuntimeError("invalid model-blind F33-F38/A4 predecessor evidence")
    for key, relative in (
        ("accepted_cases", "accepted_cases.jsonl"),
        ("rejected_cases", "rejected_cases.jsonl"),
        ("pair_audits", "pair_audits.jsonl"),
    ):
        path = F38_TASK_ROOT / relative
        if sha256(path) != f38_task["output_sha256"][key]:
            raise RuntimeError(f"F38 task-aligned artifact drift: {path}")
    for path, expected in (
        (F38_S2_ROOT / "combined_design/schedule.jsonl", f38_s2["schedule_sha256"]),
        (
            F38_S2_ROOT / "combined_design/case_schedule.jsonl",
            f38_s2["case_schedule_sha256"],
        ),
        (
            ROOT / str(f38_s2["artifact_inventory"]),
            f38_s2["artifact_inventory_sha256"],
        ),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"F38-S2 overlay drift: {path}")

    scripts = [
        Path(__file__).resolve(),
        ROOT / "scripts/finalize_kinofail_reconfirmation_f39.py",
        ROOT / "scripts/audit_kinofail_t3_runin_f38_s1_task_alignment.py",
        ROOT / "scripts/build_kinofail_t3_runin_f38_s2_overlay.py",
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
        ROOT / "scripts/run_kinofail_reconfirmation_a4_v8.py",
        ROOT / "scripts/analyze_kinofail_reconfirmation_a4_v8.py",
    ]
    if any(not path.is_file() for path in scripts):
        raise FileNotFoundError("F39 execution dependency missing")
    scale_valid = int(f35["counts"]["effective_valid_scale_pairs"])
    t3_valid = int(f38_s2["valid_cases"])
    manifest = {
        "schema_version": "kinofail.reconfirmation-f39-seal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_f38_s2_one_shot_blind_prediction",
        "passed": True,
        "model_prediction_truth_key_or_score_read": False,
        "report_regardless_of_outcome": True,
        "score_once": True,
        "scientific_contract": {
            "frozen_f0_models_unchanged": True,
            "frozen_f1_router_threshold_eta_and_statistics_unchanged": True,
            "new_scenes_materials_seeds_and_operator_parameter_ranges": True,
            "f37_and_raw_f38_failures_preserved": True,
            "f38_s2_change": (
                "pre-prediction task-alignment amendment: frozen 0.35 m footprint, "
                "decision time +0.30 s, five synchronized RGB frames, lower-55-percent "
                "crop, unchanged RGB-L1 threshold 0.015"
            ),
            "post_acquisition_pre_prediction_amendment_disclosed": True,
            "selection_uses_model_prediction_truth_or_outcome_strength": False,
            "raw_payload_modified_or_copied": False,
            "rejected_cases_excluded_complete_case_wise": True,
        },
        "counts": {
            "scale_planned_pairs": 10_560,
            "scale_valid_pairs": scale_valid,
            "scale_remaining_attrition": 10_560 - scale_valid,
            "t2_planned_and_valid_cases": 1_500,
            "t3_planned_cases": 1_500,
            "t3_valid_cases": t3_valid,
            "t3_remaining_attrition": 1_500 - t3_valid,
            "combined_conflict_planned_cases": 3_000,
            "combined_conflict_valid_cases": 1_500 + t3_valid,
            "combined_conflict_remaining_attrition": 1_500 - t3_valid,
        },
        "source_sha256": {
            "f33_final_audit": sha256(F33_AUDIT),
            "f34_final_audit": sha256(F34_AUDIT),
            "f35_scale_audit": sha256(F35_AUDIT),
            "f36_predecessor_seal": sha256(F36_SEAL),
            "f37_failure_audit": sha256(F37_AUDIT),
            "f38_raw_failure_audit": sha256(F38_RAW_AUDIT),
            "f38_task_aligned_audit": sha256(F38_TASK_AUDIT),
            "f38_s2_design_audit": sha256(F38_S2_AUDIT),
            "f38_s2_schedule": sha256(F38_S2_ROOT / "combined_design/schedule.jsonl"),
            "f38_s2_case_schedule": sha256(
                F38_S2_ROOT / "combined_design/case_schedule.jsonl"
            ),
            "f38_s2_inventory": sha256(ROOT / str(f38_s2["artifact_inventory"])),
            "a4_pre_prediction_seal": sha256(A4_SEAL),
            "execution_scripts": {
                str(path.relative_to(ROOT)): sha256(path) for path in scripts
            },
        },
        "t3_union_root": str(F38_S2_UNION),
        "target_eval_root": str(TARGET.relative_to(ROOT)),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    OUTPUT.with_name("seal_manifest.sha256").write_text(
        f"{sha256(OUTPUT)}  {OUTPUT.name}\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
