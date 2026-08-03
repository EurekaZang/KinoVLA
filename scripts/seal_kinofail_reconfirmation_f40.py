#!/usr/bin/env python3
"""Seal the F40 operational recovery before its first blind prediction."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
F39_SEAL = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f39/seal_manifest.json"
F39_ROOT = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f39"
F39_FAILURE = ROOT / "outputs/kinofail_reconfirmation_f39_failure_audit/audit.json"
F40_INPUT_ROOT = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_f40_inputs"
F40_INPUT_AUDIT = F40_INPUT_ROOT / "audit.json"
F40_SCALE_SCHEDULE = F40_INPUT_ROOT / "scale_schedule.jsonl"
F35_AUDIT = ROOT / "outputs/eval/unified_moe_v3_scale_direct_o9_f35/overlay_audit.json"
F38_RAW_AUDIT = ROOT / "outputs/kinofail_t3_runin_f38_formal_s1/final_audit.json"
F38_TASK_AUDIT = ROOT / "outputs/kinofail_t3_runin_f38_s1_task_aligned_audit/audit.json"
F38_S2_AUDIT = ROOT / "outputs/eval/unified_moe_v3_t3_runin_f38_s2/combined_design/audit.json"
A4_SEAL = ROOT / "outputs/freeze/kinofail_reconfirmation_a4_v8/seal_manifest.json"
TARGET = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f40"
OUTPUT = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f40/seal_manifest.json"

REUSABLE = (
    "conflict/t2_merged/feature_manifest.json",
    "conflict/t2_merged/features.npz",
    "conflict/t2_merged/records.jsonl",
    "conflict/valid_design/audit.json",
    "conflict/valid_design/schedule.jsonl",
    "conflict/valid_design/case_schedule.jsonl",
    "conflict/valid_design/attrition_ledger.jsonl",
    "conflict/c2_base/feature_manifest.json",
    "conflict/c2_base/features.npz",
    "conflict/c2_base/records.jsonl",
    "conflict/c2_base/geometry_manifest.json",
    "conflict/c2_base/geometry.npz",
    "conflict/features/feature_manifest.json",
    "conflict/features/features.npz",
    "conflict/features/records.jsonl",
    "conflict/features/geometry_manifest.json",
    "conflict/features/geometry.npz",
)


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
        raise FileExistsError("refusing to overwrite F40 seal or evaluation")
    f39_seal = read_json(F39_SEAL)
    f39_failure = read_json(F39_FAILURE)
    adapter = read_json(F40_INPUT_AUDIT)
    f35 = read_json(F35_AUDIT)
    raw = read_json(F38_RAW_AUDIT)
    task = read_json(F38_TASK_AUDIT)
    t3 = read_json(F38_S2_AUDIT)
    a4 = read_json(A4_SEAL)
    forbidden_f39 = [
        path
        for path in F39_ROOT.rglob("*")
        if path.is_file()
        and any(
            token in path.name.lower()
            for token in ("prediction", "truth_key", "score", "confirmatory_report")
        )
    ]
    if (
        f39_seal.get("passed") is not True
        or f39_seal.get("model_prediction_truth_key_or_score_read") is not False
        or f39_failure.get("passed") is not True
        or f39_failure.get("scientific_results_exist") is not False
        or f39_failure.get("model_prediction_truth_key_or_score_read") is not False
        or not all(f39_failure.get("checks", {}).values())
        or forbidden_f39
        or adapter.get("passed") is not True
        or adapter.get("status") != "sealed_model_blind_evaluation_alias"
        or adapter.get("model_prediction_truth_key_or_score_read") is not False
        or adapter.get("scientific_record_or_value_changed") is not False
        or adapter.get("only_added_key") != "scene_cluster"
        or adapter.get("output_sha256") != sha256(F40_SCALE_SCHEDULE)
        or not all(adapter.get("checks", {}).values())
        or f35.get("passed") is not True
        or int(f35.get("counts", {}).get("effective_valid_scale_pairs", 0)) != 10_243
        or float(f35.get("effective_scale_attrition_rate", 1.0)) >= 0.05
        or raw.get("passed") is not False
        or raw.get("model_prediction_truth_key_or_score_read") is not False
        or task.get("passed") is not True
        or task.get("counts_as_confirmatory_evidence") is not False
        or task.get("model_prediction_truth_key_or_score_read") is not False
        or int(task.get("counts", {}).get("accepted_cases", 0)) != 1_430
        or t3.get("passed") is not True
        or int(t3.get("valid_cases", 0)) != 1_430
        or t3.get("strictly_below_five_percent") is not True
        or a4.get("passed") is not True
        or a4.get("model_prediction_or_outcome_read") is not False
    ):
        raise RuntimeError("invalid model-blind F39/F40 predecessor evidence")

    manifests = {
        "t2": read_json(F39_ROOT / "conflict/t2_merged/feature_manifest.json"),
        "design": read_json(F39_ROOT / "conflict/valid_design/audit.json"),
        "base": read_json(F39_ROOT / "conflict/c2_base/feature_manifest.json"),
        "v5": read_json(F39_ROOT / "conflict/features/feature_manifest.json"),
        "base_geometry": read_json(
            F39_ROOT / "conflict/c2_base/geometry_manifest.json"
        ),
        "v5_geometry": read_json(
            F39_ROOT / "conflict/features/geometry_manifest.json"
        ),
    }
    if (
        any(manifests[key].get("passed") is not True for key in manifests)
        or manifests["t2"].get("status") != "complete"
        or int(manifests["t2"].get("counts", {}).get("cases", 0)) != 1_500
        or int(manifests["design"].get("counts", {}).get("T3_proprio_decisive", {}).get("valid_cases", 0))
        != 1_430
        or int(manifests["base"].get("counts", {}).get("cases", 0)) != 2_930
        or int(manifests["base"].get("counts", {}).get("samples", 0)) != 17_580
        or int(manifests["v5"].get("counts", {}).get("cases", 0)) != 2_930
        or int(manifests["v5"].get("counts", {}).get("samples", 0)) != 17_580
        or not all(manifests["base"].get("checks", {}).values())
        or not all(manifests["v5"].get("checks", {}).values())
    ):
        raise RuntimeError("F39 pre-prediction artifact contract failed")
    reusable_hashes = {}
    for relative in REUSABLE:
        path = F39_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        reusable_hashes[relative] = sha256(path)
    expected_payload_hashes = {
        "conflict/t2_merged/features.npz": manifests["t2"]["output_sha256"]["features"],
        "conflict/t2_merged/records.jsonl": manifests["t2"]["output_sha256"]["records"],
        "conflict/c2_base/features.npz": manifests["base"]["output_sha256"]["features"],
        "conflict/c2_base/records.jsonl": manifests["base"]["output_sha256"]["records"],
        "conflict/c2_base/geometry.npz": manifests["base_geometry"]["output_sha256"],
        "conflict/features/features.npz": manifests["v5"]["output_sha256"]["features"],
        "conflict/features/records.jsonl": manifests["v5"]["output_sha256"]["records"],
        "conflict/features/geometry.npz": manifests["v5_geometry"]["output_sha256"],
    }
    if any(reusable_hashes[path] != expected for path, expected in expected_payload_hashes.items()):
        raise RuntimeError("F39 reusable payload hash mismatch")

    scripts = [
        Path(__file__).resolve(),
        ROOT / "scripts/finalize_kinofail_reconfirmation_f40.py",
        ROOT / "scripts/audit_kinofail_reconfirmation_f39_failure.py",
        ROOT / "scripts/build_kinofail_reconfirmation_f40_scale_schedule_adapter.py",
        ROOT / "scripts/finalize_kinofail_reconfirmation_v8.py",
        ROOT / "scripts/build_kinofail_confirmatory_evaluation_bundle_inputs_v1.py",
        ROOT / "scripts/assemble_kinofail_confirmatory_blind_bundle_v1.py",
        ROOT / "scripts/predict_kinofail_unified_moe_v3_blind.py",
        ROOT / "scripts/build_kinofail_confirmatory_scoring_protocol_v1.py",
        ROOT / "scripts/score_kinofail_unified_moe_v3_confirmatory.py",
        ROOT / "scripts/analyze_kinofail_reconfirmation_a6_v7.py",
        ROOT / "scripts/run_kinofail_reconfirmation_a4_v8.py",
        ROOT / "scripts/analyze_kinofail_reconfirmation_a4_v8.py",
    ]
    if any(not path.is_file() for path in scripts):
        raise FileNotFoundError("F40 execution dependency missing")
    scale_valid = int(f35["counts"]["effective_valid_scale_pairs"])
    t3_valid = int(t3["valid_cases"])
    manifest = {
        "schema_version": "kinofail.reconfirmation-f40-seal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_f39_pre_prediction_reuse_and_f40_one_shot_prediction",
        "passed": True,
        "model_prediction_truth_key_or_score_read": False,
        "report_regardless_of_outcome": True,
        "score_once": True,
        "scientific_contract": {
            "f39_failed_before_blind_bundle_prediction_truth_or_scoring": True,
            "f39_stages_01_through_04_reused_byte_exactly": True,
            "f40_adapter_only_adds_registry_backed_scene_alias": True,
            "scientific_record_or_value_changed_by_adapter": False,
            "frozen_f0_models_unchanged": True,
            "frozen_f1_router_threshold_eta_and_statistics_unchanged": True,
            "post_acquisition_pre_prediction_task_alignment_amendment_disclosed": True,
            "selection_uses_model_prediction_truth_or_outcome_strength": False,
            "raw_failures_preserved": True,
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
            "f39_seal": sha256(F39_SEAL),
            "f39_failure_audit": sha256(F39_FAILURE),
            "f40_scale_adapter_audit": sha256(F40_INPUT_AUDIT),
            "f40_scale_schedule": sha256(F40_SCALE_SCHEDULE),
            "f35_scale_audit": sha256(F35_AUDIT),
            "f38_raw_failure_audit": sha256(F38_RAW_AUDIT),
            "f38_task_aligned_audit": sha256(F38_TASK_AUDIT),
            "f38_s2_design_audit": sha256(F38_S2_AUDIT),
            "a4_pre_prediction_seal": sha256(A4_SEAL),
            "f39_reusable_artifacts": reusable_hashes,
            "execution_scripts": {
                str(path.relative_to(ROOT)): sha256(path) for path in scripts
            },
        },
        "f39_failed_eval_root": str(F39_ROOT.relative_to(ROOT)),
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
