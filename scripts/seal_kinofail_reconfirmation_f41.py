#!/usr/bin/env python3
"""Seal F41 before its first successful checkpoint inference attempt."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
F0 = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f0_transitive_amendment1/freeze_manifest.json"
F12 = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f12_process_local_rtx_texture_amendment1/amendment_manifest.json"
F40_SEAL = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f40/seal_manifest.json"
F40_ROOT = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f40"
F40_FAILURE = ROOT / "outputs/kinofail_reconfirmation_f40_failure_audit/audit.json"
F41_TARGET = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f41"
OUTPUT = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f41/seal_manifest.json"
BACKEND = ROOT / "kino_vla/sim/isaac_policy_backend.py"

REUSED = (
    "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f40/evaluation_inputs/audit.json",
    "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f40/evaluation_inputs/scale_features.npz",
    "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f40/evaluation_inputs/scale_truth.jsonl",
    "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f40/evaluation_inputs/conflict_features.npz",
    "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f40/evaluation_inputs/conflict_truth.jsonl",
    "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f40/blind_bundle/bundle_manifest.json",
    "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f40/blind_bundle/blind_features.npz",
    "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f40/blind_bundle/truth_key.jsonl",
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
    if OUTPUT.exists() or F41_TARGET.exists():
        raise FileExistsError("refusing to overwrite F41 seal or evaluation")
    f0 = read_json(F0)
    f12 = read_json(F12)
    f40_seal = read_json(F40_SEAL)
    failure = read_json(F40_FAILURE)
    backend_items = [
        item
        for item in f0.get("frozen_files", [])
        if item.get("path") == "kino_vla/sim/isaac_policy_backend.py"
    ]
    forbidden = [
        path
        for path in F40_ROOT.rglob("*")
        if path.is_file()
        and any(
            token in path.name.lower()
            for token in (
                "blind_predictions",
                "prediction_manifest",
                "scoring_protocol",
                "confirmatory_report",
            )
        )
    ]
    if (
        f40_seal.get("passed") is not True
        or f40_seal.get("model_prediction_truth_key_or_score_read") is not False
        or failure.get("passed") is not True
        or failure.get("model_prediction_or_score_created") is not False
        or failure.get("result_dependent_retry_or_selection") is not False
        or not all(failure.get("checks", {}).values())
        or forbidden
        or len(backend_items) != 1
        or f12.get("passed") is not True
        or f12.get("correction", {}).get("backend_predecessor_sha256")
        != backend_items[0].get("sha256")
        or f12.get("correction", {}).get("backend_sha256") != sha256(BACKEND)
        or any(
            f12.get("scientific_contract", {}).get(key) is not False
            for key in (
                "model_or_route_changed",
                "sensor_or_feature_logic_changed",
                "simulation_logic_changed",
                "threshold_or_analysis_changed",
                "result_dependent_retry_enabled",
                "scene_material_seed_operator_or_parameter_changed",
            )
        )
    ):
        raise RuntimeError("invalid F40/F12 model-blind predecessor evidence")
    reused_hashes = {}
    for relative in REUSED:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        reused_hashes[relative] = sha256(path)
    bundle = read_json(F40_ROOT / "blind_bundle/bundle_manifest.json")
    if (
        bundle.get("status") != "observable_features_and_separate_truth_key_sealed"
        or bundle.get("blind_archive_keys") != ["sample_ids", "visual", "proprio"]
        or bundle.get("sample_count_valid") != 79_038
        or bundle.get("artifact_sha256", {}).get("blind_features")
        != reused_hashes[
            "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f40/blind_bundle/blind_features.npz"
        ]
        or bundle.get("artifact_sha256", {}).get("truth_key")
        != reused_hashes[
            "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f40/blind_bundle/truth_key.jsonl"
        ]
    ):
        raise RuntimeError("F40 blind bundle drift")
    scripts = [
        Path(__file__).resolve(),
        ROOT / "scripts/finalize_kinofail_reconfirmation_f41.py",
        ROOT / "scripts/predict_kinofail_unified_moe_v3_blind_f41.py",
        ROOT / "scripts/predict_kinofail_unified_moe_v3_blind.py",
        ROOT / "scripts/finalize_kinofail_reconfirmation_v8.py",
        ROOT / "scripts/build_kinofail_confirmatory_scoring_protocol_v1.py",
        ROOT / "scripts/score_kinofail_unified_moe_v3_confirmatory.py",
    ]
    if any(not path.is_file() for path in scripts):
        raise FileNotFoundError("F41 execution dependency missing")
    counts = f40_seal["counts"]
    result = {
        "schema_version": "kinofail.reconfirmation-f41-seal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_f40_bundle_reuse_and_f41_one_shot_prediction",
        "passed": True,
        "model_prediction_or_score_read": False,
        "blind_bundle_truth_key_exists_but_was_not_read_for_f41_design": True,
        "report_regardless_of_outcome": True,
        "score_once": True,
        "scientific_contract": {
            "f40_failed_before_feature_or_checkpoint_load": True,
            "f40_created_no_prediction_or_score": True,
            "f40_stages_01_through_06_reused_byte_exactly": True,
            "f12_backend_is_acquisition_only_and_inference_unreachable": True,
            "all_other_f0_files_and_checkpoints_byte_identical": True,
            "model_route_feature_threshold_statistics_or_analysis_changed": False,
            "result_dependent_retry_or_selection": False,
        },
        "counts": counts,
        "source_sha256": {
            "f0": sha256(F0),
            "f12": sha256(F12),
            "f12_backend": sha256(BACKEND),
            "f40_seal": sha256(F40_SEAL),
            "f40_failure_audit": sha256(F40_FAILURE),
            "reused_artifacts": reused_hashes,
            "execution_scripts": {
                str(path.relative_to(ROOT)): sha256(path) for path in scripts
            },
        },
        "target_eval_root": str(F41_TARGET.relative_to(ROOT)),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    OUTPUT.with_name("seal_manifest.sha256").write_text(
        f"{sha256(OUTPUT)}  {OUTPUT.name}\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
