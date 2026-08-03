#!/usr/bin/env python3
"""Seal F42 after exhaustive result-blind inference preflight."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
F0 = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f0_transitive_amendment1/freeze_manifest.json"
F1 = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f1/seal_manifest.json"
F12 = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f12_process_local_rtx_texture_amendment1/amendment_manifest.json"
FRESHNESS = ROOT / "outputs/kinofail_reconfirmation_v2/schedules/freshness_audit.json"
F41_SEAL = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f41/seal_manifest.json"
F41_ROOT = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f41"
F41_FAILURE = ROOT / "outputs/kinofail_reconfirmation_f41_failure_audit/audit.json"
F40_ROOT = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f40"
TARGET = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f42"
OUTPUT = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f42/seal_manifest.json"
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
    if OUTPUT.exists() or TARGET.exists():
        raise FileExistsError("refusing to overwrite F42 seal or evaluation")
    f0 = read_json(F0)
    f1 = read_json(F1)
    f12 = read_json(F12)
    freshness = read_json(FRESHNESS)
    f41_seal = read_json(F41_SEAL)
    failure = read_json(F41_FAILURE)
    backend_items = [
        item
        for item in f0.get("frozen_files", [])
        if item.get("path") == "kino_vla/sim/isaac_policy_backend.py"
    ]
    drifted = []
    for item in f0.get("frozen_files", []):
        path = Path(str(item["path"]))
        if not path.is_absolute():
            path = ROOT / path
        if (
            not path.is_file()
            or path.stat().st_size != int(item["bytes"])
            or sha256(path) != str(item["sha256"])
        ):
            drifted.append(str(item["path"]))
    forbidden = [
        path
        for path in F41_ROOT.rglob("*")
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
        f41_seal.get("passed") is not True
        or f41_seal.get("model_prediction_or_score_read") is not False
        or failure.get("passed") is not True
        or failure.get("model_feature_checkpoint_prediction_truth_key_or_score_read")
        is not False
        or failure.get("result_dependent_retry_or_selection") is not False
        or not all(failure.get("checks", {}).values())
        or forbidden
        or drifted != ["kino_vla/sim/isaac_policy_backend.py"]
        or len(backend_items) != 1
        or f12.get("correction", {}).get("backend_predecessor_sha256")
        != backend_items[0]["sha256"]
        or f12.get("correction", {}).get("backend_sha256") != sha256(BACKEND)
        or freshness.get("schema_version")
        != "kinofail.unified-confirmatory-schedule-freshness.v1"
        or freshness.get("passed") is not True
        or freshness.get("source_sha256") is not None
        or f1.get("input_sha256", {}).get("f0_manifest") != sha256(F0)
        or f1.get("input_sha256", {}).get("freshness_audit") != sha256(FRESHNESS)
        or F1.with_name("seal_manifest.sha256").read_text().split()[0] != sha256(F1)
    ):
        raise RuntimeError("F42 exhaustive pre-inference contract failed")
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
        or int(bundle.get("sample_count_valid", 0)) != 79_038
        or bundle.get("artifact_sha256", {}).get("blind_features")
        != reused_hashes[REUSED[-2]]
        or bundle.get("artifact_sha256", {}).get("truth_key")
        != reused_hashes[REUSED[-1]]
    ):
        raise RuntimeError("F42 reused blind bundle drift")
    scripts = [
        Path(__file__).resolve(),
        ROOT / "scripts/finalize_kinofail_reconfirmation_f42.py",
        ROOT / "scripts/predict_kinofail_unified_moe_v3_blind_f42.py",
        ROOT / "scripts/predict_kinofail_unified_moe_v3_blind_f41.py",
        ROOT / "scripts/predict_kinofail_unified_moe_v3_blind.py",
        ROOT / "scripts/finalize_kinofail_reconfirmation_v8.py",
        ROOT / "scripts/build_kinofail_confirmatory_scoring_protocol_v1.py",
        ROOT / "scripts/score_kinofail_unified_moe_v3_confirmatory.py",
    ]
    if any(not path.is_file() for path in scripts):
        raise FileNotFoundError("F42 execution dependency missing")
    result = {
        "schema_version": "kinofail.reconfirmation-f42-seal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_f42_one_shot_prediction",
        "passed": True,
        "model_prediction_or_score_read": False,
        "blind_bundle_truth_key_exists_but_was_not_read_for_f42_design": True,
        "report_regardless_of_outcome": True,
        "score_once": True,
        "scientific_contract": {
            "f40_and_f41_failures_preserved": True,
            "f41_failed_before_feature_or_checkpoint_load": True,
            "f40_stages_01_through_06_reused_byte_exactly": True,
            "f12_backend_is_acquisition_only_and_inference_unreachable": True,
            "all_other_f0_files_and_checkpoints_byte_identical": True,
            "legacy_freshness_file_modified": False,
            "legacy_freshness_binding_verified_through_sealed_f1": True,
            "model_route_feature_threshold_statistics_or_analysis_changed": False,
            "result_dependent_retry_or_selection": False,
        },
        "counts": f41_seal["counts"],
        "source_sha256": {
            "f0": sha256(F0),
            "f1": sha256(F1),
            "f12": sha256(F12),
            "freshness": sha256(FRESHNESS),
            "f41_seal": sha256(F41_SEAL),
            "f41_failure_audit": sha256(F41_FAILURE),
            "reused_artifacts": reused_hashes,
            "execution_scripts": {
                str(path.relative_to(ROOT)): sha256(path) for path in scripts
            },
        },
        "target_eval_root": str(TARGET.relative_to(ROOT)),
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
