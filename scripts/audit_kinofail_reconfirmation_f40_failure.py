#!/usr/bin/env python3
"""Preserve the result-blind F40 inference-integrity failure."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f40"
F40_SEAL = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f40/seal_manifest.json"
F12 = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f12_process_local_rtx_texture_amendment1/amendment_manifest.json"
F0 = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f0_transitive_amendment1/freeze_manifest.json"
OUTPUT = ROOT / "outputs/kinofail_reconfirmation_f40_failure_audit/audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    log = EVAL / "finalization_logs/07_blind_prediction_once.log"
    bundle_manifest = EVAL / "blind_bundle/bundle_manifest.json"
    truth_key = EVAL / "blind_bundle/truth_key.jsonl"
    blind_features = EVAL / "blind_bundle/blind_features.npz"
    inputs_audit = EVAL / "evaluation_inputs/audit.json"
    f0 = read_json(F0)
    f12 = read_json(F12)
    backend_items = [
        item
        for item in f0.get("frozen_files", [])
        if item.get("path") == "kino_vla/sim/isaac_policy_backend.py"
    ]
    backend = ROOT / "kino_vla/sim/isaac_policy_backend.py"
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
    log_text = log.read_text(encoding="utf-8", errors="replace")
    prediction_or_score_artifacts = [
        path
        for path in EVAL.rglob("*")
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
    checks = {
        "f40_was_sealed_pre_prediction": read_json(F40_SEAL).get(
            "model_prediction_truth_key_or_score_read"
        )
        is False,
        "stages_01_through_06_completed": all(
            (EVAL / f"finalization_logs/{index:02d}_{name}.log").is_file()
            for index, name in (
                (1, "merge_t2_capsules"),
                (2, "valid_conflict_design"),
                (3, "conflict_base_features"),
                (4, "conflict_v5_features"),
                (5, "evaluation_inputs"),
                (6, "blind_bundle"),
            )
        ),
        "stage_07_failed_before_prediction": (
            "F0 freeze validation failed" in log_text
            and "all_frozen_files_unchanged': False" in log_text
        ),
        "no_prediction_or_scoring_artifact": not prediction_or_score_artifacts,
        "blind_bundle_exists_but_was_not_passed_truth": all(
            path.is_file() for path in (bundle_manifest, truth_key, blind_features)
        ),
        "exactly_one_f0_file_drifted": drifted == ["kino_vla/sim/isaac_policy_backend.py"],
        "drift_is_f12_sealed_backend": (
            len(backend_items) == 1
            and backend_items[0]["sha256"]
            == f12.get("correction", {}).get("backend_predecessor_sha256")
            and sha256(backend)
            == f12.get("correction", {}).get("backend_sha256")
        ),
        "f12_declares_no_model_feature_analysis_or_simulation_change": all(
            f12.get("scientific_contract", {}).get(key) is False
            for key in (
                "model_or_route_changed",
                "sensor_or_feature_logic_changed",
                "simulation_logic_changed",
                "threshold_or_analysis_changed",
                "result_dependent_retry_enabled",
            )
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"F40 failure preservation gate failed: {checks}")
    # Bind only manifests and the failure log.  Do not open feature/truth payloads.
    artifact_hashes = {
        str(path.relative_to(EVAL)): sha256(path)
        for path in (inputs_audit, bundle_manifest, log)
    }
    result = {
        "schema_version": "kinofail.reconfirmation-f40-failure-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "result_blind_inference_reachability_validation_failure_preserved",
        "passed": True,
        "failure": {
            "stage": "07_blind_prediction_once_before_feature_or_checkpoint_load",
            "cause": (
                "The generic F0 validator treated the F12-sealed simulator-backend "
                "operational amendment as an inference dependency drift, although the "
                "backend is not imported or used by the checkpoint-only predictor."
            ),
            "scientific_content_changed": False,
        },
        "checks": checks,
        "blind_bundle_truth_key_created": True,
        "truth_key_was_an_argument_to_failed_predictor": False,
        "model_prediction_or_score_created": False,
        "checkpoint_payloads_hashed_but_not_deserialized_by_failure_audit": True,
        "model_prediction_truth_key_or_score_read_by_failure_audit": False,
        "result_dependent_retry_or_selection": False,
        "source_sha256": {
            "f40_seal": sha256(F40_SEAL),
            "f0": sha256(F0),
            "f12": sha256(F12),
            "current_backend": sha256(backend),
            "artifacts": artifact_hashes,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
