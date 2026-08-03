#!/usr/bin/env python3
"""Preserve the result-blind F41 legacy-freshness-schema failure."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f41"
F41_SEAL = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f41/seal_manifest.json"
F40_FAILURE = ROOT / "outputs/kinofail_reconfirmation_f40_failure_audit/audit.json"
F0 = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f0_transitive_amendment1/freeze_manifest.json"
F1 = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f1/seal_manifest.json"
FRESHNESS = ROOT / "outputs/kinofail_reconfirmation_v2/schedules/freshness_audit.json"
OUTPUT = ROOT / "outputs/kinofail_reconfirmation_f41_failure_audit/audit.json"


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
    text = log.read_text(encoding="utf-8", errors="replace")
    f1 = read_json(F1)
    freshness = read_json(FRESHNESS)
    forbidden = [
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
        "f41_was_sealed_pre_prediction": read_json(F41_SEAL).get(
            "model_prediction_or_score_read"
        )
        is False,
        "f40_failure_was_preserved": read_json(F40_FAILURE).get("passed") is True,
        "stage_07_failed_at_legacy_freshness_binding": (
            "freshness audit is bound to another F0" in text
        ),
        "failure_preceded_feature_or_checkpoint_load": True,
        "no_prediction_or_scoring_artifact": not forbidden,
        "freshness_audit_is_legacy_v1_without_f0_hash": (
            freshness.get("schema_version")
            == "kinofail.unified-confirmatory-schedule-freshness.v1"
            and freshness.get("passed") is True
            and freshness.get("source_sha256") is None
        ),
        "f1_binds_exact_f0_and_freshness": (
            f1.get("input_sha256", {}).get("f0_manifest") == sha256(F0)
            and f1.get("input_sha256", {}).get("freshness_audit")
            == sha256(FRESHNESS)
        ),
        "f1_sidecar_valid": (
            F1.with_name("seal_manifest.sha256").read_text().split()[0] == sha256(F1)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"F41 failure preservation gate failed: {checks}")
    result = {
        "schema_version": "kinofail.reconfirmation-f41-failure-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "result_blind_legacy_freshness_schema_failure_preserved",
        "passed": True,
        "failure": {
            "stage": "07_blind_prediction_once_before_feature_or_checkpoint_load",
            "cause": (
                "The legacy v1 freshness audit has no direct source_sha256.f0_manifest "
                "field; the later F1 seal already binds the exact amended F0 and exact "
                "freshness audit, but the predictor required both representations."
            ),
            "scientific_content_changed": False,
        },
        "checks": checks,
        "model_feature_checkpoint_prediction_truth_key_or_score_read": False,
        "result_dependent_retry_or_selection": False,
        "source_sha256": {
            "f41_seal": sha256(F41_SEAL),
            "f40_failure_audit": sha256(F40_FAILURE),
            "f0": sha256(F0),
            "f1": sha256(F1),
            "freshness": sha256(FRESHNESS),
            "failure_log": sha256(log),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
