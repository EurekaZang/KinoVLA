#!/usr/bin/env python3
"""Preserve the model-blind F39 evaluation-schedule adapter failure."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SEAL = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f39/seal_manifest.json"
RUN = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f39"
OUTPUT = ROOT / "outputs/kinofail_reconfirmation_f39_failure_audit/audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    seal = load(SEAL)
    logs = RUN / "finalization_logs"
    expected_logs = [logs / f"{index:02d}_{name}.log" for index, name in (
        (1, "merge_t2_capsules"),
        (2, "valid_conflict_design"),
        (3, "conflict_base_features"),
        (4, "conflict_v5_features"),
        (5, "evaluation_inputs"),
    )]
    stage5 = expected_logs[-1]
    text = stage5.read_text(encoding="utf-8", errors="replace")
    model_artifacts = [
        path
        for path in RUN.rglob("*")
        if path.is_file()
        and any(
            token in path.name.lower()
            for token in (
                "prediction",
                "truth_key",
                "scoring_protocol",
                "confirmatory_report",
            )
        )
    ]
    manifests = {
        "t2_merged": RUN / "conflict/t2_merged/feature_manifest.json",
        "valid_design": RUN / "conflict/valid_design/audit.json",
        "c2_base": RUN / "conflict/c2_base/feature_manifest.json",
        "c2_v5": RUN / "conflict/features/feature_manifest.json",
    }
    values = {key: load(path) for key, path in manifests.items()}
    checks = {
        "f39_was_sealed_model_blind": (
            seal.get("passed") is True
            and seal.get("model_prediction_truth_key_or_score_read") is False
        ),
        "stages_one_through_four_completed": all(
            value.get("passed") is True for value in values.values()
        ),
        "stage_five_failed_at_missing_scene_cluster": (
            "KeyError: 'scene_cluster'" in text
            and "_planned_scale" in text
        ),
        "no_blind_bundle_created": not (RUN / "blind_bundle").exists(),
        "no_prediction_truth_or_score_artifact": not model_artifacts,
        "all_five_logs_present": all(path.is_file() for path in expected_logs),
    }
    audit = {
        "schema_version": "kinofail.reconfirmation-f39-failure-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "status": "model_blind_evaluation_schedule_adapter_failure_preserved",
        "model_prediction_truth_key_or_score_read": False,
        "scientific_results_exist": False,
        "result_dependent_retry_or_selection": False,
        "failure": {
            "stage": "05_evaluation_inputs",
            "exception": "KeyError: scene_cluster",
            "cause": (
                "F35 snapshot/features were supplied with the original global Scale "
                "schedule, whose scene field is scene_family; the evaluation-input "
                "builder requires the equivalent scene_cluster alias."
            ),
            "scientific_content_changed": False,
        },
        "checks": checks,
        "source_sha256": {
            "f39_seal": sha256(SEAL),
            "stage_logs": {path.name: sha256(path) for path in expected_logs},
            "pre_prediction_manifests": {
                key: sha256(path) for key, path in manifests.items()
            },
        },
        "reusable_pre_prediction_artifacts": {
            key: str(path.parent) for key, path in manifests.items()
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
