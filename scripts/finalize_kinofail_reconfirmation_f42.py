#!/usr/bin/env python3
"""Run the sealed F42 blind prediction and one-shot score."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import finalize_kinofail_reconfirmation_v2 as base  # noqa: E402
from scripts import finalize_kinofail_reconfirmation_v8 as predecessor  # noqa: E402
from scripts.finalize_kinofail_reconfirmation_f40 import (  # noqa: E402
    ORIGINAL_EVAL,
    f40_attrition,
    read_json,
    sha256,
)


F39_ROOT = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f39"
F40_ROOT = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f40"
F41_FAILURE = ROOT / "outputs/kinofail_reconfirmation_f41_failure_audit/audit.json"
F42_SEAL = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f42/seal_manifest.json"
F42_PREDICTOR = ROOT / "scripts/predict_kinofail_unified_moe_v3_blind_f42.py"
TARGET = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f42"
REUSE = {
    "01_merge_t2_capsules": F39_ROOT / "conflict/t2_merged",
    "02_valid_conflict_design": F39_ROOT / "conflict/valid_design",
    "03_conflict_base_features": F39_ROOT / "conflict/c2_base",
    "04_conflict_v5_features": F39_ROOT / "conflict/features",
    "05_evaluation_inputs": F40_ROOT / "evaluation_inputs",
    "06_blind_bundle": F40_ROOT / "blind_bundle",
}


def validate_seal() -> dict[str, Any]:
    seal = read_json(F42_SEAL)
    sidecar = F42_SEAL.with_name("seal_manifest.sha256")
    own = Path(__file__).resolve()
    if (
        not sidecar.is_file()
        or sidecar.read_text().split()[0] != sha256(F42_SEAL)
        or seal.get("status") != "sealed_before_f42_one_shot_prediction"
        or seal.get("passed") is not True
        or seal.get("model_prediction_or_score_read") is not False
        or seal.get("source_sha256", {}).get("f41_failure_audit")
        != sha256(F41_FAILURE)
        or seal.get("source_sha256", {}).get("execution_scripts", {}).get(
            str(own.relative_to(ROOT))
        )
        != sha256(own)
    ):
        raise RuntimeError("invalid F42 pre-prediction seal")
    for relative, expected in seal["source_sha256"]["reused_artifacts"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"F42 reused artifact drift: {path}")
    return seal


def main() -> int:
    seal = validate_seal()
    if TARGET.exists():
        raise FileExistsError(f"refusing to overwrite F42 evaluation: {TARGET}")
    scenes = sorted(path.name for path in (ORIGINAL_EVAL / "shards").glob("*"))
    if len(scenes) != 30:
        raise RuntimeError(f"expected 30 original shards, found {len(scenes)}")
    original_run = base._run

    def run_with_sealed_reuse(command: list[str], name: str) -> None:
        if name in REUSE:
            flag = "--out" if name == "06_blind_bundle" else "--output"
            destination = Path(command[command.index(flag) + 1])
            source = REUSE[name]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source, target_is_directory=True)
            log_root = TARGET / "finalization_logs"
            log_root.mkdir(parents=True, exist_ok=True)
            (log_root / f"{name}.log").write_text(
                json.dumps(
                    {
                        "schema_version": "kinofail.reconfirmation-f42-preprediction-reuse.v1",
                        "created_utc": datetime.now(UTC).isoformat(),
                        "stage": name,
                        "source": str(source.relative_to(ROOT)),
                        "source_artifacts_bound_by_f42_seal": True,
                        "model_prediction_or_score_read": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            return
        if name == "07_blind_prediction_once":
            command[1] = str(F42_PREDICTOR)
        original_run(command, name)

    base.EVAL_ROOT = TARGET
    base.SHARD_ROOT = ORIGINAL_EVAL / "shards"
    base._global_snapshot_attrition_gate = f40_attrition
    base._run = run_with_sealed_reuse
    result = int(predecessor.main())
    audit_path = TARGET / "finalization_audit.json"
    audit = read_json(audit_path)
    audit["schema_version"] = "kinofail.reconfirmation-finalization.f42.v1"
    audit["f42_result_blind_operational_recovery"] = {
        "seal_sha256": sha256(F42_SEAL),
        "f41_failure_audit_sha256": sha256(F41_FAILURE),
        "f40_stages_01_through_06_reused_byte_exactly": True,
        "f42_stages_07_through_09_executed_once": True,
        "all_inference_reachable_f0_files_and_checkpoints_byte_identical": True,
        "legacy_freshness_exact_binding_verified_through_f1": True,
        "model_route_feature_threshold_statistics_or_analysis_changed": False,
        "counts": seal["counts"],
    }
    audit["f42_finalized_utc"] = datetime.now(UTC).isoformat()
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, audit_path)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
