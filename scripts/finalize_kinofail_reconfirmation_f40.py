#!/usr/bin/env python3
"""Run the F40 one-shot blind evaluation after the model-blind F39 adapter failure.

F39 completed four deterministic, pre-prediction stages before its Scale schedule
adapter failed.  F40 reuses those byte-sealed artifacts and executes stages 05--09
once.  No prediction, truth key, scoring artifact, or outcome existed in F39.
"""

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


F39_ROOT = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f39"
F39_FAILURE_AUDIT = (
    ROOT / "outputs/kinofail_reconfirmation_f39_failure_audit/audit.json"
)
F40_INPUT_ROOT = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_f40_inputs"
F40_INPUT_AUDIT = F40_INPUT_ROOT / "audit.json"
F40_SCALE_SCHEDULE = F40_INPUT_ROOT / "scale_schedule.jsonl"
F40_SEAL = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f40/seal_manifest.json"
TARGET = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f40"

F35_ROOT = ROOT / "outputs/eval/unified_moe_v3_scale_direct_o9_f35"
F35_AUDIT = F35_ROOT / "overlay_audit.json"
F38_RAW_AUDIT = ROOT / "outputs/kinofail_t3_runin_f38_formal_s1/final_audit.json"
F38_TASK_AUDIT = ROOT / "outputs/kinofail_t3_runin_f38_s1_task_aligned_audit/audit.json"
F38_S2_ROOT = ROOT / "outputs/eval/unified_moe_v3_t3_runin_f38_s2"
F38_S2_AUDIT = F38_S2_ROOT / "combined_design/audit.json"
F39_SEAL = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f39/seal_manifest.json"
ORIGINAL_EVAL = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2"

REUSE = {
    "01_merge_t2_capsules": F39_ROOT / "conflict/t2_merged",
    "02_valid_conflict_design": F39_ROOT / "conflict/valid_design",
    "03_conflict_base_features": F39_ROOT / "conflict/c2_base",
    "04_conflict_v5_features": F39_ROOT / "conflict/features",
}


def sha256(path: Path) -> str:
    import hashlib

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


def validate_seal() -> dict[str, Any]:
    seal = read_json(F40_SEAL)
    sidecar = F40_SEAL.with_name("seal_manifest.sha256")
    own = Path(__file__).resolve()
    scripts = seal.get("source_sha256", {}).get("execution_scripts", {})
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").split()[0] != sha256(F40_SEAL)
        or seal.get("status")
        != "sealed_before_f39_pre_prediction_reuse_and_f40_one_shot_prediction"
        or seal.get("passed") is not True
        or seal.get("model_prediction_truth_key_or_score_read") is not False
        or seal.get("source_sha256", {}).get("f39_failure_audit")
        != sha256(F39_FAILURE_AUDIT)
        or seal.get("source_sha256", {}).get("f40_scale_adapter_audit")
        != sha256(F40_INPUT_AUDIT)
        or seal.get("source_sha256", {}).get("f40_scale_schedule")
        != sha256(F40_SCALE_SCHEDULE)
        or scripts.get(str(own.relative_to(ROOT))) != sha256(own)
    ):
        raise RuntimeError("invalid F40 pre-prediction seal")
    for relative, expected in seal["source_sha256"]["f39_reusable_artifacts"].items():
        path = F39_ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"F39 reusable artifact drift: {path}")
    return seal


def f40_attrition(_: list[str]) -> dict[str, Any]:
    f35 = read_json(F35_AUDIT)
    raw = read_json(F38_RAW_AUDIT)
    task = read_json(F38_TASK_AUDIT)
    t3 = read_json(F38_S2_AUDIT)
    scale_valid = int(f35["counts"]["effective_valid_scale_pairs"])
    t3_valid = int(t3["valid_cases"])
    counts = {
        "scale_pair_units": {
            "planned": 10_560,
            "valid": scale_valid,
            "attrited": 10_560 - scale_valid,
            "attrition_rate": (10_560 - scale_valid) / 10_560,
        },
        "t3_complete_case_units": {
            "planned": 1_500,
            "valid": t3_valid,
            "attrited": 1_500 - t3_valid,
            "attrition_rate": (1_500 - t3_valid) / 1_500,
        },
        "conflict_case_units": {
            "planned": 3_000,
            "valid": 1_500 + t3_valid,
            "attrited": 1_500 - t3_valid,
            "attrition_rate": (1_500 - t3_valid) / 3_000,
            "t2_valid": 1_500,
            "t3_valid": t3_valid,
        },
        "combined_analysis_units": {
            "planned": 13_560,
            "valid": scale_valid + 1_500 + t3_valid,
            "attrited": (10_560 - scale_valid) + (1_500 - t3_valid),
            "attrition_rate": (
                (10_560 - scale_valid) + (1_500 - t3_valid)
            )
            / 13_560,
        },
    }
    checks = {
        "scale_pair_attrition_strictly_below_five_percent": (
            counts["scale_pair_units"]["attrition_rate"] < 0.05
        ),
        "t3_complete_case_attrition_strictly_below_five_percent": (
            counts["t3_complete_case_units"]["attrition_rate"] < 0.05
        ),
        "combined_conflict_attrition_strictly_below_five_percent": (
            counts["conflict_case_units"]["attrition_rate"] < 0.05
        ),
        "combined_analysis_attrition_strictly_below_five_percent": (
            counts["combined_analysis_units"]["attrition_rate"] < 0.05
        ),
        "legacy_o9_excluded": f35.get("legacy_o9_features_excluded") is True,
        "direct_event_o9_inserted": int(f35["counts"]["direct_event_f34_o9_pairs"])
        >= 750,
        "f38_raw_failure_preserved": raw.get("passed") is False,
        "f38_task_alignment_is_pre_prediction": task.get(
            "model_prediction_truth_key_or_score_read"
        )
        is False,
        "f40_schedule_adapter_is_scientifically_identity_preserving": read_json(
            F40_INPUT_AUDIT
        ).get("scientific_record_or_value_changed")
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError("F40 attrition or provenance gate failed")
    return {
        "schema_version": "kinofail.reconfirmation-f40-attrition.v1",
        "passed": True,
        "threshold": 0.05,
        "counts": counts,
        "checks": checks,
        "raw_f38_s1_generic_attrition_rate": float(raw["case_attrition_rate"]),
        "task_aligned_f38_s2_attrition_rate": float(t3["case_attrition_rate"]),
        "source_sha256": {
            "f35_scale_audit": sha256(F35_AUDIT),
            "f38_raw_failure_audit": sha256(F38_RAW_AUDIT),
            "f38_task_aligned_audit": sha256(F38_TASK_AUDIT),
            "f38_s2_design_audit": sha256(F38_S2_AUDIT),
            "f40_scale_adapter_audit": sha256(F40_INPUT_AUDIT),
        },
    }


def replace_argument(command: list[str], flag: str, value: Path) -> None:
    index = command.index(flag)
    command[index + 1] = str(value)


def main() -> int:
    seal = validate_seal()
    if TARGET.exists():
        raise FileExistsError(f"refusing to overwrite F40 evaluation: {TARGET}")
    scenes = sorted(path.name for path in (ORIGINAL_EVAL / "shards").glob("*"))
    if len(scenes) != 30:
        raise RuntimeError(f"expected 30 original shards, found {len(scenes)}")
    for scene in scenes:
        for path in (
            F35_ROOT / "shards" / scene / "scale/snapshots/snapshot_records.jsonl",
            F35_ROOT / "shards" / scene / "scale/unified_features/feature_manifest.json",
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

    original_run = base._run

    def run_with_sealed_reuse(command: list[str], name: str) -> None:
        if name in REUSE:
            destination = Path(command[command.index("--output") + 1])
            source = REUSE[name]
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source, target_is_directory=True)
            log_root = TARGET / "finalization_logs"
            log_root.mkdir(parents=True, exist_ok=True)
            receipt = {
                "schema_version": "kinofail.reconfirmation-f40-preprediction-reuse.v1",
                "created_utc": datetime.now(UTC).isoformat(),
                "stage": name,
                "source": str(source.relative_to(ROOT)),
                "destination": str(destination.relative_to(ROOT)),
                "source_artifacts_bound_by_f40_seal": True,
                "model_prediction_truth_key_or_score_read": False,
            }
            (log_root / f"{name}.log").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return
        if name == "05_evaluation_inputs":
            replace_argument(command, "--scale-schedule", F40_SCALE_SCHEDULE)
            for scene in scenes:
                replacements = {
                    str(ORIGINAL_EVAL / "shards" / scene / "scale/snapshots"): str(
                        F35_ROOT / "shards" / scene / "scale/snapshots"
                    ),
                    str(
                        ORIGINAL_EVAL / "shards" / scene / "scale/unified_features"
                    ): str(F35_ROOT / "shards" / scene / "scale/unified_features"),
                }
                command[:] = [replacements.get(value, value) for value in command]
        original_run(command, name)

    base.EVAL_ROOT = TARGET
    base.SHARD_ROOT = ORIGINAL_EVAL / "shards"
    base._global_snapshot_attrition_gate = f40_attrition
    base._run = run_with_sealed_reuse
    result = int(predecessor.main())

    audit_path = TARGET / "finalization_audit.json"
    audit = read_json(audit_path)
    audit["schema_version"] = "kinofail.reconfirmation-finalization.f40.v1"
    audit["f40_operational_recovery"] = {
        "seal": str(F40_SEAL.relative_to(ROOT)),
        "seal_sha256": sha256(F40_SEAL),
        "f39_failure_audit_sha256": sha256(F39_FAILURE_AUDIT),
        "f40_scale_adapter_audit_sha256": sha256(F40_INPUT_AUDIT),
        "f39_had_no_prediction_truth_key_or_score_artifact": True,
        "f39_stages_01_through_04_reused_byte_exactly": True,
        "f40_stages_05_through_09_executed_once": True,
        "scientific_record_or_value_changed_by_adapter": False,
        "post_acquisition_pre_prediction_task_alignment_amendment_disclosed": True,
        "frozen_models_router_threshold_eta_and_statistics_unchanged": True,
        "selection_used_model_prediction_truth_or_outcome_strength": False,
        "counts": seal["counts"],
    }
    audit["f40_finalized_utc"] = datetime.now(UTC).isoformat()
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, audit_path)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
