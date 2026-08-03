#!/usr/bin/env python3
"""Run the F39 one-shot blind evaluation with F35 Scale and F38-S2 T3."""

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
from scripts.seal_kinofail_reconfirmation_f39 import (  # noqa: E402
    F35_AUDIT,
    F35_ROOT,
    F37_AUDIT,
    F38_RAW_AUDIT,
    F38_S2_AUDIT,
    F38_S2_ROOT,
    F38_S2_UNION,
    F38_TASK_AUDIT,
    OUTPUT as F39_SEAL,
    TARGET,
    read_json,
    sha256,
)


ORIGINAL_EVAL = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2"
ORIGINAL_CONFLICT_SCHEDULE = (
    ROOT / "outputs/kinofail_reconfirmation_v2/schedules/conflict_schedule.jsonl"
)
T3_DESIGN = F38_S2_ROOT / "combined_design"


def validate_seal() -> dict[str, Any]:
    seal = read_json(F39_SEAL)
    sidecar = F39_SEAL.with_name("seal_manifest.sha256")
    scripts = seal.get("source_sha256", {}).get("execution_scripts", {})
    own = Path(__file__).resolve()
    if (
        not sidecar.is_file()
        or sidecar.read_text().split()[0] != sha256(F39_SEAL)
        or seal.get("status") != "sealed_before_f38_s2_one_shot_blind_prediction"
        or seal.get("passed") is not True
        or seal.get("model_prediction_truth_key_or_score_read") is not False
        or seal.get("source_sha256", {}).get("f35_scale_audit") != sha256(F35_AUDIT)
        or seal.get("source_sha256", {}).get("f37_failure_audit") != sha256(F37_AUDIT)
        or seal.get("source_sha256", {}).get("f38_raw_failure_audit")
        != sha256(F38_RAW_AUDIT)
        or seal.get("source_sha256", {}).get("f38_task_aligned_audit")
        != sha256(F38_TASK_AUDIT)
        or seal.get("source_sha256", {}).get("f38_s2_design_audit")
        != sha256(F38_S2_AUDIT)
        or scripts.get(str(own.relative_to(ROOT))) != sha256(own)
    ):
        raise RuntimeError("invalid F39 pre-prediction seal")
    return seal


def f39_attrition(_: list[str]) -> dict[str, Any]:
    f35 = read_json(F35_AUDIT)
    t3 = read_json(F38_S2_AUDIT)
    if f35.get("passed") is not True or t3.get("passed") is not True:
        raise RuntimeError("F39 corpus audit invalid")
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
        "direct_event_o9_inserted": int(
            f35["counts"]["direct_event_f34_o9_pairs"]
        )
        >= 750,
        "f38_raw_failure_preserved": read_json(F38_RAW_AUDIT).get("passed") is False,
        "f38_task_alignment_is_pre_prediction": read_json(F38_TASK_AUDIT).get(
            "model_prediction_truth_key_or_score_read"
        )
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError("F39 attrition or provenance gate failed")
    return {
        "schema_version": "kinofail.reconfirmation-f39-attrition.v1",
        "passed": True,
        "threshold": 0.05,
        "counts": counts,
        "checks": checks,
        "raw_f38_s1_generic_attrition_rate": float(
            read_json(F38_RAW_AUDIT)["case_attrition_rate"]
        ),
        "task_aligned_f38_s2_attrition_rate": float(t3["case_attrition_rate"]),
        "source_sha256": {
            "f35_scale_audit": sha256(F35_AUDIT),
            "f38_raw_failure_audit": sha256(F38_RAW_AUDIT),
            "f38_task_aligned_audit": sha256(F38_TASK_AUDIT),
            "f38_s2_design_audit": sha256(F38_S2_AUDIT),
        },
    }


def replace_argument(command: list[str], flag: str, value: Path) -> None:
    index = command.index(flag)
    command[index + 1] = str(value)


def main() -> int:
    seal = validate_seal()
    if TARGET.exists():
        raise FileExistsError(f"refusing to overwrite F39 evaluation: {TARGET}")
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
    for path in (
        T3_DESIGN / "audit.json",
        T3_DESIGN / "schedule.jsonl",
        T3_DESIGN / "case_schedule.jsonl",
        F38_S2_ROOT / "artifact_inventory.jsonl",
        ORIGINAL_CONFLICT_SCHEDULE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    original_run = base._run

    def run_with_corpora(command: list[str], name: str) -> None:
        if name == "02_valid_conflict_design":
            replace_argument(command, "--t3-design", T3_DESIGN)
            replace_argument(command, "--t3-corpus", F38_S2_UNION)
        elif name in {"03_conflict_base_features", "04_conflict_v5_features"}:
            replace_argument(command, "--t3-corpus", F38_S2_UNION)
        elif name == "05_evaluation_inputs":
            replace_argument(command, "--conflict-schedule", ORIGINAL_CONFLICT_SCHEDULE)
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
    base._global_snapshot_attrition_gate = f39_attrition
    base._run = run_with_corpora
    result = int(predecessor.main())

    audit_path = TARGET / "finalization_audit.json"
    audit = read_json(audit_path)
    audit["schema_version"] = "kinofail.reconfirmation-finalization.f39.v1"
    audit["f39_f38_s2_confirmation"] = {
        "seal": str(F39_SEAL.relative_to(ROOT)),
        "seal_sha256": sha256(F39_SEAL),
        "f35_scale_audit_sha256": sha256(F35_AUDIT),
        "f37_failure_audit_sha256": sha256(F37_AUDIT),
        "f38_raw_failure_audit_sha256": sha256(F38_RAW_AUDIT),
        "f38_task_aligned_audit_sha256": sha256(F38_TASK_AUDIT),
        "f38_s2_design_audit_sha256": sha256(F38_S2_AUDIT),
        "raw_generic_failure_preserved": True,
        "post_acquisition_pre_prediction_amendment_disclosed": True,
        "frozen_models_router_threshold_eta_and_statistics_unchanged": True,
        "selection_used_model_prediction_truth_or_outcome_strength": False,
        "counts": seal["counts"],
    }
    audit["f39_finalized_utc"] = datetime.now(UTC).isoformat()
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, audit_path)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
