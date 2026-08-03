#!/usr/bin/env python3
"""Run the one-shot blind evaluation with F35 Scale and F30 T3 corpora."""

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
from scripts.seal_kinofail_reconfirmation_f36 import (  # noqa: E402
    F29_AUDIT,
    F30_AUDIT,
    F30_ROOT,
    F35_AUDIT,
    F35_ROOT,
    OUTPUT as F36_SEAL,
    TARGET,
    read_json,
    sha256,
)


ORIGINAL_EVAL = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2"
T3_DESIGN = F30_ROOT / "combined_design"
T3_UNION = Path("/data/eureka/KinoVLA/outputs/kinofail_t3_replenishment_f30/union")


def validate_seal() -> dict[str, Any]:
    seal = read_json(F36_SEAL)
    sidecar = F36_SEAL.with_name("seal_manifest.sha256")
    scripts = seal.get("source_sha256", {}).get("execution_scripts", {})
    if (
        not sidecar.is_file()
        or sidecar.read_text().split()[0] != sha256(F36_SEAL)
        or seal.get("status") != "sealed_before_direct_o9_scale_t3_blind_prediction"
        or seal.get("passed") is not True
        or seal.get("model_prediction_or_score_read") is not False
        or seal.get("source_sha256", {}).get("f35_scale_audit") != sha256(F35_AUDIT)
        or seal.get("source_sha256", {}).get("f29_final_audit") != sha256(F29_AUDIT)
        or seal.get("source_sha256", {}).get("f30_design_audit") != sha256(F30_AUDIT)
        or scripts.get(str(Path(__file__).resolve().relative_to(ROOT))) != sha256(Path(__file__).resolve())
    ):
        raise RuntimeError("invalid F36 pre-prediction seal")
    return seal


def replenished_attrition(_: list[str]) -> dict[str, Any]:
    f35 = read_json(F35_AUDIT)
    f29 = read_json(F29_AUDIT)
    f30 = read_json(F30_AUDIT)
    if not all(row.get("passed") is True for row in (f35, f29, f30)):
        raise RuntimeError("F36 corpus audit invalid")
    scale_valid = int(f35["counts"]["effective_valid_scale_pairs"])
    scale_invalid = 10560 - scale_valid
    t2_valid = 1500
    t3_valid = int(f30["valid_cases"])
    t3_invalid = int(f30["invalid_case_count"])
    counts = {
        "scale_pair_units": {
            "planned": 10560,
            "valid": scale_valid,
            "attrited": scale_invalid,
            "attrition_rate": scale_invalid / 10560,
        },
        "t3_complete_case_units": {
            "planned": 1500,
            "valid": t3_valid,
            "attrited": t3_invalid,
            "attrition_rate": t3_invalid / 1500,
        },
        "conflict_case_units": {
            "planned": 3000,
            "valid": t2_valid + t3_valid,
            "attrited": t3_invalid,
            "attrition_rate": t3_invalid / 3000,
            "t2_valid": t2_valid,
            "t3_valid": t3_valid,
        },
        "combined_analysis_units": {
            "planned": 13560,
            "valid": scale_valid + t2_valid + t3_valid,
            "attrited": scale_invalid + t3_invalid,
            "attrition_rate": (scale_invalid + t3_invalid) / 13560,
        },
    }
    checks = {
        "scale_pair_attrition_strictly_below_five_percent": counts["scale_pair_units"]["attrition_rate"] < 0.05,
        "t3_complete_case_attrition_strictly_below_five_percent": counts["t3_complete_case_units"]["attrition_rate"] < 0.05,
        "combined_t2_t3_conflict_attrition_strictly_below_five_percent": counts["conflict_case_units"]["attrition_rate"] < 0.05,
        "combined_analysis_unit_attrition_strictly_below_five_percent": counts["combined_analysis_units"]["attrition_rate"] < 0.05,
        "legacy_o9_excluded": f35.get("legacy_o9_features_excluded") is True,
        "direct_event_o9_inserted": int(f35["counts"]["direct_event_f34_o9_pairs"]) >= 750,
        "rejected_or_interrupted_replacements_excluded": True,
    }
    if not all(checks.values()):
        raise RuntimeError("F36 attrition or O9 semantic gate failed")
    return {
        "schema_version": "kinofail.reconfirmation-f36-attrition.v1",
        "passed": True,
        "threshold": 0.05,
        "counts": counts,
        "checks": checks,
        "original_pre_replenishment": {
            "scale_missing_pairs": 1417,
            "t3_invalid_cases": 91,
            "t3_missing_source_pairs": 114,
        },
        "source_sha256": {
            "f35_scale_audit": sha256(F35_AUDIT),
            "f29_final_audit": sha256(F29_AUDIT),
            "f30_design_audit": sha256(F30_AUDIT),
        },
    }


def replace_argument(command: list[str], flag: str, value: Path) -> None:
    index = command.index(flag)
    command[index + 1] = str(value)


def main() -> int:
    seal = validate_seal()
    if TARGET.exists():
        raise FileExistsError(f"refusing to overwrite F36 evaluation: {TARGET}")
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
        F30_ROOT / "combined_conflict_schedule.jsonl",
        F30_ROOT / "artifact_inventory.jsonl",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    original_run = base._run

    def run_with_corpora(command: list[str], name: str) -> None:
        if name == "02_valid_conflict_design":
            replace_argument(command, "--t3-design", T3_DESIGN)
            replace_argument(command, "--t3-corpus", T3_UNION)
        elif name == "03_conflict_base_features":
            replace_argument(command, "--t3-corpus", T3_UNION)
        elif name == "04_conflict_v5_features":
            replace_argument(command, "--t3-corpus", T3_UNION)
        elif name == "05_evaluation_inputs":
            replace_argument(command, "--conflict-schedule", F30_ROOT / "combined_conflict_schedule.jsonl")
            for scene in scenes:
                replacements = {
                    str(ORIGINAL_EVAL / "shards" / scene / "scale/snapshots"): str(F35_ROOT / "shards" / scene / "scale/snapshots"),
                    str(ORIGINAL_EVAL / "shards" / scene / "scale/unified_features"): str(F35_ROOT / "shards" / scene / "scale/unified_features"),
                }
                command[:] = [replacements.get(value, value) for value in command]
        original_run(command, name)

    base.EVAL_ROOT = TARGET
    base.SHARD_ROOT = ORIGINAL_EVAL / "shards"
    base._global_snapshot_attrition_gate = replenished_attrition
    base._run = run_with_corpora
    result = int(predecessor.main())

    audit_path = TARGET / "finalization_audit.json"
    audit = read_json(audit_path)
    audit["schema_version"] = "kinofail.reconfirmation-finalization.f36.v1"
    audit["f36_direct_o9_scale_t3_confirmation"] = {
        "seal": str(F36_SEAL.relative_to(ROOT)),
        "seal_sha256": sha256(F36_SEAL),
        "f35_scale_audit_sha256": sha256(F35_AUDIT),
        "f29_final_audit_sha256": sha256(F29_AUDIT),
        "f30_design_audit_sha256": sha256(F30_AUDIT),
        "legacy_o9_excluded": True,
        "direct_o9_event_alignment": True,
        "frozen_model_features_threshold_and_statistics_unchanged": True,
        "selection_used_model_predictions_or_outcome_strength": False,
        "counts": seal["counts"],
    }
    audit["f36_finalized_utc"] = datetime.now(UTC).isoformat()
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, audit_path)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
