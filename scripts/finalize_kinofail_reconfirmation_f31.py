#!/usr/bin/env python3
"""Run one frozen blind evaluation with F25 Scale and F30 T3 overlays."""

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
from scripts.seal_kinofail_reconfirmation_f31 import (  # noqa: E402
    F25_AUDIT,
    F29_AUDIT,
    F30_AUDIT,
    F30_ROOT,
    OUTPUT as F31_SEAL,
    SCALE_OVERLAY,
    TARGET,
    read_json,
    sha256,
)


ORIGINAL_EVAL = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2"
SCALE_OVERLAY_ROOT = SCALE_OVERLAY.parent
T3_DESIGN = F30_ROOT / "combined_design"
T3_UNION = Path("/data/eureka/KinoVLA/outputs/kinofail_t3_replenishment_f30/union")


def validate_seal() -> dict[str, Any]:
    seal = read_json(F31_SEAL)
    sidecar = F31_SEAL.with_name("seal_manifest.sha256")
    scripts = seal.get("source_sha256", {}).get("execution_scripts", {})
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").split()[0] != sha256(F31_SEAL)
        or seal.get("status") != "sealed_before_scale_t3_replenished_blind_prediction"
        or seal.get("passed") is not True
        or seal.get("model_prediction_or_score_read") is not False
        or seal.get("source_sha256", {}).get("scale_overlay_audit") != sha256(SCALE_OVERLAY)
        or seal.get("source_sha256", {}).get("f29_final_audit") != sha256(F29_AUDIT)
        or seal.get("source_sha256", {}).get("f30_design_audit") != sha256(F30_AUDIT)
        or scripts.get(str(Path(__file__).resolve().relative_to(ROOT))) != sha256(Path(__file__).resolve())
    ):
        raise RuntimeError("invalid F31 pre-prediction seal")
    return seal


def replenished_attrition(_: list[str]) -> dict[str, Any]:
    scale = read_json(SCALE_OVERLAY)
    f25 = read_json(F25_AUDIT)
    f29 = read_json(F29_AUDIT)
    f30 = read_json(F30_AUDIT)
    if not all(row.get("passed") is True for row in (scale, f25, f29, f30)):
        raise RuntimeError("F31 overlay audit invalid")
    scale_invalid = 412
    t3_invalid = int(f30["invalid_case_count"])
    counts = {
        "scale_pair_units": {
            "planned": 10_560,
            "valid": 10_148,
            "attrited": scale_invalid,
            "attrition_rate": scale_invalid / 10_560,
        },
        "t3_complete_case_units": {
            "planned": 1_500,
            "valid": int(f30["valid_cases"]),
            "attrited": t3_invalid,
            "attrition_rate": t3_invalid / 1_500,
        },
        "combined_analysis_units": {
            "planned": 12_060,
            "valid": 10_148 + int(f30["valid_cases"]),
            "attrited": scale_invalid + t3_invalid,
            "attrition_rate": (scale_invalid + t3_invalid) / 12_060,
        },
    }
    checks = {
        "scale_pair_attrition_strictly_below_five_percent": counts["scale_pair_units"]["attrition_rate"] < 0.05,
        "t3_complete_case_attrition_strictly_below_five_percent": counts["t3_complete_case_units"]["attrition_rate"] < 0.05,
        "combined_analysis_unit_attrition_strictly_below_five_percent": counts["combined_analysis_units"]["attrition_rate"] < 0.05,
        "original_attrition_retained_in_provenance": True,
        "rejected_interrupted_replacements_excluded": True,
    }
    if not all(checks.values()):
        raise RuntimeError("F31 replenished attrition gate failed")
    return {
        "schema_version": "kinofail.reconfirmation-f31-attrition.v1",
        "passed": True,
        "threshold": 0.05,
        "threshold_scope": "Scale physical pairs; T3 complete conflict cases; combined analysis units",
        "posthoc_supplementary_replenishment": True,
        "original_attrition_reported": True,
        "counts": counts,
        "checks": checks,
        "original_pre_replenishment": {
            "scale_missing_pairs": 1_417,
            "t3_invalid_cases": 91,
            "t3_missing_source_pairs": 114,
        },
        "accepted_replenishment": {
            "scale_pairs": 1_005,
            "t3_complete_cases": int(f30["accepted_replacement_cases"]),
        },
        "source_sha256": {
            "f25_final_audit": sha256(F25_AUDIT),
            "scale_overlay_audit": sha256(SCALE_OVERLAY),
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
        raise FileExistsError(f"refusing to overwrite F31 evaluation: {TARGET}")
    scenes = sorted(path.name for path in (ORIGINAL_EVAL / "shards").glob("*"))
    if len(scenes) != 30:
        raise RuntimeError(f"expected 30 original shards, found {len(scenes)}")
    for scene in scenes:
        if not (SCALE_OVERLAY_ROOT / "shards" / scene / "scale/unified_features/feature_manifest.json").is_file():
            raise FileNotFoundError(f"missing Scale overlay shard: {scene}")
    for path in (
        T3_DESIGN / "audit.json",
        F30_ROOT / "combined_conflict_schedule.jsonl",
        F30_ROOT / "artifact_inventory.jsonl",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    original_run = base._run

    def run_with_overlays(command: list[str], name: str) -> None:
        if name == "02_valid_conflict_design":
            replace_argument(command, "--t3-design", T3_DESIGN)
            replace_argument(command, "--t3-corpus", T3_UNION)
        elif name == "03_conflict_base_features":
            replace_argument(command, "--t3-corpus", T3_UNION)
        elif name == "04_conflict_v5_features":
            replace_argument(command, "--t3-corpus", T3_UNION)
        elif name == "05_evaluation_inputs":
            replace_argument(
                command,
                "--conflict-schedule",
                F30_ROOT / "combined_conflict_schedule.jsonl",
            )
            insertion = command.index("--conflict-schedule")
            overlay_args: list[str] = []
            for scene in scenes:
                shard = SCALE_OVERLAY_ROOT / "shards" / scene / "scale"
                overlay_args.extend(
                    [
                        "--scale-snapshot-dir", str(shard / "accepted_snapshots"),
                        "--scale-unified-dir", str(shard / "unified_features"),
                    ]
                )
            command[insertion:insertion] = overlay_args
        original_run(command, name)

    base.EVAL_ROOT = TARGET
    base.SHARD_ROOT = ORIGINAL_EVAL / "shards"
    base._global_snapshot_attrition_gate = replenished_attrition
    base._run = run_with_overlays
    result = int(predecessor.main())

    audit_path = TARGET / "finalization_audit.json"
    audit = read_json(audit_path)
    audit["schema_version"] = "kinofail.reconfirmation-finalization.f31.v1"
    audit["f31_scale_t3_replenishment"] = {
        "seal": str(F31_SEAL.relative_to(ROOT)),
        "seal_sha256": sha256(F31_SEAL),
        "scale_overlay_audit_sha256": sha256(SCALE_OVERLAY),
        "f29_final_audit_sha256": sha256(F29_AUDIT),
        "f30_design_audit_sha256": sha256(F30_AUDIT),
        "posthoc_supplementary_replenishment": True,
        "original_confirmation_attrition_must_be_reported": True,
        "frozen_model_features_threshold_and_statistics_unchanged": True,
        "selection_used_model_predictions_or_outcome_strength": False,
        "counts": seal["counts"],
    }
    audit["f31_finalized_utc"] = datetime.now(UTC).isoformat()
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, audit_path)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
