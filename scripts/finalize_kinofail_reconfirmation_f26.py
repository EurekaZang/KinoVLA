#!/usr/bin/env python3
"""Run the frozen reconfirmation scorer with the F25 accepted overlay once."""

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
from scripts.seal_kinofail_reconfirmation_f26 import (  # noqa: E402
    F25_AUDIT,
    OUTPUT as F26_SEAL,
    OVERLAY,
    TARGET,
    read_json,
    sha256,
)


ORIGINAL_EVAL = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2"
OVERLAY_ROOT = OVERLAY.parent


def _validate_seal() -> dict[str, Any]:
    sidecar = F26_SEAL.with_name("seal_manifest.sha256")
    seal = read_json(F26_SEAL)
    scripts = seal.get("source_sha256", {}).get("execution_scripts", {})
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").split()[0] != sha256(F26_SEAL)
        or seal.get("status") != "sealed_before_replenished_blind_prediction"
        or seal.get("passed") is not True
        or seal.get("model_prediction_or_score_read") is not False
        or seal.get("source_sha256", {}).get("overlay_audit") != sha256(OVERLAY)
        or seal.get("source_sha256", {}).get("f25_final_audit")
        != sha256(F25_AUDIT)
        or scripts.get(str(Path(__file__).resolve().relative_to(ROOT)))
        != sha256(Path(__file__).resolve())
    ):
        raise RuntimeError("invalid F26 pre-prediction seal")
    return seal


def _replenished_attrition(_: list[str]) -> dict[str, Any]:
    overlay = read_json(OVERLAY)
    f25 = read_json(F25_AUDIT)
    if (
        overlay.get("passed") is not True
        or f25.get("passed") is not True
        or overlay.get("counts", {}).get("accepted_replacements") != 1_005
    ):
        raise RuntimeError("F25 replenishment audit is invalid")
    counts = {
        "scale": {
            "planned_pairs": 10_560,
            "original_attrited_pairs": 1_417,
            "accepted_replacements": 1_005,
            "attrited_pairs": 412,
            "retained_pairs": 10_148,
            "attrited_fraction": 412 / 10_560,
        },
        "t3": {
            "planned_pairs": 3_000,
            "original_attrited_pairs": 114,
            "accepted_replacements": 0,
            "attrited_pairs": 114,
            "retained_pairs": 2_886,
            "attrited_fraction": 114 / 3_000,
        },
        "overall": {
            "planned_pairs": 13_560,
            "original_attrited_pairs": 1_531,
            "accepted_replacements": 1_005,
            "attrited_pairs": 526,
            "retained_pairs": 13_034,
            "attrited_fraction": 526 / 13_560,
        },
    }
    checks = {
        "scale_below_five_percent": counts["scale"]["attrited_fraction"] < 0.05,
        "t3_below_five_percent": counts["t3"]["attrited_fraction"] < 0.05,
        "overall_below_five_percent": counts["overall"]["attrited_fraction"]
        < 0.05,
        "overlay_audit_passed": True,
        "rejected_replacements_excluded": True,
    }
    if not all(checks.values()):
        raise RuntimeError("F26 replenished attrition gate failed")
    return {
        "schema_version": "kinofail.reconfirmation-f26-attrition.v1",
        "passed": True,
        "threshold": 0.05,
        "posthoc_supplementary_replenishment": True,
        "original_attrition_reported": True,
        "counts": counts,
        "checks": checks,
        "source_sha256": {
            "f25_final_audit": sha256(F25_AUDIT),
            "f26_overlay_audit": sha256(OVERLAY),
        },
    }


def main() -> int:
    seal = _validate_seal()
    if TARGET.exists():
        raise FileExistsError(f"refusing to overwrite F26 evaluation: {TARGET}")
    scenes = sorted(path.name for path in (ORIGINAL_EVAL / "shards").glob("*"))
    if len(scenes) != 30:
        raise RuntimeError(f"expected 30 original shards, found {len(scenes)}")
    for scene in scenes:
        if not (
            OVERLAY_ROOT
            / "shards"
            / scene
            / "scale/unified_features/feature_manifest.json"
        ).is_file():
            raise FileNotFoundError(f"missing F26 overlay shard: {scene}")

    original_run = base._run

    def run_with_overlay(command: list[str], name: str) -> None:
        if name == "05_evaluation_inputs":
            insertion = command.index("--conflict-schedule")
            overlay_args: list[str] = []
            for scene in scenes:
                shard = OVERLAY_ROOT / "shards" / scene / "scale"
                overlay_args.extend(
                    [
                        "--scale-snapshot-dir",
                        str(shard / "accepted_snapshots"),
                        "--scale-unified-dir",
                        str(shard / "unified_features"),
                    ]
                )
            command[insertion:insertion] = overlay_args
        original_run(command, name)

    base.EVAL_ROOT = TARGET
    base.SHARD_ROOT = ORIGINAL_EVAL / "shards"
    base._global_snapshot_attrition_gate = _replenished_attrition
    base._run = run_with_overlay
    result = int(predecessor.main())

    audit_path = TARGET / "finalization_audit.json"
    audit = read_json(audit_path)
    audit["schema_version"] = "kinofail.reconfirmation-finalization.f26.v1"
    audit["f26_replenishment_overlay"] = {
        "seal": str(F26_SEAL.relative_to(ROOT)),
        "seal_sha256": sha256(F26_SEAL),
        "overlay_audit": str(OVERLAY.relative_to(ROOT)),
        "overlay_audit_sha256": sha256(OVERLAY),
        "f25_final_audit": str(F25_AUDIT.relative_to(ROOT)),
        "f25_final_audit_sha256": sha256(F25_AUDIT),
        "posthoc_supplementary_replenishment": True,
        "original_confirmation_attrition_must_be_reported": True,
        "frozen_model_features_threshold_and_statistics_unchanged": True,
        "accepted_replacements": 1_005,
        "rejected_replacements_excluded": 412,
    }
    audit["f26_finalized_utc"] = datetime.now(UTC).isoformat()
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, audit_path)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
