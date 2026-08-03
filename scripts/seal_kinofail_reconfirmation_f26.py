#!/usr/bin/env python3
"""Seal the F26 replenished evaluation before any model prediction."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = (
    ROOT
    / "outputs/eval/unified_moe_v3_replenishment_f25/accepted_overlay/overlay_audit.json"
)
F25_AUDIT = ROOT / "outputs/kinofail_replenishment_f25/final_audit.json"
TARGET = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_replenished_f26"
OUTPUT = (
    ROOT
    / "outputs/freeze/unified_moe_v3_reconfirmation_f26_replenishment_overlay/seal_manifest.json"
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
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite F26 seal: {OUTPUT}")
    forbidden = (
        [
            path
            for path in TARGET.rglob("*")
            if path.is_file()
            and (
                "prediction" in path.name.lower()
                or "score" in path.name.lower()
            )
        ]
        if TARGET.exists()
        else []
    )
    if forbidden:
        raise RuntimeError(f"F26 prediction/score exists before seal: {forbidden[:3]}")
    overlay = read_json(OVERLAY)
    f25 = read_json(F25_AUDIT)
    if (
        overlay.get("passed") is not True
        or overlay.get("model_prediction_or_score_read") is not False
        or overlay.get("counts", {}).get("accepted_replacements") != 1_005
        or overlay.get("counts", {}).get("remaining_scale_attrition") != 412
        or f25.get("passed") is not True
        or f25.get("model_prediction_feature_label_or_score_read") is not False
    ):
        raise RuntimeError("invalid F25/F26 overlay evidence")

    scripts = [
        ROOT / "scripts/finalize_kinofail_reconfirmation_f26.py",
        ROOT / "scripts/build_kinofail_replenishment_f26_overlay.py",
        ROOT / "scripts/finalize_kinofail_reconfirmation_v8.py",
        ROOT / "scripts/build_kinofail_confirmatory_evaluation_bundle_inputs_v1.py",
        ROOT / "scripts/assemble_kinofail_confirmatory_blind_bundle_v1.py",
        ROOT / "scripts/predict_kinofail_unified_moe_v3_blind.py",
        ROOT / "scripts/build_kinofail_confirmatory_scoring_protocol_v1.py",
        ROOT / "scripts/score_kinofail_unified_moe_v3_confirmatory.py",
    ]
    if any(not path.is_file() for path in scripts):
        raise FileNotFoundError("F26 execution script is missing")
    manifest = {
        "schema_version": "kinofail.reconfirmation-f26-seal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_replenished_blind_prediction",
        "passed": True,
        "posthoc_supplementary_replenishment": True,
        "original_confirmation_attrition_must_be_reported": True,
        "model_prediction_or_score_read": False,
        "scientific_contract": {
            "frozen_f0_models_unchanged": True,
            "frozen_f1_analysis_unchanged": True,
            "router_features_threshold_and_statistics_unchanged": True,
            "only_change": (
                "add the F25 model-blind accepted replacement overlay to the "
                "original frozen Scale slots"
            ),
            "rejected_replacements_excluded": True,
            "original_conflict_shards_unchanged": True,
        },
        "counts": {
            "scale_planned_groups": 10_560,
            "scale_valid_after_replenishment": 10_148,
            "scale_remaining_attrition": 412,
            "conflict_planned_groups": 3_000,
            "conflict_remaining_attrition": 114,
            "overall_planned_groups": 13_560,
            "overall_remaining_attrition": 526,
        },
        "source_sha256": {
            "overlay_audit": sha256(OVERLAY),
            "f25_final_audit": sha256(F25_AUDIT),
            "execution_scripts": {
                str(path.relative_to(ROOT)): sha256(path) for path in scripts
            },
        },
        "target_eval_root": str(TARGET.relative_to(ROOT)),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecar = OUTPUT.with_name("seal_manifest.sha256")
    sidecar.write_text(f"{sha256(OUTPUT)}  {OUTPUT.name}\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
