#!/usr/bin/env python3
"""Seal the result-blind F37 loader amendment after preserved F36 failure."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PARENT_SEAL = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f36/seal_manifest.json"
FAILURE_AUDIT = ROOT / "outputs/kinofail_reconfirmation_f36_failure_audit_v1/audit.json"
A4_SEAL = ROOT / "outputs/freeze/kinofail_reconfirmation_a4_v8/seal_manifest.json"
TARGET = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f37"
OUTPUT = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f37/seal_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    if OUTPUT.exists() or TARGET.exists():
        raise FileExistsError("refusing to overwrite F37 seal or evaluation")
    parent = read_json(PARENT_SEAL)
    failure = read_json(FAILURE_AUDIT)
    a4 = read_json(A4_SEAL)
    if (
        PARENT_SEAL.with_name("seal_manifest.sha256").read_text().split()[0] != sha256(PARENT_SEAL)
        or parent.get("passed") is not True
        or parent.get("model_prediction_or_score_read") is not False
        or failure.get("passed") is not True
        or failure.get("status") != "result_blind_loader_failure_preserved"
        or failure.get("model_prediction_or_score_read") is not False
        or failure.get("scientific_attempt_started") is not False
        or a4.get("passed") is not True
        or a4.get("status") != "sealed_before_f36_prediction_and_confirmatory_a4_outcomes"
        or parent.get("source_sha256", {}).get("a4_pre_prediction_seal") != sha256(A4_SEAL)
    ):
        raise RuntimeError("invalid F36 failure boundary or A4 seal")
    scripts = [
        ROOT / "scripts/finalize_kinofail_reconfirmation_f37.py",
        ROOT / "scripts/finalize_kinofail_reconfirmation_f36.py",
        ROOT / "scripts/audit_kinofail_reconfirmation_f36_failure.py",
        ROOT / "scripts/build_kinofail_t3_replenishment_f30_overlay.py",
        ROOT / "scripts/build_kinofail_scale_direct_o9_overlay_f35.py",
        ROOT / "scripts/finalize_kinofail_reconfirmation_v8.py",
        ROOT / "scripts/prepare_kinofail_confirmatory_valid_conflict_design_v1.py",
        ROOT / "scripts/build_kinofail_confirmatory_c2_base_features_v1.py",
        ROOT / "scripts/build_kinofail_realistic_c2_v5_features.py",
        ROOT / "scripts/build_kinofail_confirmatory_evaluation_bundle_inputs_v1.py",
        ROOT / "scripts/assemble_kinofail_confirmatory_blind_bundle_v1.py",
        ROOT / "scripts/predict_kinofail_unified_moe_v3_blind.py",
        ROOT / "scripts/build_kinofail_confirmatory_scoring_protocol_v1.py",
        ROOT / "scripts/score_kinofail_unified_moe_v3_confirmatory.py",
        ROOT / "scripts/analyze_kinofail_reconfirmation_a6_v7.py",
        ROOT / "scripts/assemble_kinofail_reconfirmation_a0_a7_v7.py",
        ROOT / "scripts/build_kinofail_reconfirmation_a4_v8_schedule.py",
        ROOT / "scripts/seal_kinofail_reconfirmation_a4_v8.py",
        ROOT / "scripts/run_kinofail_reconfirmation_a4_v8.py",
        ROOT / "scripts/isaac_collect_kinofail_reconfirmation_a4_v7.py",
        ROOT / "scripts/analyze_kinofail_reconfirmation_a4_v8.py",
    ]
    if any(not path.is_file() for path in scripts):
        raise FileNotFoundError("F37 execution dependency missing")
    manifest = {
        "schema_version": "kinofail.reconfirmation-f37-seal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_symlink_loader_amendment_blind_prediction",
        "passed": True,
        "model_prediction_or_score_read": False,
        "parent_f36_status": "result_blind_loader_failure_preserved",
        "original_confirmation_attrition_must_be_reported": True,
        "scientific_contract": {
            **dict(parent["scientific_contract"]),
            "architecture_router_features_threshold_eta_and_statistics_unchanged": True,
            "only_amendment": (
                "enumerate fixed-depth F30 episode-directory symlinks instead of "
                "Path.rglob(manifest.json), which does not descend into directory symlinks"
            ),
            "base_feature_math_unchanged": True,
            "v5_temporal_feature_math_unchanged": True,
            "case_schedule_and_union_unchanged": True,
            "no_prediction_existed_when_amendment_was_frozen": True,
            "score_once": True,
        },
        "counts": parent["counts"],
        "source_sha256": {
            **{
                key: value
                for key, value in parent["source_sha256"].items()
                if key != "execution_scripts"
            },
            "parent_f36_seal": sha256(PARENT_SEAL),
            "f36_failure_audit": sha256(FAILURE_AUDIT),
            "a4_pre_prediction_seal": sha256(A4_SEAL),
            "fixed_v5_loader": sha256(ROOT / "scripts/build_kinofail_realistic_c2_v5_features.py"),
            "execution_scripts": {
                str(path.relative_to(ROOT)): sha256(path) for path in scripts
            },
            "sealer": sha256(Path(__file__).resolve()),
        },
        "target_eval_root": str(TARGET.relative_to(ROOT)),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    OUTPUT.with_name("seal_manifest.sha256").write_text(f"{sha256(OUTPUT)}  {OUTPUT.name}\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
