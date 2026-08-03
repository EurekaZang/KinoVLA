#!/usr/bin/env python3
"""Seal the model-blind correction of attrition scope and accounting."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
F8 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f8_ext4_recovery_amendment1/"
    "amendment_manifest.json"
)
PRUNER = ROOT / "scripts/seal_and_prune_kinofail_confirmatory_shard_f4.py"
FINALIZER = ROOT / "scripts/finalize_kinofail_reconfirmation_v2.py"
SNAPSHOT_IMPLEMENTATION = ROOT / "kino_vla/data/realistic_snapshots.py"
OUT = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f9_global_attrition_amendment1"
)
MANIFEST = OUT / "amendment_manifest.json"
EXPECTED_PREDECESSOR = {
    "pruner_sha256": (
        "bb56173f56e158701064bee7f78d554f63195469a6bc928da0b3d0fa48efa6f4"
    ),
    "finalizer_sha256": (
        "b4251b4593af695dc54ffa7aeecdf63e8f9b495ca453dc457214940dd8587095"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    if MANIFEST.exists():
        raise RuntimeError("F9 is already sealed; refusing to overwrite it")
    for path in (F8, PRUNER, FINALIZER, SNAPSHOT_IMPLEMENTATION):
        if not path.is_file():
            raise FileNotFoundError(path)
    f8 = _json(F8)
    if (
        f8.get("passed") is not True
        or f8.get("status") != "sealed_before_f8_recovery_execution"
    ):
        raise RuntimeError("F8 predecessor is invalid")

    feature_root = (
        ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2/shards"
    )
    feature_files = (
        [path for path in feature_root.rglob("*") if path.is_file()]
        if feature_root.exists()
        else []
    )
    prediction_files = list(
        (
            ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2"
        ).glob("**/*prediction*")
    )
    if feature_files or prediction_files:
        raise RuntimeError(
            "features or predictions exist; F9 can no longer be sealed"
        )

    source = Path(__file__).resolve()
    manifest = {
        "schema_version": (
            "kinofail.reconfirmation-f9-global-attrition-amendment.v1"
        ),
        "status": (
            "sealed_before_any_reconfirmation_feature_extraction"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "scope": (
            "correct missing-pair accounting and enforce the already frozen "
            "five-percent attrition ceiling per battery and overall"
        ),
        "predecessor_f8": str(F8.relative_to(ROOT)),
        "predecessor_f8_sha256": _sha256(F8),
        "source": str(source.relative_to(ROOT)),
        "source_sha256": _sha256(source),
        "identified_issue": {
            "snapshot_presence_accounting": (
                "pairs with neither manifest were absent from by_pair and "
                "therefore absent from skipped_incomplete_pairs"
            ),
            "pruner_scope_mismatch": (
                "the operational F4 wrapper enforced five percent within "
                "each scene although the frozen protocol specifies each "
                "battery and overall"
            ),
            "identified_from": (
                "source-level accounting and acquisition presence metadata"
            ),
            "features_predictions_scores_or_checkpoints_used": False,
        },
        "predecessor": EXPECTED_PREDECESSOR,
        "successor": {
            "pruner": str(PRUNER.relative_to(ROOT)),
            "pruner_sha256": _sha256(PRUNER),
            "finalizer": str(FINALIZER.relative_to(ROOT)),
            "finalizer_sha256": _sha256(FINALIZER),
            "snapshot_implementation": str(
                SNAPSHOT_IMPLEMENTATION.relative_to(ROOT)
            ),
            "snapshot_implementation_sha256": _sha256(
                SNAPSHOT_IMPLEMENTATION
            ),
            "snapshot_implementation_changed": False,
        },
        "correction": {
            "per_scene_receipt_records_exact_missing_pair_count": True,
            "missing_pair_count_rule": (
                "planned pairs minus selected complete pairs minus "
                "pair-level temporal exclusions"
            ),
            "per_scene_five_percent_gate_removed": True,
            "battery_and_overall_five_percent_gate_added_to_finalizer": True,
            "threshold": 0.05,
            "threshold_scope": "per_battery_and_overall",
            "raw_shard_pruned_only_after_derived_hash_chain_validation": True,
        },
        "scientific_contract": {
            "feature_or_sample_selection_changed": False,
            "scene_material_seed_operator_or_parameter_changed": False,
            "simulation_or_sensor_logic_changed": False,
            "model_or_route_changed": False,
            "attrition_threshold_changed": False,
            "statistical_analysis_changed": False,
            "result_dependent_retry_or_selection_changed": False,
        },
        "timing_evidence": {
            "some_model_blind_physical_acquisition_existed": True,
            "reconfirmation_feature_files_existed_at_seal": False,
            "reconfirmation_prediction_files_existed_at_seal": False,
            "reconfirmation_score_files_existed_at_seal": False,
        },
    }
    _write_json(MANIFEST, manifest)
    print(
        json.dumps(
            {
                "manifest": str(MANIFEST),
                "manifest_sha256": _sha256(MANIFEST),
                "pruner_sha256": _sha256(PRUNER),
                "finalizer_sha256": _sha256(FINALIZER),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
