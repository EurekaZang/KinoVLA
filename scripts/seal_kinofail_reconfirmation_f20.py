#!/usr/bin/env python3
"""Seal the model-blind F20 finalizer module-reference correction."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f20_finalizer_reference_amendment1/"
    "amendment_manifest.json"
)
F19 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f19_storage_postprocess_amendment1/"
    "amendment_manifest.json"
)
FAILURE_LOG = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/orchestration_logs/f19/"
    "finalizer_v5.log"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    paths = {
        "finalizer_v6": (
            ROOT / "scripts/finalize_kinofail_reconfirmation_v6.py"
        ),
        "failed_finalizer_v5": (
            ROOT / "scripts/finalize_kinofail_reconfirmation_v5.py"
        ),
        "finalizer_v4": (
            ROOT / "scripts/finalize_kinofail_reconfirmation_v4.py"
        ),
        "sealer_v1": Path(__file__).resolve(),
        "failure_log": FAILURE_LOG,
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("F20 input is missing")
    failure = FAILURE_LOG.read_text(encoding="utf-8")
    if (
        "AttributeError" not in failure
        or "finalize_kinofail_reconfirmation_v4" not in failure
        or "has no attribute 'base'" not in failure
    ):
        raise RuntimeError("unexpected F19 finalizer failure")
    f19 = json.loads(F19.read_text(encoding="utf-8"))
    if f19.get("passed") is not True:
        raise RuntimeError("F19 is not sealed")
    manifest = {
        "schema_version": (
            "kinofail.reconfirmation-f20-finalizer-reference-amendment.v1"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_during_scene05_acquisition_before_finalization",
        "passed": True,
        "predecessor_f19": str(F19.relative_to(ROOT)),
        "predecessor_f19_sha256": _sha256(F19),
        "incident": {
            "classification": "finalizer_wrapper_module_reference_error",
            "failed_entrypoint": str(paths["failed_finalizer_v5"]),
            "failed_before_predecessor_main": True,
            "failed_before_waiting_or_scientific_steps": True,
            "model_feature_prediction_score_or_label_loaded": False,
        },
        "correction": {
            **{
                f"{name}_sha256": _sha256(path)
                for name, path in paths.items()
            },
            "module_reference_only_changed": True,
            "predecessor_finalizer_main_reused": True,
            "failed_before_predecessor_main": True,
            "base_finalizer_reference": (
                "finalize_v4.predecessor.base"
            ),
        },
        "scientific_contract": {
            "model_feature_route_threshold_or_analysis_changed": False,
            "collection_schedule_simulation_or_sensor_changed": False,
            "existing_scientific_artifact_modified": False,
            "prediction_or_score_used": False,
            "result_dependent_retry_or_selection_enabled": False,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, OUT)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
