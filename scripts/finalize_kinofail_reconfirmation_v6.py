#!/usr/bin/env python3
"""F20 finalizer with the sealed F19 module-reference correction."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import finalize_kinofail_reconfirmation_v4 as predecessor
from scripts import finalize_kinofail_reconfirmation_v5 as failed_predecessor
from scripts import run_kinofail_reconfirmation_scene_pipeline_v4 as pipeline


BASE_FINALIZER = predecessor.predecessor.base
F20 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f20_finalizer_reference_amendment1/"
    "amendment_manifest.json"
)
RECOVERY_AUDIT = failed_predecessor.RECOVERY_AUDIT
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


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _validate_f20() -> dict[str, Any]:
    amendment = _json(F20)
    correction = amendment.get("correction", {})
    scientific = amendment.get("scientific_contract", {})
    paths = {
        "finalizer_v6": Path(__file__).resolve(),
        "failed_finalizer_v5": Path(failed_predecessor.__file__).resolve(),
        "finalizer_v4": Path(predecessor.__file__).resolve(),
        "sealer_v1": ROOT / "scripts/seal_kinofail_reconfirmation_f20.py",
        "failure_log": FAILURE_LOG,
    }
    if (
        amendment.get("schema_version")
        != "kinofail.reconfirmation-f20-finalizer-reference-amendment.v1"
        or amendment.get("status")
        != "sealed_during_scene05_acquisition_before_finalization"
        or amendment.get("passed") is not True
        or amendment.get("predecessor_f19_sha256")
        != _sha256(pipeline.AMENDMENT)
        or any(
            not path.is_file()
            or correction.get(f"{name}_sha256") != _sha256(path)
            for name, path in paths.items()
        )
        or correction.get("module_reference_only_changed") is not True
        or correction.get("predecessor_finalizer_main_reused") is not True
        or correction.get("failed_before_predecessor_main") is not True
        or scientific.get("model_feature_route_threshold_or_analysis_changed")
        is not False
        or scientific.get("collection_schedule_simulation_or_sensor_changed")
        is not False
        or scientific.get("existing_scientific_artifact_modified") is not False
        or scientific.get("prediction_or_score_used") is not False
    ):
        raise RuntimeError("F20 finalizer-reference amendment is invalid")
    return amendment


def _validate_recovery() -> dict[str, Any]:
    amendment = pipeline._validate_f19()
    audit = _json(RECOVERY_AUDIT)
    if (
        audit.get("schema_version")
        != "kinofail.reconfirmation-f19-scene04-recovery.v1"
        or audit.get("state") != "terminal"
        or audit.get("passed") is not True
        or audit.get("physical_acquisition_reexecuted") is not False
        or audit.get("scale_or_t3_postprocess_reexecuted") is not False
        or audit.get("model_or_prediction_loaded") is not False
        or audit.get("scientific_content_changed") is not False
        or audit.get("result_dependent_retry_or_selection") is not False
        or audit.get("operational_amendment_sha256")
        != _sha256(pipeline.AMENDMENT)
    ):
        raise RuntimeError("F19 scene04 recovery audit is invalid")
    return amendment


def main() -> int:
    f20 = _validate_f20()
    f19 = _validate_recovery()
    result = int(predecessor.main())
    audit_path = BASE_FINALIZER.EVAL_ROOT / "finalization_audit.json"
    audit = _json(audit_path)
    audit["schema_version"] = "kinofail.reconfirmation-finalization.v6"
    audit["f19_storage_postprocess_amendment"] = {
        "path": str(pipeline.AMENDMENT),
        "sha256": _sha256(pipeline.AMENDMENT),
        "status": f19["status"],
        "scene04_recovery_audit": str(RECOVERY_AUDIT),
        "scene04_recovery_audit_sha256": _sha256(RECOVERY_AUDIT),
        "scientific_content_changed": False,
    }
    audit["f20_finalizer_reference_amendment"] = {
        "path": str(F20),
        "sha256": _sha256(F20),
        "status": f20["status"],
        "failed_finalizer_log": str(FAILURE_LOG),
        "failed_finalizer_log_sha256": _sha256(FAILURE_LOG),
        "module_reference_only_changed": True,
        "scientific_content_changed": False,
    }
    audit["f20_finalizer_wrapped_utc"] = datetime.now(UTC).isoformat()
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, audit_path)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
