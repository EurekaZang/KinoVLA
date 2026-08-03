#!/usr/bin/env python3
"""F19-aware finalizer after storage-only scene04 postprocess recovery."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import finalize_kinofail_reconfirmation_v4 as predecessor
from scripts import run_kinofail_reconfirmation_scene_pipeline_v4 as pipeline


RECOVERY_AUDIT = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/orchestration/"
    "confirm_v2_production_scene_04/f19_storage_postprocess_recovery.json"
)


def _validate_recovery() -> dict:
    amendment = pipeline._validate_f19()
    audit = predecessor.base.base._json(RECOVERY_AUDIT)
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
        != predecessor._sha256(pipeline.AMENDMENT)
    ):
        raise RuntimeError("F19 scene04 recovery audit is invalid")
    return amendment


def main() -> int:
    amendment = _validate_recovery()
    result = int(predecessor.main())
    audit_path = predecessor.base.base.EVAL_ROOT / "finalization_audit.json"
    audit = predecessor.base.base._json(audit_path)
    audit["schema_version"] = "kinofail.reconfirmation-finalization.v5"
    audit["f19_storage_postprocess_amendment"] = {
        "path": str(pipeline.AMENDMENT),
        "sha256": predecessor._sha256(pipeline.AMENDMENT),
        "status": amendment["status"],
        "scene04_recovery_audit": str(RECOVERY_AUDIT),
        "scene04_recovery_audit_sha256": predecessor._sha256(
            RECOVERY_AUDIT
        ),
        "scientific_content_changed": False,
    }
    audit["f19_finalizer_wrapped_utc"] = datetime.now(UTC).isoformat()
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, audit_path)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
