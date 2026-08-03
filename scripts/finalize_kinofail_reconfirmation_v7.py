#!/usr/bin/env python3
"""F21-aware finalizer reusing the complete frozen F20 finalization chain."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import finalize_kinofail_reconfirmation_v6 as predecessor
from scripts import run_kinofail_reconfirmation_scene_pipeline_v5 as pipeline
from scripts import run_kinofail_reconfirmation_t2_atomic_f21 as t2


RECOVERY_AUDIT = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/orchestration/"
    "confirm_v2_production_scene_05/f21_scene05_recovery.json"
)


def _validate_recovery() -> dict:
    amendment = pipeline._validate_f21()
    audit = t2.read_json(RECOVERY_AUDIT)
    if (
        audit.get("schema_version")
        != "kinofail.reconfirmation-f21-scene05-recovery.v1"
        or audit.get("t2_recovery_state") != "terminal"
        or audit.get("t2_recovery_passed") is not True
        or audit.get("existing_completed_case_reexecuted") is not False
        or audit.get("model_feature_prediction_score_or_label_read") is not False
        or audit.get("scientific_content_changed") is not False
        or audit.get("operational_amendment_sha256") != t2.sha256(t2.F21)
    ):
        raise RuntimeError("F21 scene05 recovery audit is invalid")
    return amendment


def main() -> int:
    amendment = _validate_recovery()
    result = int(predecessor.main())
    audit_path = predecessor.BASE_FINALIZER.EVAL_ROOT / "finalization_audit.json"
    audit = predecessor._json(audit_path)
    audit["schema_version"] = "kinofail.reconfirmation-finalization.v7"
    audit["f21_atomic_t2_amendment"] = {
        "path": str(t2.F21),
        "sha256": t2.sha256(t2.F21),
        "status": amendment["status"],
        "scene05_recovery_audit": str(RECOVERY_AUDIT),
        "scene05_recovery_audit_sha256": t2.sha256(RECOVERY_AUDIT),
        "fresh_isaac_process_per_t2_case": True,
        "atomic_case_commit": True,
        "scientific_content_changed": False,
    }
    audit["f21_finalizer_wrapped_utc"] = datetime.now(UTC).isoformat()
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, audit_path)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
