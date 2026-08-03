#!/usr/bin/env python3
"""F23-aware finalizer reusing the complete F21 v7 finalization chain."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import finalize_kinofail_reconfirmation_v7 as predecessor
from scripts import run_kinofail_reconfirmation_scene_pipeline_v6 as pipeline
from scripts import run_kinofail_reconfirmation_t2_atomic_f21 as t2


def main() -> int:
    amendment = pipeline._validate_f23()
    result = int(predecessor.main())
    audit_path = (
        predecessor.predecessor.BASE_FINALIZER.EVAL_ROOT
        / "finalization_audit.json"
    )
    audit = predecessor.predecessor._json(audit_path)
    audit["schema_version"] = "kinofail.reconfirmation-finalization.v8"
    audit["f23_preflight_order_amendment"] = {
        "path": str(pipeline.F23),
        "sha256": t2.sha256(pipeline.F23),
        "status": amendment["status"],
        "f17_validation_before_t2_entrypoint_swap": True,
        "scientific_content_changed": False,
    }
    audit["f23_finalizer_wrapped_utc"] = datetime.now(UTC).isoformat()
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, audit_path)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
