#!/usr/bin/env python3
"""F17-aware wrapper around the frozen one-shot v2 finalizer."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import finalize_kinofail_reconfirmation_v2 as base
from scripts import run_kinofail_reconfirmation_scene_pipeline_v3 as pipeline


def main() -> int:
    amendment = pipeline._validate_f17()
    result = int(base.main())
    audit_path = base.EVAL_ROOT / "finalization_audit.json"
    audit = base._json(audit_path)
    audit["schema_version"] = "kinofail.reconfirmation-finalization.v3"
    audit["f17_work_conserving_storage_amendment"] = {
        "path": str(pipeline.AMENDMENT),
        "sha256": base._sha256(pipeline.AMENDMENT),
        "status": amendment["status"],
        "maximum_concurrent_isaac_processes": 3,
        "logical_partition_count": 4,
        "fresh_isaac_process_per_pair": True,
        "scientific_content_changed": False,
    }
    audit["f17_finalizer_wrapped_utc"] = datetime.now(UTC).isoformat()
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, audit_path)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
