#!/usr/bin/env python3
"""F18-aware finalizer using the authenticated cache of the pruned F14 audit."""

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

from scripts import finalize_kinofail_reconfirmation_v3 as predecessor
from scripts import recover_kinofail_reconfirmation_f14_audit_cache_v1 as recovery


F18 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f18_pruned_audit_cache_amendment1/"
    "amendment_manifest.json"
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


def _validate_f18() -> dict[str, Any]:
    amendment = _json(F18)
    correction = amendment.get("correction", {})
    paths = {
        "finalizer_v4": Path(__file__).resolve(),
        "finalizer_v3": Path(predecessor.__file__).resolve(),
        "finalizer_v2": Path(predecessor.base.__file__).resolve(),
        "recovery_script_v1": Path(recovery.__file__).resolve(),
        "f14_audit_cache": recovery.CACHE,
        "f14_raw_inventory": recovery.RAW_INVENTORY,
        "sealer_v1": (
            ROOT / "scripts/seal_kinofail_reconfirmation_f18.py"
        ),
    }
    scientific = amendment.get("scientific_contract", {})
    if (
        amendment.get("schema_version")
        != "kinofail.reconfirmation-f18-pruned-audit-cache-amendment.v1"
        or amendment.get("status")
        != "sealed_during_scene04_acquisition_before_any_prediction"
        or amendment.get("passed") is not True
        or amendment.get("predecessor_f17_sha256")
        != _sha256(predecessor.pipeline.AMENDMENT)
        or any(
            not path.is_file()
            or correction.get(f"{name}_sha256") != _sha256(path)
            for name, path in paths.items()
        )
        or correction.get("f14_audit_cache_bytes")
        != recovery.CACHE.stat().st_size
        or correction.get("byte_identical_to_pruned_original") is not True
        or correction.get("finalizer_scientific_steps_changed") is not False
        or scientific.get("prediction_or_score_used") is not False
        or scientific.get("model_feature_route_threshold_or_analysis_changed")
        is not False
        or scientific.get("collection_schedule_simulation_or_sensor_changed")
        is not False
        or scientific.get("existing_scientific_artifact_modified") is not False
    ):
        raise RuntimeError("F18 pruned-audit-cache amendment is invalid")
    return amendment


def main() -> int:
    amendment = _validate_f18()
    predecessor.base.F14_RECOVERY_AUDIT = recovery.CACHE
    result = int(predecessor.main())
    audit_path = predecessor.base.EVAL_ROOT / "finalization_audit.json"
    audit = predecessor.base._json(audit_path)
    audit["schema_version"] = "kinofail.reconfirmation-finalization.v4"
    audit["f18_pruned_audit_cache_amendment"] = {
        "path": str(F18),
        "sha256": _sha256(F18),
        "status": amendment["status"],
        "f14_audit_cache": str(recovery.CACHE),
        "f14_audit_cache_sha256": _sha256(recovery.CACHE),
        "byte_identical_to_pruned_original": True,
        "finalizer_scientific_steps_changed": False,
    }
    audit["f18_finalizer_wrapped_utc"] = datetime.now(UTC).isoformat()
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, audit_path)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
