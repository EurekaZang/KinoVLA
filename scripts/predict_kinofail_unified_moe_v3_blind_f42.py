#!/usr/bin/env python3
"""Blind predictor with sealed F12 and legacy-v1 freshness compatibility."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import predict_kinofail_unified_moe_v3_blind as base  # noqa: E402
from scripts.predict_kinofail_unified_moe_v3_blind_f41 import (  # noqa: E402
    F12,
    validate_f0_with_f12,
)


ORIGINAL_LOAD_JSON = base._load_json


def argument_path(flag: str) -> Path:
    return Path(sys.argv[sys.argv.index(flag) + 1]).resolve()


def main() -> int:
    f0_path = argument_path("--f0-manifest")
    f1_path = argument_path("--f1-manifest")
    freshness_path = argument_path("--freshness-audit")
    f1 = ORIGINAL_LOAD_JSON(f1_path)
    freshness = ORIGINAL_LOAD_JSON(freshness_path)
    if (
        freshness.get("schema_version")
        != "kinofail.unified-confirmatory-schedule-freshness.v1"
        or freshness.get("passed") is not True
        or freshness.get("source_sha256") is not None
        or f1.get("input_sha256", {}).get("f0_manifest")
        != base.sha256_file(f0_path)
        or f1.get("input_sha256", {}).get("freshness_audit")
        != base.sha256_file(freshness_path)
    ):
        raise RuntimeError("legacy freshness compatibility contract failed")

    def load_json_with_legacy_binding(path: Path) -> dict[str, Any]:
        value = ORIGINAL_LOAD_JSON(path)
        if path.resolve() == freshness_path:
            value = dict(value)
            value["source_sha256"] = {"f0_manifest": base.sha256_file(f0_path)}
        return value

    base.validate_f0 = validate_f0_with_f12
    base._load_json = load_json_with_legacy_binding
    result = int(base.main())

    output = argument_path("--output-dir")
    manifest_path = output / "prediction_manifest.json"
    manifest = ORIGINAL_LOAD_JSON(manifest_path)
    manifest["schema_version"] = "kinofail.unified-moe-blind-predictions.f42.v1"
    manifest["f42_preinference_compatibility_validation"] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "f12_amendment": str(F12.relative_to(ROOT)),
        "f12_amendment_sha256": base.sha256_file(F12),
        "acquisition_backend_imported_or_used_by_predictor": False,
        "legacy_freshness_schema": freshness["schema_version"],
        "legacy_freshness_file_modified": False,
        "exact_f0_and_freshness_binding_verified_through_f1": True,
        "all_inference_reachable_f0_files_and_checkpoints_byte_identical": True,
        "model_route_feature_threshold_or_statistics_changed": False,
    }
    manifest["source_sha256"]["predictor_f42"] = base.sha256_file(
        Path(__file__).resolve()
    )
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    manifest_path.with_name("prediction_manifest.sha256").write_text(
        f"{base.sha256_file(manifest_path)}  prediction_manifest.json\n"
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
