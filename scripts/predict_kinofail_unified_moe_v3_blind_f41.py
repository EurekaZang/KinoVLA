#!/usr/bin/env python3
"""Inference-only successor accepting the already sealed F12 backend amendment.

The F12 simulator backend is acquisition-only and unreachable from blind inference.
All inference-reachable F0 files and every frozen checkpoint remain byte-identical.
"""

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


F12 = ROOT / "outputs/freeze/unified_moe_v3_reconfirmation_f12_process_local_rtx_texture_amendment1/amendment_manifest.json"
BACKEND_RELATIVE = "kino_vla/sim/isaac_policy_backend.py"
ORIGINAL_VALIDATE_F0 = base.validate_f0


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def validate_f0_with_f12(path: Path) -> tuple[dict[str, Any], dict[str, bool]]:
    manifest, checks = ORIGINAL_VALIDATE_F0(path)
    if all(checks.values()):
        return manifest, checks
    if any(value is not True for key, value in checks.items() if key != "all_frozen_files_unchanged"):
        raise RuntimeError(f"non-file F0 contract drift: {checks}")
    f12 = read_json(F12)
    correction = f12.get("correction", {})
    scientific = f12.get("scientific_contract", {})
    backend_items = [
        item for item in manifest.get("frozen_files", []) if item.get("path") == BACKEND_RELATIVE
    ]
    if len(backend_items) != 1:
        raise RuntimeError("F0 must bind exactly one simulator backend")
    frozen_backend = backend_items[0]
    backend = ROOT / BACKEND_RELATIVE
    if (
        f12.get("passed") is not True
        or f12.get("status") != "sealed_before_resumption_after_rtx_texture_race"
        or correction.get("backend") != BACKEND_RELATIVE
        or correction.get("backend_predecessor_sha256") != frozen_backend.get("sha256")
        or correction.get("backend_sha256") != base.sha256_file(backend)
        or correction.get("process_local_namespace") is not True
        or any(
            scientific.get(key) is not False
            for key in (
                "model_or_route_changed",
                "sensor_or_feature_logic_changed",
                "simulation_logic_changed",
                "threshold_or_analysis_changed",
                "result_dependent_retry_enabled",
                "scene_material_seed_operator_or_parameter_changed",
            )
        )
        or "kino_vla.sim.isaac_policy_backend" in sys.modules
    ):
        raise RuntimeError("F12 does not authorize the acquisition-only backend drift")
    for item in manifest.get("frozen_files", []):
        if item.get("path") == BACKEND_RELATIVE:
            continue
        frozen_path = Path(str(item["path"]))
        if not frozen_path.is_absolute():
            frozen_path = ROOT / frozen_path
        if (
            not frozen_path.is_file()
            or frozen_path.stat().st_size != int(item["bytes"])
            or base.sha256_file(frozen_path) != str(item["sha256"])
        ):
            raise RuntimeError(f"inference-reachable F0 artifact drift: {frozen_path}")
    checks["all_frozen_files_unchanged"] = True
    checks["f12_acquisition_only_backend_amendment_valid"] = True
    checks["all_other_f0_files_unchanged"] = True
    checks["simulator_backend_not_imported_by_predictor"] = True
    return manifest, checks


def main() -> int:
    base.validate_f0 = validate_f0_with_f12
    result = int(base.main())
    output_index = sys.argv.index("--output-dir") + 1
    output = Path(sys.argv[output_index]).resolve()
    manifest_path = output / "prediction_manifest.json"
    manifest = read_json(manifest_path)
    manifest["schema_version"] = "kinofail.unified-moe-blind-predictions.f41.v1"
    manifest["f41_inference_reachability_validation"] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "f12_amendment": str(F12.relative_to(ROOT)),
        "f12_amendment_sha256": base.sha256_file(F12),
        "f12_backend_sha256": base.sha256_file(ROOT / BACKEND_RELATIVE),
        "backend_imported_or_used_by_predictor": False,
        "all_other_f0_files_and_checkpoints_byte_identical": True,
        "model_route_feature_threshold_or_statistics_changed": False,
    }
    manifest["source_sha256"]["predictor_f41"] = base.sha256_file(Path(__file__).resolve())
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    manifest_path.with_name("prediction_manifest.sha256").write_text(
        f"{base.sha256_file(manifest_path)}  prediction_manifest.json\n"
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
