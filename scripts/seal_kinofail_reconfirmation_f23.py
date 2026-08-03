#!/usr/bin/env python3
"""Seal the F23 validation-order correction before scene06 acquisition."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import recover_kinofail_reconfirmation_scene05_f21 as process
from scripts import run_kinofail_reconfirmation_t2_atomic_f21 as t2


F23 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f23_preflight_order_amendment1/"
    "amendment_manifest.json"
)
F21 = t2.F21
F22 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f22_orphan_wrapper_amendment1/"
    "amendment_manifest.json"
)
F21_ALL_SCENES_AUDIT = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/orchestration/"
    "all_scenes_pipeline_f21.jsonl"
)
F21_ALL_SCENES_LOG = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/orchestration_logs/f21/"
    "remaining_scenes_pipeline_v5.log"
)
SCENE06 = "confirm_v2_production_scene_06"


def main() -> int:
    if F23.exists():
        raise FileExistsError(F23)
    f21 = t2._validate_f21()
    f22 = t2.read_json(F22)
    if f22.get("passed") is not True:
        raise RuntimeError("F22 predecessor is invalid")
    paths = {
        "scene_pipeline_v6": (
            ROOT / "scripts/run_kinofail_reconfirmation_scene_pipeline_v6.py"
        ),
        "all_scenes_v6": (
            ROOT
            / "scripts/run_kinofail_reconfirmation_all_scene_pipelines_v6.py"
        ),
        "finalizer_v8": (
            ROOT / "scripts/finalize_kinofail_reconfirmation_v8.py"
        ),
        "restart_f23": (
            ROOT / "scripts/restart_kinofail_reconfirmation_f23.py"
        ),
        "sealer_f23": Path(__file__).resolve(),
    }
    for path in (
        *paths.values(),
        F21,
        F22,
        F21_ALL_SCENES_AUDIT,
        F21_ALL_SCENES_LOG,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if t2._prediction_files():
        raise RuntimeError("prediction or score exists before F23 seal")
    if process._exact_token_processes(
        "run_kinofail_reconfirmation_all_scene_pipelines_v5.py"
    ) or process._exact_token_processes(
        "run_kinofail_reconfirmation_scene_pipeline_v5.py"
    ):
        raise RuntimeError("failed v5 process is unexpectedly live")

    lines = [
        json.loads(line)
        for line in F21_ALL_SCENES_AUDIT.read_text().splitlines()
        if line
    ]
    last = lines[-1]
    scene_root = (
        ROOT / "outputs/kinofail_reconfirmation_v2/orchestration" / SCENE06
    )
    physical = (
        Path("/data/eureka/KinoVLA")
        / "outputs/kinofail_reconfirmation_v2/corpus_ext4"
        / SCENE06
    )
    receipt = (
        ROOT
        / "outputs/kinofail_reconfirmation_v2/prune_receipts"
        / SCENE06
        / "completed.json"
    )
    if (
        last.get("scene_id") != SCENE06
        or last.get("returncode") != 1
        or receipt.exists()
        or scene_root.exists()
        or physical.exists()
    ):
        raise RuntimeError("scene06 is not an untouched preflight failure")

    manifest = {
        "schema_version": (
            "kinofail.reconfirmation-f23-preflight-order-amendment.v1"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": (
            "sealed_after_scene06_preflight_failure_before_acquisition"
        ),
        "passed": True,
        "predecessor_f21": str(F21.relative_to(ROOT)),
        "predecessor_f21_sha256": t2.sha256(F21),
        "predecessor_f22": str(F22.relative_to(ROOT)),
        "predecessor_f22_sha256": t2.sha256(F22),
        "incident": {
            "classification": "f17_validation_after_f21_entrypoint_swap",
            "scene_id": SCENE06,
            "all_scenes_audit": str(F21_ALL_SCENES_AUDIT),
            "all_scenes_audit_sha256": t2.sha256(F21_ALL_SCENES_AUDIT),
            "all_scenes_log": str(F21_ALL_SCENES_LOG),
            "all_scenes_log_sha256": t2.sha256(F21_ALL_SCENES_LOG),
            "failed_row": last,
            "scene_pipeline_state_existed": False,
            "scene_physical_corpus_existed": False,
            "scene_receipt_existed": False,
            "prediction_or_score_existed": False,
            "identified_without_model_outputs": True,
        },
        "correction": {
            **{
                f"{name}_sha256": t2.sha256(path)
                for name, path in paths.items()
            },
            "f17_validation_before_t2_entrypoint_swap": True,
            "f21_atomic_t2_entrypoint_preserved": True,
            "predecessor_f17_or_f19_source_modified": False,
        },
        "scientific_contract": {
            "schedule_protocol_or_case_membership_changed": False,
            "physics_sensor_render_or_seed_changed": False,
            "existing_completed_case_reexecuted": False,
            "feature_model_route_threshold_or_analysis_changed": False,
            "result_dependent_retry_enabled": False,
        },
    }
    F23.parent.mkdir(parents=True, exist_ok=True)
    temporary = F23.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, F23)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
