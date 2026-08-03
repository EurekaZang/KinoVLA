#!/usr/bin/env python3
"""Seal the model-blind F17 scheduling and storage-only amendment."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f17_work_conserving_storage_amendment1"
)
MANIFEST = OUT / "amendment_manifest.json"
PREDECESSOR = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f16_t2_production00_liveness_amendment1/"
    "amendment_manifest.json"
)
SCENE03_RECEIPT = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/prune_receipts/"
    "confirm_v2_production_scene_03/completed.json"
)
SCENE04_CORPUS = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/corpus_ext4/"
    "confirm_v2_production_scene_04"
)
SCRATCH_LINK = (
    ROOT / "outputs/kinofail_reconfirmation_v2/corpus_ext4"
)
SCRATCH_BACKUP = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/"
    ".corpus_ext4.f17_migration_backup"
)
SCRATCH_DESTINATION = Path(
    "/data/eureka/KinoVLA/outputs/kinofail_reconfirmation_v2/corpus_ext4"
)
EVAL_ROOT = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2"


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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _inventory(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = 0
    bytes_total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest.update(
            f"{relative}\0{size}\0{_sha256(path)}\n".encode("utf-8")
        )
        files += 1
        bytes_total += size
    return {
        "file_count": files,
        "regular_file_bytes": bytes_total,
        "relative_path_size_content_inventory_sha256": digest.hexdigest(),
    }


def main() -> int:
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    for path in (PREDECESSOR, SCENE03_RECEIPT):
        if not path.is_file():
            raise FileNotFoundError(path)
    if _json(SCENE03_RECEIPT).get("state") != "completed":
        raise RuntimeError("scene03 is not sealed and pruned")
    if SCENE04_CORPUS.exists():
        raise RuntimeError("scene04 acquisition started before F17 seal")
    predictions = sorted(
        str(path.relative_to(ROOT))
        for path in EVAL_ROOT.glob("**/*prediction*")
    )
    if predictions:
        raise RuntimeError("prediction artifacts exist before F17 seal")
    if (
        not SCRATCH_LINK.is_symlink()
        or SCRATCH_LINK.resolve() != SCRATCH_DESTINATION.resolve()
        or not SCRATCH_BACKUP.is_dir()
        or not SCRATCH_DESTINATION.is_dir()
    ):
        raise RuntimeError("F17 scratch migration is not ready for sealing")

    source_inventory = _inventory(SCRATCH_BACKUP)
    destination_inventory = _inventory(SCRATCH_DESTINATION)
    if source_inventory != destination_inventory:
        raise RuntimeError("scratch source and destination inventories differ")

    paths = {
        "scene_pipeline_v3": (
            ROOT / "scripts/run_kinofail_reconfirmation_scene_pipeline_v3.py"
        ),
        "scene_pipeline_v2": (
            ROOT / "scripts/run_kinofail_reconfirmation_scene_pipeline_v2.py"
        ),
        "pair_runner_v2": (
            ROOT / "scripts/run_kinofail_reconfirmation_pair_partition_v2.py"
        ),
        "t2_runner_v2": (
            ROOT / "scripts/run_kinofail_reconfirmation_t2_scene_v2.py"
        ),
        "slot_pool_v1": (
            ROOT / "scripts/kinofail_reconfirmation_slot_pool_v1.py"
        ),
        "slotted_partition_v1": (
            ROOT
            / "scripts/run_kinofail_reconfirmation_slotted_partition_v1.py"
        ),
        "slotted_t2_v1": (
            ROOT / "scripts/run_kinofail_reconfirmation_slotted_t2_v1.py"
        ),
        "all_scenes_v3": (
            ROOT
            / "scripts/run_kinofail_reconfirmation_all_scene_pipelines_v3.py"
        ),
        "finalizer_v3": (
            ROOT / "scripts/finalize_kinofail_reconfirmation_v3.py"
        ),
        "finalizer_v2": (
            ROOT / "scripts/finalize_kinofail_reconfirmation_v2.py"
        ),
        "sealer_v1": Path(__file__).resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    correction = {
        f"{name}_sha256": _sha256(path) for name, path in paths.items()
    }
    correction.update(
        {
            "maximum_concurrent_isaac_processes_before": 3,
            "maximum_concurrent_isaac_processes_after": 3,
            "logical_partition_count_before": 4,
            "logical_partition_count_after": 4,
            "fresh_isaac_process_per_pair": True,
            "scale_t3_stage_barrier_preserved": True,
            "within_partition_pair_order_preserved": True,
            "partition_membership_preserved": True,
            "scratch_symlink_required": True,
            "scratch_source": str(SCRATCH_BACKUP),
            "scratch_destination": str(SCRATCH_DESTINATION),
            "scratch_filesystem": "ext4",
            "source_inventory": source_inventory,
            "destination_inventory": destination_inventory,
        }
    )
    manifest = {
        "schema_version": (
            "kinofail.reconfirmation-f17-work-conserving-storage-amendment.v1"
        ),
        "status": "sealed_after_scene03_before_scene04_acquisition",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "predecessor_f16": str(PREDECESSOR.relative_to(ROOT)),
        "predecessor_f16_sha256": _sha256(PREDECESSOR),
        "activation": {
            "last_scene_under_predecessor_pipeline": (
                "confirm_v2_production_scene_03"
            ),
            "first_scene_under_f17_pipeline": (
                "confirm_v2_production_scene_04"
            ),
            "scene03_receipt": str(SCENE03_RECEIPT),
            "scene03_receipt_sha256": _sha256(SCENE03_RECEIPT),
            "scene04_physical_files_existed_at_seal": False,
        },
        "evidence": {
            "identified_from": (
                "process liveness, GPU/CPU/memory/I/O utilization, launcher "
                "timestamps, and orchestration occupancy only"
            ),
            "model_feature_value_label_outcome_prediction_or_score_used": False,
            "prediction_artifacts_existed_at_seal": False,
            "completed_production_scene_logs_used_for_timing_only": [
                "confirm_v2_production_scene_01",
                "confirm_v2_production_scene_02",
            ],
            "pytest_command": (
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q "
                "tests/test_reconfirmation_f17_work_conserving.py "
                "tests/test_reconfirmation_f13_launcher_eligibility.py "
                "tests/test_reconfirmation_f15_partition_supersession.py"
            ),
            "pytest_result": "10 passed",
        },
        "correction": correction,
        "scientific_contract": {
            "schedule_or_protocol_changed": False,
            "pair_scene_material_seed_or_operator_changed": False,
            "simulation_or_sensor_logic_changed": False,
            "rgb_pixels_or_proprio_values_changed": False,
            "feature_model_route_threshold_or_analysis_changed": False,
            "result_dependent_retry_enabled": False,
            "existing_observation_or_summary_modified": False,
        },
        "scope": (
            "work-conserving scheduling of the four original logical "
            "partitions through the already-sealed maximum of three fresh "
            "Isaac processes, plus content-identical ext4 scratch relocation"
        ),
    }
    _write_json(MANIFEST, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
