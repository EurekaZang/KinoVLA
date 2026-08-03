#!/usr/bin/env python3
"""F19 successor adding storage-aware model-blind postprocess path handling."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_kinofail_reconfirmation_scene_pipeline_v3 as predecessor


AMENDMENT = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f19_storage_postprocess_amendment1/"
    "amendment_manifest.json"
)
STORAGE_T2 = (
    ROOT
    / "scripts/extract_kinofail_confirmatory_t2_features_storage_v1.py"
)
FROZEN_T2 = (
    ROOT / "scripts/extract_kinofail_confirmatory_t2_features_v1.py"
)
DATA_CORPUS_ROOT = Path(
    "/data/eureka/KinoVLA/outputs/kinofail_reconfirmation_v2/corpus_ext4"
)
_ORIGINAL_RUN = predecessor.base._run


def _validate_f19() -> dict[str, Any]:
    amendment = predecessor.base._json(AMENDMENT)
    correction = amendment.get("correction", {})
    scientific = amendment.get("scientific_contract", {})
    paths = {
        "scene_pipeline_v4": Path(__file__).resolve(),
        "scene_pipeline_v3": Path(predecessor.__file__).resolve(),
        "scene_pipeline_v2": Path(predecessor.base.__file__).resolve(),
        "storage_t2_wrapper_v1": STORAGE_T2,
        "frozen_t2_wrapper_v1": FROZEN_T2,
        "frozen_t2_source_v1": (
            ROOT / "scripts/extract_kinofail_realistic_c1_causal_features_v1.py"
        ),
        "all_scenes_v4": (
            ROOT
            / "scripts/run_kinofail_reconfirmation_all_scene_pipelines_v4.py"
        ),
        "scene04_recovery_v1": (
            ROOT / "scripts/recover_kinofail_reconfirmation_scene04_f19.py"
        ),
        "finalizer_v5": (
            ROOT / "scripts/finalize_kinofail_reconfirmation_v5.py"
        ),
        "sealer_v1": ROOT / "scripts/seal_kinofail_reconfirmation_f19.py",
        "scene04_prestate": (
            ROOT
            / "outputs/kinofail_reconfirmation_v2/orchestration/"
            "confirm_v2_production_scene_04/"
            "pipeline_state.f19_pre_recovery.json"
        ),
    }
    if (
        amendment.get("schema_version")
        != "kinofail.reconfirmation-f19-storage-postprocess-amendment.v1"
        or amendment.get("status")
        != "sealed_after_scene04_failure_before_model_or_recovery"
        or amendment.get("passed") is not True
        or amendment.get("predecessor_f18_sha256")
        != predecessor.base._sha256(
            ROOT
            / "outputs/freeze/"
            "unified_moe_v3_reconfirmation_f18_pruned_audit_cache_amendment1/"
            "amendment_manifest.json"
        )
        or any(
            not path.is_file()
            or correction.get(f"{name}_sha256")
            != predecessor.base._sha256(path)
            for name, path in paths.items()
        )
        or correction.get("logical_repo_root") != str(ROOT)
        or correction.get("physical_repo_root") != "/data/eureka/KinoVLA"
        or correction.get("t2_feature_entrypoint_only_changed") is not True
        or correction.get("pruner_storage_arguments_only_changed") is not True
        or scientific.get("feature_values_or_order_changed") is not False
        or scientific.get("model_route_threshold_or_analysis_changed") is not False
        or scientific.get("collection_simulation_sensor_or_schedule_changed")
        is not False
        or scientific.get("result_dependent_retry_enabled") is not False
    ):
        raise RuntimeError("F19 storage-postprocess amendment is invalid")
    return amendment


def rewrite_storage_command(name: str, command: list[str]) -> list[str]:
    rewritten = list(command)
    if name == "t2_features":
        if (
            Path(rewritten[1]).resolve() != FROZEN_T2.resolve()
            or not STORAGE_T2.is_file()
        ):
            raise RuntimeError("unexpected frozen T2 feature command")
        rewritten[1] = str(STORAGE_T2)
    elif name == "seal_and_prune":
        root_index = rewritten.index("--corpus-root") + 1
        shard_index = rewritten.index("--corpus-shard") + 1
        physical_root = Path(rewritten[root_index]).resolve()
        physical_shard = Path(rewritten[shard_index]).resolve()
        if (
            physical_root != DATA_CORPUS_ROOT
            or physical_shard.parent != DATA_CORPUS_ROOT
        ):
            raise RuntimeError("unexpected scratch mirror for F19 prune")
        rewritten[root_index] = str(physical_root)
        rewritten[shard_index] = str(physical_shard)
    return rewritten


def _storage_aware_run(
    name: str,
    command: list[str],
    *,
    log_dir: Path,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    return _ORIGINAL_RUN(
        name,
        rewrite_storage_command(name, command),
        log_dir=log_dir,
        environment=environment,
    )


def install_storage_overrides() -> None:
    predecessor.base._run = _storage_aware_run


def _augment_state(scene: str, amendment: dict[str, Any]) -> None:
    state_path = predecessor.base.ORCHESTRATION_ROOT / scene / "pipeline_state.json"
    if not state_path.is_file():
        return
    state = predecessor.base._json(state_path)
    state["schema_version"] = "kinofail.reconfirmation-scene-pipeline.v4"
    state["f19_storage_postprocess_amendment"] = {
        "path": str(AMENDMENT),
        "sha256": predecessor.base._sha256(AMENDMENT),
        "status": amendment["status"],
        "t2_feature_entrypoint_only_changed": True,
        "pruner_storage_arguments_only_changed": True,
        "scientific_content_changed": False,
    }
    state["f19_state_wrapped_utc"] = datetime.now(UTC).isoformat()
    predecessor.base._write_json(state_path, state)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scene", required=True)
    known, _ = parser.parse_known_args()
    amendment = _validate_f19()
    install_storage_overrides()
    try:
        return int(predecessor.main())
    finally:
        _augment_state(known.scene, amendment)


if __name__ == "__main__":
    raise SystemExit(main())
