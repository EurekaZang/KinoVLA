#!/usr/bin/env python3
"""Work-conserving successor for one sealed reconfirmation scene.

Scientific collection remains delegated to the exact F12--F16 launchers and
collectors.  Four logical partitions are started together, while a local
broker admits at most three fresh Isaac child processes at any instant.
Scale/T3 stage boundaries, pair membership, within-partition order, failure
terminality, post-processing, and pruning remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_kinofail_reconfirmation_scene_pipeline_v2 as base
from scripts.kinofail_reconfirmation_slot_pool_v1 import start_slot_broker


AMENDMENT = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f17_work_conserving_storage_amendment1/"
    "amendment_manifest.json"
)
SLOTTED_PARTITION = (
    ROOT / "scripts/run_kinofail_reconfirmation_slotted_partition_v1.py"
)
SLOTTED_T2 = ROOT / "scripts/run_kinofail_reconfirmation_slotted_t2_v1.py"
MAX_CONCURRENT_ISAAC_PROCESSES = 3
LOGICAL_PARTITION_COUNT = 4


def _validate_f17() -> dict[str, Any]:
    amendment = base._json(AMENDMENT)
    correction = amendment.get("correction", {})
    scientific = amendment.get("scientific_contract", {})
    predecessor = Path(str(amendment.get("predecessor_f16", "")))
    if not predecessor.is_absolute():
        predecessor = ROOT / predecessor
    paths = {
        "scene_pipeline_v3": Path(__file__).resolve(),
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
        "slotted_partition_v1": SLOTTED_PARTITION,
        "slotted_t2_v1": SLOTTED_T2,
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
        "sealer_v1": (
            ROOT / "scripts/seal_kinofail_reconfirmation_f17.py"
        ),
    }
    if (
        amendment.get("schema_version")
        != "kinofail.reconfirmation-f17-work-conserving-storage-amendment.v1"
        or amendment.get("status")
        != "sealed_after_scene03_before_scene04_acquisition"
        or amendment.get("passed") is not True
        or not predecessor.is_file()
        or amendment.get("predecessor_f16_sha256")
        != base._sha256(predecessor)
        or correction.get("maximum_concurrent_isaac_processes_before") != 3
        or correction.get("maximum_concurrent_isaac_processes_after") != 3
        or correction.get("logical_partition_count_before") != 4
        or correction.get("logical_partition_count_after") != 4
        or correction.get("fresh_isaac_process_per_pair") is not True
        or correction.get("scale_t3_stage_barrier_preserved") is not True
        or correction.get("scratch_symlink_required") is not True
        or Path(base.SCRATCH_CORPUS_ROOT).resolve()
        != Path(str(correction.get("scratch_destination", ""))).resolve()
        or any(
            not path.is_file()
            or correction.get(f"{name}_sha256") != base._sha256(path)
            for name, path in paths.items()
        )
        or scientific.get("schedule_or_protocol_changed") is not False
        or scientific.get("pair_scene_material_seed_or_operator_changed")
        is not False
        or scientific.get("simulation_or_sensor_logic_changed") is not False
        or scientific.get("rgb_pixels_or_proprio_values_changed") is not False
        or scientific.get("feature_model_route_threshold_or_analysis_changed")
        is not False
        or scientific.get("result_dependent_retry_enabled") is not False
        or scientific.get("existing_observation_or_summary_modified")
        is not False
    ):
        raise RuntimeError("F17 work-conserving storage amendment is invalid")
    return amendment


def _pair_command(
    *,
    paths: dict[str, Path],
    battery: str,
    partition_index: int,
) -> list[str]:
    prefix = "scale" if battery == "scale" else "t3"
    return [
        sys.executable,
        str(SLOTTED_PARTITION),
        "--schedule",
        str(paths[f"{prefix}_schedule"]),
        "--scene-registry",
        str(base.REGISTRY),
        "--protocol",
        str(paths[f"{prefix}_protocol"]),
        "--asset-lock",
        str(base.ASSET_LOCK),
        "--corpus-root",
        str(paths["physical"]),
        "--operational-amendment",
        str(base.OPERATIONAL_AMENDMENT),
        "--partition-index",
        str(partition_index),
        "--partition-count",
        str(LOGICAL_PARTITION_COUNT),
    ]


def _run_jobs(
    jobs: list[tuple[str, list[str]]],
    *,
    log_dir: Path,
    environment: dict[str, str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(
                base._run,
                name,
                command,
                log_dir=log_dir,
                environment=environment,
            ): name
            for name, command in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    return results


def _collect(
    scene: str,
    paths: dict[str, Path],
    *,
    log_dir: Path,
) -> list[dict[str, Any]]:
    broker = start_slot_broker(MAX_CONCURRENT_ISAAC_PROCESSES)
    environment = os.environ.copy()
    environment.update(broker.environment)
    try:
        scale_jobs = [
            (
                "t2",
                [
                    sys.executable,
                    str(SLOTTED_T2),
                    "--schedule",
                    str(paths["t2_schedule"]),
                    "--scene-registry",
                    str(base.REGISTRY),
                    "--asset-lock",
                    str(base.ASSET_LOCK),
                    "--protocol",
                    str(paths["t2_protocol"]),
                    "--out",
                    str(paths["physical"] / "c2_t2"),
                    "--scene",
                    scene,
                ],
            )
        ]
        scale_jobs.extend(
            (
                f"scale_p{index}",
                _pair_command(
                    paths=paths,
                    battery="scale",
                    partition_index=index,
                ),
            )
            for index in range(LOGICAL_PARTITION_COUNT)
        )
        results = _run_jobs(
            scale_jobs,
            log_dir=log_dir,
            environment=environment,
        )
        t3_jobs = [
            (
                f"t3_p{index}",
                _pair_command(
                    paths=paths,
                    battery="t3",
                    partition_index=index,
                ),
            )
            for index in range(LOGICAL_PARTITION_COUNT)
        ]
        results.extend(
            _run_jobs(
                t3_jobs,
                log_dir=log_dir,
                environment=environment,
            )
        )
        return results
    finally:
        broker.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--partition-count", type=int, default=4)
    args = parser.parse_args()
    if args.partition_count != LOGICAL_PARTITION_COUNT:
        raise ValueError("F17 preserves the four frozen logical partitions")
    amendment = _validate_f17()
    paths = base._scene_paths(args.scene)
    if not paths["shard"].is_dir():
        raise FileNotFoundError(paths["shard"])
    receipt = base.RECEIPT_ROOT / args.scene / "completed.json"
    if receipt.is_file() and base._json(receipt).get("state") == "completed":
        print(json.dumps({"scene": args.scene, "status": "already_complete"}))
        return 0
    base._prepare_storage(paths)

    log_dir = base.ORCHESTRATION_ROOT / args.scene / "logs"
    state_path = base.ORCHESTRATION_ROOT / args.scene / "pipeline_state.json"
    state: dict[str, Any] = {
        "schema_version": "kinofail.reconfirmation-scene-pipeline.v3",
        "scene_id": args.scene,
        "started_utc": datetime.now(UTC).isoformat(),
        "logical_partition_count": LOGICAL_PARTITION_COUNT,
        "maximum_concurrent_isaac_processes": (
            MAX_CONCURRENT_ISAAC_PROCESSES
        ),
        "scheduler": "work_conserving_process_shared_slot_broker",
        "fresh_isaac_process_per_counterfactual_pair": True,
        "model_checkpoint_loaded": False,
        "operational_amendment": str(AMENDMENT),
        "operational_amendment_sha256": base._sha256(AMENDMENT),
        "launcher_eligibility_filter": amendment.get("scope", ""),
        "stages": {},
    }
    base._write_json(state_path, state)
    try:
        acquisition = _collect(args.scene, paths, log_dir=log_dir)
        state["stages"]["acquisition"] = acquisition
        base._write_json(state_path, state)
        postprocess = base._postprocess(args.scene, paths, log_dir=log_dir)
        state["stages"]["model_blind_postprocess"] = postprocess
        base._write_json(state_path, state)
        capsule = base._build_conflict_capsule(
            args.scene,
            paths,
            log_dir=log_dir,
        )
        state["stages"]["conflict_evidence_capsule"] = capsule
        base._write_json(state_path, state)
        prune = base._prune(args.scene, paths, log_dir=log_dir)
        state["stages"]["prune"] = prune
        state["status"] = "complete"
        state["completed_utc"] = datetime.now(UTC).isoformat()
        base._write_json(state_path, state)
    except BaseException as error:
        state["status"] = "terminal_failure"
        state["failed_utc"] = datetime.now(UTC).isoformat()
        state["error"] = f"{type(error).__name__}: {error}"
        base._write_json(state_path, state)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
