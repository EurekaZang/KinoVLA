#!/usr/bin/env python3
"""Execute one sealed reconfirmation scene through acquisition and pruning.

At most three Isaac Sim processes run concurrently.  Formal acquisition
launchers retain every terminal failure and prohibit retries.  Model-blind
feature extraction follows acquisition; no model checkpoint is loaded here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEDULE_ROOT = ROOT / "outputs/kinofail_reconfirmation_v2/schedules"
REGISTRY = ROOT / "outputs/kinofail_reconfirmation_v2/scene_registry.json"
ASSET_LOCK = (
    ROOT
    / "outputs/assets/terrain_pbr_confirmatory_v2/terrain_assets.lock.json"
)
LOGICAL_CORPUS_ROOT = ROOT / "outputs/kinofail_reconfirmation_v2/corpus"
SCRATCH_CORPUS_ROOT = Path(
    ROOT / "outputs/kinofail_reconfirmation_v2/corpus_ext4"
)
CONFLICT_CAPSULE_ROOT = (
    ROOT / "outputs/kinofail_reconfirmation_v2/conflict_capsules_ext4"
)
OPERATIONAL_AMENDMENT = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f12_process_local_rtx_texture_amendment1/"
    "amendment_manifest.json"
)
F13_OPERATIONAL_AMENDMENT = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f13_launcher_eligibility_amendment1/"
    "amendment_manifest.json"
)
DERIVED_ROOT = (
    ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2/shards"
)
RECEIPT_ROOT = (
    ROOT / "outputs/kinofail_reconfirmation_v2/prune_receipts"
)
ORCHESTRATION_ROOT = (
    ROOT / "outputs/kinofail_reconfirmation_v2/orchestration"
)
MAX_CONCURRENT_ISAAC_PROCESSES = 3


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


def _validate_f13() -> dict[str, Any]:
    amendment = _json(F13_OPERATIONAL_AMENDMENT)
    correction = amendment.get("correction", {})
    if (
        amendment.get("schema_version")
        != "kinofail.reconfirmation-f13-launcher-eligibility-amendment.v1"
        or amendment.get("status")
        != "sealed_before_resumption_of_scene_01_post_acquisition"
        or amendment.get("passed") is not True
        or correction.get("scene_pipeline_sha256")
        != _sha256(Path(__file__).resolve())
        or correction.get("snapshot_entrypoint_sha256")
        != _sha256(
            ROOT / "scripts/build_kinofail_realistic_snapshot_dev.py"
        )
        or correction.get("capsule_builder_sha256")
        != _sha256(
            ROOT
            / "scripts/build_kinofail_reconfirmation_conflict_capsule_v1.py"
        )
        or correction.get("f13_pruner_sha256")
        != _sha256(
            ROOT
            / "scripts/seal_and_prune_kinofail_confirmatory_shard_f13.py"
        )
    ):
        raise RuntimeError("F13 launcher eligibility amendment is invalid")
    return amendment


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run(
    name: str,
    command: list[str],
    *,
    log_dir: Path,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    log_path = log_dir / f"{name}.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat()
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "name": name,
        "command": command,
        "started_utc": started,
        "completed_utc": datetime.now(UTC).isoformat(),
        "returncode": int(completed.returncode),
        "log": str(log_path),
    }


def _scene_paths(scene: str) -> dict[str, Path]:
    shard = SCHEDULE_ROOT / "scenes" / scene
    return {
        "shard": shard,
        "physical": SCRATCH_CORPUS_ROOT / scene,
        "logical": LOGICAL_CORPUS_ROOT / scene,
        "derived": DERIVED_ROOT / scene,
        "scale_schedule": shard / "scale/schedule.jsonl",
        "scale_protocol": shard / "scale/collection_protocol.json",
        "scale_snapshot_protocol": shard / "scale/snapshot_protocol.json",
        "t3_schedule": shard / "c2_t3/schedule.jsonl",
        "t3_protocol": shard / "c2_t3/collection_protocol.json",
        "t3_snapshot_protocol": shard / "c2_t3/snapshot_protocol.json",
        "t2_schedule": SCHEDULE_ROOT / "c2_t2/schedule.jsonl",
        "t2_protocol": SCHEDULE_ROOT / "c2_t2/collection_protocol.json",
    }


def _prepare_storage(paths: dict[str, Path]) -> None:
    physical = paths["physical"]
    logical = paths["logical"]
    physical.mkdir(parents=True, exist_ok=True)
    LOGICAL_CORPUS_ROOT.mkdir(parents=True, exist_ok=True)
    if logical.is_symlink():
        if logical.resolve() != physical.resolve():
            raise RuntimeError("logical corpus link points to another scene shard")
    elif logical.exists():
        raise RuntimeError("logical corpus shard must be a bounded symlink")
    else:
        logical.symlink_to(physical, target_is_directory=True)


def _pair_launch_command(
    *,
    paths: dict[str, Path],
    battery: str,
    partition_index: int,
    partition_count: int,
) -> list[str]:
    prefix = "scale" if battery == "scale" else "t3"
    return [
        sys.executable,
        str(ROOT / "scripts/run_kinofail_reconfirmation_pair_partition_v2.py"),
        "--schedule",
        str(paths[f"{prefix}_schedule"]),
        "--scene-registry",
        str(REGISTRY),
        "--protocol",
        str(paths[f"{prefix}_protocol"]),
        "--asset-lock",
        str(ASSET_LOCK),
        "--corpus-root",
        str(paths["physical"]),
        "--operational-amendment",
        str(OPERATIONAL_AMENDMENT),
        "--partition-index",
        str(partition_index),
        "--partition-count",
        str(partition_count),
    ]


def _collect(
    scene: str,
    paths: dict[str, Path],
    *,
    log_dir: Path,
    partition_count: int,
) -> list[dict[str, Any]]:
    jobs = [
        (
            "t2",
            [
                sys.executable,
                str(ROOT / "scripts/run_kinofail_reconfirmation_t2_scene_v2.py"),
                "--schedule",
                str(paths["t2_schedule"]),
                "--scene-registry",
                str(REGISTRY),
                "--asset-lock",
                str(ASSET_LOCK),
                "--protocol",
                str(paths["t2_protocol"]),
                "--out",
                str(paths["physical"] / "c2_t2"),
                "--scene",
                scene,
            ],
        )
    ]
    jobs.extend(
        (
            f"scale_p{index}",
            _pair_launch_command(
                paths=paths,
                battery="scale",
                partition_index=index,
                partition_count=partition_count,
            ),
        )
        for index in range(partition_count)
    )
    results = []
    with ThreadPoolExecutor(
        max_workers=MAX_CONCURRENT_ISAAC_PROCESSES
    ) as executor:
        futures = {
            executor.submit(_run, name, command, log_dir=log_dir): name
            for name, command in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    t3_jobs = [
        (
            f"t3_p{index}",
            _pair_launch_command(
                paths=paths,
                battery="t3",
                partition_index=index,
                partition_count=partition_count,
            ),
        )
        for index in range(partition_count)
    ]
    with ThreadPoolExecutor(
        max_workers=MAX_CONCURRENT_ISAAC_PROCESSES
    ) as executor:
        futures = {
            executor.submit(_run, name, command, log_dir=log_dir): name
            for name, command in t3_jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    return results


def _postprocess(
    scene: str,
    paths: dict[str, Path],
    *,
    log_dir: Path,
) -> list[dict[str, Any]]:
    derived = paths["derived"]
    commands = [
        (
            "scale_snapshots",
            [
                sys.executable,
                str(ROOT / "scripts/build_kinofail_realistic_snapshot_dev.py"),
                "--protocol",
                str(paths["scale_snapshot_protocol"]),
                "--repo-root",
                str(ROOT),
            ],
        ),
        (
            "scale_visual_features",
            [
                sys.executable,
                str(ROOT / "scripts/extract_kinofail_realistic_features.py"),
                "--snapshot-dir",
                str(derived / "scale/snapshots"),
                "--output-dir",
                str(derived / "scale/features"),
                "--batch-size",
                "64",
            ],
        ),
        (
            "scale_unified_features",
            [
                sys.executable,
                str(
                    ROOT
                    / "scripts/build_kinofail_unified_invariant_features_v1.py"
                ),
                "--snapshot-dir",
                str(derived / "scale/snapshots"),
                "--visual-feature-dir",
                str(derived / "scale/features"),
                "--output-dir",
                str(derived / "scale/unified_features"),
            ],
        ),
        (
            "t3_snapshots",
            [
                sys.executable,
                str(ROOT / "scripts/build_kinofail_realistic_snapshot_dev.py"),
                "--protocol",
                str(paths["t3_snapshot_protocol"]),
                "--repo-root",
                str(ROOT),
            ],
        ),
        (
            "t3_visual_features",
            [
                sys.executable,
                str(ROOT / "scripts/extract_kinofail_realistic_features.py"),
                "--snapshot-dir",
                str(derived / "c2_t3/snapshots"),
                "--output-dir",
                str(derived / "c2_t3/features"),
                "--batch-size",
                "64",
            ],
        ),
        (
            "t3_unified_features",
            [
                sys.executable,
                str(
                    ROOT
                    / "scripts/build_kinofail_unified_invariant_features_v1.py"
                ),
                "--snapshot-dir",
                str(derived / "c2_t3/snapshots"),
                "--visual-feature-dir",
                str(derived / "c2_t3/features"),
                "--output-dir",
                str(derived / "c2_t3/unified_features"),
            ],
        ),
        (
            "t2_features",
            [
                sys.executable,
                str(
                    ROOT
                    / "scripts/extract_kinofail_confirmatory_t2_features_v1.py"
                ),
                "--corpus",
                str(paths["physical"] / "c2_t2"),
                "--output",
                str(derived / "c2_t2/features"),
                "--scene-id",
                scene,
                "--batch-size",
                "64",
            ],
        ),
    ]
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    results = []
    for name, command in commands:
        result = _run(
            name,
            command,
            log_dir=log_dir,
            environment=environment,
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        if result["returncode"] != 0:
            raise RuntimeError(f"model-blind postprocess failed: {name}")
    return results


def _prune(
    scene: str,
    paths: dict[str, Path],
    *,
    log_dir: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(
            ROOT
            / "scripts/seal_and_prune_kinofail_confirmatory_shard_f13.py"
        ),
        "--scene-id",
        scene,
        "--schedule-root",
        str(SCHEDULE_ROOT),
        "--corpus-root",
        str(SCRATCH_CORPUS_ROOT),
        "--corpus-shard",
        str(paths["physical"]),
        "--derived-root",
        str(DERIVED_ROOT),
        "--t2-feature-manifest",
        str(paths["derived"] / "c2_t2/features/feature_manifest.json"),
        "--receipt-root",
        str(RECEIPT_ROOT),
        "--execute",
        "--prune-snapshot-arrays",
    ]
    result = _run("seal_and_prune", command, log_dir=log_dir)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["returncode"] != 0:
        raise RuntimeError("scene seal-and-prune failed")
    return result


def _build_conflict_capsule(
    scene: str,
    paths: dict[str, Path],
    *,
    log_dir: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(
            ROOT
            / "scripts/build_kinofail_reconfirmation_conflict_capsule_v1.py"
        ),
        "--scene-id",
        scene,
        "--corpus-scene",
        str(paths["physical"]),
        "--schedule-shard",
        str(paths["shard"]),
        "--t2-feature-dir",
        str(paths["derived"] / "c2_t2/features"),
        "--output-root",
        str(CONFLICT_CAPSULE_ROOT),
    ]
    result = _run("conflict_evidence_capsule", command, log_dir=log_dir)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["returncode"] != 0:
        raise RuntimeError("conflict evidence capsule failed")
    manifest_path = CONFLICT_CAPSULE_ROOT / scene / "capsule_manifest.json"
    manifest = _json(manifest_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("passed") is not True
        or manifest.get("scene_id") != scene
        or manifest.get("model_or_prediction_loaded") is not False
    ):
        raise RuntimeError("conflict evidence capsule did not seal")
    return {
        **result,
        "manifest": str(manifest_path),
        "counts": manifest["counts"],
        "t2": manifest["t2"],
        "t3": manifest["t3"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--partition-count", type=int, default=4)
    args = parser.parse_args()
    if args.partition_count != 4:
        raise ValueError("the deterministic schedule uses four partitions")
    f13 = _validate_f13()

    paths = _scene_paths(args.scene)
    if not paths["shard"].is_dir():
        raise FileNotFoundError(paths["shard"])
    receipt = RECEIPT_ROOT / args.scene / "completed.json"
    if receipt.is_file() and _json(receipt).get("state") == "completed":
        print(json.dumps({"scene": args.scene, "status": "already_complete"}))
        return 0
    _prepare_storage(paths)

    log_dir = ORCHESTRATION_ROOT / args.scene / "logs"
    state_path = ORCHESTRATION_ROOT / args.scene / "pipeline_state.json"
    state = {
        "schema_version": "kinofail.reconfirmation-scene-pipeline.v2",
        "scene_id": args.scene,
        "started_utc": datetime.now(UTC).isoformat(),
        "partition_count": args.partition_count,
        "maximum_concurrent_isaac_processes": (
            MAX_CONCURRENT_ISAAC_PROCESSES
        ),
        "model_checkpoint_loaded": False,
        "operational_amendment": str(F13_OPERATIONAL_AMENDMENT),
        "operational_amendment_sha256": _sha256(
            F13_OPERATIONAL_AMENDMENT
        ),
        "launcher_eligibility_filter": f13.get("scope", ""),
        "stages": {},
    }
    _write_json(state_path, state)
    try:
        acquisition = _collect(
            args.scene,
            paths,
            log_dir=log_dir,
            partition_count=args.partition_count,
        )
        state["stages"]["acquisition"] = acquisition
        _write_json(state_path, state)
        postprocess = _postprocess(args.scene, paths, log_dir=log_dir)
        state["stages"]["model_blind_postprocess"] = postprocess
        _write_json(state_path, state)
        capsule = _build_conflict_capsule(
            args.scene,
            paths,
            log_dir=log_dir,
        )
        state["stages"]["conflict_evidence_capsule"] = capsule
        _write_json(state_path, state)
        prune = _prune(args.scene, paths, log_dir=log_dir)
        state["stages"]["prune"] = prune
        state["status"] = "complete"
        state["completed_utc"] = datetime.now(UTC).isoformat()
        _write_json(state_path, state)
    except BaseException as error:
        state["status"] = "terminal_failure"
        state["failed_utc"] = datetime.now(UTC).isoformat()
        state["error"] = f"{type(error).__name__}: {error}"
        _write_json(state_path, state)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
