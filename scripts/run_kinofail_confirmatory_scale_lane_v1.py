#!/usr/bin/env python3
"""Collect, derive, seal, and prune a disjoint lane of Scale scene shards.

Two instances cover disjoint scene slices.  Each instance runs the two frozen
pair partitions concurrently, then builds the F0-frozen snapshot/visual/
invariant features and invokes the pre-Scale-sealed F4 retention gate.
No model, checkpoint, prediction, or endpoint statistic is loaded.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
REGISTRY = ROOT / "outputs/kinofail_confirmatory_v1/scene_registry.json"
SCHEDULE_ROOT = (
    ROOT
    / "outputs/kinofail_confirmatory_v1/"
    "schedules_f2_scene_source_amendment"
)
CORPUS_ROOT = ROOT / "outputs/kinofail_confirmatory_v1/corpus"
DERIVED_ROOT = ROOT / "outputs/eval/unified_moe_v3_confirmatory_v1/shards"
T2_FEATURE_ROOT = (
    ROOT
    / "outputs/eval/unified_moe_v3_confirmatory_v1/"
    "conflict/t2_scene_features"
)
CONFLICT_AUDIT = (
    ROOT
    / "outputs/eval/unified_moe_v3_confirmatory_v1/"
    "conflict/postprocess_audit.json"
)
F3_MANIFEST = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_confirmatory_f3_runtime_namespace/"
    "amendment_manifest.json"
)
F4_MANIFEST = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_confirmatory_f4_prune_attrition/"
    "amendment_manifest.json"
)
PRUNE_RECEIPTS = (
    ROOT / "outputs/kinofail_confirmatory_v1/prune_receipts"
)
ORCHESTRATION_ROOT = (
    ROOT / "outputs/kinofail_confirmatory_v1/scale_orchestration"
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


def _wait_for_conflict_postprocess(poll_seconds: int) -> None:
    last_report = 0.0
    while True:
        if CONFLICT_AUDIT.is_file():
            audit = _json(CONFLICT_AUDIT)
            if (
                audit.get("passed") is True
                and audit.get("model_or_checkpoint_loaded") is False
                and audit.get("predictions_generated") is False
            ):
                return
            raise RuntimeError("conflict postprocess audit did not pass")
        now = time.monotonic()
        if now - last_report >= 600:
            print(
                json.dumps({"stage": "waiting_for_conflict_postprocess"}),
                flush=True,
            )
            last_report = now
        time.sleep(poll_seconds)


def _run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    with log_path.open("wb") as stream:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def _tail(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")[-4000:]


def _require_success(command: list[str], log_path: Path) -> None:
    returncode = _run(command, log_path)
    if returncode != 0:
        raise RuntimeError(
            f"stage failed ({returncode}): {log_path}\n{_tail(log_path)}"
        )


def _terminal_partition_audit(
    *, scene: Path, partition: int
) -> dict[str, Any]:
    path = scene / "launcher_audits" / f"partition_{partition}_of_2.json"
    audit = _json(path)
    selected = [str(value) for value in audit.get("selected_pair_ids", [])]
    attempts = [
        str(row["counterfactual_group_id"])
        for row in audit.get("attempts", [])
    ]
    if (
        audit.get("partition_index") != partition
        or len(selected) != 176
        or len(attempts) != 176
        or set(selected) != set(attempts)
    ):
        raise RuntimeError(f"incomplete Scale partition audit: {path}")
    return audit


def _manifest_passed(path: Path) -> bool:
    if not path.is_file():
        return False
    value = _json(path)
    return (
        value.get("passed") is True
        or value.get("status") == "complete"
    )


def _write_lane_audit(
    *,
    path: Path,
    lane_id: str,
    scenes: list[str],
    attempts: list[dict[str, Any]],
) -> None:
    value = {
        "schema_version": "kinofail.confirmatory-scale-lane.v1",
        "updated_utc": datetime.now(UTC).isoformat(),
        "lane_id": lane_id,
        "scenes": scenes,
        "model_or_checkpoint_loaded": False,
        "predictions_generated": False,
        "f3_manifest_sha256": _sha256(F3_MANIFEST),
        "f4_manifest_sha256": _sha256(F4_MANIFEST),
        "attempts": attempts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--start-scene-index", type=int, required=True)
    parser.add_argument("--end-scene-index", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        raise ValueError("poll interval is too small")
    registry = _json(REGISTRY)
    all_scenes = [str(row["scene_id"]) for row in registry["scenes"]]
    if (
        len(all_scenes) != 30
        or len(set(all_scenes)) != 30
        or not 0
        <= args.start_scene_index
        < args.end_scene_index
        <= len(all_scenes)
    ):
        raise RuntimeError("invalid frozen scene-registry lane")
    scenes = all_scenes[args.start_scene_index : args.end_scene_index]
    lane_audit_path = ORCHESTRATION_ROOT / f"{args.lane_id}.json"
    attempts: list[dict[str, Any]] = []
    if lane_audit_path.is_file():
        prior = _json(lane_audit_path)
        if prior.get("lane_id") != args.lane_id or prior.get("scenes") != scenes:
            raise RuntimeError("existing Scale lane audit mismatch")
        attempts = list(prior.get("attempts", []))
    completed_scenes = {str(row["scene_id"]) for row in attempts}

    _wait_for_conflict_postprocess(args.poll_seconds)
    if (
        _json(F3_MANIFEST).get("status") != "sealed"
        or _json(F4_MANIFEST).get("status")
        != "sealed_before_scale_acquisition"
    ):
        raise RuntimeError("F3/F4 operational amendments are not sealed")

    log_root = ORCHESTRATION_ROOT / "logs" / args.lane_id
    for scene_id in scenes:
        if scene_id in completed_scenes:
            continue
        logical_scene = CORPUS_ROOT / scene_id
        receipt = PRUNE_RECEIPTS / scene_id / "completed.json"
        if receipt.is_file():
            record = {
                "scene_id": scene_id,
                "status": "already_sealed_and_pruned",
                "completed_utc": datetime.now(UTC).isoformat(),
            }
            attempts.append(record)
            _write_lane_audit(
                path=lane_audit_path,
                lane_id=args.lane_id,
                scenes=scenes,
                attempts=attempts,
            )
            continue

        shard = SCHEDULE_ROOT / "scenes" / scene_id / "scale"
        collection_commands = []
        for partition in (0, 1):
            collection_commands.append(
                (
                    [
                        str(PYTHON),
                        "scripts/run_kinofail_confirmatory_scale_shard_v1.py",
                        "--schedule",
                        str(shard / "schedule.jsonl"),
                        "--scene-registry",
                        str(REGISTRY),
                        "--asset-lock",
                        str(
                            ROOT
                            / "outputs/assets/terrain_pbr_confirmatory_v1/"
                            "terrain_assets.lock.json"
                        ),
                        "--protocol",
                        str(shard / "collection_protocol.json"),
                        "--corpus-root",
                        str(logical_scene),
                        "--partition-index",
                        str(partition),
                    ],
                    log_root
                    / f"{scene_id}_collect_partition_{partition}.log",
                )
            )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(_run, command, log_path)
                for command, log_path in collection_commands
            ]
            returncodes = [future.result() for future in futures]

        partition_audits = [
            _terminal_partition_audit(
                scene=logical_scene, partition=partition
            )
            for partition in (0, 1)
        ]
        passed_pairs = sum(
            row.get("passed") is True
            for audit in partition_audits
            for row in audit["attempts"]
        )
        failed_pairs = 352 - passed_pairs
        if failed_pairs > int(0.05 * 352):
            raise RuntimeError(
                f"Scale attrition exceeds frozen gate in {scene_id}: "
                f"{failed_pairs}/352"
            )

        derived = DERIVED_ROOT / scene_id / "scale"
        snapshots = derived / "snapshots"
        features = derived / "features"
        unified = derived / "unified_features"
        if not _manifest_passed(snapshots / "extraction_audit.json"):
            _require_success(
                [
                    str(PYTHON),
                    "scripts/build_kinofail_realistic_snapshot_dev.py",
                    "--protocol",
                    str(shard / "snapshot_protocol.json"),
                    "--repo-root",
                    str(ROOT),
                ],
                log_root / f"{scene_id}_snapshots.log",
            )
        if not _manifest_passed(features / "feature_manifest.json"):
            _require_success(
                [
                    str(PYTHON),
                    "scripts/extract_kinofail_realistic_features.py",
                    "--snapshot-dir",
                    str(snapshots),
                    "--output-dir",
                    str(features),
                    "--batch-size",
                    "64",
                ],
                log_root / f"{scene_id}_visual_features.log",
            )
        if not _manifest_passed(unified / "feature_manifest.json"):
            _require_success(
                [
                    str(PYTHON),
                    "scripts/build_kinofail_unified_invariant_features_v1.py",
                    "--snapshot-dir",
                    str(snapshots),
                    "--visual-feature-dir",
                    str(features),
                    "--output-dir",
                    str(unified),
                ],
                log_root / f"{scene_id}_unified_features.log",
            )

        physical_scene = logical_scene.resolve()
        physical_root = physical_scene.parent
        prune_command = [
            str(PYTHON),
            "scripts/seal_and_prune_kinofail_confirmatory_shard_f4.py",
            "--scene-id",
            scene_id,
            "--schedule-root",
            str(SCHEDULE_ROOT),
            "--corpus-root",
            str(physical_root),
            "--corpus-shard",
            str(physical_scene),
            "--derived-root",
            str(DERIVED_ROOT),
            "--t2-feature-manifest",
            str(
                T2_FEATURE_ROOT / scene_id / "feature_manifest.json"
            ),
            "--receipt-root",
            str(PRUNE_RECEIPTS),
        ]
        _require_success(
            prune_command,
            log_root / f"{scene_id}_prune_preflight.log",
        )
        _require_success(
            prune_command
            + [
                "--execute",
                "--prune-snapshot-arrays",
            ],
            log_root / f"{scene_id}_prune_execute.log",
        )
        completed_receipt = _json(receipt)
        if (
            completed_receipt.get("state") != "completed"
            or completed_receipt.get("corpus_shard_removed") is not True
        ):
            raise RuntimeError(f"Scale shard was not pruned: {scene_id}")
        record = {
            "scene_id": scene_id,
            "status": "complete_sealed_pruned",
            "collection_returncodes": returncodes,
            "passed_pairs": passed_pairs,
            "failed_pairs": failed_pairs,
            "snapshot_records": _json(
                snapshots / "extraction_audit.json"
            )["counts"]["snapshot_records"],
            "prune_receipt_sha256": _sha256(receipt),
            "completed_utc": datetime.now(UTC).isoformat(),
            "retry_authorized": False,
        }
        attempts.append(record)
        completed_scenes.add(scene_id)
        _write_lane_audit(
            path=lane_audit_path,
            lane_id=args.lane_id,
            scenes=scenes,
            attempts=attempts,
        )
        print(json.dumps(record, sort_keys=True), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
