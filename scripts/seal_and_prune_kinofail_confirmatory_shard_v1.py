#!/usr/bin/env python3
"""Seal verified confirmatory features, then prune exactly one raw scene shard.

Deletion is opt-in (``--execute``), restricted to a direct child of the frozen
confirmatory corpus root, and allowed only after both Scale and C2-T3 snapshot,
visual-feature, and unified-feature hash chains pass.  A T2 feature manifest is
also mandatory because the shared-prefix arm is not represented by the generic
snapshot bundles.  The script writes a complete raw-file hash inventory and a
prepared receipt before deletion, followed by a completion receipt afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEDULE_ROOT = ROOT / "outputs/kinofail_confirmatory_v1/schedules"
DEFAULT_CORPUS_ROOT = ROOT / "outputs/kinofail_confirmatory_v1/corpus"
DEFAULT_DERIVED_ROOT = (
    ROOT / "outputs/eval/unified_moe_v3_confirmatory_v1/shards"
)
DEFAULT_RECEIPT_ROOT = (
    ROOT / "outputs/kinofail_confirmatory_v1/prune_receipts"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _assert_no_symlinks(path: Path, *, include_descendants: bool) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise RuntimeError(f"symlink is forbidden in prune target chain: {current}")
        if current.parent == current:
            break
        current = current.parent
    if include_descendants:
        for descendant in path.rglob("*"):
            if descendant.is_symlink():
                raise RuntimeError(
                    f"symlink is forbidden inside prune target: {descendant}"
                )


def _validate_target(
    *, corpus_root: Path, corpus_shard: Path, scene_id: str
) -> None:
    if not corpus_root.is_dir():
        raise FileNotFoundError(corpus_root)
    if not corpus_shard.is_dir():
        raise FileNotFoundError(corpus_shard)
    _assert_no_symlinks(corpus_root, include_descendants=False)
    _assert_no_symlinks(corpus_shard, include_descendants=True)
    if corpus_shard.parent.resolve() != corpus_root.resolve():
        raise ValueError("corpus shard must be a direct child of the allowed root")
    if corpus_shard.name != scene_id:
        raise ValueError("corpus shard basename must equal the frozen scene ID")


def _validate_schedule_shard(
    *, schedule_root: Path, scene_id: str
) -> tuple[dict[str, Any], Path]:
    shard_root = schedule_root / "scenes" / scene_id
    manifest_path = shard_root / "shard_manifest.json"
    manifest = _json(manifest_path)
    if (
        manifest.get("schema_version")
        != "kinofail.unified-confirmatory-shard-manifest.v1"
        or manifest.get("passed") is not True
        or manifest.get("scene_id") != scene_id
    ):
        raise RuntimeError("invalid confirmatory schedule shard manifest")
    standard = {
        "scale_schedule": shard_root / "scale/schedule.jsonl",
        "scale_collection_protocol": shard_root
        / "scale/collection_protocol.json",
        "scale_snapshot_protocol": shard_root / "scale/snapshot_protocol.json",
        "t2_case_schedule": shard_root / "c2_t2/case_schedule.jsonl",
        "t3_schedule": shard_root / "c2_t3/schedule.jsonl",
        "t3_case_schedule": shard_root / "c2_t3/case_schedule.jsonl",
        "t3_collection_protocol": shard_root
        / "c2_t3/collection_protocol.json",
        "t3_snapshot_protocol": shard_root / "c2_t3/snapshot_protocol.json",
        "conflict_schedule": shard_root / "conflict_schedule.jsonl",
    }
    for key, path in standard.items():
        if not path.is_file() or manifest["artifacts"].get(key) != _sha256(path):
            raise RuntimeError(f"schedule shard artifact hash mismatch: {key}")
    return manifest, manifest_path


def _validate_bundle(
    *,
    scene_id: str,
    snapshot_dir: Path,
    feature_dir: Path,
    unified_dir: Path,
    expected_pairs: int,
) -> dict[str, Any]:
    records_path = snapshot_dir / "snapshot_records.jsonl"
    snapshots_path = snapshot_dir / "snapshots.npz"
    audit_path = snapshot_dir / "extraction_audit.json"
    visual_path = feature_dir / "features.npz"
    visual_manifest_path = feature_dir / "feature_manifest.json"
    unified_path = unified_dir / "features.npz"
    unified_manifest_path = unified_dir / "feature_manifest.json"
    for path in (
        records_path,
        snapshots_path,
        audit_path,
        visual_path,
        visual_manifest_path,
        unified_path,
        unified_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.is_symlink():
            raise RuntimeError(f"derived artifact may not be a symlink: {path}")
    audit = _json(audit_path)
    visual_manifest = _json(visual_manifest_path)
    unified_manifest = _json(unified_manifest_path)
    if (
        audit.get("passed") is not True
        or audit.get("checks", {}).get(
            "all_selected_or_excluded_pairs_accounted_for"
        )
        is not True
        or int(audit.get("skipped_incomplete_pairs", -1)) != 0
    ):
        raise RuntimeError("snapshot bundle is incomplete or failed")
    accounted_pairs = len(audit.get("pair_audits", [])) + len(
        audit.get("temporal_alignment_exclusions", [])
    )
    if accounted_pairs != expected_pairs:
        raise RuntimeError(
            f"snapshot bundle accounts for {accounted_pairs}, expected {expected_pairs}"
        )
    if (
        visual_manifest.get("status") != "complete"
        or visual_manifest.get("output_sha256", {}).get("features")
        != _sha256(visual_path)
        or visual_manifest.get("source_sha256", {}).get("snapshot_records")
        != _sha256(records_path)
        or visual_manifest.get("source_sha256", {}).get("snapshots")
        != _sha256(snapshots_path)
        or visual_manifest.get("source_sha256", {}).get("extraction_audit")
        != _sha256(audit_path)
    ):
        raise RuntimeError("visual feature hash chain failed")
    if (
        unified_manifest.get("status") != "complete"
        or unified_manifest.get("output_sha256", {}).get("features")
        != _sha256(unified_path)
        or unified_manifest.get("source_sha256", {}).get("snapshot_records")
        != _sha256(records_path)
        or unified_manifest.get("source_sha256", {}).get("snapshots")
        != _sha256(snapshots_path)
        or unified_manifest.get("source_sha256", {}).get("snapshot_audit")
        != _sha256(audit_path)
        or unified_manifest.get("source_sha256", {}).get("visual_features")
        != _sha256(visual_path)
        or unified_manifest.get("source_sha256", {}).get("visual_manifest")
        != _sha256(visual_manifest_path)
    ):
        raise RuntimeError("unified feature hash chain failed")
    records = _records(records_path)
    observed_scenes = {
        str(
            row.get("scene_id")
            or row.get("scene_family")
            or row.get("scene_cluster")
        )
        for row in records
    }
    if observed_scenes != {scene_id}:
        raise RuntimeError("snapshot records are not restricted to the scene shard")
    expected_ids = np.asarray(
        [str(row["sample_id"]) for row in records], dtype=str
    )
    with np.load(visual_path, allow_pickle=False) as archive:
        visual_ids = archive["sample_ids"].astype(str)
    with np.load(unified_path, allow_pickle=False) as archive:
        unified_ids = archive["sample_ids"].astype(str)
    if (
        not np.array_equal(expected_ids, visual_ids)
        or not np.array_equal(expected_ids, unified_ids)
    ):
        raise RuntimeError("derived sample IDs are misaligned")
    return {
        "snapshot_records_sha256": _sha256(records_path),
        "snapshots_sha256": _sha256(snapshots_path),
        "snapshot_audit_sha256": _sha256(audit_path),
        "visual_features_sha256": _sha256(visual_path),
        "visual_manifest_sha256": _sha256(visual_manifest_path),
        "unified_features_sha256": _sha256(unified_path),
        "unified_manifest_sha256": _sha256(unified_manifest_path),
        "samples": len(records),
        "accounted_pairs": accounted_pairs,
    }


def _validate_t2_manifest(path: Path, *, scene_id: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    manifest = _json(path)
    if manifest.get("status") != "complete" and manifest.get("passed") is not True:
        raise RuntimeError("T2 feature manifest is not complete")
    scene_value = (
        manifest.get("scene_id")
        or manifest.get("scene_cluster")
        or manifest.get("scope", {}).get("scene_id")
    )
    if scene_value != scene_id:
        raise RuntimeError("T2 feature manifest is not bound to this scene")
    output_hashes = manifest.get("output_sha256", {})
    if not isinstance(output_hashes, dict) or not output_hashes:
        raise RuntimeError("T2 feature manifest has no output hashes")
    verified = {}
    for key, expected in output_hashes.items():
        candidates = [
            path.parent / str(key),
            path.parent / f"{key}.npz",
            path.parent / f"{key}.jsonl",
        ]
        artifact = next((value for value in candidates if value.is_file()), None)
        if artifact is None or _sha256(artifact) != str(expected):
            raise RuntimeError(f"T2 output hash mismatch: {key}")
        verified[str(key)] = {
            "path": str(artifact),
            "sha256": str(expected),
        }
    return {
        "manifest_sha256": _sha256(path),
        "verified_outputs": verified,
    }


def _inventory(root: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    total = 0
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = str(path.relative_to(root))
        size = path.stat().st_size
        total += size
        rows.append(
            {
                "path": relative,
                "bytes": size,
                "sha256": _sha256(path),
            }
        )
    if not rows:
        raise RuntimeError("raw shard is empty")
    return rows, total


def seal_and_prune(
    *,
    scene_id: str,
    schedule_root: Path,
    corpus_root: Path,
    corpus_shard: Path,
    scale_snapshot_dir: Path,
    scale_feature_dir: Path,
    scale_unified_dir: Path,
    t3_snapshot_dir: Path,
    t3_feature_dir: Path,
    t3_unified_dir: Path,
    t2_feature_manifest: Path,
    receipt_root: Path,
    execute: bool,
    prune_snapshot_arrays: bool,
) -> dict[str, Any]:
    """Validate one scene and optionally perform the exact bounded deletion."""

    schedule_root = schedule_root.resolve()
    # Preserve the lexical target until the symlink checks have completed.
    corpus_root = corpus_root.absolute()
    corpus_shard = corpus_shard.absolute()
    receipt_root = receipt_root.absolute()
    _validate_target(
        corpus_root=corpus_root, corpus_shard=corpus_shard, scene_id=scene_id
    )
    shard_manifest, shard_manifest_path = _validate_schedule_shard(
        schedule_root=schedule_root, scene_id=scene_id
    )
    scale = _validate_bundle(
        scene_id=scene_id,
        snapshot_dir=scale_snapshot_dir.resolve(),
        feature_dir=scale_feature_dir.resolve(),
        unified_dir=scale_unified_dir.resolve(),
        expected_pairs=int(shard_manifest["counts"]["scale_pairs"]),
    )
    t3 = _validate_bundle(
        scene_id=scene_id,
        snapshot_dir=t3_snapshot_dir.resolve(),
        feature_dir=t3_feature_dir.resolve(),
        unified_dir=t3_unified_dir.resolve(),
        expected_pairs=int(shard_manifest["counts"]["t3_pairs"]),
    )
    t2 = _validate_t2_manifest(
        t2_feature_manifest.resolve(), scene_id=scene_id
    )
    raw_inventory, raw_bytes = _inventory(corpus_shard)
    audit = {
        "schema_version": "kinofail.confirmatory-shard-prune-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "scene_id": scene_id,
        "passed": True,
        "execute_requested": execute,
        "schedule_shard_manifest": str(shard_manifest_path),
        "schedule_shard_manifest_sha256": _sha256(shard_manifest_path),
        "scale": scale,
        "t3": t3,
        "t2": t2,
        "raw_file_count": len(raw_inventory),
        "raw_bytes": raw_bytes,
        "prune_snapshot_arrays": prune_snapshot_arrays,
    }
    if not execute:
        return audit

    scene_receipt = receipt_root / scene_id
    if scene_receipt.exists() or scene_receipt.is_symlink():
        raise FileExistsError(f"refusing to overwrite prune receipt: {scene_receipt}")
    scene_receipt.mkdir(parents=True, exist_ok=False)
    inventory_path = scene_receipt / "raw_inventory.jsonl"
    with inventory_path.open("x", encoding="utf-8") as stream:
        for row in raw_inventory:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    prepared = {
        **audit,
        "state": "prepared_before_deletion",
        "raw_inventory": str(inventory_path),
        "raw_inventory_sha256": _sha256(inventory_path),
    }
    prepared_path = scene_receipt / "prepared.json"
    prepared_path.write_text(
        json.dumps(prepared, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.rmtree(corpus_shard)
    pruned_snapshot_paths = []
    if prune_snapshot_arrays:
        for snapshot_dir in (scale_snapshot_dir, t3_snapshot_dir):
            target = snapshot_dir.resolve() / "snapshots.npz"
            if target.is_file():
                target.unlink()
                pruned_snapshot_paths.append(str(target))
    completed = {
        **prepared,
        "state": "completed",
        "completed_utc": datetime.now(UTC).isoformat(),
        "corpus_shard_removed": not corpus_shard.exists(),
        "pruned_snapshot_arrays": pruned_snapshot_paths,
    }
    completed_path = scene_receipt / "completed.json"
    completed_path.write_text(
        json.dumps(completed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--schedule-root", type=Path, default=DEFAULT_SCHEDULE_ROOT)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--corpus-shard", type=Path)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--scale-snapshot-dir", type=Path)
    parser.add_argument("--scale-feature-dir", type=Path)
    parser.add_argument("--scale-unified-dir", type=Path)
    parser.add_argument("--t3-snapshot-dir", type=Path)
    parser.add_argument("--t3-feature-dir", type=Path)
    parser.add_argument("--t3-unified-dir", type=Path)
    parser.add_argument("--t2-feature-manifest", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, default=DEFAULT_RECEIPT_ROOT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prune-snapshot-arrays", action="store_true")
    args = parser.parse_args()
    scene_root = args.derived_root / args.scene_id
    result = seal_and_prune(
        scene_id=args.scene_id,
        schedule_root=args.schedule_root,
        corpus_root=args.corpus_root,
        corpus_shard=args.corpus_shard
        or args.corpus_root / args.scene_id,
        scale_snapshot_dir=args.scale_snapshot_dir
        or scene_root / "scale/snapshots",
        scale_feature_dir=args.scale_feature_dir
        or scene_root / "scale/features",
        scale_unified_dir=args.scale_unified_dir
        or scene_root / "scale/unified_features",
        t3_snapshot_dir=args.t3_snapshot_dir
        or scene_root / "c2_t3/snapshots",
        t3_feature_dir=args.t3_feature_dir
        or scene_root / "c2_t3/features",
        t3_unified_dir=args.t3_unified_dir
        or scene_root / "c2_t3/unified_features",
        t2_feature_manifest=args.t2_feature_manifest,
        receipt_root=args.receipt_root,
        execute=args.execute,
        prune_snapshot_arrays=args.prune_snapshot_arrays,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
