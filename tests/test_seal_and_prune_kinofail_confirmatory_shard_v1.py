from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.seal_and_prune_kinofail_confirmatory_shard_v1 import (
    seal_and_prune,
)


SCENE_ID = "confirm_v1_life_scene_00"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _schedule_fixture(root: Path) -> Path:
    shard = root / "scenes" / SCENE_ID
    artifacts = {
        "scale_schedule": shard / "scale/schedule.jsonl",
        "scale_collection_protocol": shard / "scale/collection_protocol.json",
        "scale_snapshot_protocol": shard / "scale/snapshot_protocol.json",
        "t2_case_schedule": shard / "c2_t2/case_schedule.jsonl",
        "t3_schedule": shard / "c2_t3/schedule.jsonl",
        "t3_case_schedule": shard / "c2_t3/case_schedule.jsonl",
        "t3_collection_protocol": shard / "c2_t3/collection_protocol.json",
        "t3_snapshot_protocol": shard / "c2_t3/snapshot_protocol.json",
        "conflict_schedule": shard / "conflict_schedule.jsonl",
    }
    hashes = {}
    for key, path in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        hashes[key] = _sha(path)
    _write_json(
        shard / "shard_manifest.json",
        {
            "schema_version": "kinofail.unified-confirmatory-shard-manifest.v1",
            "scene_id": SCENE_ID,
            "passed": True,
            "counts": {"scale_pairs": 1, "t3_pairs": 1},
            "artifacts": hashes,
        },
    )
    return root


def _bundle(root: Path) -> tuple[Path, Path, Path]:
    snapshots = root / "snapshots"
    features = root / "features"
    unified = root / "unified_features"
    snapshots.mkdir(parents=True)
    features.mkdir(parents=True)
    unified.mkdir(parents=True)
    records = [
        {
            "sample_id": "sample_0",
            "scene_family": SCENE_ID,
        },
        {
            "sample_id": "sample_1",
            "scene_family": SCENE_ID,
        },
    ]
    records_path = snapshots / "snapshot_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    np.savez_compressed(
        snapshots / "snapshots.npz",
        sample_0__rgb=np.zeros((1, 2, 2, 3), dtype=np.uint8),
        sample_0__proprio=np.zeros((2, 2), dtype=np.float32),
        sample_1__rgb=np.ones((1, 2, 2, 3), dtype=np.uint8),
        sample_1__proprio=np.ones((2, 2), dtype=np.float32),
    )
    _write_json(
        snapshots / "extraction_audit.json",
        {
            "passed": True,
            "checks": {"all_selected_or_excluded_pairs_accounted_for": True},
            "skipped_incomplete_pairs": 0,
            "pair_audits": [{"counterfactual_group_id": "cf_1"}],
            "temporal_alignment_exclusions": [],
        },
    )
    sample_ids = np.asarray(["sample_0", "sample_1"])
    np.savez_compressed(
        features / "features.npz",
        sample_ids=sample_ids,
        visual=np.zeros((2, 2), dtype=np.float32),
        proprio=np.zeros((2, 2), dtype=np.float32),
    )
    _write_json(
        features / "feature_manifest.json",
        {
            "status": "complete",
            "output_sha256": {"features": _sha(features / "features.npz")},
            "source_sha256": {
                "snapshot_records": _sha(records_path),
                "snapshots": _sha(snapshots / "snapshots.npz"),
                "extraction_audit": _sha(
                    snapshots / "extraction_audit.json"
                ),
            },
        },
    )
    np.savez_compressed(
        unified / "features.npz",
        sample_ids=sample_ids,
        visual=np.zeros((2, 2), dtype=np.float32),
        proprio=np.zeros((2, 2), dtype=np.float32),
    )
    _write_json(
        unified / "feature_manifest.json",
        {
            "status": "complete",
            "output_sha256": {"features": _sha(unified / "features.npz")},
            "source_sha256": {
                "snapshot_records": _sha(records_path),
                "snapshots": _sha(snapshots / "snapshots.npz"),
                "snapshot_audit": _sha(
                    snapshots / "extraction_audit.json"
                ),
                "visual_features": _sha(features / "features.npz"),
                "visual_manifest": _sha(
                    features / "feature_manifest.json"
                ),
            },
        },
    )
    return snapshots, features, unified


def _fixture(tmp_path: Path) -> dict[str, Path]:
    schedule_root = _schedule_fixture(tmp_path / "schedules")
    corpus_root = tmp_path / "corpus"
    corpus_shard = corpus_root / SCENE_ID
    corpus_shard.mkdir(parents=True)
    (corpus_shard / "raw.bin").write_bytes(b"raw-confirmatory-episode")
    scale_snapshot, scale_features, scale_unified = _bundle(
        tmp_path / "derived/scale"
    )
    t3_snapshot, t3_features, t3_unified = _bundle(
        tmp_path / "derived/t3"
    )
    t2_dir = tmp_path / "derived/t2"
    t2_dir.mkdir(parents=True)
    np.savez_compressed(
        t2_dir / "features.npz",
        values=np.zeros((1, 1), dtype=np.float32),
    )
    _write_json(
        t2_dir / "feature_manifest.json",
        {
            "status": "complete",
            "scene_id": SCENE_ID,
            "output_sha256": {"features": _sha(t2_dir / "features.npz")},
        },
    )
    return {
        "schedule_root": schedule_root,
        "corpus_root": corpus_root,
        "corpus_shard": corpus_shard,
        "scale_snapshot": scale_snapshot,
        "scale_features": scale_features,
        "scale_unified": scale_unified,
        "t3_snapshot": t3_snapshot,
        "t3_features": t3_features,
        "t3_unified": t3_unified,
        "t2_manifest": t2_dir / "feature_manifest.json",
        "receipt_root": tmp_path / "receipts",
    }


def _call(paths: dict[str, Path], *, execute: bool) -> dict:
    return seal_and_prune(
        scene_id=SCENE_ID,
        schedule_root=paths["schedule_root"],
        corpus_root=paths["corpus_root"],
        corpus_shard=paths["corpus_shard"],
        scale_snapshot_dir=paths["scale_snapshot"],
        scale_feature_dir=paths["scale_features"],
        scale_unified_dir=paths["scale_unified"],
        t3_snapshot_dir=paths["t3_snapshot"],
        t3_feature_dir=paths["t3_features"],
        t3_unified_dir=paths["t3_unified"],
        t2_feature_manifest=paths["t2_manifest"],
        receipt_root=paths["receipt_root"],
        execute=execute,
        prune_snapshot_arrays=False,
    )


def test_dry_run_is_non_destructive(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    result = _call(paths, execute=False)
    assert result["passed"] is True
    assert paths["corpus_shard"].is_dir()
    assert not paths["receipt_root"].exists()


def test_execute_removes_only_exact_scene_and_writes_receipts(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    sibling = paths["corpus_root"] / "confirm_v1_life_scene_01"
    sibling.mkdir()
    (sibling / "keep.bin").write_bytes(b"keep")
    result = _call(paths, execute=True)
    assert result["state"] == "completed"
    assert not paths["corpus_shard"].exists()
    assert (sibling / "keep.bin").read_bytes() == b"keep"
    receipt = paths["receipt_root"] / SCENE_ID
    assert (receipt / "prepared.json").is_file()
    assert (receipt / "completed.json").is_file()
    assert (receipt / "raw_inventory.jsonl").is_file()


def test_rejects_symlink_corpus_target(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    real = paths["corpus_root"] / "real_scene"
    paths["corpus_shard"].rename(real)
    paths["corpus_shard"].symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        _call(paths, execute=False)


def test_rejects_tampered_unified_features(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    with (paths["scale_unified"] / "features.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(RuntimeError, match="unified feature hash"):
        _call(paths, execute=False)
