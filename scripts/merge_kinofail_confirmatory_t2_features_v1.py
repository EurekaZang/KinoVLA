#!/usr/bin/env python3
"""Merge valid per-scene T2 feature shards without copying raw RGB."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-feature-dir", type=Path, action="append", required=True)
    parser.add_argument("--scene-corpus-dir", type=Path, action="append", required=True)
    parser.add_argument("--t2-schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    feature_dirs = [path.resolve() for path in args.scene_feature_dir]
    corpus_dirs = [path.resolve() for path in args.scene_corpus_dir]
    if len(feature_dirs) != len(corpus_dirs):
        raise RuntimeError("T2 feature and corpus directories must be paired")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    schedule_path = args.t2_schedule.resolve()
    schedule_rows = _jsonl(schedule_path)
    if (
        len(schedule_rows) != 1_500
        or len({str(row["case_id"]) for row in schedule_rows}) != 1_500
    ):
        raise RuntimeError("frozen T2 schedule must contain 1,500 unique cases")
    schedule_by_case = {str(row["case_id"]): row for row in schedule_rows}

    rows: list[dict[str, Any]] = []
    ids: list[np.ndarray] = []
    visual: list[np.ndarray] = []
    proprio: list[np.ndarray] = []
    sources = {}
    for feature_dir, corpus_dir in zip(feature_dirs, corpus_dirs, strict=True):
        manifest_path = feature_dir / "feature_manifest.json"
        records_path = feature_dir / "records.jsonl"
        features_path = feature_dir / "features.npz"
        manifest = _json(manifest_path)
        scene_id = str(manifest.get("scene_id", ""))
        if (
            manifest.get("passed") is not True
            or not scene_id
            or manifest.get("output_sha256", {}).get("records")
            != _sha256(records_path)
            or manifest.get("output_sha256", {}).get("features")
            != _sha256(features_path)
        ):
            raise RuntimeError(f"invalid T2 feature shard: {feature_dir}")
        shard_rows = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if {str(row["scene_cluster"]) for row in shard_rows} != {scene_id}:
            raise RuntimeError(f"T2 shard mixes scenes: {feature_dir}")
        with np.load(features_path, allow_pickle=False) as archive:
            shard_ids = archive["sample_ids"].astype(str)
            shard_visual = np.asarray(archive["visual"], dtype=np.float32)
            shard_proprio = np.asarray(archive["proprio"], dtype=np.float32)
        if shard_ids.tolist() != [str(row["sample_id"]) for row in shard_rows]:
            raise RuntimeError(f"T2 shard rows/features are misaligned: {feature_dir}")
        for row in shard_rows:
            case_id = str(row["case_id"])
            if case_id not in schedule_by_case:
                raise RuntimeError(f"unplanned T2 case in feature shard: {case_id}")
            planned = schedule_by_case[case_id]
            if (
                str(planned["scene_cluster"]) != scene_id
                or str(row["scene_cluster"]) != scene_id
            ):
                raise RuntimeError(f"T2 case/scene schedule mismatch: {case_id}")
            case_dir = corpus_dir / case_id
            if not case_dir.is_dir():
                raise FileNotFoundError(case_dir)
            row["source_case_dir"] = str(case_dir)
            # Appearance swaps deliberately change ``material_family``.  The
            # crossed bootstrap cluster is the preregistered physical context
            # and therefore must come only from the frozen case schedule.
            row["cluster_material"] = str(planned["cluster_material"])
            row["material_id"] = str(planned["material_id"])
            row["cell"] = "T2_vision_decisive"
            row["split"] = "confirmatory"
        rows.extend(shard_rows)
        ids.append(shard_ids)
        visual.append(shard_visual)
        proprio.append(shard_proprio)
        sources[scene_id] = {
            "feature_manifest": _sha256(manifest_path),
            "records": _sha256(records_path),
            "features": _sha256(features_path),
        }
    if not rows:
        raise RuntimeError("no valid T2 feature shards")
    order = sorted(
        range(len(rows)),
        key=lambda index: (
            str(rows[index]["scene_cluster"]),
            str(rows[index]["case_id"]),
            str(rows[index]["target_operator"]),
            str(rows[index]["appearance_view_id"]),
        ),
    )
    all_ids = np.concatenate(ids)[order]
    all_visual = np.concatenate(visual)[order]
    all_proprio = np.concatenate(proprio)[order]
    rows = [rows[index] for index in order]
    if (
        len(set(all_ids.tolist())) != len(all_ids)
        or all_ids.tolist() != [str(row["sample_id"]) for row in rows]
    ):
        raise RuntimeError("merged T2 IDs are duplicate or misaligned")
    cases = {str(row["case_id"]) for row in rows}
    if len(cases) < 1_425 or len(cases) > 1_500 or len(rows) != 6 * len(cases):
        raise RuntimeError("T2 attrition exceeds the frozen 5% gate")

    output.mkdir(parents=True, exist_ok=False)
    records_path = output / "records.jsonl"
    features_path = output / "features.npz"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    np.savez_compressed(
        features_path,
        sample_ids=all_ids,
        visual=all_visual,
        proprio=all_proprio,
    )
    manifest = {
        "schema_version": "kinofail.confirmatory-t2-merged-features.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "passed": True,
        "counts": {
            "scenes": len({str(row["scene_cluster"]) for row in rows}),
            "cases": len(cases),
            "samples": len(rows),
            "invalid_planned_cases": 1_500 - len(cases),
        },
        "source_sha256": sources,
        "t2_schedule": {
            "path": str(schedule_path),
            "sha256": _sha256(schedule_path),
        },
        "output_sha256": {
            "records": _sha256(records_path),
            "features": _sha256(features_path),
        },
    }
    (output / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
