#!/usr/bin/env python3
"""Merge disjoint consumed C2 bundles for v5 LOSO development only."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = [path.resolve() for path in args.source]
    rows: list[dict] = []
    visual_parts: list[np.ndarray] = []
    geometry_parts: list[np.ndarray] = []
    proprio_parts: list[np.ndarray] = []
    source_hashes: dict[str, str] = {}
    scene_sets: list[set[str]] = []
    for source in sources:
        manifest_path = source / "feature_manifest.json"
        geometry_manifest_path = source / "geometry_manifest.json"
        records_path = source / "records.jsonl"
        features_path = source / "features.npz"
        geometry_path = source / "geometry.npz"
        manifest = _json(manifest_path)
        geometry_manifest = _json(geometry_manifest_path)
        if (
            manifest.get("passed") is not True
            or geometry_manifest.get("passed") is not True
            or _sha(records_path)
            != manifest["output_sha256"]["records"]
            or _sha(features_path)
            != manifest["output_sha256"]["features"]
            or _sha(geometry_path)
            != geometry_manifest["output_sha256"]
        ):
            raise RuntimeError(f"invalid source bundle: {source}")
        source_rows = _jsonl(records_path)
        source_ids = [str(row["sample_id"]) for row in source_rows]
        with np.load(features_path, allow_pickle=False) as archive:
            if archive["sample_ids"].astype(str).tolist() != source_ids:
                raise RuntimeError(f"feature misalignment: {source}")
            visual_parts.append(
                np.asarray(archive["visual"], dtype=np.float32)
            )
            proprio_parts.append(
                np.asarray(archive["proprio"], dtype=np.float32)
            )
        with np.load(geometry_path, allow_pickle=False) as archive:
            if archive["sample_ids"].astype(str).tolist() != source_ids:
                raise RuntimeError(f"geometry misalignment: {source}")
            geometry_parts.append(
                np.asarray(archive["geometry"], dtype=np.float32)
            )
        rows.extend(source_rows)
        scene_sets.append(
            {str(row["scene_cluster"]) for row in source_rows}
        )
        source_hashes[str(source)] = _sha(manifest_path)
    if any(
        left & right
        for index, left in enumerate(scene_sets)
        for right in scene_sets[index + 1 :]
    ):
        raise RuntimeError("development source scenes are not disjoint")
    visual = np.concatenate(visual_parts)
    geometry = np.concatenate(geometry_parts)
    proprio = np.concatenate(proprio_parts)
    sample_ids = np.asarray([str(row["sample_id"]) for row in rows])
    checks = {
        "source_bundles_valid": True,
        "source_scenes_disjoint": True,
        "sample_ids_unique": len(set(sample_ids.tolist()))
        == len(sample_ids),
        "six_scene_loso_battery": len(
            {str(row["scene_cluster"]) for row in rows}
        )
        == 6,
        "three_domains_present": len(
            {str(row["domain"]) for row in rows}
        )
        == 3,
        "both_directions_present": {
            str(row["cell"]) for row in rows
        }
        == {"T2_vision_decisive", "T3_proprio_decisive"},
        "proprio_dimension_80": proprio.shape[1] == 80,
        "all_values_finite": bool(
            np.isfinite(visual).all()
            and np.isfinite(geometry).all()
            and np.isfinite(proprio).all()
        ),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records_path = output / "records.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    features_path = output / "features.npz"
    np.savez_compressed(
        features_path,
        sample_ids=sample_ids,
        visual=visual,
        geometry=geometry,
        proprio=proprio,
    )
    manifest = {
        "schema_version": (
            "kinofail.realistic-c2-v5-development-merge.v1"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "development_only": True,
        "formal_confirmation_outcomes_used_for_development": True,
        "counts": {
            "samples": len(rows),
            "cases": len({str(row["case_id"]) for row in rows}),
            "scene_clusters": len(
                {str(row["scene_cluster"]) for row in rows}
            ),
            "domains": len({str(row["domain"]) for row in rows}),
        },
        "source_feature_manifest_sha256": source_hashes,
        "output_sha256": {
            "records": _sha(records_path),
            "features": _sha(features_path),
        },
    }
    (output / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
