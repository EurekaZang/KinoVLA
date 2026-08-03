#!/usr/bin/env python3
"""Assemble scene-disjoint C2 v2 development data.

Only the six non-test scene clusters are admitted. Opened confirmations from
the three held-out scenes are intentionally excluded from model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _load(
    feature_dir: Path, geometry_dir: Path
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    feature_manifest_path = feature_dir / "feature_manifest.json"
    geometry_manifest_path = geometry_dir / "geometry_manifest.json"
    feature_manifest = _json(feature_manifest_path)
    geometry_manifest = _json(geometry_manifest_path)
    records_path = feature_dir / "records.jsonl"
    features_path = feature_dir / "features.npz"
    geometry_path = geometry_dir / "geometry.npz"
    if (
        feature_manifest.get("passed") is not True
        or geometry_manifest.get("passed") is not True
        or _sha(records_path) != feature_manifest["output_sha256"]["records"]
        or _sha(features_path) != feature_manifest["output_sha256"]["features"]
        or _sha(geometry_path) != geometry_manifest["output_sha256"]
    ):
        raise RuntimeError("C2 v2 source feature provenance mismatch")
    rows = _jsonl(records_path)
    features = np.load(features_path, allow_pickle=False)
    geometry = np.load(geometry_path, allow_pickle=False)
    sample_ids = [str(row["sample_id"]) for row in rows]
    if (
        features["sample_ids"].astype(str).tolist() != sample_ids
        or geometry["sample_ids"].astype(str).tolist() != sample_ids
    ):
        raise RuntimeError("C2 v2 source feature streams are misaligned")
    return (
        rows,
        np.asarray(features["visual"], dtype=np.float32),
        np.asarray(features["proprio"], dtype=np.float32),
        np.asarray(geometry["geometry"], dtype=np.float32),
        {
            "feature_manifest": _sha(feature_manifest_path),
            "geometry_manifest": _sha(geometry_manifest_path),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--c1-development-features", type=Path, required=True
    )
    parser.add_argument(
        "--c1-development-geometry", type=Path, required=True
    )
    parser.add_argument("--c2-v1-features", type=Path, required=True)
    parser.add_argument("--c2-v1-geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = [
        _load(
            args.c1_development_features.resolve(),
            args.c1_development_geometry.resolve(),
        ),
        _load(
            args.c2_v1_features.resolve(),
            args.c2_v1_geometry.resolve(),
        ),
    ]
    rows: list[dict[str, Any]] = []
    visual: list[np.ndarray] = []
    proprio: list[np.ndarray] = []
    geometry: list[np.ndarray] = []
    source_hashes: list[dict[str, str]] = []
    for source_index, (source_rows, v, p, g, hashes) in enumerate(sources):
        source_hashes.append(hashes)
        for index, row in enumerate(source_rows):
            if source_index == 0:
                value = {
                    **row,
                    "cell": "T2_vision_decisive",
                    "development_origin": "structured_v3_development",
                }
            else:
                if row.get("cell") != "T3_proprio_decisive":
                    continue
                if str(row["scene_cluster"]) in {
                    "indoor_kitchen_140",
                    "indoor_office_186",
                    "forest_river_walk_metric_g07",
                }:
                    continue
                value = {
                    **row,
                    "development_origin": (
                        "preserved_c2_v1_non_test_t3"
                    ),
                }
            value["source_split"] = str(row["split"])
            value["split"] = "development"
            rows.append(value)
            visual.append(v[index])
            proprio.append(p[index])
            geometry.append(g[index])
    order = sorted(
        range(len(rows)),
        key=lambda index: (
            str(rows[index]["scene_cluster"]),
            str(rows[index]["cell"]),
            str(rows[index]["case_id"]),
            str(rows[index]["target_operator"]),
            str(rows[index]["appearance_view_id"]),
        ),
    )
    rows = [rows[index] for index in order]
    visual_array = np.stack([visual[index] for index in order]).astype(
        np.float32
    )
    proprio_array = np.stack([proprio[index] for index in order]).astype(
        np.float32
    )
    geometry_array = np.stack([geometry[index] for index in order]).astype(
        np.float32
    )
    checks = {
        "balanced_within_each_direction": {
            label: sum(
                row["attribution_category"] == label for row in rows
            )
            for label in sorted(
                {str(row["attribution_category"]) for row in rows}
            )
        }
        == {
            "adhesion": 36,
            "compliant_terrain": 36,
            "invisible_obstacle": 54,
            "low_friction": 54,
        },
        "two_direction_cells": {
            str(row["cell"]) for row in rows
        }
        == {"T2_vision_decisive", "T3_proprio_decisive"},
        "six_scene_clusters": len(
            {str(row["scene_cluster"]) for row in rows}
        )
        == 6,
        "heldout_test_scenes_absent": {
            str(row["scene_cluster"]) for row in rows
        }.isdisjoint(
            {
                "indoor_kitchen_140",
                "indoor_office_186",
                "forest_river_walk_metric_g07",
            }
        ),
        "all_finite": bool(
            np.isfinite(visual_array).all()
            and np.isfinite(proprio_array).all()
            and np.isfinite(geometry_array).all()
        ),
        "unique_sample_ids": len(
            {str(row["sample_id"]) for row in rows}
        )
        == len(rows),
    }
    if not all(checks.values()):
        raise RuntimeError(f"C2 v2 development audit failed: {checks}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records_path = output / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    features_path = output / "features.npz"
    np.savez_compressed(
        features_path,
        sample_ids=np.asarray([row["sample_id"] for row in rows]),
        visual=visual_array,
        proprio=proprio_array,
        geometry=geometry_array,
    )
    manifest = {
        "schema_version": "kinofail.realistic-c2-development-features.v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "samples": len(rows),
            "cases": len({str(row["case_id"]) for row in rows}),
            "scene_clusters": 6,
            "t2_cases": len(
                {
                    str(row["case_id"])
                    for row in rows
                    if row["cell"] == "T2_vision_decisive"
                }
            ),
            "t3_cases": len(
                {
                    str(row["case_id"])
                    for row in rows
                    if row["cell"] == "T3_proprio_decisive"
                }
            ),
        },
        "source_sha256": source_hashes,
        "output_sha256": {
            "records": _sha(records_path),
            "features": _sha(features_path),
        },
    }
    manifest_path = output / "feature_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
