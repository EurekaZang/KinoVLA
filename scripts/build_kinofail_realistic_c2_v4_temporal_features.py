#!/usr/bin/env python3
"""Apply the frozen C2 v4 temporal/proprio contract to a v3 feature bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from kino_vla.eval.c2_temporal import (
    FOOTPRINT_MARGIN_M,
    MAX_END_SKEW_S,
    POST_ENCOUNTER_DELAY_S,
    PROPRIO_SIGNAL_INDICES,
    WINDOW_S,
    WINDOW_SAMPLES,
    geometry_aligned_proprio_summary,
    project_proprio,
)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--t3-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.base_features.resolve()
    corpus = args.t3_corpus.resolve()
    output = args.output.resolve()
    feature_manifest_path = base / "feature_manifest.json"
    geometry_manifest_path = base / "geometry_manifest.json"
    records_path = base / "records.jsonl"
    features_path = base / "features.npz"
    geometry_path = base / "geometry.npz"
    feature_manifest = _json(feature_manifest_path)
    geometry_manifest = _json(geometry_manifest_path)
    if (
        feature_manifest.get("passed") is not True
        or geometry_manifest.get("passed") is not True
        or _sha(records_path)
        != feature_manifest["output_sha256"]["records"]
        or _sha(features_path)
        != feature_manifest["output_sha256"]["features"]
        or _sha(geometry_path) != geometry_manifest["output_sha256"]
    ):
        raise RuntimeError("invalid base C2 feature bundle")
    rows = _jsonl(records_path)
    with np.load(features_path, allow_pickle=False) as archive:
        sample_ids = np.asarray(archive["sample_ids"]).astype(str)
        visual = np.asarray(archive["visual"], dtype=np.float32)
        proprio = np.asarray(archive["proprio"], dtype=np.float32)
    with np.load(geometry_path, allow_pickle=False) as archive:
        geometry_ids = np.asarray(archive["sample_ids"]).astype(str)
        geometry = np.asarray(archive["geometry"], dtype=np.float32)
    expected_ids = np.asarray(
        [str(row["sample_id"]) for row in rows]
    )
    if (
        not np.array_equal(sample_ids, expected_ids)
        or not np.array_equal(geometry_ids, expected_ids)
    ):
        raise RuntimeError("base C2 feature rows are misaligned")

    episode_dirs = {
        path.parent.name: path.parent
        for path in corpus.rglob("manifest.json")
    }
    cache: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    corrected = np.asarray(proprio, dtype=np.float32).copy()
    for index, row in enumerate(rows):
        if str(row["cell"]) != "T3_proprio_decisive":
            continue
        episode_id = str(row["proprio_source_episode_id"])
        if episode_id not in episode_dirs:
            raise RuntimeError(f"missing T3 episode: {episode_id}")
        if episode_id not in cache:
            cache[episode_id] = geometry_aligned_proprio_summary(
                episode_dirs[episode_id]
            )
        corrected[index] = cache[episode_id][0]
    projected = project_proprio(corrected)
    checks = {
        "base_bundle_hash_locked": True,
        "sample_order_preserved": True,
        "all_t3_episodes_geometry_aligned": (
            len(cache)
            == len(
                {
                    str(row["proprio_source_episode_id"])
                    for row in rows
                    if row["cell"] == "T3_proprio_decisive"
                }
            )
        ),
        "all_values_finite": bool(
            np.isfinite(visual).all()
            and np.isfinite(geometry).all()
            and np.isfinite(projected).all()
        ),
        "projected_proprio_dimension_130": projected.shape[1] == 130,
    }
    output.mkdir(parents=True, exist_ok=False)
    output_records = output / "records.jsonl"
    output_records.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    output_features = output / "features.npz"
    np.savez_compressed(
        output_features,
        sample_ids=expected_ids,
        visual=visual,
        proprio=projected,
    )
    output_geometry = output / "geometry.npz"
    np.savez_compressed(
        output_geometry,
        sample_ids=expected_ids,
        geometry=geometry,
    )
    source = {
        "feature_manifest_sha256": _sha(feature_manifest_path),
        "geometry_manifest_sha256": _sha(geometry_manifest_path),
        "records_sha256": _sha(records_path),
        "features_sha256": _sha(features_path),
        "geometry_sha256": _sha(geometry_path),
    }
    temporal_contract = {
        "event": (
            "first outcome-independent overlap between measured Go2 base "
            "footprint margin and frozen operator region"
        ),
        "footprint_margin_m": FOOTPRINT_MARGIN_M,
        "post_encounter_delay_s": POST_ENCOUNTER_DELAY_S,
        "window_s": WINDOW_S,
        "window_samples": WINDOW_SAMPLES,
        "maximum_end_skew_s": MAX_END_SKEW_S,
        "retained_signal_indices": list(PROPRIO_SIGNAL_INDICES),
        "excluded": [
            "gravity_xyz",
            "absolute_odom_xy",
            "absolute_odom_heading",
        ],
    }
    out_feature_manifest = {
        "schema_version": "kinofail.realistic-c2-v4-features.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "temporal_contract_corrected",
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "samples": len(rows),
            "cases": len({str(row["case_id"]) for row in rows}),
            "scene_clusters": len(
                {str(row["scene_cluster"]) for row in rows}
            ),
            "domains": len({str(row["domain"]) for row in rows}),
            "t3_source_episodes": len(cache),
        },
        "temporal_contract": temporal_contract,
        "source_sha256": source,
        "episode_alignment": {
            episode_id: audit
            for episode_id, (_, audit) in sorted(cache.items())
        },
        "output_sha256": {
            "records": _sha(output_records),
            "features": _sha(output_features),
        },
    }
    out_geometry_manifest = {
        "schema_version": "kinofail.realistic-c2-v4-geometry.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": {
            "sample_order_preserved": True,
            "all_values_finite": bool(np.isfinite(geometry).all()),
        },
        "counts": {
            "samples": len(rows),
            "dimension": int(geometry.shape[1]),
        },
        "source_geometry_manifest_sha256": _sha(
            geometry_manifest_path
        ),
        "output_sha256": _sha(output_geometry),
    }
    (output / "feature_manifest.json").write_text(
        json.dumps(out_feature_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "geometry_manifest.json").write_text(
        json.dumps(out_geometry_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": all(checks.values()),
                "checks": checks,
                "counts": out_feature_manifest["counts"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
