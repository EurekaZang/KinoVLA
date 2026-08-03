#!/usr/bin/env python3
"""Apply the C2 v5 post-interaction invariant proprio contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from kino_vla.eval.c2_temporal_v5 import (
    FOOTPRINT_MARGIN_M,
    MAX_END_SKEW_S,
    POST_ENCOUNTER_DELAY_S,
    WINDOW_SAMPLES,
    geometry_aligned_invariant_summary,
    invariant_summary,
)


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


def index_t3_episode_dirs(corpus: Path) -> dict[str, Path]:
    """Index the frozen union without traversing episode-directory symlinks.

    F30 materializes its model-blind union as real scene/material/operator
    directories whose episode leaves are symlinks.  ``Path.rglob`` does not
    descend through those directory symlinks, so enumerate the fixed hierarchy
    and validate each linked episode by its manifest instead.
    """

    episodes: dict[str, Path] = {}
    for episode_dir in sorted(corpus.resolve().glob("*/c2_t3/*/*/*")):
        manifest = episode_dir / "manifest.json"
        if not manifest.is_file():
            continue
        episode_id = episode_dir.name
        prior = episodes.get(episode_id)
        if prior is not None and prior.resolve() != episode_dir.resolve():
            raise RuntimeError(f"duplicate T3 episode id: {episode_id}")
        episodes[episode_id] = episode_dir
    return episodes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument(
        "--t2-corpus", type=Path, action="append", required=True
    )
    parser.add_argument("--t3-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.base_features.resolve()
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
        or _sha(geometry_path)
        != geometry_manifest["output_sha256"]
    ):
        raise RuntimeError("invalid base feature bundle")
    rows = _jsonl(records_path)
    with np.load(features_path, allow_pickle=False) as archive:
        sample_ids = archive["sample_ids"].astype(str)
        visual = np.asarray(archive["visual"], dtype=np.float32)
    with np.load(geometry_path, allow_pickle=False) as archive:
        geometry_ids = archive["sample_ids"].astype(str)
        geometry = np.asarray(archive["geometry"], dtype=np.float32)
    expected_ids = np.asarray(
        [str(row["sample_id"]) for row in rows]
    )
    if (
        not np.array_equal(sample_ids, expected_ids)
        or not np.array_equal(geometry_ids, expected_ids)
    ):
        raise RuntimeError("base feature rows are misaligned")
    t2_cases = {
        path.parent.name: path
        for root in args.t2_corpus
        for path in root.resolve().glob("*/observables.npz")
    }
    t3_episodes = index_t3_episode_dirs(args.t3_corpus)
    cache: dict[str, tuple[np.ndarray, dict]] = {}
    proprio = []
    for row in rows:
        if row["cell"] == "T2_vision_decisive":
            case_id = str(row["case_id"])
            path = t2_cases[case_id]
            key = (
                f"{case_id}__{row['target_operator']}__"
                f"{row['appearance_view_id']}__proprio"
            )
            with np.load(path, allow_pickle=False) as archive:
                window = np.asarray(
                    archive[key], dtype=np.float32
                )
            proprio.append(invariant_summary(window))
            continue
        episode_id = str(
            row.get("proprio_source_episode_id")
            or f"{row['source_physics_group_id']}_anomaly"
        )
        if episode_id not in cache:
            cache[episode_id] = (
                geometry_aligned_invariant_summary(
                    t3_episodes[episode_id]
                )
            )
        proprio.append(cache[episode_id][0])
    proprio_array = np.stack(proprio).astype(np.float32)
    checks = {
        "base_hash_locked": True,
        "sample_order_preserved": True,
        "all_values_finite": bool(
            np.isfinite(visual).all()
            and np.isfinite(geometry).all()
            and np.isfinite(proprio_array).all()
        ),
        "invariant_proprio_dimension_80": (
            proprio_array.shape[1] == 80
        ),
        "all_t3_episodes_aligned": len(cache)
        == len(
            {
                str(
                    row.get("proprio_source_episode_id")
                    or f"{row['source_physics_group_id']}_anomaly"
                )
                for row in rows
                if row["cell"] == "T3_proprio_decisive"
            }
        ),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    out_records = output / "records.jsonl"
    out_records.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    out_features = output / "features.npz"
    np.savez_compressed(
        out_features,
        sample_ids=expected_ids,
        visual=visual,
        proprio=proprio_array,
    )
    out_geometry = output / "geometry.npz"
    np.savez_compressed(
        out_geometry,
        sample_ids=expected_ids,
        geometry=geometry,
    )
    manifest = {
        "schema_version": "kinofail.realistic-c2-v5-features.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "samples": len(rows),
            "cases": len({str(row["case_id"]) for row in rows}),
            "scene_clusters": len(
                {str(row["scene_cluster"]) for row in rows}
            ),
            "t3_source_episodes": len(cache),
        },
        "temporal_contract": {
            "footprint_margin_m": FOOTPRINT_MARGIN_M,
            "post_encounter_delay_s": POST_ENCOUNTER_DELAY_S,
            "window_samples": WINDOW_SAMPLES,
            "maximum_end_skew_s": MAX_END_SKEW_S,
            "signals": [
                "angular_speed",
                "linear_acceleration_norm",
                "odom_speed",
                "slip_ratio",
                "base_height",
                "tilt",
                "effort_ratio",
                "support_ratio"
            ],
            "forbidden": [
                "outcome time",
                "fall state",
                "truth label",
                "scene/domain ID"
            ]
        },
        "episode_alignment": {
            key: value[1] for key, value in sorted(cache.items())
        },
        "source_sha256": {
            "feature_manifest": _sha(feature_manifest_path),
            "geometry_manifest": _sha(geometry_manifest_path),
        },
        "output_sha256": {
            "records": _sha(out_records),
            "features": _sha(out_features),
        },
    }
    geometry_manifest_out = {
        "schema_version": "kinofail.realistic-c2-v5-geometry.v1",
        "passed": all(checks.values()),
        "output_sha256": _sha(out_geometry),
        "counts": {
            "samples": len(rows),
            "dimension": int(geometry.shape[1])
        }
    }
    (output / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "geometry_manifest.json").write_text(
        json.dumps(
            geometry_manifest_out, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                "checks": checks,
                "counts": manifest["counts"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if manifest["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
