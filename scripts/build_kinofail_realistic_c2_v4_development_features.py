#!/usr/bin/env python3
"""Build the 12-scene C2 v4 development pool after the v3 failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from eval_kinofail_realistic_c2_bidirectional_v2 import (
    _load_development,
    _load_test,
)
from kino_vla.eval.c2_temporal import project_proprio


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-development", type=Path, required=True)
    parser.add_argument("--v3-failure-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = args.prior_development.resolve()
    failure = args.v3_failure_features.resolve()
    (
        prior_rows,
        prior_visual,
        prior_geometry,
        prior_proprio,
        prior_manifest,
    ) = _load_development(prior)
    (
        failure_rows,
        failure_visual,
        failure_geometry,
        failure_proprio,
        failure_manifest,
        failure_geometry_manifest,
    ) = _load_test(failure, failure)
    prior_scenes = {
        str(row["scene_cluster"]) for row in prior_rows
    }
    failure_scenes = {
        str(row["scene_cluster"]) for row in failure_rows
    }
    rows = [
        {
            **row,
            "split": "development",
            "v4_development_source": source,
        }
        for source, source_rows in (
            ("prior_nine_scene_development", prior_rows),
            ("audited_v3_formal_failure", failure_rows),
        )
        for row in source_rows
    ]
    visual = np.concatenate([prior_visual, failure_visual], axis=0)
    geometry = np.concatenate(
        [prior_geometry, failure_geometry], axis=0
    )
    proprio = np.concatenate(
        [project_proprio(prior_proprio), failure_proprio], axis=0
    )
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
    visual = visual[order]
    geometry = geometry[order]
    proprio = proprio[order]
    sample_ids = np.asarray(
        [str(row["sample_id"]) for row in rows]
    )
    checks = {
        "source_scene_sets_disjoint": prior_scenes.isdisjoint(
            failure_scenes
        ),
        "twelve_development_scenes": (
            len(prior_scenes | failure_scenes) == 12
        ),
        "three_domains": (
            {str(row["domain"]) for row in rows}
            == {"life", "production", "wild"}
        ),
        "both_directions": (
            {str(row["cell"]) for row in rows}
            == {"T2_vision_decisive", "T3_proprio_decisive"}
        ),
        "all_values_finite": bool(
            np.isfinite(visual).all()
            and np.isfinite(geometry).all()
            and np.isfinite(proprio).all()
        ),
        "projected_proprio_dimension_130": proprio.shape[1] == 130,
        "v3_failure_is_development_only": all(
            row["split"] == "development" for row in rows
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
        visual=visual.astype(np.float32),
        geometry=geometry.astype(np.float32),
        proprio=proprio.astype(np.float32),
    )
    manifest = {
        "schema_version": (
            "kinofail.realistic-c2-v4-development-features.v1"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "samples": len(rows),
            "cases": len({str(row["case_id"]) for row in rows}),
            "scene_clusters": len(prior_scenes | failure_scenes),
            "domains": 3,
            "cases_per_direction": {
                cell: len(
                    {
                        str(row["case_id"])
                        for row in rows
                        if row["cell"] == cell
                    }
                )
                for cell in (
                    "T2_vision_decisive",
                    "T3_proprio_decisive",
                )
            },
        },
        "scene_clusters": sorted(prior_scenes | failure_scenes),
        "formal_v3_failure_used_for_v4_development": True,
        "v4_confirmation_outcomes_used": False,
        "source_sha256": {
            "prior_feature_manifest": _sha(prior_manifest),
            "v3_failure_feature_manifest": _sha(failure_manifest),
            "v3_failure_geometry_manifest": _sha(
                failure_geometry_manifest
            ),
        },
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
