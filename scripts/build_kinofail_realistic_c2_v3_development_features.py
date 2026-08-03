#!/usr/bin/env python3
"""Promote the audited C2 v2 failure into the scene-disjoint v3 dev pool."""

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
    directory: Path,
    geometry_directory: Path | None = None,
) -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, str],
]:
    manifest_path = directory / "feature_manifest.json"
    records_path = directory / "records.jsonl"
    features_path = directory / "features.npz"
    manifest = _json(manifest_path)
    if (
        manifest.get("passed") is not True
        or _sha(records_path) != manifest["output_sha256"]["records"]
        or _sha(features_path) != manifest["output_sha256"]["features"]
    ):
        raise RuntimeError(f"invalid C2 feature source: {directory}")
    rows = _jsonl(records_path)
    archive = np.load(features_path, allow_pickle=False)
    sample_ids = [str(row["sample_id"]) for row in rows]
    if archive["sample_ids"].astype(str).tolist() != sample_ids:
        raise RuntimeError(f"misaligned C2 feature source: {directory}")
    if geometry_directory is None:
        geometry = np.asarray(archive["geometry"], dtype=np.float32)
        geometry_manifest_sha = ""
    else:
        geometry_manifest_path = geometry_directory / "geometry_manifest.json"
        geometry_path = geometry_directory / "geometry.npz"
        geometry_manifest = _json(geometry_manifest_path)
        if (
            geometry_manifest.get("passed") is not True
            or _sha(geometry_path) != geometry_manifest["output_sha256"]
        ):
            raise RuntimeError(
                f"invalid C2 geometry source: {geometry_directory}"
            )
        geometry_archive = np.load(geometry_path, allow_pickle=False)
        if geometry_archive["sample_ids"].astype(str).tolist() != sample_ids:
            raise RuntimeError(
                f"misaligned C2 geometry source: {geometry_directory}"
            )
        geometry = np.asarray(
            geometry_archive["geometry"], dtype=np.float32
        )
        geometry_manifest_sha = _sha(geometry_manifest_path)
    return (
        rows,
        np.asarray(archive["visual"], dtype=np.float32),
        geometry,
        np.asarray(archive["proprio"], dtype=np.float32),
        {
            "feature_manifest": _sha(manifest_path),
            "records": _sha(records_path),
            "features": _sha(features_path),
            "geometry_manifest": geometry_manifest_sha,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-development", type=Path, required=True)
    parser.add_argument("--v2-confirmation", type=Path, required=True)
    parser.add_argument("--v2-geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = _load(args.base_development.resolve())
    failure = _load(
        args.v2_confirmation.resolve(),
        args.v2_geometry.resolve(),
    )
    base_rows, base_visual, base_geometry, base_proprio, base_hash = base
    fail_rows, fail_visual, fail_geometry, fail_proprio, fail_hash = failure
    base_scenes = {str(row["scene_cluster"]) for row in base_rows}
    fail_scenes = {str(row["scene_cluster"]) for row in fail_rows}
    base_ids = {str(row["sample_id"]) for row in base_rows}
    fail_ids = {str(row["sample_id"]) for row in fail_rows}
    if (
        base_visual.shape[1:] != fail_visual.shape[1:]
        or base_geometry.shape[1:] != fail_geometry.shape[1:]
        or base_proprio.shape[1:] != fail_proprio.shape[1:]
    ):
        raise RuntimeError("C2 v3 source feature dimensions differ")

    rows = [
        {
            **row,
            "split": "development",
            "v3_development_source": source,
        }
        for source, values in (
            ("pre_v2_scene_disjoint_development", base_rows),
            ("audited_v2_confirmation_failure", fail_rows),
        )
        for row in values
    ]
    visual = np.concatenate([base_visual, fail_visual], axis=0)
    geometry = np.concatenate([base_geometry, fail_geometry], axis=0)
    proprio = np.concatenate([base_proprio, fail_proprio], axis=0)
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
        [str(row["sample_id"]) for row in rows], dtype=str
    )

    checks = {
        "source_manifests_passed_and_hash_locked": True,
        "source_scene_clusters_disjoint": base_scenes.isdisjoint(fail_scenes),
        "source_sample_ids_disjoint": base_ids.isdisjoint(fail_ids),
        "nine_development_scene_clusters": (
            len(base_scenes | fail_scenes) == 9
        ),
        "three_development_domains": (
            {str(row["domain"]) for row in rows}
            == {"life", "production", "wild"}
        ),
        "both_conflict_directions_present": (
            {str(row["cell"]) for row in rows}
            == {"T2_vision_decisive", "T3_proprio_decisive"}
        ),
        "four_balanced_categories": (
            len(
                {
                    str(row["attribution_category"])
                    for row in rows
                }
            )
            == 4
        ),
        "all_values_finite": bool(
            np.isfinite(visual).all()
            and np.isfinite(geometry).all()
            and np.isfinite(proprio).all()
        ),
        "v2_failure_explicitly_development_only": all(
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
        visual=visual,
        geometry=geometry,
        proprio=proprio,
    )
    counts = {
        "samples": len(rows),
        "cases": len({str(row["case_id"]) for row in rows}),
        "scene_clusters": len(base_scenes | fail_scenes),
        "domains": len({str(row["domain"]) for row in rows}),
        "cases_per_direction": {
            cell: len(
                {
                    str(row["case_id"])
                    for row in rows
                    if row["cell"] == cell
                }
            )
            for cell in sorted({str(row["cell"]) for row in rows})
        },
    }
    manifest = {
        "schema_version": (
            "kinofail.realistic-c2-v3-development-features.v1"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "passed": all(checks.values()),
        "checks": checks,
        "counts": counts,
        "scene_clusters": sorted(base_scenes | fail_scenes),
        "formal_v2_outcomes_used_for_v3_development": True,
        "claim_boundary": (
            "The failed C2 v2 confirmation is retained unchanged and is "
            "used only to develop v3. No v3 confirmation scene or outcome "
            "is present in this feature pool."
        ),
        "source_sha256": {
            "base_development": base_hash,
            "audited_v2_confirmation_failure": fail_hash,
        },
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
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                "counts": counts,
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if manifest["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
