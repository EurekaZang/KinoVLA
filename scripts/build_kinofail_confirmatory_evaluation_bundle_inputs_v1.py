#!/usr/bin/env python3
"""Build observable-only feature shards and complete planned truth ledgers.

Incomplete acquisition is handled at the frozen physical group/case level.
Valid groups contribute exactly six observable samples.  Invalid planned
groups contribute a truth-key placeholder only, so attrition remains visible
to the one-shot scorer while no nonexistent feature is presented to a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


SCALE_GROUPS = 10_560
CONFLICT_GROUPS = 3_000
MAX_ATTRITION = 0.05
VIEWS = ("primary", "swap_01", "swap_02")
CATEGORY = {
    "O2_compliance": "compliant_terrain",
    "O4_tether": "adhesion",
    "O7_visual_remap": "low_friction",
    "O8_invisible_collider": "invisible_obstacle",
}


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


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(path)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_unified_shard(
    *,
    unified_dir: Path,
    records_path: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    manifest_path = unified_dir / "feature_manifest.json"
    features_path = unified_dir / "features.npz"
    manifest = _json(manifest_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("output_sha256", {}).get("features")
        != _sha256(features_path)
        or manifest.get("source_sha256", {}).get("snapshot_records")
        != _sha256(records_path)
    ):
        raise RuntimeError(f"invalid scale unified feature shard: {unified_dir}")
    rows = _jsonl(records_path)
    with np.load(features_path, allow_pickle=False) as archive:
        ids = archive["sample_ids"].astype(str)
        visual = np.asarray(archive["visual"], dtype=np.float32)
        proprio = np.asarray(archive["proprio"], dtype=np.float32)
    if (
        ids.tolist() != [str(row["sample_id"]) for row in rows]
        or visual.shape != (len(rows), 1536)
        or proprio.shape != (len(rows), 80)
        or not np.isfinite(visual).all()
        or not np.isfinite(proprio).all()
    ):
        raise RuntimeError(f"misaligned/nonfinite scale shard: {unified_dir}")
    return rows, ids, visual, proprio


def _planned_scale(
    schedule_path: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    rows = _jsonl(schedule_path)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["counterfactual_group_id"])].append(row)
    if (
        len(rows) != 2 * SCALE_GROUPS
        or len(by_group) != SCALE_GROUPS
        or any(
            len(group) != 2
            or {str(row["condition"]) for row in group}
            != {"nominal_counterfactual", "anomaly"}
            for group in by_group.values()
        )
    ):
        raise RuntimeError("scale schedule is not the frozen paired design")
    expected: dict[str, dict[str, dict[str, Any]]] = {}
    for group_id, group in by_group.items():
        samples = {}
        for episode in group:
            for view in episode["appearance_views"]:
                view_id = str(view["appearance_view_id"])
                sample_id = f"{episode['episode_id']}__{view_id}"
                samples[sample_id] = {
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "dataset": "scale",
                    "scene": str(episode["scene_cluster"]),
                    "material": str(view["material_family"]),
                    "cluster_material": str(episode["cluster_material"]),
                    "truth": str(episode["attribution_category"]),
                    "valid": True,
                    "operator": str(episode["target_operator"]),
                    "cell": "scale",
                    "condition": str(episode["condition"]),
                    "appearance_view_id": view_id,
                }
        if len(samples) != 6:
            raise RuntimeError(f"scale group does not define six samples: {group_id}")
        expected[group_id] = samples
    return expected


def _planned_conflict(
    schedule_path: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    rows = _jsonl(schedule_path)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)
    if (
        len(rows) != 6 * CONFLICT_GROUPS
        or len(by_case) != CONFLICT_GROUPS
        or any(len(group) != 6 for group in by_case.values())
    ):
        raise RuntimeError("conflict schedule is not the frozen 3,000-case design")
    expected: dict[str, dict[str, dict[str, Any]]] = {}
    for case_id, group in by_case.items():
        samples = {}
        for row in group:
            operator = str(row["candidate_operator"])
            view_id = str(row["appearance_view_id"])
            sample_id = f"{case_id}__{operator}__{view_id}"
            samples[sample_id] = {
                "sample_id": sample_id,
                "group_id": case_id,
                "dataset": "conflict",
                "scene": str(row["scene_cluster"]),
                # Replaced from the observable record for valid appearance swaps.
                "material": str(row["material_id"]),
                "cluster_material": str(row["cluster_material"]),
                "truth": CATEGORY[operator],
                "valid": True,
                "operator": operator,
                "cell": str(row["cell"]),
                "appearance_view_id": view_id,
            }
        if len(samples) != 6:
            raise RuntimeError(f"conflict case does not define six samples: {case_id}")
        expected[case_id] = samples
    return expected


def _complete_groups(
    *,
    expected: dict[str, dict[str, dict[str, Any]]],
    actual_rows: list[dict[str, Any]],
    dataset: str,
) -> tuple[set[str], list[dict[str, Any]], dict[str, int]]:
    actual_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in actual_rows:
        group_id = str(
            row["counterfactual_group_id"]
            if dataset == "scale"
            else row["case_id"]
        )
        if group_id not in expected:
            raise RuntimeError(f"unplanned {dataset} group in features: {group_id}")
        actual_by_group[group_id].append(row)
    valid_groups = {
        group_id
        for group_id, rows in actual_by_group.items()
        if len(rows) == 6
        and {str(row["sample_id"]) for row in rows}
        == set(expected[group_id])
    }
    invalid_groups = set(expected) - valid_groups
    truth: list[dict[str, Any]] = []
    for group_id in sorted(expected):
        if group_id in valid_groups:
            observed = {
                str(row["sample_id"]): row for row in actual_by_group[group_id]
            }
            for sample_id, planned in sorted(expected[group_id].items()):
                row = dict(planned)
                actual = observed[sample_id]
                if (
                    str(actual.get("scene_cluster", actual.get("scene_family")))
                    != row["scene"]
                    or str(actual["attribution_category"]) != row["truth"]
                ):
                    raise RuntimeError(
                        f"{dataset} observed metadata differs from schedule: {sample_id}"
                    )
                row["material"] = str(actual["material_family"])
                truth.append(row)
        else:
            exemplar = next(iter(expected[group_id].values()))
            truth.append(
                {
                    **exemplar,
                    "sample_id": f"invalid::{dataset}::{group_id}",
                    "valid": False,
                    "attrition_reason": "incomplete_six_sample_physical_group",
                }
            )
    counts = {
        "planned_groups": len(expected),
        "valid_groups": len(valid_groups),
        "invalid_groups": len(invalid_groups),
    }
    return valid_groups, truth, counts


def _filter_arrays(
    *,
    rows: list[dict[str, Any]],
    ids: np.ndarray,
    visual: np.ndarray,
    proprio: np.ndarray,
    valid_groups: set[str],
    dataset: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = np.asarray(
        [
            str(
                row["counterfactual_group_id"]
                if dataset == "scale"
                else row["case_id"]
            )
            in valid_groups
            for row in rows
        ],
        dtype=bool,
    )
    selected_ids = ids[keep]
    selected_visual = visual[keep]
    selected_proprio = proprio[keep]
    order = np.argsort(selected_ids, kind="stable")
    return (
        selected_ids[order],
        selected_visual[order],
        selected_proprio[order],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-schedule", type=Path, required=True)
    parser.add_argument("--scale-snapshot-dir", type=Path, action="append", required=True)
    parser.add_argument("--scale-unified-dir", type=Path, action="append", required=True)
    parser.add_argument("--conflict-schedule", type=Path, required=True)
    parser.add_argument("--conflict-feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    scale_snapshot_dirs = [path.resolve() for path in args.scale_snapshot_dir]
    scale_unified_dirs = [path.resolve() for path in args.scale_unified_dir]
    if len(scale_snapshot_dirs) != len(scale_unified_dirs):
        raise RuntimeError("scale snapshot/unified directories must be paired")

    scale_rows: list[dict[str, Any]] = []
    scale_ids: list[np.ndarray] = []
    scale_visual: list[np.ndarray] = []
    scale_proprio: list[np.ndarray] = []
    scale_sources = {}
    for snapshot_dir, unified_dir in zip(
        scale_snapshot_dirs, scale_unified_dirs, strict=True
    ):
        records_path = snapshot_dir / "snapshot_records.jsonl"
        rows, ids, visual, proprio = _load_unified_shard(
            unified_dir=unified_dir,
            records_path=records_path,
        )
        scale_rows.extend(rows)
        scale_ids.append(ids)
        scale_visual.append(visual)
        scale_proprio.append(proprio)
        scale_sources[str(unified_dir)] = {
            "records": _sha256(records_path),
            "features": _sha256(unified_dir / "features.npz"),
            "manifest": _sha256(unified_dir / "feature_manifest.json"),
        }
    if not scale_rows:
        raise RuntimeError("no scale feature shards")
    all_scale_ids = np.concatenate(scale_ids)
    all_scale_visual = np.concatenate(scale_visual)
    all_scale_proprio = np.concatenate(scale_proprio)
    if len(set(all_scale_ids.tolist())) != len(all_scale_ids):
        raise RuntimeError("duplicate scale sample IDs")

    conflict_dir = args.conflict_feature_dir.resolve()
    conflict_records_path = conflict_dir / "records.jsonl"
    conflict_features_path = conflict_dir / "features.npz"
    conflict_manifest_path = conflict_dir / "feature_manifest.json"
    conflict_manifest = _json(conflict_manifest_path)
    if (
        conflict_manifest.get("passed") is not True
        or conflict_manifest.get("output_sha256", {}).get("records")
        != _sha256(conflict_records_path)
        or conflict_manifest.get("output_sha256", {}).get("features")
        != _sha256(conflict_features_path)
    ):
        raise RuntimeError("invalid conflict feature bundle")
    conflict_rows = _jsonl(conflict_records_path)
    with np.load(conflict_features_path, allow_pickle=False) as archive:
        conflict_ids = archive["sample_ids"].astype(str)
        conflict_visual = np.asarray(archive["visual"], dtype=np.float32)
        conflict_proprio = np.asarray(archive["proprio"], dtype=np.float32)
    if (
        conflict_ids.tolist()
        != [str(row["sample_id"]) for row in conflict_rows]
        or conflict_visual.shape != (len(conflict_rows), 1536)
        or conflict_proprio.shape != (len(conflict_rows), 80)
    ):
        raise RuntimeError("misaligned conflict feature bundle")

    scale_expected = _planned_scale(args.scale_schedule.resolve())
    conflict_expected = _planned_conflict(args.conflict_schedule.resolve())
    valid_scale, scale_truth, scale_counts = _complete_groups(
        expected=scale_expected,
        actual_rows=scale_rows,
        dataset="scale",
    )
    valid_conflict, conflict_truth, conflict_counts = _complete_groups(
        expected=conflict_expected,
        actual_rows=conflict_rows,
        dataset="conflict",
    )
    for counts in (scale_counts, conflict_counts):
        if counts["invalid_groups"] / counts["planned_groups"] > MAX_ATTRITION:
            raise RuntimeError(f"frozen five-percent attrition gate failed: {counts}")

    scale_arrays = _filter_arrays(
        rows=scale_rows,
        ids=all_scale_ids,
        visual=all_scale_visual,
        proprio=all_scale_proprio,
        valid_groups=valid_scale,
        dataset="scale",
    )
    conflict_arrays = _filter_arrays(
        rows=conflict_rows,
        ids=conflict_ids,
        visual=conflict_visual,
        proprio=conflict_proprio,
        valid_groups=valid_conflict,
        dataset="conflict",
    )
    if (
        len(scale_arrays[0]) != 6 * len(valid_scale)
        or len(conflict_arrays[0]) != 6 * len(valid_conflict)
    ):
        raise RuntimeError("filtered feature count violates the six-sample group rule")

    output.mkdir(parents=True, exist_ok=False)
    scale_features_out = output / "scale_features.npz"
    conflict_features_out = output / "conflict_features.npz"
    scale_truth_out = output / "scale_truth.jsonl"
    conflict_truth_out = output / "conflict_truth.jsonl"
    np.savez_compressed(
        scale_features_out,
        sample_ids=scale_arrays[0],
        visual=scale_arrays[1],
        proprio=scale_arrays[2],
    )
    np.savez_compressed(
        conflict_features_out,
        sample_ids=conflict_arrays[0],
        visual=conflict_arrays[1],
        proprio=conflict_arrays[2],
    )
    _write_jsonl(scale_truth_out, scale_truth)
    _write_jsonl(conflict_truth_out, conflict_truth)
    audit = {
        "schema_version": "kinofail.unified-confirmatory-evaluation-inputs.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "case_level_complete_case_rule": True,
        "maximum_attrition_rate": MAX_ATTRITION,
        "counts": {
            "scale": scale_counts,
            "conflict": conflict_counts,
        },
        "source_sha256": {
            "scale_schedule": _sha256(args.scale_schedule.resolve()),
            "conflict_schedule": _sha256(args.conflict_schedule.resolve()),
            "scale_shards": scale_sources,
            "conflict_records": _sha256(conflict_records_path),
            "conflict_features": _sha256(conflict_features_path),
            "conflict_manifest": _sha256(conflict_manifest_path),
        },
        "output_sha256": {
            "scale_features": _sha256(scale_features_out),
            "conflict_features": _sha256(conflict_features_out),
            "scale_truth": _sha256(scale_truth_out),
            "conflict_truth": _sha256(conflict_truth_out),
        },
    }
    _write_json(output / "audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
