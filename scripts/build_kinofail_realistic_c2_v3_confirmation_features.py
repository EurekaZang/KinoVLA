#!/usr/bin/env python3
"""Build the fresh C2 v3 feature and generic-HOG confirmation battery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from extract_kinofail_realistic_c2_geometry_v2 import _hog_summary
from kino_vla.data.runtime_manifest import schedule_record_sha256
from kino_vla.eval.realistic_multimodal import (
    proprio_summary,
    temporal_clip_summary,
)
from kino_vla.map.clip_appearance import ClipAppearanceEncoder


ROOT = Path(__file__).resolve().parents[1]
VIEWS = ("primary", "swap_01", "swap_02")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
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


def _rgb_sequence(
    episode_dir: Path,
    manifest: dict[str, Any],
    view: str,
) -> tuple[np.ndarray, list[str]]:
    entries = manifest["artifacts"]["rgb_views"][view]
    if len(entries) < 5:
        raise RuntimeError(
            f"incomplete C2 v3 T3 RGB sequence: {episode_dir} {view}"
        )
    selected = entries[-5:]
    frames = []
    paths = []
    for entry in selected:
        relative = str(entry["path"])
        path = episode_dir / relative
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        paths.append(relative)
    return np.stack(frames).astype(np.uint8), paths


def _validated_anomaly(
    schedule_by_group: dict[str, list[dict[str, Any]]],
    group_id: str,
    corpus: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, np.ndarray]:
    candidates = [
        row
        for row in schedule_by_group[group_id]
        if row["condition"] == "anomaly"
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"missing anomaly schedule row: {group_id}")
    schedule = candidates[0]
    episode_dir = corpus / Path(
        schedule["required_outputs"]["episode_manifest"]
    ).parent
    manifest_path = episode_dir / "manifest.json"
    manifest = _json(manifest_path)
    if (
        manifest.get("artifact_state") != "validated"
        or manifest.get("evaluation_eligible") is not True
        or manifest.get("episode_id") != schedule["episode_id"]
        or manifest.get("counterfactual_group_id") != group_id
        or manifest.get("schedule_record_sha256")
        != schedule_record_sha256(schedule)
        or manifest.get("operator_readback", {}).get("qa_passed")
        is not True
        or manifest.get("runtime_validation", {}).get("passed")
        is not True
    ):
        raise RuntimeError(
            f"invalid C2 v3 T3 runtime episode: {group_id}"
        )
    proprio_path = episode_dir / "proprio.npz"
    if _sha(proprio_path) != manifest["artifacts"]["proprio"]["sha256"]:
        raise RuntimeError(
            f"C2 v3 T3 proprio hash mismatch: {group_id}"
        )
    archive = np.load(proprio_path, allow_pickle=False)
    proprio = np.asarray(archive["features"], dtype=np.float32)
    return schedule, manifest, episode_dir, proprio


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c1-features", type=Path, required=True)
    parser.add_argument("--c1-corpus", type=Path, required=True)
    parser.add_argument("--t3-design", type=Path, required=True)
    parser.add_argument("--t3-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    c1_features = args.c1_features.resolve()
    c1_corpus = args.c1_corpus.resolve()
    t3_design = args.t3_design.resolve()
    t3_corpus = args.t3_corpus.resolve()
    output = args.output.resolve()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    c1_manifest_path = c1_features / "feature_manifest.json"
    c1_records_path = c1_features / "records.jsonl"
    c1_features_path = c1_features / "features.npz"
    c1_manifest = _json(c1_manifest_path)
    if (
        c1_manifest.get("passed") is not True
        or _sha(c1_records_path)
        != c1_manifest["output_sha256"]["records"]
        or _sha(c1_features_path)
        != c1_manifest["output_sha256"]["features"]
    ):
        raise RuntimeError("invalid C2 v3 T2 feature source")
    c1_rows = _jsonl(c1_records_path)
    c1_archive = np.load(c1_features_path, allow_pickle=False)
    if c1_archive["sample_ids"].astype(str).tolist() != [
        str(row["sample_id"]) for row in c1_rows
    ]:
        raise RuntimeError("misaligned C2 v3 T2 features")
    c1_visual = np.asarray(
        c1_archive["visual"], dtype=np.float32
    )
    c1_proprio = np.asarray(
        c1_archive["proprio"], dtype=np.float32
    )
    c1_geometry = []
    for row in c1_rows:
        frames = []
        for relative in row["rgb_paths"]:
            path = (
                c1_corpus
                / str(row["case_id"])
                / str(relative)
            )
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        rgb = np.stack(frames).astype(np.uint8)
        c1_geometry.append(
            _hog_summary(
                rgb,
                int(row["rgb_crop"]["pixel_y_start"]),
            )
        )
    c1_geometry_array = np.stack(c1_geometry).astype(np.float32)

    t3_schedule_path = t3_design / "schedule.jsonl"
    t3_cases_path = t3_design / "case_schedule.jsonl"
    t3_audit_path = t3_design / "audit.json"
    t3_audit = _json(t3_audit_path)
    if (
        t3_audit.get("passed") is not True
        or _sha(t3_schedule_path) != t3_audit["schedule_sha256"]
        or _sha(t3_cases_path)
        != t3_audit["case_schedule_sha256"]
    ):
        raise RuntimeError("invalid C2 v3 T3 design")
    t3_schedule = _jsonl(t3_schedule_path)
    t3_cases = _jsonl(t3_cases_path)
    schedule_by_group: dict[str, list[dict[str, Any]]] = {}
    for row in t3_schedule:
        schedule_by_group.setdefault(
            str(row["counterfactual_group_id"]), []
        ).append(row)

    unique_sequences: list[np.ndarray] = []
    unique_geometry: list[np.ndarray] = []
    t3_rows: list[dict[str, Any]] = []
    t3_proprio: list[np.ndarray] = []
    visual_index: dict[tuple[str, str], int] = {}
    runtime_manifest_hashes: set[str] = set()
    for case in t3_cases:
        o7 = _validated_anomaly(
            schedule_by_group,
            str(case["o7_source_physics_group_id"]),
            t3_corpus,
        )
        o8 = _validated_anomaly(
            schedule_by_group,
            str(case["o8_source_physics_group_id"]),
            t3_corpus,
        )
        o7_schedule, o7_manifest, o7_dir, o7_proprio = o7
        o8_schedule, o8_manifest, o8_dir, o8_proprio = o8
        runtime_manifest_hashes.update(
            {
                _sha(o7_dir / "manifest.json"),
                _sha(o8_dir / "manifest.json"),
            }
        )
        for view in VIEWS:
            rgb, rgb_paths = _rgb_sequence(
                o7_dir, o7_manifest, view
            )
            crop_start = int(np.floor(0.45 * rgb.shape[1]))
            key = (str(case["case_id"]), view)
            visual_index[key] = len(unique_sequences)
            unique_sequences.append(
                np.ascontiguousarray(
                    rgb[:, crop_start:, :, :]
                )
            )
            unique_geometry.append(
                _hog_summary(rgb, crop_start)
            )
            shared_rgb_sha = _array_sha(rgb)
            for schedule, manifest, episode_dir, proprio, category in (
                (
                    o7_schedule,
                    o7_manifest,
                    o7_dir,
                    o7_proprio,
                    "low_friction",
                ),
                (
                    o8_schedule,
                    o8_manifest,
                    o8_dir,
                    o8_proprio,
                    "invisible_obstacle",
                ),
            ):
                operator = str(schedule["target_operator"])
                t3_proprio.append(proprio_summary(proprio))
                t3_rows.append(
                    {
                        "sample_id": (
                            f"{case['case_id']}__{operator}__{view}"
                        ),
                        "case_id": str(case["case_id"]),
                        "cell": "T3_proprio_decisive",
                        "scene_cluster": str(
                            case["scene_cluster"]
                        ),
                        "domain": str(case["domain"]),
                        "split": "test",
                        "profile_index": int(
                            case["profile_index"]
                        ),
                        "severity": str(case["severity"]),
                        "target_operator": operator,
                        "attribution_category": category,
                        "appearance_view_id": view,
                        "material_family": next(
                            value["material_family"]
                            for value in schedule[
                                "appearance_views"
                            ]
                            if value["appearance_view_id"] == view
                        ),
                        "shared_rgb_sha256": shared_rgb_sha,
                        "visual_source_episode_id": str(
                            o7_manifest["episode_id"]
                        ),
                        "proprio_source_episode_id": str(
                            manifest["episode_id"]
                        ),
                        "source_physics_group_id": str(
                            schedule["counterfactual_group_id"]
                        ),
                        "rgb_paths": [
                            str(
                                (
                                    Path(
                                        o7_schedule[
                                            "required_outputs"
                                        ]["episode_manifest"]
                                    ).parent
                                    / relative
                                )
                            )
                            for relative in rgb_paths
                        ],
                        "rgb_crop": {
                            "x_fraction": [0.0, 1.0],
                            "y_fraction": [0.45, 1.0],
                            "pixel_y_start": crop_start,
                        },
                    }
                )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    encoder = ClipAppearanceEncoder()
    frame_embeddings: np.ndarray | None = None
    images: list[np.ndarray] = []
    locations: list[tuple[int, int]] = []

    def flush() -> None:
        nonlocal frame_embeddings, images, locations
        if not images:
            return
        embedded = encoder.embed_batch(images).astype(np.float32)
        if frame_embeddings is None:
            frame_embeddings = np.empty(
                (
                    len(unique_sequences),
                    5,
                    embedded.shape[1],
                ),
                dtype=np.float32,
            )
        for location, value in zip(
            locations, embedded, strict=True
        ):
            frame_embeddings[location] = value
        images = []
        locations = []

    for sequence_index, sequence in enumerate(unique_sequences):
        for frame_index, frame in enumerate(sequence):
            images.append(frame)
            locations.append((sequence_index, frame_index))
            if len(images) >= args.batch_size:
                flush()
    flush()
    if frame_embeddings is None:
        raise RuntimeError("C2 v3 T3 CLIP extraction is empty")
    unique_visual = np.stack(
        [
            temporal_clip_summary(value)
            for value in frame_embeddings
        ]
    ).astype(np.float32)
    unique_geometry_array = np.stack(
        unique_geometry
    ).astype(np.float32)
    t3_visual = np.stack(
        [
            unique_visual[
                visual_index[
                    (
                        str(row["case_id"]),
                        str(row["appearance_view_id"]),
                    )
                ]
            ]
            for row in t3_rows
        ]
    )
    t3_geometry = np.stack(
        [
            unique_geometry_array[
                visual_index[
                    (
                        str(row["case_id"]),
                        str(row["appearance_view_id"]),
                    )
                ]
            ]
            for row in t3_rows
        ]
    )
    t3_proprio_array = np.stack(t3_proprio).astype(np.float32)

    rows = [
        {
            **row,
            "cell": "T2_vision_decisive",
            "split": "test",
            "shared_rgb_sha256": None,
        }
        for row in c1_rows
    ] + t3_rows
    visual = np.concatenate([c1_visual, t3_visual], axis=0)
    geometry = np.concatenate(
        [c1_geometry_array, t3_geometry], axis=0
    )
    proprio = np.concatenate(
        [c1_proprio, t3_proprio_array], axis=0
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
        [str(row["sample_id"]) for row in rows], dtype=str
    )

    by_t3_view: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        if row["cell"] == "T3_proprio_decisive":
            by_t3_view.setdefault(
                (
                    str(row["case_id"]),
                    str(row["appearance_view_id"]),
                ),
                [],
            ).append(index)
    checks = {
        "expected_180_samples": len(rows) == 180,
        "thirty_cases_fifteen_per_direction": (
            len({str(row["case_id"]) for row in rows}) == 30
            and all(
                len(
                    {
                        str(row["case_id"])
                        for row in rows
                        if row["cell"] == cell
                    }
                )
                == 15
                for cell in (
                    "T2_vision_decisive",
                    "T3_proprio_decisive",
                )
            )
        ),
        "four_balanced_categories": all(
            sum(
                row["attribution_category"] == category
                for row in rows
            )
            == 45
            for category in (
                "adhesion",
                "compliant_terrain",
                "invisible_obstacle",
                "low_friction",
            )
        ),
        "three_fresh_scenes_and_domains": (
            len({str(row["scene_cluster"]) for row in rows})
            == 3
            and len({str(row["domain"]) for row in rows}) == 3
        ),
        "t3_visual_features_byte_identical": all(
            len(indices) == 2
            and np.array_equal(
                visual[indices[0]], visual[indices[1]]
            )
            and np.array_equal(
                geometry[indices[0]], geometry[indices[1]]
            )
            for indices in by_t3_view.values()
        ),
        "all_values_finite": bool(
            np.isfinite(visual).all()
            and np.isfinite(geometry).all()
            and np.isfinite(proprio).all()
        ),
        "runtime_manifests_validated": (
            len(runtime_manifest_hashes) == 30
        ),
    }
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
        proprio=proprio,
    )
    geometry_path = output / "geometry.npz"
    np.savez_compressed(
        geometry_path,
        sample_ids=sample_ids,
        geometry=geometry,
    )
    feature_manifest = {
        "schema_version": (
            "kinofail.realistic-c2-v3-confirmation-features.v1"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "samples": len(rows),
            "cases": len(
                {str(row["case_id"]) for row in rows}
            ),
            "cases_per_direction": 15,
            "classes": 4,
            "scene_clusters": 3,
            "domains": 3,
        },
        "input_contract": {
            "T2": (
                "identical proprioception across O2/O4 causes; "
                "cause geometry changes in RTX"
            ),
            "T3": (
                "O7 RTX sequence copied byte-for-byte across O7/O8 "
                "candidates; actual anomaly proprioception differs"
            ),
            "forbidden_metadata_features": True,
        },
        "source_sha256": {
            "c1_feature_manifest": _sha(c1_manifest_path),
            "t3_schedule": _sha(t3_schedule_path),
            "t3_case_schedule": _sha(t3_cases_path),
            "t3_design_audit": _sha(t3_audit_path),
        },
        "output_sha256": {
            "records": _sha(records_path),
            "features": _sha(features_path),
        },
    }
    feature_manifest_path = output / "feature_manifest.json"
    feature_manifest_path.write_text(
        json.dumps(
            feature_manifest, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    geometry_checks = {
        "sample_order_matches_features": True,
        "all_values_finite": bool(np.isfinite(geometry).all()),
        "expected_sample_count": len(geometry) == len(rows),
        "t3_visual_identity_preserved": checks[
            "t3_visual_features_byte_identical"
        ],
    }
    geometry_manifest = {
        "schema_version": (
            "kinofail.realistic-c2-geometry-features.v3"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(geometry_checks.values()),
        "checks": geometry_checks,
        "descriptor": {
            "name": "grayscale_HOG_temporal_summary",
            "roi": "full width, RGB rows 45%-100%",
            "resize": [80, 48],
            "temporal_summary": [
                "mean",
                "last",
                "last-minus-first",
            ],
            "uses_labels_or_material_metadata": False,
        },
        "counts": {
            "samples": len(rows),
            "dimension": int(geometry.shape[1]),
            "unique_t3_visual_sequences": len(
                unique_sequences
            ),
        },
        "source_sha256": {
            "feature_manifest": _sha(feature_manifest_path),
            "records": _sha(records_path),
        },
        "output": str(geometry_path.relative_to(ROOT)),
        "output_sha256": _sha(geometry_path),
    }
    geometry_manifest_path = output / "geometry_manifest.json"
    geometry_manifest_path.write_text(
        json.dumps(
            geometry_manifest, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": (
                    feature_manifest["passed"]
                    and geometry_manifest["passed"]
                ),
                "counts": feature_manifest["counts"],
                "feature_manifest": str(feature_manifest_path),
                "geometry_manifest": str(
                    geometry_manifest_path
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return (
        0
        if (
            feature_manifest["passed"]
            and geometry_manifest["passed"]
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
