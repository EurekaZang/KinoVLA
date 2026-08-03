#!/usr/bin/env python3
"""Build the matched two-direction realistic C2 conflict battery.

T2 reuses the prospective C1 O2/O4 battery: proprioception is byte-identical
and physical appearance differs.  T3 pairs actual O7/O8 physical rollouts but
assigns the same RTX sequence to both labels, making vision byte-identical.
All samples use the same frozen Go2-front ground ROI and CLIP encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.realistic_multimodal import (  # noqa: E402
    proprio_summary,
    temporal_clip_summary,
)
from kino_vla.map.clip_appearance import ClipAppearanceEncoder  # noqa: E402


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
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
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


def _score(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _id(prefix: str, *parts: str) -> str:
    return f"{prefix}_{_score(*parts)[:20]}"


def _select_groups(
    records: list[dict[str, Any]], scenes: list[str]
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(
        dict
    )
    for row in records:
        if (
            row["condition"] == "anomaly"
            and row["target_operator"]
            in {"O7_visual_remap", "O8_invisible_collider"}
        ):
            key = (
                str(row["scene_family"]),
                str(row["target_operator"]),
                str(row["counterfactual_group_id"]),
            )
            by_key[key][str(row["appearance_intervention_id"])] = row
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (scene, operator, group_id), views in by_key.items():
        if set(views) != set(VIEWS):
            raise RuntimeError(f"incomplete source appearance group: {group_id}")
        primary = views["primary"]
        groups[(scene, operator)].append(
            {
                "group_id": group_id,
                "severity": str(primary["severity_id"]),
                "views": views,
            }
        )

    selected: list[dict[str, Any]] = []
    for scene in scenes:
        chosen: dict[str, list[dict[str, Any]]] = {}
        for operator in ("O7_visual_remap", "O8_invisible_collider"):
            values = groups[(scene, operator)]
            moderate = sorted(
                [value for value in values if value["severity"] == "moderate"],
                key=lambda value: _score(
                    "c2_t3_formal_v1", scene, operator, value["group_id"]
                ),
            )
            severe = sorted(
                [value for value in values if value["severity"] == "severe"],
                key=lambda value: _score(
                    "c2_t3_formal_v1", scene, operator, value["group_id"]
                ),
            )
            if len(moderate) < 2 or len(severe) < 1:
                raise RuntimeError(f"insufficient balanced T3 sources: {scene} {operator}")
            chosen[operator] = [moderate[0], moderate[1], severe[0]]
        for profile_index, severity in enumerate(
            ("moderate", "moderate", "severe")
        ):
            left = chosen["O7_visual_remap"][profile_index]
            right = chosen["O8_invisible_collider"][profile_index]
            if left["severity"] != severity or right["severity"] != severity:
                raise RuntimeError("T3 severity matching failed")
            selected.append(
                {
                    "case_id": _id(
                        "c2t3",
                        "c2_t3_formal_v1",
                        scene,
                        left["group_id"],
                        right["group_id"],
                    ),
                    "scene_cluster": scene,
                    "profile_index": profile_index,
                    "severity": severity,
                    "o7": left,
                    "o8": right,
                }
            )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c1-features", type=Path, required=True)
    parser.add_argument("--scale-snapshots", type=Path, required=True)
    parser.add_argument("--scene-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    c1_dir = args.c1_features.resolve()
    scale_dir = args.scale_snapshots.resolve()
    registry_path = args.scene_registry.resolve()
    output = args.output.resolve()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    c1_manifest_path = c1_dir / "feature_manifest.json"
    c1_manifest = _json(c1_manifest_path)
    c1_records_path = c1_dir / "records.jsonl"
    c1_features_path = c1_dir / "features.npz"
    if c1_manifest.get("passed") is not True:
        raise RuntimeError("C2 refuses an incomplete C1 feature cache")
    if (
        _sha(c1_records_path) != c1_manifest["output_sha256"]["records"]
        or _sha(c1_features_path) != c1_manifest["output_sha256"]["features"]
    ):
        raise RuntimeError("C1 feature provenance mismatch")
    c1_rows = _jsonl(c1_records_path)
    c1_archive = np.load(c1_features_path, allow_pickle=False)
    if c1_archive["sample_ids"].astype(str).tolist() != [
        str(row["sample_id"]) for row in c1_rows
    ]:
        raise RuntimeError("C1 feature rows are misaligned")
    c1_visual = np.asarray(c1_archive["visual"], dtype=np.float32)
    c1_proprio = np.asarray(c1_archive["proprio"], dtype=np.float32)

    scale_records_path = scale_dir / "snapshot_records.jsonl"
    scale_arrays_path = scale_dir / "snapshots.npz"
    scale_audit_path = scale_dir / "extraction_audit.json"
    scale_audit = _json(scale_audit_path)
    if scale_audit.get("passed") is not True:
        raise RuntimeError("C2 refuses a failed scale snapshot audit")
    scale_rows = _jsonl(scale_records_path)
    registry = _json(registry_path)
    scene_info = {
        str(row["scene_id"]): {
            "split": str(row["split"]),
            "domain": str(row["domain"]),
        }
        for row in registry["scenes"]
    }
    scenes = sorted(scene_info)
    t3_cases = _select_groups(scale_rows, scenes)
    if len(t3_cases) != 27:
        raise RuntimeError("C2 T3 design must contain 27 matched cases")

    scale_archive = np.load(scale_arrays_path, allow_pickle=False)
    unique_sequences: list[np.ndarray] = []
    sequence_keys: list[tuple[str, str]] = []
    t3_rows: list[dict[str, Any]] = []
    t3_proprio: list[np.ndarray] = []
    pair_schedule: list[dict[str, Any]] = []
    visual_index_by_key: dict[tuple[str, str], int] = {}
    for case in t3_cases:
        scene = str(case["scene_cluster"])
        info = scene_info[scene]
        for view in VIEWS:
            visual_source = case["o7"]["views"][view]
            rgb = np.asarray(
                scale_archive[f"{visual_source['sample_id']}__rgb"],
                dtype=np.uint8,
            )
            crop_start = int(np.floor(0.45 * rgb.shape[1]))
            cropped = np.ascontiguousarray(rgb[:, crop_start:, :, :])
            visual_key = (str(case["case_id"]), view)
            visual_index_by_key[visual_key] = len(unique_sequences)
            unique_sequences.append(cropped)
            sequence_keys.append(visual_key)
            shared_rgb_sha = _array_sha(rgb)
            for operator, source_key, category in (
                ("O7_visual_remap", "o7", "low_friction"),
                ("O8_invisible_collider", "o8", "invisible_obstacle"),
            ):
                source = case[source_key]["views"][view]
                proprio = np.asarray(
                    scale_archive[f"{source['sample_id']}__proprio"],
                    dtype=np.float32,
                )
                t3_proprio.append(proprio_summary(proprio))
                t3_rows.append(
                    {
                        "sample_id": (
                            f"{case['case_id']}__{operator}__{view}"
                        ),
                        "case_id": case["case_id"],
                        "cell": "T3_proprio_decisive",
                        "scene_cluster": scene,
                        "domain": info["domain"],
                        "split": info["split"],
                        "profile_index": case["profile_index"],
                        "severity": case["severity"],
                        "target_operator": operator,
                        "attribution_category": category,
                        "appearance_view_id": view,
                        "material_family": visual_source["material_family"],
                        "shared_rgb_sha256": shared_rgb_sha,
                        "visual_source_sample_id": visual_source["sample_id"],
                        "proprio_source_sample_id": source["sample_id"],
                        "source_physics_group_id": source[
                            "counterfactual_group_id"
                        ],
                        "rgb_crop": {
                            "x_fraction": [0.0, 1.0],
                            "y_fraction": [0.45, 1.0],
                            "pixel_y_start": crop_start,
                        },
                    }
                )
        pair_schedule.append(
            {
                "case_id": case["case_id"],
                "scene_cluster": scene,
                "domain": info["domain"],
                "split": info["split"],
                "profile_index": case["profile_index"],
                "severity": case["severity"],
                "o7_source_physics_group_id": case["o7"]["group_id"],
                "o8_source_physics_group_id": case["o8"]["group_id"],
                "shared_visual_source": "O7 RTX sequence copied byte-for-byte",
                "visual_intervention_advances_physics": False,
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
                (len(unique_sequences), 5, embedded.shape[1]), dtype=np.float32
            )
        for location, value in zip(locations, embedded, strict=True):
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
        raise RuntimeError("C2 T3 CLIP extraction produced no features")
    unique_visual = np.stack(
        [temporal_clip_summary(value) for value in frame_embeddings]
    ).astype(np.float32)
    t3_visual = np.stack(
        [
            unique_visual[
                visual_index_by_key[
                    (str(row["case_id"]), str(row["appearance_view_id"]))
                ]
            ]
            for row in t3_rows
        ]
    )
    t3_proprio_array = np.stack(t3_proprio).astype(np.float32)

    combined_rows: list[dict[str, Any]] = []
    combined_visual: list[np.ndarray] = []
    combined_proprio: list[np.ndarray] = []
    for index, row in enumerate(c1_rows):
        combined_rows.append(
            {
                **row,
                "cell": "T2_vision_decisive",
                "shared_rgb_sha256": None,
            }
        )
        combined_visual.append(c1_visual[index])
        combined_proprio.append(c1_proprio[index])
    for index, row in enumerate(t3_rows):
        combined_rows.append(row)
        combined_visual.append(t3_visual[index])
        combined_proprio.append(t3_proprio_array[index])
    order = sorted(
        range(len(combined_rows)),
        key=lambda index: (
            str(combined_rows[index]["split"]),
            str(combined_rows[index]["scene_cluster"]),
            str(combined_rows[index]["cell"]),
            str(combined_rows[index]["case_id"]),
            str(combined_rows[index]["target_operator"]),
            str(combined_rows[index]["appearance_view_id"]),
        ),
    )
    combined_rows = [combined_rows[index] for index in order]
    visual = np.stack([combined_visual[index] for index in order])
    proprio = np.stack([combined_proprio[index] for index in order])

    t3_hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in t3_rows:
        t3_hashes[(row["case_id"], row["appearance_view_id"])].add(
            row["shared_rgb_sha256"]
        )
    checks = {
        "c1_feature_audit_passed": c1_manifest["passed"] is True,
        "27_t2_cases": len({row["case_id"] for row in c1_rows}) == 27,
        "27_t3_cases": len({row["case_id"] for row in t3_rows}) == 27,
        "nine_scene_clusters_each_direction": all(
            len(
                {
                    row["scene_cluster"]
                    for row in combined_rows
                    if row["cell"] == cell
                }
            )
            == 9
            for cell in ("T2_vision_decisive", "T3_proprio_decisive")
        ),
        "t3_rgb_byte_identical_across_labels": all(
            len(values) == 1 for values in t3_hashes.values()
        ),
        "t2_proprio_byte_identical_across_labels": bool(
            c1_manifest["checks"]["proprio_hash_identical_within_every_case"]
        ),
        "same_feature_pipeline_both_directions": (
            c1_manifest["encoder"] == encoder.model_id
            and c1_manifest["input_contract"]["crop_xy_fraction"]
            == [0.0, 0.45, 1.0, 1.0]
        ),
        "balanced_four_classes": len(
            {
                category: sum(
                    row["attribution_category"] == category
                    for row in combined_rows
                )
                for category in {
                    row["attribution_category"] for row in combined_rows
                }
            }
        )
        == 4
        and len(
            {
                sum(
                    row["attribution_category"] == category
                    for row in combined_rows
                )
                for category in {
                    row["attribution_category"] for row in combined_rows
                }
            }
        )
        == 1,
    }
    if not all(checks.values()):
        raise RuntimeError(f"C2 bidirectional feature audit failed: {checks}")

    output.mkdir(parents=True, exist_ok=True)
    schedule_path = output / "t3_pair_schedule.jsonl"
    schedule_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in pair_schedule),
        encoding="utf-8",
    )
    records_path = output / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in combined_rows),
        encoding="utf-8",
    )
    features_path = output / "features.npz"
    np.savez_compressed(
        features_path,
        sample_ids=np.asarray([row["sample_id"] for row in combined_rows]),
        visual=visual.astype(np.float32),
        proprio=proprio.astype(np.float32),
    )
    manifest = {
        "schema_version": "kinofail.realistic-c2-bidirectional-features.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "passed": all(checks.values()),
        "checks": checks,
        "encoder": encoder.model_id,
        "input_contract": {
            "camera": "Go2 front RTX camera",
            "crop_xy_fraction": [0.0, 0.45, 1.0, 1.0],
            "frames": 5,
            "temporal_visual_summary": (
                "mean + last + last-minus-first CLIP embedding"
            ),
            "temporal_proprio_summary": (
                "mean/std/min/max/q25/q75/first/last/delta/slope"
            ),
        },
        "counts": {
            "samples": len(combined_rows),
            "cases": len({row["case_id"] for row in combined_rows}),
            "cases_per_direction": 27,
            "scene_clusters": 9,
            "domains": 3,
            "classes": 4,
        },
        "source_sha256": {
            "c1_feature_manifest": _sha(c1_manifest_path),
            "c1_records": _sha(c1_records_path),
            "c1_features": _sha(c1_features_path),
            "scale_snapshot_records": _sha(scale_records_path),
            "scale_snapshots": _sha(scale_arrays_path),
            "scale_extraction_audit": _sha(scale_audit_path),
            "scene_registry": _sha(registry_path),
        },
        "output_sha256": {
            "t3_pair_schedule": _sha(schedule_path),
            "records": _sha(records_path),
            "features": _sha(features_path),
        },
        "claim_boundary": (
            "The T3 arm is an observable counterfactual: actual O7 and O8 "
            "proprioceptive rollouts are evaluated under a byte-identical RTX "
            "sequence. It establishes visual insufficiency under this matched "
            "condition, not that every natural O7/O8 image is identical."
        ),
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
                **manifest["counts"],
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
