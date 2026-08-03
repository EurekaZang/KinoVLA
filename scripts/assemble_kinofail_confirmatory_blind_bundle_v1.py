#!/usr/bin/env python3
"""Assemble write-once blind features and a physically grouped truth key.

Feature shards contain observable arrays only.  Metadata ledgers are consumed
in a separate path and written only to ``truth_key.jsonl``; the model-facing
NPZ has exactly ``sample_ids``, ``visual``, and ``proprio``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = {
    "adhesion",
    "compliant_terrain",
    "effort_decay",
    "external_push",
    "high_centering",
    "invisible_obstacle",
    "low_friction",
    "nominal",
    "obs_bias",
    "overload",
    "region_collapse",
}
PLANNED_GROUPS = {"scale": 10_560, "conflict": 3_000}


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
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(path)
    return rows


def _validate_f1(path: Path) -> dict[str, Any]:
    manifest = _json(path)
    sidecar = path.with_name("seal_manifest.sha256")
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").split()[0] != _sha256(path)
        or manifest.get("status")
        != "sealed_before_anomaly_collection_and_model_inference"
        or manifest.get("confirmatory") is not True
    ):
        raise RuntimeError("invalid F1 seal")
    return manifest


def _feature_shards(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids: list[np.ndarray] = []
    visual: list[np.ndarray] = []
    proprio: list[np.ndarray] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"sample_ids", "visual", "proprio"}:
                raise RuntimeError(f"unexpected feature keys in {path}: {archive.files}")
            shard_ids = archive["sample_ids"].astype(str)
            shard_visual = np.asarray(archive["visual"], dtype=np.float32)
            shard_proprio = np.asarray(archive["proprio"], dtype=np.float32)
        if (
            shard_ids.ndim != 1
            or shard_visual.shape != (len(shard_ids), 1536)
            or shard_proprio.shape != (len(shard_ids), 80)
            or not np.isfinite(shard_visual).all()
            or not np.isfinite(shard_proprio).all()
        ):
            raise RuntimeError(f"invalid observable feature shard: {path}")
        ids.append(shard_ids)
        visual.append(shard_visual)
        proprio.append(shard_proprio)
    if not ids:
        raise RuntimeError("no feature shards supplied")
    return (
        np.concatenate(ids),
        np.concatenate(visual),
        np.concatenate(proprio),
    )


def _truth_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for source in _jsonl(path):
            row = dict(source)
            required = {
                "sample_id",
                "group_id",
                "dataset",
                "scene",
                "material",
                "cluster_material",
                "truth",
                "valid",
            }
            missing = sorted(required - set(row))
            if missing:
                raise RuntimeError(f"{path}: truth row missing {missing}")
            row["sample_id"] = str(row["sample_id"])
            row["group_id"] = str(row["group_id"])
            row["dataset"] = str(row["dataset"])
            row["scene"] = str(row["scene"])
            row["material"] = str(row["material"])
            row["cluster_material"] = str(row["cluster_material"])
            row["truth"] = str(row["truth"])
            row["valid"] = bool(row["valid"])
            if row["dataset"] not in PLANNED_GROUPS or row["truth"] not in ONTOLOGY:
                raise RuntimeError(f"{path}: invalid dataset/truth")
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f1-manifest", type=Path, required=True)
    parser.add_argument("--feature-shard", type=Path, action="append", required=True)
    parser.add_argument("--truth-ledger", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite blind bundle: {output}")
    f1_path = args.f1_manifest.resolve()
    _validate_f1(f1_path)
    feature_paths = [path.resolve() for path in args.feature_shard]
    truth_paths = [path.resolve() for path in args.truth_ledger]
    sample_ids, visual, proprio = _feature_shards(feature_paths)
    rows = _truth_rows(truth_paths)

    if len(sample_ids) != len(set(sample_ids.tolist())):
        raise RuntimeError("feature sample IDs are not globally unique")
    truth_ids = [row["sample_id"] for row in rows]
    if len(truth_ids) != len(set(truth_ids)):
        raise RuntimeError("truth sample IDs are not globally unique")
    valid_truth_ids = {row["sample_id"] for row in rows if row["valid"]}
    if set(sample_ids.tolist()) != valid_truth_ids:
        missing = sorted(valid_truth_ids - set(sample_ids.tolist()))[:5]
        extra = sorted(set(sample_ids.tolist()) - valid_truth_ids)[:5]
        raise RuntimeError(f"feature/truth alignment failed; missing={missing}, extra={extra}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    group_counts = Counter(group_rows[0]["dataset"] for group_rows in groups.values())
    if group_counts != Counter(PLANNED_GROUPS):
        raise RuntimeError(f"planned group counts differ: {dict(group_counts)}")
    for group_id, group_rows in groups.items():
        constant = (
            {row["dataset"] for row in group_rows},
            {row["scene"] for row in group_rows},
            {row["cluster_material"] for row in group_rows},
            {row["valid"] for row in group_rows},
        )
        if any(len(values) != 1 for values in constant):
            raise RuntimeError(f"group metadata/validity is not constant: {group_id}")
        if group_rows[0]["valid"] and len(group_rows) != 6:
            raise RuntimeError(f"valid physical group does not have six samples: {group_id}")

    attrition = {}
    for dataset, planned in PLANNED_GROUPS.items():
        invalid = sum(
            rows_for_group[0]["dataset"] == dataset
            and not rows_for_group[0]["valid"]
            for rows_for_group in groups.values()
        )
        attrition[dataset] = {
            "planned_groups": planned,
            "invalid_groups": invalid,
            "rate": invalid / planned,
            "passed": invalid / planned <= 0.05,
        }
    if not all(row["passed"] for row in attrition.values()):
        raise RuntimeError(f"confirmatory attrition exceeds 5%: {attrition}")

    order = np.argsort(sample_ids, kind="stable")
    sample_ids = sample_ids[order]
    visual = visual[order]
    proprio = proprio[order]
    rows.sort(key=lambda row: row["sample_id"])

    output.mkdir(parents=True, exist_ok=False)
    features_path = output / "blind_features.npz"
    truth_path = output / "truth_key.jsonl"
    np.savez_compressed(
        features_path,
        sample_ids=sample_ids,
        visual=visual,
        proprio=proprio,
    )
    truth_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "kinofail.unified-confirmatory-blind-bundle.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "observable_features_and_separate_truth_key_sealed",
        "blind_archive_keys": ["sample_ids", "visual", "proprio"],
        "sample_count_valid": len(sample_ids),
        "planned_groups": PLANNED_GROUPS,
        "attrition": attrition,
        "source_sha256": {
            "f1_manifest": _sha256(f1_path),
            "feature_shards": {str(path): _sha256(path) for path in feature_paths},
            "truth_ledgers": {str(path): _sha256(path) for path in truth_paths},
            "assembler": _sha256(Path(__file__).resolve()),
        },
        "artifact_sha256": {
            "blind_features": _sha256(features_path),
            "truth_key": _sha256(truth_path),
        },
    }
    manifest_path = output / "bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
