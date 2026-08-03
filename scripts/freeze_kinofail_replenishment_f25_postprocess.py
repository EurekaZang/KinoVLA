#!/usr/bin/env python3
"""Freeze F25 snapshot/feature extraction before any replacement prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "outputs/kinofail_reconfirmation_v2"


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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f25-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    args = parser.parse_args()
    f25_root = args.f25_root.resolve()
    corpus_root = args.corpus_root.resolve()
    eval_root = args.eval_root.resolve()
    freeze_path = f25_root / "freeze_manifest.json"
    freeze = _json(freeze_path)
    if (
        freeze.get("cohort_id") != "F25-v1"
        or freeze.get("status") != "sealed_before_collection"
        or freeze.get("model_prediction_or_score_read") is not False
    ):
        raise RuntimeError("invalid F25 physical freeze")
    forbidden = list(eval_root.rglob("*prediction*")) + list(
        eval_root.rglob("*score*")
    ) if eval_root.exists() else []
    if forbidden:
        raise RuntimeError("prediction or score artifacts already exist in F25 eval root")

    collection_protocols = {
        Path(str(row["path"])).parts[-3]: ROOT / str(row["path"])
        for row in freeze["protocols"]
    }
    protocols: list[dict[str, str]] = []
    for scene_id in sorted(collection_protocols):
        source = (
            ORIGINAL
            / "schedules/scenes"
            / scene_id
            / "scale/snapshot_protocol.json"
        )
        schedule = f25_root / "schedules" / scene_id / "scale/schedule.jsonl"
        collection = _json(collection_protocols[scene_id])
        protocol = _json(source)
        protocol.update(
            {
                "protocol_id": f"replenishment-f25-scale-snapshot-{scene_id}",
                "status": "frozen_before_model_blind_feature_extraction",
                "source_corpus_root": str(corpus_root / scene_id),
                "source_schedule": str(schedule.relative_to(ROOT)),
                "source_schedule_sha256": _sha256(schedule),
                "output_dir": str(
                    (eval_root / "shards" / scene_id / "scale/snapshots").relative_to(ROOT)
                ),
            }
        )
        protocol["selection"] = {
            **protocol["selection"],
            "required_collection_protocol_id": collection["protocol_id"],
        }
        protocol["freeze_provenance"] = {
            **protocol["freeze_provenance"],
            "collector": "scripts/isaac_collect_kinofail_replenishment_f24_v1.py",
            "collector_sha256": _sha256(
                ROOT / "scripts/isaac_collect_kinofail_replenishment_f24_v1.py"
            ),
            "f25_physical_freeze": str(freeze_path.relative_to(ROOT)),
            "f25_physical_freeze_sha256": _sha256(freeze_path),
            "model_outcomes_available_at_freeze": False,
            "physical_collection_started_at_freeze": True,
        }
        protocol["publication_guard"] = {
            "may_satisfy_realistic_a0_a7": True,
            "reason": (
                "post-hoc F25 replenishment; report beside the original independent "
                "confirmation rather than replacing its attrition disclosure"
            ),
        }
        path = f25_root / "snapshot_protocols" / scene_id / "scale.json"
        _write_json(path, protocol)
        protocols.append(
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
        )

    scripts = [
        ROOT / "scripts/build_kinofail_realistic_snapshot_dev.py",
        ROOT / "scripts/extract_kinofail_realistic_features.py",
        ROOT / "scripts/build_kinofail_unified_invariant_features_v1.py",
    ]
    manifest = {
        "schema_version": "kinofail.f25-postprocess-freeze.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_model_blind_feature_extraction",
        "passed": True,
        "model_prediction_label_outcome_or_score_read": False,
        "physical_collection_complete_at_freeze": False,
        "physical_freeze": str(freeze_path.relative_to(ROOT)),
        "physical_freeze_sha256": _sha256(freeze_path),
        "corpus_root": str(corpus_root),
        "eval_root": str(eval_root.relative_to(ROOT)),
        "snapshot_protocols": protocols,
        "feature_scripts": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for path in scripts
        ],
        "selection": {
            "complete_runtime_validated_pairs_only": True,
            "all_manifest_appearance_views": True,
            "result_dependent_pair_selection": False,
            "original_frozen_temporal_alignment_reused": True,
        },
    }
    output = f25_root / "postprocess_freeze_manifest.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite postprocess freeze: {output}")
    _write_json(output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
