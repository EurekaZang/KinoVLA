#!/usr/bin/env python3
"""Freeze the A6 provenance-only amendment before any valid A6 collection.

The v2 collection protocol omitted ``allowed.geometry_profiles`` even though the immutable
schedule contains five geometry profiles.  Runtime validation therefore rejected the first
completed control episode.  This amendment adds exactly the scheduled values, changes no
schedule, physics, capture, analysis threshold, or model, and requires a fresh corpus root.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    base_collection_path = ROOT / "configs/data/kinofail_realistic_a6_boundary_formal_v2.json"
    collection_path = ROOT / "configs/data/kinofail_realistic_a6_boundary_formal_v3.json"
    schedule_path = ROOT / "outputs/kinofail_realistic/design_a6_boundary_v2/schedule.jsonl"
    invalid_root = ROOT / "outputs/kinofail_realistic/corpus_a6_boundary_v2"
    base_analysis_path = ROOT / "configs/eval/kinofail_realistic_a6_analysis_v1.json"
    analysis_path = ROOT / "configs/eval/kinofail_realistic_a6_analysis_v2.json"
    freeze_path = ROOT / "outputs/eval/realistic_a0_a7_v4/a6_analysis_protocol_freeze_v2.json"

    rows = [json.loads(line) for line in schedule_path.read_text(encoding="utf-8").splitlines() if line]
    scheduled_geometry = sorted({str(row["geometry_profile"]) for row in rows})
    base_collection = _json(base_collection_path)
    if "geometry_profiles" in base_collection.get("allowed", {}):
        raise RuntimeError("v2 unexpectedly already authorizes geometry profiles")
    if base_collection["schedule_sha256"] != _sha(schedule_path):
        raise RuntimeError("immutable A6 schedule hash changed")

    invalid_manifests = []
    for manifest_path in sorted(invalid_root.rglob("manifest.json")) if invalid_root.exists() else []:
        manifest = _json(manifest_path)
        invalid_manifests.append(
            {
                "episode_id": manifest.get("episode_id"),
                "issues": manifest.get("runtime_validation", {}).get("issues", []),
                "manifest_sha256": _sha(manifest_path),
            }
        )
    if not invalid_manifests or not all(
        "formal_protocol_geometry_profile_not_authorized" in row["issues"]
        for row in invalid_manifests
    ):
        raise RuntimeError("missing expected fail-closed v2 provenance evidence")

    collection = deepcopy(base_collection)
    collection["protocol_id"] = "kinofail_realistic_a6_boundary_150_v3"
    collection["frozen_utc"] = datetime.now(UTC).isoformat()
    collection["allowed"]["geometry_profiles"] = scheduled_geometry
    collection["supersedes_invalid_protocol"] = str(base_collection_path.relative_to(ROOT))
    collection["amendment_reason"] = (
        "Provenance-only correction: authorize the five geometry_profile values already present "
        "in the immutable v2 schedule. No schedule, operator, parameter, scene, capture, outcome "
        "criterion, or analysis setting changes."
    )
    collection["invalid_predecessor_evidence"] = {
        "corpus_root": str(invalid_root.relative_to(ROOT)),
        "completed_manifests": invalid_manifests,
        "valid_pair_summaries": 0,
        "model_or_boundary_results_available": False,
        "fresh_corpus_root_required": True,
    }
    _write_new(collection_path, collection)

    analysis = deepcopy(_json(base_analysis_path))
    analysis["protocol_id"] = "kinofail_realistic_a6_analysis_v2"
    analysis["status"] = "frozen_after_provenance_amendment_before_valid_a6_simulation"
    analysis["collection_protocol"] = str(collection_path.relative_to(ROOT))
    analysis["corpus_root"] = "outputs/kinofail_realistic/corpus_a6_boundary_v3"
    analysis["amendment"] = {
        "base_analysis_protocol": str(base_analysis_path.relative_to(ROOT)),
        "base_analysis_protocol_sha256": _sha(base_analysis_path),
        "changed": "collection protocol and fresh corpus-root bindings only",
        "analysis_logic_or_threshold_changed": False,
    }
    _write_new(analysis_path, analysis)

    freeze = {
        "schema_version": "kinofail.realistic-a6-analysis-freeze.v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_before_valid_a6_simulation",
        "collection_protocol": str(collection_path.relative_to(ROOT)),
        "collection_protocol_sha256": _sha(collection_path),
        "analysis_protocol": str(analysis_path.relative_to(ROOT)),
        "analysis_protocol_sha256": _sha(analysis_path),
        "schedule": str(schedule_path.relative_to(ROOT)),
        "schedule_sha256": _sha(schedule_path),
        "invalid_predecessor_results_excluded": True,
        "a8_in_scope": False,
    }
    _write_new(freeze_path, freeze)
    print(
        json.dumps(
            {
                "collection_protocol": str(collection_path),
                "collection_sha256": _sha(collection_path),
                "analysis_protocol": str(analysis_path),
                "analysis_sha256": _sha(analysis_path),
                "authorized_geometry_profiles": scheduled_geometry,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
