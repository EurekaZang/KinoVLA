#!/usr/bin/env python3
"""Freeze the nine-pair realistic A7 visual/camera replay before model outcomes."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "configs/eval/kinofail_realistic_a7_matched_design_v1.json"
BASE_SCHEDULE = (
    ROOT / "outputs/kinofail_realistic/design_scale_v8_replication_v1/full_schedule.jsonl"
)
REGISTRY = ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_realistic_a7_visual_replay_v1.py"
RUNTIME_MANIFEST = ROOT / "kino_vla/data/runtime_manifest.py"
OUT_DIR = ROOT / "outputs/kinofail_realistic/design_a7_visual_replay_v1"
OUT_SCHEDULE = OUT_DIR / "schedule.jsonl"
OUT_PROTOCOL = ROOT / "configs/data/kinofail_realistic_a7_visual_replay_formal_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    if OUT_SCHEDULE.exists() or OUT_PROTOCOL.exists():
        raise FileExistsError(
            "A7 visual-replay freeze already exists; refusing post-outcome overwrite"
        )
    design = _json(DESIGN)
    if design.get("model_outcomes_available_at_freeze") is not False:
        raise RuntimeError("A7 design is not certified outcome-blind")
    if _sha256(BASE_SCHEDULE) != str(design["base_schedule_sha256"]):
        raise RuntimeError("A7 base schedule hash mismatch")
    cell = design["design"]["visual_camera_replay"]
    selected_ids = set(cell["counterfactual_group_ids"])
    rows = [
        row for row in _jsonl(BASE_SCHEDULE)
        if row["counterfactual_group_id"] in selected_ids
    ]
    if len(rows) != 2 * len(selected_ids):
        raise RuntimeError("A7 visual-replay selection is not nine complete pairs")

    frozen_rows = []
    for source in rows:
        row = dict(source)
        row["benchmark_id"] = "kinofail_realistic_a7_visual_replay_v1"
        row["physical_nuisance_profile_index"] = 2
        row["a7_ablation"] = {
            "cells": [
                "default_grid_vs_realistic_scene",
                "procedural_texture_vs_scanned_pbr",
                "world_follow_vs_body_fixed_camera",
            ],
            "render_arms": list(cell["render_arms"]),
            "matched_base_counterfactual_group_id": source["counterfactual_group_id"],
            "matched_without_physics_advance": True,
        }
        frozen_rows.append(row)
    frozen_rows.sort(key=lambda row: (row["counterfactual_group_id"], row["condition"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SCHEDULE.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in frozen_rows),
        encoding="utf-8",
    )
    schedule_sha256 = _sha256(OUT_SCHEDULE)
    scenes = sorted({str(row["scene_family"]) for row in frozen_rows})
    protocol = {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": "kinofail_realistic_a7_visual_replay_formal_v1",
        "status": "frozen",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "benchmark_id": "kinofail_realistic_a7_visual_replay_v1",
        "scope": (
            "18 physical episodes / 9 pairs / O5 moderate / 9 scenes / nuisance profile 2 / "
            "four timestamp-matched render arms; A8 excluded."
        ),
        "schedule_path": str(OUT_SCHEDULE.relative_to(ROOT)),
        "schedule_sha256": schedule_sha256,
        "scene_registry_path": str(REGISTRY.relative_to(ROOT)),
        "scene_registry_sha256": _sha256(REGISTRY),
        "collector_path": str(COLLECTOR.relative_to(ROOT)),
        "collector_sha256": _sha256(COLLECTOR),
        "runtime_manifest_path": str(RUNTIME_MANIFEST.relative_to(ROOT)),
        "runtime_manifest_sha256": _sha256(RUNTIME_MANIFEST),
        "a7_design_path": str(DESIGN.relative_to(ROOT)),
        "a7_design_sha256": _sha256(DESIGN),
        "base_schedule_path": str(BASE_SCHEDULE.relative_to(ROOT)),
        "base_schedule_sha256": _sha256(BASE_SCHEDULE),
        "allowed": {
            "counterfactual_group_ids": sorted(selected_ids),
            "target_operators": ["O5_payload"],
            "scene_families": scenes,
            "physical_realizations": sorted(
                {str(row["physical_realization"]) for row in frozen_rows}
            ),
            "geometry_profiles": sorted(
                {str(row["geometry_profile"]) for row in frozen_rows}
            ),
            "conditions": ["anomaly", "nominal_counterfactual"],
            "severity_ids": ["moderate"],
        },
        "collection_contract": {
            "a8_in_scope": False,
            "counterfactual_pairs": len(selected_ids),
            "physical_episodes": len(frozen_rows),
            "scene_families": len(scenes),
            "physical_nuisance_profile_index": 2,
            "render_arms": list(cell["render_arms"]),
            "physics_advance_between_render_arms": False,
            "pair_shared_scene_camera_controller_operator_and_nuisance": True,
            "negative_or_null_results_retained": True,
            "model_outcomes_available_at_freeze": False,
        },
    }
    OUT_PROTOCOL.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schedule": str(OUT_SCHEDULE.relative_to(ROOT)),
                "schedule_sha256": schedule_sha256,
                "protocol": str(OUT_PROTOCOL.relative_to(ROOT)),
                "protocol_sha256": _sha256(OUT_PROTOCOL),
                "pairs": len(selected_ids),
                "episodes": len(frozen_rows),
                "render_arms": list(cell["render_arms"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
