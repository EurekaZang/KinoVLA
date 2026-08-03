#!/usr/bin/env python3
"""Freeze the final 90-pair O9 semantic repair after excluded pilots."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from kino_vla.data.o9_semantics import DEFAULT_THRESHOLDS


ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = (
    ROOT / "outputs/kinofail_realistic/design_o9_semantic_repair_v4/schedule.jsonl"
)
DESIGN_AUDIT = (
    ROOT
    / "outputs/kinofail_realistic/design_o9_semantic_repair_v4/design_audit.json"
)
SEMANTIC_AUDIT = (
    ROOT / "outputs/eval/realistic_a0_a7_v6/o9_semantic_audit_v1.json"
)
PILOTS = [
    ROOT
    / f"outputs/kinofail_realistic/corpus_o9_semantic_repair_{version}/"
    f"pair_summaries/{pair_id}.json"
    for version, pair_id in [
        ("v1", "cf_126a04c6a2399393cdc4"),
        ("v2", "cf_126a04c6a2399393cdc4"),
        ("v3", "cf_126a04c6a2399393cdc4"),
        ("v3", "cf_19a18ca10278db9c830b"),
    ]
]
COLLECTOR = (
    ROOT / "scripts/isaac_collect_kinofail_realistic_o9_semantic_repair_v4.py"
)
SEMANTIC_MODULE = ROOT / "kino_vla/data/o9_semantics.py"
RUNTIME_MANIFEST = ROOT / "kino_vla/data/runtime_manifest.py"
SCENE_REGISTRY = ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json"
PROTOCOL = (
    ROOT / "configs/data/kinofail_realistic_o9_semantic_repair_formal_v4.json"
)
CORPUS = ROOT / "outputs/kinofail_realistic/corpus_o9_semantic_repair_v4"
BENCHMARK_ID = "kinofail_realistic_o9_semantic_repair_v4"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> None:
    existing = sorted((CORPUS / "pair_summaries").glob("*.json"))
    if existing:
        raise RuntimeError(
            "refusing to freeze after O9 semantic-repair-v4 outcomes exist: "
            f"{len(existing)} pair summaries"
        )
    rows = jsonl(SCHEDULE)
    design = json.loads(DESIGN_AUDIT.read_text(encoding="utf-8"))
    if design.get("status") != "passed" or design.get("benchmark_id") != BENCHMARK_ID:
        raise RuntimeError("v4 design audit mismatch")
    pair_ids = sorted({row["counterfactual_group_id"] for row in rows})
    protocol = {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": "kinofail_realistic_o9_semantic_repair_v4",
        "benchmark_id": BENCHMARK_ID,
        "status": "frozen",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Final complete 90-pair scale-v8 O9 replacement with direct "
            "belly-contact admission and pair-shared 0.08 m/s diagnostic creep."
        ),
        "collector_path": str(COLLECTOR.relative_to(ROOT)),
        "collector_sha256": sha256(COLLECTOR),
        "schedule_path": str(SCHEDULE.relative_to(ROOT)),
        "schedule_sha256": sha256(SCHEDULE),
        "runtime_manifest_path": str(RUNTIME_MANIFEST.relative_to(ROOT)),
        "runtime_manifest_sha256": sha256(RUNTIME_MANIFEST),
        "scene_registry_path": str(SCENE_REGISTRY.relative_to(ROOT)),
        "scene_registry_sha256": sha256(SCENE_REGISTRY),
        "allowed": {
            "conditions": ["anomaly", "nominal_counterfactual"],
            "counterfactual_group_ids": pair_ids,
            "target_operators": ["O9_high_centering"],
            "scene_families": sorted({row["scene_family"] for row in rows}),
            "physical_realizations": ["pallet_edge"],
            "geometry_profiles": ["cross_path_pallet_edge_belly_crossbar"],
            "severity_ids": ["moderate", "severe"],
        },
        "collection_contract": {
            "counterfactual_pairs": 90,
            "physical_episodes": 180,
            "appearance_views_per_episode": 3,
            "independent_statistical_unit": "counterfactual_group_id",
            "a8_in_scope": False,
            "separate_repair_root_required": True,
        },
        "semantic_admission": {
            "definition": (
                "sustained Go2 base/belly load on a route-transverse pallet "
                "crossbar with partial foot unloading; head- or limb-only "
                "collision is rejected"
            ),
            "module_path": str(SEMANTIC_MODULE.relative_to(ROOT)),
            "module_sha256": sha256(SEMANTIC_MODULE),
            "thresholds": DEFAULT_THRESHOLDS,
            "geometry": {
                "moderate_height_m": 0.36,
                "moderate_width_m": 0.12,
                "severe_height_m": 0.36,
                "severe_width_m": 0.14,
                "spawn_base_height_m": 0.43,
                "start_progress_m": 0.35,
            },
            "pair_shared_forward_speed_mps": 0.08,
        },
        "design_provenance": {
            "design_audit_path": str(DESIGN_AUDIT.relative_to(ROOT)),
            "design_audit_sha256": sha256(DESIGN_AUDIT),
            "source_semantic_audit_path": str(SEMANTIC_AUDIT.relative_to(ROOT)),
            "source_semantic_audit_sha256": sha256(SEMANTIC_AUDIT),
            "excluded_pilot_locks": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                for path in PILOTS
            ],
            "pilot_disposition": (
                "All pilot pairs are excluded. They calibrated only nominal route "
                "stability and direct link-contact mechanism admission; no "
                "attribution-model prediction was available or inspected."
            ),
            "v4_change": (
                "Moderate ridge height 0.35 -> 0.36 m after direct telemetry "
                "showed limb-only contact at 0.35 m; width and residual-support "
                "severity remain distinct."
            ),
            "model_predictions_used_to_choose_geometry_thresholds_or_speed": False,
            "v4_outcomes_present_at_freeze": False,
        },
    }
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "protocol": str(PROTOCOL),
                "protocol_sha256": sha256(PROTOCOL),
                "schedule_sha256": protocol["schedule_sha256"],
                "collector_sha256": protocol["collector_sha256"],
                "pairs": len(pair_ids),
                "status": protocol["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
