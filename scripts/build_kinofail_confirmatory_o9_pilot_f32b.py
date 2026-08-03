#!/usr/bin/env python3
"""Freeze the second, permanently excluded O9 mechanism pilot.

F32 proved direct belly contact in 12/12 pairs but lost three controls to
nominal route instability.  F32b changes only the pair-shared start nuisance
band, using fresh IDs, physics seeds, and appearances.  Attribution models,
features, labels, and endpoint statistics remain unread.
"""

from __future__ import annotations

import copy
import json
import random
from datetime import UTC, datetime
from pathlib import Path

from scripts import build_kinofail_confirmatory_o9_pilot_f32 as base


ROOT = base.ROOT
OUTPUT = ROOT / "outputs/kinofail_confirmatory_o9_pilot_f32b"
CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_confirmatory_o9_pilot_f32b/corpus")
RUNNER = ROOT / "scripts/run_kinofail_confirmatory_o9_pilot_f32b.py"
F32_AUDIT = ROOT / "outputs/kinofail_confirmatory_o9_pilot_f32/final_audit.json"
SEED_BASE = 2_331_000_000


def clone_pair(source: list[dict], index: int) -> list[dict]:
    previous_seed_base = base.SEED_BASE
    base.SEED_BASE = SEED_BASE
    try:
        rows = base.clone_pair(source, index)
    finally:
        base.SEED_BASE = previous_seed_base
    old_pair_id = str(rows[0]["counterfactual_group_id"])
    new_pair_id = base.make_id(
        "cf_f32b_",
        {
            "source": source[0]["counterfactual_group_id"],
            "seed": SEED_BASE + index,
            "pilot_attempt": 2,
            "nuisance_revision": "narrow_start_band",
        },
    )
    rows = base.replace_token(rows, old_pair_id, new_pair_id)
    seed = SEED_BASE + index
    lateral = round(random.Random(seed + 17).uniform(-0.003, 0.003), 6)
    heading = round(random.Random(seed + 31).uniform(-0.002, 0.002), 6)
    for row in rows:
        condition = str(row["condition"])
        row.update(
            {
                "benchmark_id": "kinofail_confirmatory_o9_pilot_f32b",
                "counterfactual_group_id": new_pair_id,
                "episode_id": f"{new_pair_id}_{'nominal' if condition == 'nominal_counterfactual' else 'anomaly'}",
                "design_mode": "excluded_physical_mechanism_pilot_revision",
                "split": "excluded_o9_mechanism_pilot_f32b",
                "geometry_id": f"{row['scene_id']}::transverse_pallet_belly_crossbar_f32b",
                "physical_nuisance": {
                    **copy.deepcopy(row["physical_nuisance"]),
                    "start_lateral_offset_m": lateral,
                    "start_heading_offset_rad": heading,
                },
                "o9_semantic_recollection": {
                    **copy.deepcopy(row["o9_semantic_recollection"]),
                    "phase": "excluded_pilot_revision_f32b",
                    "pilot_pair_or_seed_reused": False,
                    "adaptation_basis": "F32 direct semantics passed 12/12; narrow start nuisance to reduce nominal-only instability",
                },
            }
        )
    return rows


def main() -> int:
    if OUTPUT.exists() or CORPUS.exists():
        raise FileExistsError("refusing to overwrite F32b pilot or corpus")
    f32_audit = json.loads(F32_AUDIT.read_text())
    if (
        f32_audit.get("status") != "final"
        or f32_audit.get("passed") is not False
        or f32_audit.get("model_prediction_feature_label_or_score_read") is not False
        or f32_audit.get("counts", {}).get("scheduled_pairs") != 12
    ):
        raise RuntimeError("F32 failure audit is missing or inconsistent")
    semantic_passes = sum(
        bool(row.get("evidence", {}).get("anomaly_semantic", {}).get("passed"))
        for row in f32_audit.get("results", [])
    )
    if semantic_passes != 12:
        raise RuntimeError("F32 did not establish the direct O9 mechanism in all pilot pairs")

    source_rows = base.read_jsonl(base.SOURCE_SCHEDULE)
    groups: dict[str, list[dict]] = {}
    for row in source_rows:
        if row["target_operator"] == "O9_high_centering":
            groups.setdefault(str(row["counterfactual_group_id"]), []).append(row)
    selected = base.select_pilot(groups)
    pilot_rows = [row for index, source in enumerate(selected) for row in clone_pair(source, index)]
    pilot_rows.sort(key=lambda row: (row["counterfactual_group_id"], row["condition"]))
    if len(pilot_rows) != 24:
        raise RuntimeError("F32b schedule size mismatch")

    design_path = OUTPUT / "design.json"
    design = {
        "schema_version": "kinofail.confirmatory-o9-pilot-f32b-design.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "fixed_before_second_pilot_collection",
        "scientific_role": "excluded model-blind physical-mechanism and nominal-stability pilot revision",
        "evaluation_eligible": False,
        "all_pilot_pairs_permanently_excluded_from_final_confirmation": True,
        "pairs": 12,
        "physical_episodes": 24,
        "domains": sorted({row["domain"] for row in pilot_rows}),
        "scenes": sorted({row["scene_id"] for row in pilot_rows}),
        "semantic_thresholds": base.DEFAULT_THRESHOLDS,
        "f32_observation": {
            "audit": str(F32_AUDIT.relative_to(ROOT)),
            "audit_sha256": base.sha256(F32_AUDIT),
            "direct_semantic_passes": semantic_passes,
            "nominal_stable_pair_passes": f32_audit["counts"]["direct_semantic_and_nominal_stability_passed_pairs"],
            "model_prediction_or_score_read": False,
        },
        "single_prespecified_change": {
            "start_lateral_offset_m": [-0.003, 0.003],
            "start_heading_offset_rad": [-0.002, 0.002],
            "unchanged_start_progress_m": 0.35,
            "unchanged_forward_speed_mps": 0.08,
            "unchanged_spawn_base_height_m": 0.43,
            "unchanged_o9_geometry_and_semantic_thresholds": True,
        },
        "fresh_pair_ids_physics_seeds_and_appearances": True,
        "model_prediction_feature_label_or_score_read": False,
        "source_schedule": str(base.SOURCE_SCHEDULE.relative_to(ROOT)),
        "source_schedule_sha256": base.sha256(base.SOURCE_SCHEDULE),
    }
    base.write_json(design_path, design)
    schedule_path = OUTPUT / "schedule.jsonl"
    base.write_jsonl(schedule_path, pilot_rows)
    pair_ids = sorted({str(row["counterfactual_group_id"]) for row in pilot_rows})

    protocol_path = OUTPUT / "collection_protocol.json"
    protocol = {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": "kinofail-confirmatory-o9-direct-pilot-f32",
        "protocol_variant": "f32b-narrow-start-nuisance",
        "benchmark_id": "kinofail_confirmatory_o9_pilot_f32b",
        "status": "frozen",
        "confirmatory": False,
        "collector_path": str(base.COLLECTOR.relative_to(ROOT)),
        "collector_sha256": base.sha256(base.COLLECTOR),
        "schedule_path": str(schedule_path.relative_to(ROOT)),
        "schedule_sha256": base.sha256(schedule_path),
        "runtime_manifest_path": str(base.RUNTIME.relative_to(ROOT)),
        "runtime_manifest_sha256": base.sha256(base.RUNTIME),
        "scene_registry_path": str(base.REGISTRY.relative_to(ROOT)),
        "scene_registry_sha256": base.sha256(base.REGISTRY),
        "material_lock_path": str(base.ASSET_LOCK.relative_to(ROOT)),
        "material_lock_sha256": base.sha256(base.ASSET_LOCK),
        "design_path": str(design_path.relative_to(ROOT)),
        "design_sha256": base.sha256(design_path),
        "allowed": {
            "conditions": ["nominal_counterfactual", "anomaly"],
            "counterfactual_group_ids": pair_ids,
            "geometry_profiles": ["transverse_pallet_belly_crossbar"],
            "physical_realizations": ["pallet_edge"],
            "scene_families": sorted({str(row["scene_family"]) for row in pilot_rows}),
            "severity_ids": ["hard", "moderate"],
            "target_operators": ["O9_high_centering"],
        },
        "collection_contract": {
            "a8_in_scope": False,
            "counterfactual_pairs": 12,
            "physical_episodes": 24,
            "appearance_views_per_episode": 3,
            "operator_seed_equals_physics_seed": True,
            "per_pair_physical_nuisance_exact_and_shared": True,
            "direct_o9_semantic_gate_required": True,
            "pilot_excluded_from_final_evaluation": True,
            "result_dependent_retry_forbidden": True,
        },
    }
    base.write_json(protocol_path, protocol)
    dependencies = [
        base.COLLECTOR,
        RUNNER,
        ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v1.py",
        base.SEMANTICS,
        base.RUNTIME,
    ]
    seal = {
        "schema_version": "kinofail.confirmatory-o9-pilot-f32b-seal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_collection",
        "passed": True,
        "model_prediction_feature_label_or_score_read": False,
        "result_dependent_retry": False,
        "maximum_concurrent_isaac_processes": 3,
        "counterfactual_pairs": 12,
        "physical_episodes": 24,
        "corpus_root": str(CORPUS),
        "schedule": str(schedule_path.relative_to(ROOT)),
        "schedule_sha256": base.sha256(schedule_path),
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": base.sha256(protocol_path),
        "design": str(design_path.relative_to(ROOT)),
        "design_sha256": base.sha256(design_path),
        "scene_registry": str(base.REGISTRY.relative_to(ROOT)),
        "scene_registry_sha256": base.sha256(base.REGISTRY),
        "material_lock": str(base.ASSET_LOCK.relative_to(ROOT)),
        "material_lock_sha256": base.sha256(base.ASSET_LOCK),
        "semantic_module": str(base.SEMANTICS.relative_to(ROOT)),
        "semantic_module_sha256": base.sha256(base.SEMANTICS),
        "f32_audit": str(F32_AUDIT.relative_to(ROOT)),
        "f32_audit_sha256": base.sha256(F32_AUDIT),
        "dependencies": [{"path": str(path), "sha256": base.sha256(path)} for path in dependencies],
    }
    seal_path = OUTPUT / "seal_manifest.json"
    base.write_json(seal_path, seal)
    (OUTPUT / "seal_manifest.sha256").write_text(f"{base.sha256(seal_path)}  {seal_path.name}\n")
    print(json.dumps(seal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
