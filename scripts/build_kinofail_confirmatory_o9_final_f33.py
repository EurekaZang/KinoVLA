#!/usr/bin/env python3
"""Freeze the 960-pair direct-contact O9 confirmation after F32 passes."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kino_vla.data.o9_semantics import DEFAULT_THRESHOLDS
from scripts.build_kinofail_confirmatory_o9_pilot_f32 import (
    ASSET_LOCK,
    COLLECTOR,
    REGISTRY,
    ROOT,
    RUNTIME,
    SEMANTICS,
    SOURCE_SCHEDULE,
    SURFACE_STATES,
    canonical,
    make_id,
    read_jsonl,
    replace_token,
    sha256,
    write_json,
    write_jsonl,
)


PILOT = ROOT / "outputs/kinofail_confirmatory_o9_pilot_f32c"
OUTPUT = ROOT / "outputs/kinofail_confirmatory_o9_final_f33"
CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_confirmatory_o9_final_f33/corpus")
RUNNER = ROOT / "scripts/run_kinofail_confirmatory_o9_final_f33.py"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_confirmatory_o9_direct_v2.py"
SEED_BASE = 2_340_000_000


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def fresh_views(source: list[dict[str, Any]], seed: int, pair_id: str) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for index, item in enumerate(source):
        row = copy.deepcopy(item)
        rng = random.Random(seed + 1009 * index)
        row.update(
            {
                "appearance_seed": seed,
                "surface_state": SURFACE_STATES[rng.randrange(len(SURFACE_STATES))],
                "uv_scale": round(rng.uniform(0.68, 1.42), 6),
                "uv_rotation_deg": int(rng.choice((0, 90, 180, 270))),
                "uv_offset": [round(rng.random(), 6), round(rng.random(), 6)],
                "albedo_brightness_multiplier": round(rng.uniform(0.78, 1.22), 6),
                "normal_strength": round(rng.uniform(0.70, 1.30), 6),
                "roughness_multiplier": round(rng.uniform(0.72, 1.28), 6),
            }
        )
        row["appearance_id"] = make_id(
            "app_f33_", {"pair_id": pair_id, "view": index, "seed": seed, "row": row}
        )
        views.append(row)
    return views


def clone_pair(source: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    if len(source) != 2 or {row["condition"] for row in source} != {
        "nominal_counterfactual",
        "anomaly",
    }:
        raise RuntimeError("source O9 pair is incomplete")
    original_id = str(source[0]["counterfactual_group_id"])
    seed = SEED_BASE + index
    pair_id = make_id(
        "cf_f33_", {"original": original_id, "seed": seed, "final_attempt": 1}
    )
    views = fresh_views(list(source[0]["appearance_views"]), seed, pair_id)
    primary = next(row for row in views if row["is_primary"] is True)
    source_lambda = float(source[0]["parameter_interpolation"]["lambda"])
    continuum = max(0.0, min(1.0, (source_lambda - 0.2025) / (0.96375 - 0.2025)))
    width = round(0.12 + 0.02 * continuum, 6)
    residual = round(0.45 - 0.35 * continuum, 6)
    geometry_spec = {
        "kind": "route_transverse_pallet_belly_crossbar",
        "ridge_height_m": 0.36,
        "ridge_width_m": width,
        "residual_support": residual,
    }
    output: list[dict[str, Any]] = []
    for original in source:
        row = replace_token(copy.deepcopy(original), original_id, pair_id)
        condition = str(row["condition"])
        row.update(
            {
                "benchmark_id": "kinofail_confirmatory_o9_final_f33",
                "counterfactual_group_id": pair_id,
                "episode_id": f"{pair_id}_{'nominal' if condition == 'nominal_counterfactual' else 'anomaly'}",
                "design_mode": "prospective_direct_o9_confirmation",
                "split": "confirmatory_o9_direct",
                "evaluation_eligible": False,
                "artifact_state": "planned",
                "operator_seed": seed,
                "physical_seed": seed,
                "runtime_seed": seed,
                "appearance_seed": seed,
                "appearance_views": copy.deepcopy(views),
                "appearance_id": primary["appearance_id"],
                "surface_state": primary["surface_state"],
                "uv_scale": primary["uv_scale"],
                "uv_rotation_deg": primary["uv_rotation_deg"],
                "uv_offset": primary["uv_offset"],
                "albedo_brightness_multiplier": primary["albedo_brightness_multiplier"],
                "normal_strength": primary["normal_strength"],
                "roughness_multiplier": primary["roughness_multiplier"],
                "texture_swap_group_id": make_id("ts_f33_", {"pair": pair_id}),
                "geometry_id": f"{row['scene_id']}::transverse_pallet_belly_crossbar_f33",
                "geometry_profile": "transverse_pallet_belly_crossbar",
                "geometry_hash": hashlib.sha256(canonical(geometry_spec).encode()).hexdigest(),
                "physical_realization": "pallet_edge",
                "physical_nuisance": {
                    "profile_index": index,
                    "start_progress_m": 0.35,
                    "start_lateral_offset_m": round(random.Random(seed + 17).uniform(-0.003, 0.003), 6),
                    "start_heading_offset_rad": round(random.Random(seed + 31).uniform(-0.002, 0.002), 6),
                    "forward_speed_mps": 0.18,
                    "controller_target_lateral_offset_m": 0.0,
                    "physics_seed": seed,
                    "pair_shared": True,
                    "spawn_base_height_m": 0.43,
                },
                "parameter_interpolation": {
                    "source_lambda": source_lambda,
                    "direct_o9_continuum": round(continuum, 8),
                    "rule": "height=0.36; width=0.12+0.02*t; residual_support=0.45-0.35*t",
                },
                "o9_semantic_recollection": {
                    "phase": "final_confirmation",
                    "source_counterfactual_group_id": original_id,
                    "model_prediction_or_score_read": False,
                    "direct_contact_required": True,
                    "pilot_pair_or_seed_reused": False,
                },
            }
        )
        row["physics_parameters"] = (
            {
                "residual_support": 1.0,
                "ridge_height_m": 0.0,
                "ridge_width_m": 0.0,
            }
            if condition == "nominal_counterfactual"
            else {
                "residual_support": residual,
                "ridge_height_m": 0.36,
                "ridge_width_m": width,
            }
        )
        output.append(row)
    return sorted(output, key=lambda row: row["condition"] == "anomaly")


def main() -> int:
    if OUTPUT.exists() or CORPUS.exists():
        raise FileExistsError("refusing to overwrite F33 final design or corpus")
    pilot_seal_path = PILOT / "seal_manifest.json"
    pilot_audit_path = PILOT / "aligned_final_audit.json"
    pilot_seal = read_json(pilot_seal_path)
    pilot_audit = read_json(pilot_audit_path)
    if (
        pilot_seal.get("passed") is not True
        or pilot_audit.get("passed") is not True
        or pilot_audit.get("model_prediction_feature_label_outcome_or_score_read") is not False
        or pilot_audit.get("counts", {}).get("strict_semantic_and_matched_horizon_passed_pairs") != 12
    ):
        raise RuntimeError("F32c matched-horizon physical mechanism pilot did not pass all gates")

    source_rows = read_jsonl(SOURCE_SCHEDULE)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in source_rows:
        if row["target_operator"] == "O9_high_centering":
            groups.setdefault(str(row["counterfactual_group_id"]), []).append(row)
    if len(groups) != 960:
        raise RuntimeError(f"expected 960 source O9 pairs, found {len(groups)}")
    ordered = [groups[key] for key in sorted(groups)]
    final_rows = [row for index, source in enumerate(ordered) for row in clone_pair(source, index)]
    final_rows.sort(key=lambda row: (row["counterfactual_group_id"], row["condition"]))
    if len(final_rows) != 1920:
        raise RuntimeError("F33 schedule row count mismatch")

    design_path = OUTPUT / "design.json"
    design = {
        "schema_version": "kinofail.confirmatory-o9-final-f33-design.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "fixed_before_final_o9_collection",
        "scientific_role": "prospective direct-contact replacement for every expanded Scale O9 slot",
        "confirmatory": True,
        "pairs": 960,
        "physical_episodes": 1920,
        "independent_statistical_unit": "counterfactual_group_id",
        "scene_count": 30,
        "domain_count": 3,
        "parameter_points_per_scene": 16,
        "semantic_thresholds": DEFAULT_THRESHOLDS,
        "parameter_interval": {
            "ridge_height_m": [0.36, 0.36],
            "ridge_width_m": [0.12, 0.14],
            "residual_support": [0.10, 0.45],
            "continuum_points": 16,
        },
        "admission": {
            "direct_o9_semantics_required": True,
            "nominal_must_not_fall_through_matched_decision_time": True,
            "nominal_max_route_deviation_m_through_matched_decision_time": 0.55,
            "minimum_decision_time_s": 0.42,
            "post_decision_nominal_outcome_reported_but_not_used_for_attribution_admission": True,
            "allowed_nonphysical_runtime_issue_suffixes": [
                "appearance_effect_too_small",
                "rgb_spatial_contrast_too_low",
            ],
        },
        "minimum_accepted_pairs_for_scale_attrition_strictly_below_five_percent": 750,
        "effective_non_o9_scale_pairs_before_f33": 9283,
        "model_prediction_feature_label_outcome_or_score_read": False,
        "architecture_router_features_threshold_eta_and_analysis_unchanged": True,
        "pilot_pairs_or_seeds_reused": False,
        "source_schedule": str(SOURCE_SCHEDULE.relative_to(ROOT)),
        "source_schedule_sha256": sha256(SOURCE_SCHEDULE),
        "pilot_seal": str(pilot_seal_path.relative_to(ROOT)),
        "pilot_seal_sha256": sha256(pilot_seal_path),
        "pilot_audit": str(pilot_audit_path.relative_to(ROOT)),
        "pilot_audit_sha256": sha256(pilot_audit_path),
    }
    write_json(design_path, design)
    schedule_path = OUTPUT / "schedule.jsonl"
    write_jsonl(schedule_path, final_rows)
    pair_ids = sorted({str(row["counterfactual_group_id"]) for row in final_rows})
    protocol_path = OUTPUT / "collection_protocol.json"
    protocol = {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": "kinofail-confirmatory-o9-direct-final-f33",
        "benchmark_id": "kinofail_confirmatory_o9_final_f33",
        "status": "frozen",
        "confirmatory": True,
        "collector_path": str(COLLECTOR.relative_to(ROOT)),
        "collector_sha256": sha256(COLLECTOR),
        "schedule_path": str(schedule_path.relative_to(ROOT)),
        "schedule_sha256": sha256(schedule_path),
        "runtime_manifest_path": str(RUNTIME.relative_to(ROOT)),
        "runtime_manifest_sha256": sha256(RUNTIME),
        "scene_registry_path": str(REGISTRY.relative_to(ROOT)),
        "scene_registry_sha256": sha256(REGISTRY),
        "material_lock_path": str(ASSET_LOCK.relative_to(ROOT)),
        "material_lock_sha256": sha256(ASSET_LOCK),
        "design_path": str(design_path.relative_to(ROOT)),
        "design_sha256": sha256(design_path),
        "allowed": {
            "conditions": ["nominal_counterfactual", "anomaly"],
            "counterfactual_group_ids": pair_ids,
            "geometry_profiles": ["transverse_pallet_belly_crossbar"],
            "physical_realizations": ["pallet_edge"],
            "scene_families": sorted({str(row["scene_family"]) for row in final_rows}),
            "severity_ids": ["hard", "moderate"],
            "target_operators": ["O9_high_centering"],
        },
        "collection_contract": {
            "a8_in_scope": False,
            "counterfactual_pairs": 960,
            "physical_episodes": 1920,
            "appearance_views_per_episode": 3,
            "operator_seed_equals_physics_seed": True,
            "per_pair_physical_nuisance_exact_and_shared": True,
            "outcome_strength_is_reported_not_gated": True,
            "direct_o9_semantic_gate_required": True,
            "result_dependent_retry_forbidden": True,
        },
    }
    write_json(protocol_path, protocol)
    dependencies = [
        COLLECTOR,
        RUNNER,
        ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v1.py",
        ROOT / "kino_vla/data/o9_pair_alignment.py",
        SEMANTICS,
        RUNTIME,
    ]
    seal = {
        "schema_version": "kinofail.confirmatory-o9-final-f33-seal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_collection",
        "passed": True,
        "model_prediction_feature_label_outcome_or_score_read": False,
        "result_dependent_retry": False,
        "fresh_isaac_process_per_pair": True,
        "maximum_concurrent_isaac_processes": 3,
        "counterfactual_pairs": 960,
        "physical_episodes": 1920,
        "minimum_accepted_pairs": 750,
        "corpus_root": str(CORPUS),
        "schedule": str(schedule_path.relative_to(ROOT)),
        "schedule_sha256": sha256(schedule_path),
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": sha256(protocol_path),
        "design": str(design_path.relative_to(ROOT)),
        "design_sha256": sha256(design_path),
        "scene_registry": str(REGISTRY.relative_to(ROOT)),
        "scene_registry_sha256": sha256(REGISTRY),
        "material_lock": str(ASSET_LOCK.relative_to(ROOT)),
        "material_lock_sha256": sha256(ASSET_LOCK),
        "semantic_module": str(SEMANTICS.relative_to(ROOT)),
        "semantic_module_sha256": sha256(SEMANTICS),
        "pilot_seal": str(pilot_seal_path.relative_to(ROOT)),
        "pilot_seal_sha256": sha256(pilot_seal_path),
        "pilot_audit": str(pilot_audit_path.relative_to(ROOT)),
        "pilot_audit_sha256": sha256(pilot_audit_path),
        "dependencies": [{"path": str(path), "sha256": sha256(path)} for path in dependencies],
    }
    seal_path = OUTPUT / "seal_manifest.json"
    write_json(seal_path, seal)
    (OUTPUT / "seal_manifest.sha256").write_text(f"{sha256(seal_path)}  {seal_path.name}\n")
    print(json.dumps(seal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
