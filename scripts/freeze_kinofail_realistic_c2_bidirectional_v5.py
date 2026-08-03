#!/usr/bin/env python3
"""Freeze C2 v5 collection and evaluation before confirmation outcomes."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
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
        if line
    ]


def _write(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    registry = (
        ROOT
        / "configs/data"
        / "kinofail_realistic_c2_v5_confirmation_scene_registry.json"
    )
    t2_design = (
        ROOT
        / "outputs/kinofail_realistic"
        / "design_c2_bidirectional_confirmation_v5_t2"
    )
    t3_design = (
        ROOT
        / "outputs/kinofail_realistic"
        / "design_c2_bidirectional_confirmation_v5_t3"
    )
    t2_schedule = t2_design / "schedule.jsonl"
    t3_schedule = t3_design / "schedule.jsonl"
    t3_cases_path = t3_design / "case_schedule.jsonl"
    t2_audit_path = t2_design / "audit.json"
    t3_audit_path = t3_design / "audit.json"
    t2_audit = _json(t2_audit_path)
    t3_audit = _json(t3_audit_path)
    t2_rows = _jsonl(t2_schedule)
    t3_rows = _jsonl(t3_schedule)
    t3_cases = _jsonl(t3_cases_path)
    if (
        t2_audit.get("passed") is not True
        or t3_audit.get("passed") is not True
        or _sha(t2_schedule) != t2_audit["schedule_sha256"]
        or _sha(t3_schedule) != t3_audit["schedule_sha256"]
        or _sha(t3_cases_path)
        != t3_audit["case_schedule_sha256"]
        or len(t2_rows) != 15
        or len(t3_rows) != 60
        or len(t3_cases) != 15
    ):
        raise RuntimeError("invalid C2 v5 design")

    registry_value = _json(registry)
    scenes = {
        str(row["scene_id"]) for row in registry_value["scenes"]
    }
    if (
        len(scenes) != 3
        or {str(row["domain"]) for row in registry_value["scenes"]}
        != {"life", "production", "wild"}
    ):
        raise RuntimeError("invalid C2 v5 scene registry")
    for row in registry_value["scenes"]:
        episode = ROOT / str(row["episode_usd"])
        audit_path = ROOT / str(row["compiled_audit"])
        if (
            _sha(episode) != row["episode_sha256"]
            or _sha(audit_path) != row["compiled_audit_sha256"]
            or _json(audit_path).get("passed") is not True
        ):
            raise RuntimeError(
                f"scene provenance mismatch: {row['scene_id']}"
            )
    prior_scenes: set[str] = set()
    for path in (
        ROOT / "outputs/eval"
    ).glob("c2_bidirectional_v[1-4]*/**/records.jsonl"):
        prior_scenes.update(
            str(row["scene_cluster"]) for row in _jsonl(path)
        )
    for path in (
        ROOT / "outputs/kinofail_realistic"
    ).glob("design_c2_bidirectional_confirmation_v[1-4]*/schedule.jsonl"):
        prior_scenes.update(
            str(row.get("scene_cluster") or row.get("scene_family"))
            for row in _jsonl(path)
        )
    if scenes & prior_scenes:
        raise RuntimeError(
            f"C2 v5 scenes are not fresh: {sorted(scenes & prior_scenes)}"
        )

    development = (
        ROOT
        / "outputs/eval/c2_bidirectional_v5"
        / "development_features_six_scene"
    )
    development_manifest = development / "feature_manifest.json"
    development_report = (
        ROOT
        / "outputs/eval/c2_bidirectional_v5"
        / "development_screen_six_scene_loso_v5"
        / "development_report.json"
    )
    development_value = _json(development_report)
    ours = "structured_post_interaction_router_v5"
    best = str(development_value["best_baseline_method"])
    if (
        _json(development_manifest).get("passed") is not True
        or development_value["schema_version"]
        != "kinofail.realistic-c2-v5-development-screen.v1"
        or development_value.get("v3_failure_used_as_development")
        is not True
        or development_value.get("v4_confirmation_outcomes_used")
        is not True
        or development_value.get("scene_disjoint_loso") is not True
        or len(development_value["scene_clusters"]) != 6
        or development_value["family_gate_accuracy"] != 1.0
        or development_value["results"][ours]["balanced_accuracy"]
        <= development_value["results"][best]["balanced_accuracy"]
        or development_value["results"][ours]["worst_scene_accuracy"]
        <= development_value["results"][best]["worst_scene_accuracy"]
        or development_value["feature_manifest_sha256"]
        != _sha(development_manifest)
    ):
        raise RuntimeError("C2 v5 development is not freeze-ready")

    outputs = {
        "t2": (
            ROOT
            / "configs/eval"
            / "kinofail_realistic_c2_v5_t2_collection.json"
        ),
        "t3": (
            ROOT
            / "configs/eval"
            / "kinofail_realistic_c2_v5_t3_collection.json"
        ),
        "evaluation": (
            ROOT
            / "configs/eval"
            / "kinofail_realistic_c2_bidirectional_formal_v5.json"
        ),
    }
    outcome_paths = (
        ROOT
        / "outputs/kinofail_realistic"
        / "corpus_c2_bidirectional_confirmation_v5_t2",
        ROOT
        / "outputs/kinofail_realistic"
        / "corpus_c2_bidirectional_confirmation_v5_t3",
        ROOT
        / "outputs/eval/c2_bidirectional_v5/formal_features",
        ROOT / "outputs/eval/c2_bidirectional_v5/formal",
    )
    if any(path.exists() for path in (*outputs.values(), *outcome_paths)):
        raise RuntimeError(
            "C2 v5 protocol must be frozen before any v5 outcome"
        )

    asset_lock = (
        ROOT / "outputs/assets/terrain_pbr_v1/terrain_assets.lock.json"
    )
    t2_collector = (
        ROOT
        / "scripts/isaac_collect_kinofail_realistic_c1_causal_v1.py"
    )
    t2_runner = (
        ROOT / "scripts/run_kinofail_realistic_c1_causal_v1.py"
    )
    visual_helper = ROOT / "kino_vla/sim/c1_causal_visuals.py"
    now = datetime.now(UTC).isoformat()
    t2_protocol = {
        "schema_version": (
            "kinofail.realistic-c2-v5-t2-collection-protocol.v1"
        ),
        "protocol_id": "kinofail-realistic-c2-v5-t2-collection",
        "status": "frozen_before_v5_confirmation_collection",
        "created_utc": now,
        "design_tag": "c2_bidirectional_confirmation_v5_t2",
        "schedule_sha256": _sha(t2_schedule),
        "design_audit_sha256": _sha(t2_audit_path),
        "scene_registry_sha256": _sha(registry),
        "asset_lock_sha256": _sha(asset_lock),
        "collector_sha256": _sha(t2_collector),
        "runner_sha256": _sha(t2_runner),
        "visual_helper_sha256": _sha(visual_helper),
        "outcomes_available_at_freeze": False,
        "collection_contract": {
            "cases": 15,
            "samples_after_three_views": 90,
            "scene_clusters": 3,
            "domains": 3,
            "shared_proprioception_across_causes": True,
            "operator_physics_after_decision_boundary": True,
        },
    }
    _write(outputs["t2"], t2_protocol)

    t3_collector = (
        ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v3.py"
    )
    t3_runner = (
        ROOT / "scripts/run_kinofail_realistic_c2_v4_t3_isolated.py"
    )
    runtime_manifest = ROOT / "kino_vla/data/runtime_manifest.py"
    group_ids = sorted(
        {
            str(row["counterfactual_group_id"])
            for row in t3_rows
        }
    )
    t3_protocol = {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": "kinofail-realistic-c2-v5-t3-collection",
        "status": "frozen",
        "frozen_utc": now,
        "benchmark_id": "c2_bidirectional_confirmation_v5_t3",
        "scope": (
            "60 physical episodes / 30 nominal-anomaly O7/O8 "
            "pairs on three fresh C2 scene instances."
        ),
        "schedule_path": str(t3_schedule.relative_to(ROOT)),
        "schedule_sha256": _sha(t3_schedule),
        "scene_registry_path": str(registry.relative_to(ROOT)),
        "scene_registry_sha256": _sha(registry),
        "collector_path": str(t3_collector.relative_to(ROOT)),
        "collector_sha256": _sha(t3_collector),
        "runner_path": str(t3_runner.relative_to(ROOT)),
        "runner_sha256": _sha(t3_runner),
        "runtime_manifest_path": str(runtime_manifest.relative_to(ROOT)),
        "runtime_manifest_sha256": _sha(runtime_manifest),
        "design_audit_sha256": _sha(t3_audit_path),
        "case_schedule_sha256": _sha(t3_cases_path),
        "outcomes_available_at_freeze": False,
        "allowed": {
            "counterfactual_group_ids": group_ids,
            "target_operators": [
                "O7_visual_remap",
                "O8_invisible_collider",
            ],
            "scene_families": sorted(scenes),
            "physical_realizations": sorted(
                {
                    str(row["physical_realization"])
                    for row in t3_rows
                }
            ),
            "geometry_profiles": sorted(
                {str(row["geometry_profile"]) for row in t3_rows}
            ),
            "severity_ids": ["moderate", "severe"],
            "conditions": ["anomaly", "nominal_counterfactual"],
        },
        "collection_contract": {
            "physical_episodes": 60,
            "counterfactual_pairs": 30,
            "appearance_views_per_episode": 3,
            "scene_families": 3,
            "domains": 3,
            "operators": 2,
            "formal_collection_status": "formal_pilot",
            "runtime_schema": "kinofail.realistic-runtime.v5",
            "outcome_strength_is_reported_not_gated": True,
            "per_scene_parameter_tuning_forbidden": True,
            "isaac_processes": 30,
            "pairs_per_process": 1,
            "maximum_frame_highlight_fraction": 0.995,
            "maximum_mean_highlight_fraction": 0.45,
            "a8_in_scope": False,
        },
    }
    _write(outputs["t3"], t3_protocol)

    evaluator = (
        ROOT
        / "scripts/eval_kinofail_realistic_c2_bidirectional_v5.py"
    )
    feature_builder = (
        ROOT / "scripts/build_kinofail_realistic_c2_v5_features.py"
    )
    temporal = ROOT / "kino_vla/eval/c2_temporal_v5.py"
    evaluation_protocol = {
        "schema_version": (
            "kinofail.realistic-c2-bidirectional-protocol.v5"
        ),
        "protocol_id": (
            "kinofail-realistic-c2-bidirectional-confirmation-v5"
        ),
        "status": "frozen_before_v5_confirmation_collection",
        "created_utc": now,
        "model_predictions_available_at_freeze": False,
        "v5_confirmation_outcomes_available_at_freeze": False,
        "evaluator_sha256": _sha(evaluator),
        "feature_builder_sha256": _sha(feature_builder),
        "temporal_contract_sha256": _sha(temporal),
        "development_feature_manifest_sha256": _sha(
            development_manifest
        ),
        "development_report_sha256": _sha(development_report),
        "scene_registry_sha256": _sha(registry),
        "t2_schedule_sha256": _sha(t2_schedule),
        "t3_schedule_sha256": _sha(t3_schedule),
        "t2_collection_protocol_sha256": _sha(outputs["t2"]),
        "t3_collection_protocol_sha256": _sha(outputs["t3"]),
        "frozen_model_contract": {
            "selected_method": ours,
            "proprio_dimension": 80,
            "footprint_margin_m": 0.35,
            "post_encounter_delay_s": 0.30,
            "window_samples": 21,
            "outcome_aligned_window": False,
            "scene_or_operator_metadata_allowed": False,
        },
        "bootstrap": {
            "draws": 20000,
            "seed": 2026072451,
            "unit": (
                "scene cluster with matched case resampling nested "
                "within scene"
            ),
        },
        "acceptance": {
            "minimum_balanced_accuracy": 0.90,
            "minimum_worst_direction_accuracy": 0.85,
            "minimum_worst_scene_accuracy": 0.80,
            "minimum_texture_swap_hard_consistency": 0.80,
            "beat_every_baseline_point_estimate": True,
            "minimum_delta_ci95_lower": -0.05,
            "maximum_wrong_modality_direction_accuracy": 0.60,
        },
        "claim_boundary": (
            "C2 v5 tests bidirectional modality conflict on four "
            "attribution categories, three appearance interventions per "
            "cause, and three C2-fresh realistic scene instances. "
            "Together with six-scene LOSO development, it supports "
            "post-interaction structured modality selection under the "
            "frozen T2/T3 battery; it does not claim universal optimality "
            "across all eleven operators."
        ),
    }
    _write(outputs["evaluation"], evaluation_protocol)
    print(
        json.dumps(
            {
                "passed": True,
                "fresh_scenes": sorted(scenes),
                "prior_c2_scenes": len(prior_scenes),
                "t2_protocol": str(outputs["t2"]),
                "t2_protocol_sha256": _sha(outputs["t2"]),
                "t3_protocol": str(outputs["t3"]),
                "t3_protocol_sha256": _sha(outputs["t3"]),
                "evaluation_protocol": str(outputs["evaluation"]),
                "evaluation_protocol_sha256": _sha(
                    outputs["evaluation"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
