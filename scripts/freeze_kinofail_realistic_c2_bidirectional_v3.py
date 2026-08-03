#!/usr/bin/env python3
"""Freeze both C2 v3 acquisition streams and the final evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--t2-output",
        type=Path,
        default=ROOT
        / "configs/eval"
        / "kinofail_realistic_c2_v3_t2_collection.json",
    )
    parser.add_argument(
        "--t3-output",
        type=Path,
        default=ROOT
        / "configs/eval"
        / "kinofail_realistic_c2_v3_t3_collection.json",
    )
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        default=ROOT
        / "configs/eval"
        / "kinofail_realistic_c2_bidirectional_formal_v3.json",
    )
    args = parser.parse_args()
    outputs = [
        args.t2_output.resolve(),
        args.t3_output.resolve(),
        args.evaluation_output.resolve(),
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError(
            f"refusing to overwrite frozen C2 v3 protocol: {outputs}"
        )
    forbidden_outcomes = (
        ROOT
        / "outputs/kinofail_realistic"
        / "corpus_c2_bidirectional_confirmation_v3_t2",
        ROOT
        / "outputs/kinofail_realistic"
        / "corpus_c2_bidirectional_confirmation_v3_t3",
        ROOT
        / "outputs/eval/c2_bidirectional_v3/formal_features",
        ROOT / "outputs/eval/c2_bidirectional_v3/formal",
    )
    if any(path.exists() for path in forbidden_outcomes):
        raise RuntimeError(
            "C2 v3 outcomes exist; freeze must precede collection"
        )

    registry = (
        ROOT
        / "configs/data"
        / "kinofail_realistic_c2_v3_confirmation_scene_registry.json"
    )
    asset_lock = (
        ROOT
        / "outputs/assets/terrain_pbr_v1/terrain_assets.lock.json"
    )
    t2_design = (
        ROOT
        / "outputs/kinofail_realistic"
        / "design_c2_bidirectional_confirmation_v3_t2"
    )
    t3_design = (
        ROOT
        / "outputs/kinofail_realistic"
        / "design_c2_bidirectional_confirmation_v3_t3"
    )
    t2_schedule = t2_design / "schedule.jsonl"
    t2_audit_path = t2_design / "audit.json"
    t3_schedule = t3_design / "schedule.jsonl"
    t3_case_schedule = t3_design / "case_schedule.jsonl"
    t3_audit_path = t3_design / "audit.json"
    t2_audit = _json(t2_audit_path)
    t3_audit = _json(t3_audit_path)
    t2_rows = _jsonl(t2_schedule)
    t3_rows = _jsonl(t3_schedule)
    t3_cases = _jsonl(t3_case_schedule)
    if (
        t2_audit.get("passed") is not True
        or t3_audit.get("passed") is not True
        or _sha(t2_schedule) != t2_audit["schedule_sha256"]
        or _sha(t3_schedule) != t3_audit["schedule_sha256"]
        or _sha(t3_case_schedule)
        != t3_audit["case_schedule_sha256"]
        or len(t2_rows) != 15
        or len(t3_rows) != 60
        or len(t3_cases) != 15
    ):
        raise RuntimeError("C2 v3 frozen design is invalid")

    development_dir = (
        ROOT
        / "outputs/eval/c2_bidirectional_v3"
        / "development_features_nine_scene"
    )
    development_manifest_path = (
        development_dir / "feature_manifest.json"
    )
    development_report_path = (
        ROOT
        / "outputs/eval/c2_bidirectional_v3"
        / "development_screen_nine_scene_loso"
        / "development_report.json"
    )
    development_manifest = _json(development_manifest_path)
    development_report = _json(development_report_path)
    ours = "structured_modality_sparse_router_v3"
    if (
        development_manifest.get("passed") is not True
        or development_manifest.get(
            "formal_v2_outcomes_used_for_v3_development"
        )
        is not True
        or development_report.get(
            "v3_confirmation_outcomes_used"
        )
        is not False
        or development_report.get("scene_disjoint_loso") is not True
        or len(development_report["scene_clusters"]) != 9
        or development_report["family_gate_accuracy"] != 1.0
        or development_report["results"][ours][
            "balanced_accuracy"
        ]
        <= development_report["results"][
            development_report["best_baseline_method"]
        ]["balanced_accuracy"]
        or development_report["feature_manifest_sha256"]
        != _sha(development_manifest_path)
    ):
        raise RuntimeError(
            "C2 v3 development evidence is not freeze-ready"
        )

    c1_collector = (
        ROOT
        / "scripts/isaac_collect_kinofail_realistic_c1_causal_v1.py"
    )
    c1_runner = (
        ROOT / "scripts/run_kinofail_realistic_c1_causal_v1.py"
    )
    visual_helper = ROOT / "kino_vla/sim/c1_causal_visuals.py"
    t3_collector = (
        ROOT
        / "scripts"
        / "isaac_collect_kinofail_realistic_c2_v3_t3_scene.py"
    )
    t3_runner = (
        ROOT / "scripts/run_kinofail_realistic_c2_v3_t3.py"
    )
    runtime_manifest = ROOT / "kino_vla/data/runtime_manifest.py"
    t2_protocol = {
        "schema_version": (
            "kinofail.realistic-c2-v3-t2-collection-protocol.v1"
        ),
        "protocol_id": (
            "kinofail-realistic-c2-v3-t2-collection"
        ),
        "status": "frozen",
        "created_utc": datetime.now(UTC).isoformat(),
        "design_tag": t2_audit["design_tag"],
        "schedule_sha256": _sha(t2_schedule),
        "design_audit_sha256": _sha(t2_audit_path),
        "scene_registry_sha256": _sha(registry),
        "asset_lock_sha256": _sha(asset_lock),
        "collector_sha256": _sha(c1_collector),
        "runner_sha256": _sha(c1_runner),
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
    _write(outputs[0], t2_protocol)

    group_ids = sorted(
        {
            str(row["counterfactual_group_id"])
            for row in t3_rows
        }
    )
    t3_protocol = {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": (
            "kinofail-realistic-c2-v3-t3-collection"
        ),
        "status": "frozen",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "benchmark_id": t3_audit["design_tag"],
        "scope": (
            "60 physical episodes / 30 nominal-anomaly O7/O8 "
            "pairs on three fresh realistic scene instances."
        ),
        "schedule_path": str(t3_schedule.relative_to(ROOT)),
        "schedule_sha256": _sha(t3_schedule),
        "scene_registry_path": str(registry.relative_to(ROOT)),
        "scene_registry_sha256": _sha(registry),
        "collector_path": str(t3_collector.relative_to(ROOT)),
        "collector_sha256": _sha(t3_collector),
        "runner_path": str(t3_runner.relative_to(ROOT)),
        "runner_sha256": _sha(t3_runner),
        "runtime_manifest_path": str(
            runtime_manifest.relative_to(ROOT)
        ),
        "runtime_manifest_sha256": _sha(runtime_manifest),
        "design_audit_sha256": _sha(t3_audit_path),
        "case_schedule_sha256": _sha(t3_case_schedule),
        "allowed": {
            "counterfactual_group_ids": group_ids,
            "target_operators": [
                "O7_visual_remap",
                "O8_invisible_collider",
            ],
            "scene_families": sorted(
                {
                    str(row["scene_family"]) for row in t3_rows
                }
            ),
            "physical_realizations": sorted(
                {
                    str(row["physical_realization"])
                    for row in t3_rows
                }
            ),
            "geometry_profiles": sorted(
                {
                    str(row["geometry_profile"])
                    for row in t3_rows
                }
            ),
            "severity_ids": ["moderate", "severe"],
            "conditions": [
                "anomaly",
                "nominal_counterfactual",
            ],
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
            "a8_in_scope": False,
        },
    }
    _write(outputs[1], t3_protocol)

    evaluator = (
        ROOT
        / "scripts/eval_kinofail_realistic_c2_bidirectional_v3.py"
    )
    feature_builder = (
        ROOT
        / "scripts"
        / "build_kinofail_realistic_c2_v3_confirmation_features.py"
    )
    clip_extractor = (
        ROOT
        / "scripts/extract_kinofail_realistic_c1_causal_features_v1.py"
    )
    evaluation_protocol = {
        "schema_version": (
            "kinofail.realistic-c2-bidirectional-protocol.v3"
        ),
        "protocol_id": (
            "kinofail-realistic-c2-bidirectional-confirmation-v3"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": (
            "frozen_after_nine_scene_loso_before_v3_collection"
        ),
        "outcomes_available_at_freeze": False,
        "scene_registry_sha256": _sha(registry),
        "asset_lock_sha256": _sha(asset_lock),
        "t2_collection_protocol_sha256": _sha(outputs[0]),
        "t3_collection_protocol_sha256": _sha(outputs[1]),
        "t2_schedule_sha256": _sha(t2_schedule),
        "t3_schedule_sha256": _sha(t3_schedule),
        "t3_case_schedule_sha256": _sha(t3_case_schedule),
        "development_feature_manifest_sha256": _sha(
            development_manifest_path
        ),
        "development_report_sha256": _sha(
            development_report_path
        ),
        "evaluator_sha256": _sha(evaluator),
        "feature_builder_sha256": _sha(feature_builder),
        "clip_extractor_sha256": _sha(clip_extractor),
        "model_roster": [
            "vision_only",
            "proprio_only",
            "early_fusion_linear",
            "early_fusion_rbf",
            "late_fusion",
            ours,
        ],
        "frozen_architecture": {
            "family_gate": (
                "balanced logistic(C=1) on proprioception only"
            ),
            "T2_specialist": (
                "balanced logistic(C=0.1) on CLIP+generic HOG"
            ),
            "T3_specialist": (
                "balanced logistic(C=1) on proprioception"
            ),
            "composition": (
                "hard observable-family route; exactly one specialist "
                "is evaluated per sample"
            ),
            "forbidden_inputs": [
                "truth label",
                "operator ID",
                "cell ID",
                "scene ID",
                "domain",
                "split",
                "severity",
                "material ID",
                "formal correctness",
            ],
        },
        "development_result": {
            "v2_failure_used_as_development": True,
            "v3_confirmation_outcomes_used": False,
            "nine_scene_loso": True,
            "family_gate_accuracy": development_report[
                "family_gate_accuracy"
            ],
            "structured_balanced_accuracy": development_report[
                "results"
            ][ours]["balanced_accuracy"],
            "best_baseline": development_report[
                "best_baseline_method"
            ],
            "best_baseline_balanced_accuracy": development_report[
                "results"
            ][development_report["best_baseline_method"]][
                "balanced_accuracy"
            ],
        },
        "formal_design": {
            "fresh_scene_instances": 3,
            "domains": 3,
            "cases_per_direction": 15,
            "matched_cases": 30,
            "samples": 180,
            "appearance_views_per_cause": 3,
            "T2": (
                "byte-identical proprioception across causal visual "
                "alternatives"
            ),
            "T3": (
                "byte-identical O7 RTX sequence across actual O7/O8 "
                "proprioceptive rollouts"
            ),
        },
        "acceptance": {
            "minimum_balanced_accuracy": 0.88,
            "minimum_worst_direction_accuracy": 0.82,
            "minimum_worst_scene_accuracy": 0.80,
            "minimum_texture_swap_hard_consistency": 0.80,
            "beat_every_baseline_point_estimate": True,
            "minimum_delta_ci95_lower": -0.05,
            "maximum_wrong_modality_direction_accuracy": 0.60,
        },
        "bootstrap": {
            "unit": (
                "scene cluster with matched case resampling nested "
                "within scene"
            ),
            "draws": 20000,
            "seed": 2026072429,
        },
        "claim_boundary": (
            "C2 v3 tests bidirectional modality conflict on four "
            "attribution categories, three appearance interventions per "
            "cause, and three scene instances absent from all v3 "
            "development. It does not claim universal optimality across "
            "all eleven operators."
        ),
    }
    _write(outputs[2], evaluation_protocol)
    print(
        json.dumps(
            {
                "t2_protocol": str(outputs[0]),
                "t2_sha256": _sha(outputs[0]),
                "t3_protocol": str(outputs[1]),
                "t3_sha256": _sha(outputs[1]),
                "evaluation_protocol": str(outputs[2]),
                "evaluation_sha256": _sha(outputs[2]),
                "outcomes_available_at_freeze": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
