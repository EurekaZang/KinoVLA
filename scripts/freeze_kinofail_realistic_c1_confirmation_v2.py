#!/usr/bin/env python3
"""Freeze the untouched C1 confirmation protocol after the v1 gate failure."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--design-dir",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/design_c1_causal_confirmation_v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_realistic_c1_causal_confirmation_v2.json",
    )
    args = parser.parse_args()
    design_dir = args.design_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol: {output}")
    schedule = design_dir / "schedule.jsonl"
    audit_path = design_dir / "audit.json"
    audit = _json(audit_path)
    if audit.get("passed") is not True:
        raise RuntimeError("C1 confirmation design audit did not pass")
    if _sha(schedule) != audit["schedule_sha256"]:
        raise RuntimeError("C1 confirmation schedule/audit mismatch")
    rows = [
        json.loads(line)
        for line in schedule.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != 21 or {row["split"] for row in rows} != {"test"}:
        raise RuntimeError("C1 confirmation schedule is not 21 test-only cases")
    if {row["benchmark_id"] for row in rows} != {audit["design_tag"]}:
        raise RuntimeError("C1 confirmation design tag mismatch")

    registry = (
        ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json"
    )
    asset_lock = ROOT / "outputs/assets/terrain_pbr_v1/terrain_assets.lock.json"
    collector = (
        ROOT / "scripts/isaac_collect_kinofail_realistic_c1_causal_v1.py"
    )
    helper = ROOT / "kino_vla/sim/c1_causal_visuals.py"
    runner = ROOT / "scripts/run_kinofail_realistic_c1_causal_v1.py"
    clip_extractor = (
        ROOT / "scripts/extract_kinofail_realistic_c1_causal_features_v1.py"
    )
    geometry_extractor = (
        ROOT / "scripts/extract_kinofail_realistic_c1_geometry_v2.py"
    )
    analyzer = (
        ROOT / "scripts/analyze_kinofail_realistic_c1_confirmation_v2.py"
    )
    schedule_builder = (
        ROOT / "scripts/build_kinofail_realistic_c1_extension_schedule_v2.py"
    )
    prior_report_path = (
        ROOT / "outputs/eval/c1_causal_formal_v1/formal/report.json"
    )
    prior_report = _json(prior_report_path)
    if prior_report.get("passed") is not False:
        raise RuntimeError("C1 v2 protocol expects the preserved v1 gate failure")
    failed = [
        key for key, value in prior_report["checks"].items() if value is False
    ]
    if failed != ["minimum_texture_swap_hard_consistency"]:
        raise RuntimeError(f"unexpected C1 v1 failure set: {failed}")
    development_feature_manifest = (
        ROOT / "outputs/eval/c1_causal_formal_v1/features/feature_manifest.json"
    )
    development_geometry_manifest = (
        ROOT / "outputs/eval/c1_causal_formal_v1/geometry_v2/geometry_manifest.json"
    )
    development_geometry = _json(development_geometry_manifest)
    if development_geometry.get("passed") is not True:
        raise RuntimeError("C1 v2 development geometry is incomplete")

    protocol = {
        "schema_version": "kinofail.realistic-c1-confirmation-protocol.v2",
        "protocol_id": "kinofail-realistic-c1-causal-confirmation-v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": (
            "formal_frozen_after_v1_failure_before_confirmation_outcomes"
        ),
        "design_tag": audit["design_tag"],
        "schedule": str(schedule.relative_to(ROOT)),
        "schedule_sha256": _sha(schedule),
        "design_audit": str(audit_path.relative_to(ROOT)),
        "design_audit_sha256": _sha(audit_path),
        "scene_registry": str(registry.relative_to(ROOT)),
        "scene_registry_sha256": _sha(registry),
        "asset_lock": str(asset_lock.relative_to(ROOT)),
        "asset_lock_sha256": _sha(asset_lock),
        "schedule_builder_sha256": _sha(schedule_builder),
        "collector": str(collector.relative_to(ROOT)),
        "collector_sha256": _sha(collector),
        "visual_helper": str(helper.relative_to(ROOT)),
        "visual_helper_sha256": _sha(helper),
        "runner": str(runner.relative_to(ROOT)),
        "runner_sha256": _sha(runner),
        "feature_extractor": str(clip_extractor.relative_to(ROOT)),
        "feature_extractor_sha256": _sha(clip_extractor),
        "geometry_extractor": str(geometry_extractor.relative_to(ROOT)),
        "geometry_extractor_sha256": _sha(geometry_extractor),
        "analyzer": str(analyzer.relative_to(ROOT)),
        "analyzer_sha256": _sha(analyzer),
        "development_feature_manifest_sha256": _sha(
            development_feature_manifest
        ),
        "development_geometry_manifest_sha256": _sha(
            development_geometry_manifest
        ),
        "prior_v1": {
            "report": str(prior_report_path.relative_to(ROOT)),
            "report_sha256": _sha(prior_report_path),
            "preserved_passed": False,
            "failed_gate": failed[0],
            "core_macro_scene_auc": prior_report["visual"][
                "primary_macro_within_scene_roc_auc"
            ],
            "proprio_auc": prior_report["proprio"]["roc_auc"],
            "texture_hard_consistency": prior_report["visual"][
                "texture_swap"
            ]["hard_consistency"],
            "role_for_v2": (
                "all 27 v1 cases are development-only for representation and "
                "hyperparameter selection; they are never pooled with the "
                "untouched confirmation outcomes"
            ),
        },
        "sample_design": {
            "fresh_shared_prefix_cases": len(rows),
            "scene_clusters": len({row["scene_cluster"] for row in rows}),
            "domains": sorted({row["domain"] for row in rows}),
            "split": "test only",
            "causes_per_case": 2,
            "appearance_views_per_cause": 3,
            "rgb_frames_per_sample": 5,
            "proprio_samples_per_sample": 21,
            "statistical_unit": "scene cluster",
            "nested_repetitions": (
                "seven fresh reset/nuisance profiles per scene and three "
                "appearance interventions per cause"
            ),
            "fresh_case_ids_disjoint_from_v1": True,
        },
        "causal_contract": {
            "same_physics_rollout_for_o2_and_o4": True,
            "proprio_bytes_must_be_identical_across_labels": True,
            "render_only_counterfactual_must_not_advance_physics": True,
            "same_pbr_material_view_for_both_labels": True,
            "class_dependent_color_or_texture_forbidden": True,
            "cause_geometry_has_no_collision": True,
            "decision_boundary_precedes_operator_region": True,
        },
        "frozen_analysis": {
            "training_data": "all 27 preserved C1 v1 cases",
            "confirmation_data": (
                "21 fresh cases from this protocol; opened once"
            ),
            "visual_input": (
                "Go2-front RTX RGB; full width and rows 45%-100%"
            ),
            "visual_model": (
                "L2 logistic regression on concatenated frozen CLIP temporal "
                "summary and label-agnostic grayscale-HOG temporal summary"
            ),
            "visual_C": 0.001,
            "proprio_C": 0.001,
            "hard_prediction_threshold": 0.5,
            "selection_provenance": (
                "C and descriptor fixed from leave-one-scene-out screening on "
                "the already-open v1 cases only"
            ),
            "uncertainty": {
                "method": "scene-cluster bootstrap",
                "draws": 20000,
                "seed": 2026072414,
            },
        },
        "acceptance": {
            "c1_confirmation": {
                "all_case_manifests_pass": True,
                "minimum_scene_clusters": 3,
                "minimum_domains": 3,
                "minimum_macro_scene_auc": 0.90,
                "minimum_per_scene_auc": 0.85,
                "minimum_scene_ci95_lower": 0.85,
                "minimum_balanced_accuracy": 0.85,
                "minimum_texture_hard_consistency": 0.90,
                "proprio_byte_identity_rate": 1.0,
                "proprio_auc": 0.5,
                "maximum_paired_proprio_probability_delta": 1e-12,
            }
        },
        "claim_boundary": (
            "This confirmation establishes the constructive O2/O4 arm of C1 "
            "under fresh trajectory nuisance and appearance interventions on "
            "three held-out realistic scene/domain clusters: paired pre-contact "
            "proprioception is exactly non-identifying, whereas generic RGB "
            "appearance plus geometry is identifying and material-consistent. "
            "It is not a universal open-world identifiability theorem."
        ),
        "outcomes_available_at_freeze": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output.relative_to(ROOT)),
                "protocol_sha256": _sha(output),
                "case_count": len(rows),
                "scene_clusters": protocol["sample_design"][
                    "scene_clusters"
                ],
                "outcomes_available_at_freeze": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
