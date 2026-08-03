#!/usr/bin/env python3
"""Freeze the structured C1 v3 confirmation before formal outcomes exist."""

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
        / "outputs/kinofail_realistic/design_c1_causal_confirmation_v3",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_realistic_c1_causal_confirmation_v3.json",
    )
    args = parser.parse_args()
    design_dir = args.design_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    schedule = design_dir / "schedule.jsonl"
    audit_path = design_dir / "audit.json"
    audit = _json(audit_path)
    rows = [
        json.loads(line)
        for line in schedule.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if (
        audit.get("passed") is not True
        or _sha(schedule) != audit["schedule_sha256"]
        or len(rows) != 21
        or {row["split"] for row in rows} != {"test"}
    ):
        raise RuntimeError("C1 v3 formal design is incomplete")

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
        ROOT / "scripts/analyze_kinofail_realistic_c1_confirmation_v3.py"
    )
    builder = (
        ROOT / "scripts/build_kinofail_realistic_c1_confirmation_schedule_v3.py"
    )
    development_feature_manifest = (
        ROOT
        / "outputs/eval/c1_causal_structured_v3_development/features"
        / "feature_manifest.json"
    )
    development_geometry_manifest = (
        ROOT
        / "outputs/eval/c1_causal_structured_v3_development/geometry"
        / "geometry_manifest.json"
    )
    development_protocol = (
        ROOT
        / "configs/data/kinofail_realistic_c1_causal_structured_v3_development.json"
    )
    prior_reports = [
        ROOT / "outputs/eval/c1_causal_formal_v1/formal/report.json",
        ROOT / "outputs/eval/c1_causal_confirmation_v2/formal/report.json",
    ]
    if not all(_json(path).get("passed") is False for path in prior_reports):
        raise RuntimeError("C1 v3 requires both preserved prior failures")

    protocol = {
        "schema_version": "kinofail.realistic-c1-confirmation-protocol.v3",
        "protocol_id": "kinofail-realistic-c1-causal-confirmation-v3",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": (
            "formal_frozen_after_v1_v2_failures_before_v3_outcomes"
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
        "schedule_builder_sha256": _sha(builder),
        "collector": str(collector.relative_to(ROOT)),
        "collector_sha256": _sha(collector),
        "visual_helper": str(helper.relative_to(ROOT)),
        "visual_helper_sha256": _sha(helper),
        "runner_sha256": _sha(runner),
        "feature_extractor_sha256": _sha(clip_extractor),
        "geometry_extractor_sha256": _sha(geometry_extractor),
        "analyzer": str(analyzer.relative_to(ROOT)),
        "analyzer_sha256": _sha(analyzer),
        "development_protocol_sha256": _sha(development_protocol),
        "development_feature_manifest_sha256": _sha(
            development_feature_manifest
        ),
        "development_geometry_manifest_sha256": _sha(
            development_geometry_manifest
        ),
        "development_source_description": (
            "12 structured-v3 development cases: two fresh nuisance profiles "
            "in each of six non-test realistic scenes; formal v3 test scenes "
            "and case IDs were absent from model selection"
        ),
        "preserved_negative_evidence": [
            {
                "report": str(path.relative_to(ROOT)),
                "report_sha256": _sha(path),
                "passed": False,
            }
            for path in prior_reports
        ],
        "sample_design": {
            "fresh_shared_prefix_cases": 21,
            "scene_clusters": 3,
            "domains": ["life", "production", "wild"],
            "split": "test only",
            "causes_per_case": 2,
            "appearance_views_per_cause": 3,
            "rgb_frames_per_sample": 5,
            "proprio_samples_per_sample": 21,
            "statistical_unit": "scene cluster",
            "nested_repetitions": (
                "seven trajectory nuisance profiles per scene and three "
                "material interventions per cause"
            ),
        },
        "causal_contract": {
            "same_physics_rollout_for_o2_and_o4": True,
            "proprio_bytes_must_be_identical_across_labels": True,
            "render_only_counterfactual_must_not_advance_physics": True,
            "same_pbr_material_view_for_both_labels": True,
            "all_imageable_descendants_visibility_is_explicit": True,
            "four_render_only_settle_frames_after_each_counterfactual_switch": True,
            "class_dependent_color_or_texture_forbidden": True,
            "cause_geometry_has_no_collision": True,
            "decision_boundary_precedes_operator_region": True,
        },
        "frozen_analysis": {
            "visual_input": (
                "Go2-front RTX RGB; full width and rows 45%-100%"
            ),
            "visual_model": (
                "RBF SVC on frozen CLIP mean+last+delta temporal features; "
                "generic HOG is extracted for audit but excluded by frozen "
                "leave-one-scene-out selection"
            ),
            "visual_C": 10.0,
            "visual_gamma": 0.0001,
            "proprio_C": 0.001,
            "hard_prediction_threshold": 0.5,
            "selection_provenance": (
                "six-scene leave-one-scene-out development screen; selected "
                "for highest macro scene AUC with strong minimum-scene AUC"
            ),
            "development_loso": {
                "macro_scene_auc": 0.9722222222222223,
                "minimum_scene_auc": 0.9166666666666667,
                "macro_accuracy": 0.8055555555555557,
                "texture_hard_consistency": 0.7916666666666666,
            },
            "uncertainty": {
                "method": "scene-cluster bootstrap",
                "draws": 20000,
                "seed": 2026072415,
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
                "minimum_balanced_accuracy": 0.75,
                "minimum_texture_hard_consistency": 0.75,
                "proprio_byte_identity_rate": 1.0,
                "proprio_auc": 0.5,
                "maximum_paired_proprio_probability_delta": 1e-12,
            }
        },
        "claim_boundary": (
            "C1 v3 tests constructive pre-contact O2/O4 identifiability under "
            "fresh trajectory and material nuisance on three held-out realistic "
            "scene/domain clusters. It requires strong rank identifiability and "
            "nontrivial hard texture consistency while deliberately retaining "
            "headroom; it is not a universal open-world theorem."
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
                "cases": len(rows),
                "outcomes_available_at_freeze": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
