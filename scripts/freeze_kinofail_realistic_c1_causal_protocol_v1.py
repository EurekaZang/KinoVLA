#!/usr/bin/env python3
"""Freeze a provenance-bound realistic C1 causal battery protocol."""

from __future__ import annotations

import argparse
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


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument(
        "--status",
        choices=("development_preflight", "formal_frozen"),
        required=True,
    )
    parser.add_argument(
        "--excluded-development-root",
        action="append",
        default=[],
    )
    args = parser.parse_args()
    design_dir = _resolve(args.design_dir)
    output = _resolve(args.output)
    schedule = design_dir / "schedule.jsonl"
    audit_path = design_dir / "audit.json"
    collector = ROOT / "scripts/isaac_collect_kinofail_realistic_c1_causal_v1.py"
    helper = ROOT / "kino_vla/sim/c1_causal_visuals.py"
    extractor = (
        ROOT / "scripts/extract_kinofail_realistic_c1_causal_features_v1.py"
    )
    analyzer = ROOT / "scripts/analyze_kinofail_realistic_c1_causal_v1.py"
    runner = ROOT / "scripts/run_kinofail_realistic_c1_causal_v1.py"
    registry = (
        ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json"
    )
    asset_lock = ROOT / "outputs/assets/terrain_pbr_v1/terrain_assets.lock.json"
    audit = _json(audit_path)
    if audit.get("passed") is not True:
        raise RuntimeError("C1 causal design audit did not pass")
    if _sha(schedule) != audit["schedule_sha256"]:
        raise RuntimeError("C1 schedule no longer matches its design audit")
    rows = [
        json.loads(line)
        for line in schedule.read_text(encoding="utf-8").splitlines()
        if line
    ]
    design_tags = {str(row["benchmark_id"]) for row in rows}
    if design_tags != {str(audit["design_tag"])}:
        raise RuntimeError("C1 schedule/audit design-tag mismatch")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    protocol = {
        "schema_version": "kinofail.realistic-c1-causal-protocol.v1",
        "protocol_id": args.protocol_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": args.status,
        "design_tag": audit["design_tag"],
        "schedule": str(schedule.relative_to(ROOT)),
        "schedule_sha256": _sha(schedule),
        "design_audit": str(audit_path.relative_to(ROOT)),
        "design_audit_sha256": _sha(audit_path),
        "scene_registry": str(registry.relative_to(ROOT)),
        "scene_registry_sha256": _sha(registry),
        "asset_lock": str(asset_lock.relative_to(ROOT)),
        "asset_lock_sha256": _sha(asset_lock),
        "collector": str(collector.relative_to(ROOT)),
        "collector_sha256": _sha(collector),
        "visual_helper": str(helper.relative_to(ROOT)),
        "visual_helper_sha256": _sha(helper),
        "feature_extractor": str(extractor.relative_to(ROOT)),
        "feature_extractor_sha256": _sha(extractor),
        "analyzer": str(analyzer.relative_to(ROOT)),
        "analyzer_sha256": _sha(analyzer),
        "runner": str(runner.relative_to(ROOT)),
        "runner_sha256": _sha(runner),
        "sample_design": {
            "shared_physics_prefix_cases": len(rows),
            "scene_clusters": len({row["scene_cluster"] for row in rows}),
            "domains": sorted({row["domain"] for row in rows}),
            "splits": sorted({row["split"] for row in rows}),
            "causes_per_case": 2,
            "appearance_views_per_cause": 3,
            "rgb_frames_per_sample": 5,
            "proprio_samples_per_sample": 21,
            "statistical_unit": "scene cluster",
            "nested_repetitions": "three reset profiles and three appearance views",
        },
        "causal_contract": {
            "same_physics_rollout_for_o2_and_o4": True,
            "proprio_bytes_must_be_identical_across_labels": True,
            "render_only_counterfactual_must_not_advance_physics": True,
            "same_pbr_material_view_for_both_labels": True,
            "class_dependent_color_or_texture_forbidden": True,
            "class_signal": [
                "soft-surface metric deformation",
                "overlapping protective film and metric creases",
            ],
            "cause_geometry_has_no_collision": True,
            "decision_boundary_precedes_operator_region": True,
        },
        "frozen_analysis": {
            "visual_input": (
                "Go2-front RTX RGB; full width and rows 45%-100% only"
            ),
            "visual_encoder": "local frozen CLIP ViT-B/32",
            "temporal_summary": "mean + last + last-minus-first over five frames",
            "proprio_summary": (
                "mean/std/min/max/q25/q75/first/last/delta/slope"
            ),
            "classifier": "L2 logistic regression",
            "candidate_C": [0.01, 0.1, 1.0, 10.0],
            "selection": (
                "maximize validation macro within-scene ROC-AUC; ties choose smaller C"
            ),
            "test_policy": "held-out test split opened once after selection",
            "primary_visual_estimand": (
                "macro within-scene ROC-AUC over three held-out scene clusters"
            ),
            "uncertainty": {
                "method": "scene-cluster bootstrap",
                "draws": 20000,
                "seed": 2026072411,
            },
            "texture_consistency": (
                "within-case hard prediction agreement and probability dispersion "
                "across the three shared material views"
            ),
        },
        "acceptance": {
            "all_case_manifests_pass": True,
            "minimum_test_scene_clusters": 3,
            "minimum_test_domains": 3,
            "proprio_byte_identity_rate": 1.0,
            "maximum_paired_proprio_probability_delta": 1e-12,
            "proprio_roc_auc_target": 0.5,
            "minimum_visual_macro_within_scene_auc": 0.90,
            "minimum_visual_per_scene_auc": 0.85,
            "minimum_visual_scene_bootstrap_ci95_lower": 0.85,
            "minimum_texture_swap_hard_consistency": 0.90,
        },
        "claim_boundary": (
            "This battery establishes the constructive O2/O4 arm of C1: before "
            "contact, paired proprioception contains no label information while "
            "front-camera physical appearance can distinguish the causes across "
            "unseen realistic scenes and PBR materials. It is not a universal "
            "open-world identifiability claim."
        ),
        "outcomes_available_at_freeze": False,
        "excluded_development_data": (
            []
            if args.status == "development_preflight"
            else [
                {
                    "root": root,
                    "counts_as_formal_evidence": False,
                    "reason": (
                        "engineering visibility/runtime preflight used before "
                        "fresh formal case IDs and seeds were generated"
                    ),
                }
                for root in (
                    args.excluded_development_root
                    or [
                        "outputs/kinofail_realistic/development_c1_causal_v1",
                        "outputs/kinofail_realistic/development_c1_causal_v2",
                    ]
                )
            ]
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "output": str(output.relative_to(ROOT)),
        "protocol_sha256": _sha(output),
        "status": args.status,
        "design_tag": audit["design_tag"],
        "case_count": len(rows),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
