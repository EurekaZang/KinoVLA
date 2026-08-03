#!/usr/bin/env python3
"""Freeze C2 v2 before collecting or extracting its formal outcomes."""

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
        / "outputs/kinofail_realistic"
        / "design_c2_bidirectional_confirmation_v2_t2",
    )
    parser.add_argument(
        "--development-features",
        type=Path,
        default=ROOT
        / "outputs/eval/c2_bidirectional_v2"
        / "development_features_scene_disjoint",
    )
    parser.add_argument(
        "--development-report",
        type=Path,
        default=ROOT
        / "outputs/eval/c2_bidirectional_v2"
        / "development_screen_frozen_candidate"
        / "development_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "configs/eval/kinofail_realistic_c2_bidirectional_formal_v2.json",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    formal_corpus = (
        ROOT
        / "outputs/kinofail_realistic"
        / "corpus_c2_bidirectional_confirmation_v2_t2"
    )
    formal_features = (
        ROOT
        / "outputs/eval/c2_bidirectional_v2/formal_features"
    )
    formal_report = (
        ROOT
        / "outputs/eval/c2_bidirectional_v2/formal/report.json"
    )
    if (
        formal_corpus.exists()
        or formal_features.exists()
        or formal_report.exists()
    ):
        raise RuntimeError(
            "C2 v2 formal outcomes exist; freeze must precede collection"
        )

    design_dir = args.design_dir.resolve()
    schedule = design_dir / "schedule.jsonl"
    audit_path = design_dir / "audit.json"
    audit = _json(audit_path)
    schedule_rows = [
        json.loads(line)
        for line in schedule.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if (
        audit.get("passed") is not True
        or _sha(schedule) != audit["schedule_sha256"]
        or len(schedule_rows) != 21
        or {str(row["split"]) for row in schedule_rows} != {"test"}
    ):
        raise RuntimeError("C2 v2 T2 design is incomplete")
    development_dir = args.development_features.resolve()
    development_manifest_path = (
        development_dir / "feature_manifest.json"
    )
    development_manifest = _json(development_manifest_path)
    development_report_path = args.development_report.resolve()
    development_report = _json(development_report_path)
    if (
        development_manifest.get("passed") is not True
        or development_report.get("formal_outcomes_used") is not False
        or development_report.get("scene_disjoint_loso") is not True
        or development_report["feature_manifest_sha256"]
        != _sha(development_manifest_path)
    ):
        raise RuntimeError("C2 v2 development provenance is invalid")

    registry = (
        ROOT
        / "configs/data/kinofail_realistic_scale_scene_registry_v1.json"
    )
    asset_lock = (
        ROOT / "outputs/assets/terrain_pbr_v1/terrain_assets.lock.json"
    )
    prior_t3_schedule = (
        ROOT
        / "outputs/eval/c2_bidirectional_formal_v1"
        / "features/t3_pair_schedule.jsonl"
    )
    scale_snapshots = (
        ROOT / "outputs/eval/realistic_a0_a7_v6/snapshots"
    )
    paths = {
        "schedule_builder": (
            ROOT
            / "scripts/build_kinofail_realistic_c2_t2_schedule_v2.py"
        ),
        "collector": (
            ROOT
            / "scripts/isaac_collect_kinofail_realistic_c1_causal_v1.py"
        ),
        "visual_helper": ROOT / "kino_vla/sim/c1_causal_visuals.py",
        "runner": (
            ROOT / "scripts/run_kinofail_realistic_c1_causal_v1.py"
        ),
        "clip_extractor": (
            ROOT
            / "scripts/extract_kinofail_realistic_c1_causal_features_v1.py"
        ),
        "geometry_extractor": (
            ROOT
            / "scripts/extract_kinofail_realistic_c2_geometry_v2.py"
        ),
        "formal_feature_builder": (
            ROOT
            / "scripts/build_kinofail_realistic_c2_confirmation_features_v2.py"
        ),
        "development_feature_builder": (
            ROOT
            / "scripts/build_kinofail_realistic_c2_v2_development_features.py"
        ),
        "evaluator": (
            ROOT
            / "scripts/eval_kinofail_realistic_c2_bidirectional_v2.py"
        ),
    }
    protocol = {
        "schema_version": (
            "kinofail.realistic-c2-bidirectional-protocol.v2"
        ),
        "protocol_id": (
            "kinofail-realistic-c2-bidirectional-confirmation-v2"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": (
            "frozen_after_scene_disjoint_development_before_formal_collection"
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
        "prior_t3_schedule": str(prior_t3_schedule.relative_to(ROOT)),
        "prior_t3_schedule_sha256": _sha(prior_t3_schedule),
        "scale_snapshot_records_sha256": _sha(
            scale_snapshots / "snapshot_records.jsonl"
        ),
        "scale_snapshots_sha256": _sha(
            scale_snapshots / "snapshots.npz"
        ),
        "scale_extraction_audit_sha256": _sha(
            scale_snapshots / "extraction_audit.json"
        ),
        **{
            f"{name}_sha256": _sha(path)
            for name, path in paths.items()
        },
        "development_feature_manifest": str(
            development_manifest_path.relative_to(ROOT)
        ),
        "development_feature_manifest_sha256": _sha(
            development_manifest_path
        ),
        "development_report": str(
            development_report_path.relative_to(ROOT)
        ),
        "development_report_sha256": _sha(
            development_report_path
        ),
        "development_result": {
            "formal_outcomes_used": False,
            "six_scene_loso": True,
            "structured_balanced_accuracy": development_report[
                "results"
            ]["structured_conditional_router_v2"][
                "balanced_accuracy"
            ],
            "best_baseline": development_report[
                "best_baseline_method"
            ],
            "best_baseline_balanced_accuracy": development_report[
                "results"
            ][development_report["best_baseline_method"]][
                "balanced_accuracy"
            ],
            "family_gate_accuracy": development_report[
                "family_gate_accuracy"
            ],
        },
        "model_roster": [
            "vision_only",
            "proprio_only",
            "early_fusion_linear",
            "early_fusion_rbf",
            "late_fusion",
            "structured_conditional_router_v2",
        ],
        "frozen_architecture": {
            "shared_observable_inputs": [
                "Go2-front five-frame ground-ROI CLIP summary",
                "same-ROI generic grayscale HOG temporal summary",
                "21-sample proprioceptive temporal summary",
            ],
            "strong_early_fusion_baseline": (
                "RBF SVC(C=100,gamma=1e-4), selected in the same "
                "six-scene LOSO development grid"
            ),
            "structured_family_gate": (
                "ExtraTrees(512,max_features=sqrt,min_samples_leaf=2) "
                "on all observable inputs"
            ),
            "structured_t2_specialist": (
                "RBF SVC(C=0.1,gamma=scale) on visual CLIP+generic HOG"
            ),
            "structured_t3_specialist": (
                "balanced logistic regression(C=1) on proprioception"
            ),
            "composition": (
                "family probability multiplied by the corresponding "
                "within-family specialist probability"
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
        "formal_design": {
            "T2": (
                "21 fresh shared-physics cases; proprioception is "
                "byte-identical across O2/O4 labels"
            ),
            "T3": (
                "21 previously unused matched O7/O8 physics cases; "
                "RTX sequence is byte-identical across labels"
            ),
            "test_cases": 42,
            "test_samples": 252,
            "test_scene_clusters": 3,
            "test_domains": 3,
            "appearance_views_per_label_case": 3,
            "scene_clusters_disjoint_from_development": True,
            "statistical_unit": (
                "scene cluster with matched-case resampling nested "
                "within scene"
            ),
        },
        "bootstrap": {
            "draws": 20000,
            "seed": 2026072417,
        },
        "acceptance": {
            "minimum_balanced_accuracy": 0.88,
            "minimum_worst_direction_accuracy": 0.82,
            "minimum_worst_scene_accuracy": 0.80,
            "minimum_texture_swap_hard_consistency": 0.80,
            "beats_every_baseline_point": True,
            "minimum_delta_ci95_lower": -0.05,
            "maximum_wrong_modality_direction_accuracy": 0.60,
        },
        "outcomes_available_at_freeze": False,
        "formal_test_used_for_model_or_baseline_selection": False,
        "claim_boundary": (
            "C2 v2 supports learned observable evidence routing on the "
            "predeclared O2/O4 and O7/O8 bidirectional conflict battery "
            "across three held-out realistic domains. It does not claim "
            "universal open-world conflict resolution. Scores are expected "
            "to retain benchmark headroom rather than saturate."
        ),
    }
    # Collector compatibility fields use these exact names.
    protocol["collector_sha256"] = _sha(paths["collector"])
    protocol["visual_helper_sha256"] = _sha(paths["visual_helper"])
    protocol["evaluator_sha256"] = _sha(paths["evaluator"])
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
                "formal_cases": 42,
                "outcomes_available_at_freeze": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
