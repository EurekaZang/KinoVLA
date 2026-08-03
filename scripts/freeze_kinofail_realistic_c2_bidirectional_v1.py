#!/usr/bin/env python3
"""Freeze C2 after validation-only architecture selection and before test."""

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
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    feature_dir = args.features.resolve()
    development_path = args.development_report.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest_path = feature_dir / "feature_manifest.json"
    manifest = _json(manifest_path)
    development = _json(development_path)
    evaluator = ROOT / "scripts/eval_kinofail_realistic_c2_bidirectional_v1.py"
    builder = (
        ROOT / "scripts/build_kinofail_realistic_c2_bidirectional_features_v1.py"
    )
    if manifest.get("passed") is not True:
        raise RuntimeError("C2 feature manifest did not pass")
    if development.get("test_outcomes_used") is not False:
        raise RuntimeError("C2 development report is not validation-only")
    if development["evaluation_split"] != "val":
        raise RuntimeError("C2 model selection did not use the validation split")
    if development["feature_manifest_sha256"] != _sha(manifest_path):
        raise RuntimeError("C2 development/feature provenance mismatch")
    if development["evaluator_sha256"] != _sha(evaluator):
        raise RuntimeError("C2 evaluator changed after validation screen")
    protocol = {
        "schema_version": "kinofail.realistic-c2-bidirectional-protocol.v1",
        "protocol_id": "kinofail-realistic-c2-bidirectional-formal-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_after_validation_before_first_test_evaluation",
        "feature_manifest": str(manifest_path.relative_to(ROOT)),
        "feature_manifest_sha256": _sha(manifest_path),
        "feature_builder": str(builder.relative_to(ROOT)),
        "feature_builder_sha256": _sha(builder),
        "evaluator": str(evaluator.relative_to(ROOT)),
        "evaluator_sha256": _sha(evaluator),
        "development_report": str(development_path.relative_to(ROOT)),
        "development_report_sha256": _sha(development_path),
        "selected_C": development["selected_C"],
        "best_baseline_method": development["best_baseline_method"],
        "model_roster": [
            "vision_only",
            "proprio_only",
            "early_fusion",
            "late_fusion",
            "structured_evidence_router",
        ],
        "structured_method": {
            "family_gate": (
                "observable joint feature gate: T2 vision-decisive versus "
                "T3 proprio-decisive"
            ),
            "T2_expert": "visual-only O2/O4 classifier",
            "T3_expert": "proprio-only O7/O8 classifier",
            "probability_composition": (
                "family probability times the corresponding binary expert"
            ),
            "deployment_inputs": [
                "five Go2-front ground-ROI CLIP embeddings",
                "21-sample proprioceptive temporal summary",
            ],
            "forbidden_deployment_inputs": [
                "truth label",
                "operator ID",
                "cell ID",
                "scene ID",
                "domain",
                "split",
                "severity",
                "material ID",
                "test correctness",
            ],
        },
        "test_design": {
            "directions": {
                "T2": "proprio byte-identical; physical visual cue differs",
                "T3": "RTX sequence byte-identical; actual physical proprio differs",
            },
            "test_scene_clusters": 3,
            "test_domains": 3,
            "test_cases": 18,
            "test_samples": 108,
            "appearance_views_per_label_case": 3,
            "statistical_unit": (
                "scene cluster with matched case resampling nested within scene"
            ),
        },
        "bootstrap": {
            "draws": 20000,
            "seed": 2026072412,
        },
        "acceptance": {
            "minimum_balanced_accuracy": 0.90,
            "minimum_worst_direction_accuracy": 0.85,
            "minimum_texture_swap_hard_consistency": 0.90,
            "beats_vision_only_point": True,
            "beats_proprio_only_point": True,
            "ours_minus_best_unstructured_ci_lower_gt_0": True,
        },
        "test_outcomes_available_at_freeze": False,
        "test_used_for_model_or_baseline_selection": False,
        "claim_boundary": (
            "C2 is supported only for the predeclared O2/O4 and O7/O8 "
            "bidirectional conflict battery in three held-out realistic domains. "
            "The result demonstrates learned observable evidence routing; it does "
            "not claim universal open-world conflict resolution."
        ),
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
                "selected_C": protocol["selected_C"],
                "best_baseline_method": protocol["best_baseline_method"],
                "test_outcomes_available_at_freeze": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
