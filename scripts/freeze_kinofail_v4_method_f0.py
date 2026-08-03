#!/usr/bin/env python3
"""Freeze the selected KiNO v4 method before independent confirmation.

The script consumes development data only.  It fits one final KiNO checkpoint
and favorable battery-specific baselines, records every relevant input/code
hash, and writes an immutable F0 bundle.  Confirmation observations must not
exist when this script is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.conflict_invariant_kino import (  # noqa: E402
    ConflictInvariantKiNO,
    invariant_relative_proprio_features,
    physical_group_weights,
    visual_context_features,
)
from scripts.develop_kinofail_conflict_invariant_kino_v4 import (  # noqa: E402
    _load_conflict,
    _load_scale,
)


DEFAULT_OUTPUT = ROOT / "outputs/freeze/kino_v4_dinov2large_terrain448_f0"
T2_DINO = (
    ROOT
    / "outputs/eval/kino_t2_dinov2_large_448_terrain_v4_development/features.npz"
)
CONFLICT_DINO = (
    ROOT
    / "outputs/eval/kino_conflict_dinov2_large_448_terrain_v4_development/features.npz"
)
DEVELOPMENT_REPORT = (
    ROOT
    / "outputs/eval/kino_conflict_invariant_v4_development_dinov2large_terrain448_eta011/report.json"
)
BASELINE_REPORT = (
    ROOT
    / "outputs/eval/kino_v4_fair_baselines_dinov2large_terrain448_eta011_development/report.json"
)
STATISTICS_REPORT = (
    ROOT
    / "outputs/eval/kino_v4_paired_statistics_dinov2large_terrain448_eta011_development/report.json"
)
CONFIRMATION_ROOT = ROOT / "outputs/kinofail_kino_v4_confirmation_v1"

ROUTE_THRESHOLD = 0.11
VISUAL_REGULARIZATION = 0.03
MODEL_SEED = 2026080351
BASELINE_SEED = 2026080361
DINO_MODEL_ID = "facebook/dinov2-large"
DINO_REVISION = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _baseline_estimator(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=512,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=-1,
    )


def _load_dino(
    path: Path,
    expected_ids: np.ndarray,
    *,
    pool: str = "ground_mean",
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["sample_ids"].astype(str)
        values = np.asarray(archive[f"visual_{pool}"], dtype=np.float32)
    if not np.array_equal(ids, expected_ids.astype(str)):
        raise RuntimeError(f"DINO cache does not align: {path}")
    if values.ndim != 2 or values.shape[1] != 2048:
        raise RuntimeError(f"unexpected DINO feature shape: {values.shape}")
    return values


def _fit_kino(scale: dict[str, Any], conflict: dict[str, Any]) -> ConflictInvariantKiNO:
    t2 = conflict["vision_decisive"]
    t2_detail = _load_dino(T2_DINO, conflict["sample_ids"][t2])
    scale_detail = np.zeros((len(scale["sample_ids"]), 2048), dtype=np.float32)
    conflict_detail = np.zeros(
        (len(conflict["sample_ids"]), 2048), dtype=np.float32
    )
    conflict_detail[t2] = t2_detail
    return ConflictInvariantKiNO.fit(
        np.concatenate([scale["visual"], conflict["visual"]], axis=0),
        np.concatenate([scale["full"], conflict["full"]], axis=0),
        np.concatenate([scale["invariant"], conflict["invariant"]], axis=0),
        np.concatenate([scale["labels"], conflict["labels"]], axis=0),
        np.concatenate([scale["groups"], conflict["groups"]], axis=0),
        np.concatenate(
            [
                np.zeros(len(scale["sample_ids"]), dtype=bool),
                conflict["vision_decisive"],
            ]
        ),
        seed=MODEL_SEED,
        route_threshold=ROUTE_THRESHOLD,
        detail_visual=np.concatenate([scale_detail, conflict_detail], axis=0),
        visual_regularization=VISUAL_REGULARIZATION,
        include_base_visual_in_specialist=True,
    )


def _fit_baselines(
    scale: dict[str, Any], conflict: dict[str, Any]
) -> dict[str, Any]:
    full_conflict_dino = _load_dino(CONFLICT_DINO, conflict["sample_ids"])
    bundle: dict[str, Any] = {
        "schema_version": "kinofail.kino-v4-frozen-baselines.v1",
        "seed": BASELINE_SEED,
        "datasets": {},
    }
    for dataset_index, (name, source, visual) in enumerate(
        (
            ("scale", scale, visual_context_features(scale["visual"])),
            (
                "conflict",
                conflict,
                np.concatenate(
                    [visual_context_features(conflict["visual"]), full_conflict_dino],
                    axis=1,
                ),
            ),
        )
    ):
        proprio = invariant_relative_proprio_features(
            source["invariant"], source["full"]
        )
        early = np.concatenate([visual, proprio], axis=1)
        weights = physical_group_weights(source["groups"])
        offset = dataset_index * 100
        bundle["datasets"][name] = {
            "classes": sorted(set(source["labels"].tolist())),
            "visual": _baseline_estimator(BASELINE_SEED + offset + 1).fit(
                visual, source["labels"], sample_weight=weights
            ),
            "proprioception": _baseline_estimator(
                BASELINE_SEED + offset + 2
            ).fit(proprio, source["labels"], sample_weight=weights),
            "joint_early_fusion": _baseline_estimator(
                BASELINE_SEED + offset + 3
            ).fit(early, source["labels"], sample_weight=weights),
            "visual_dimension": int(visual.shape[1]),
            "proprio_dimension": int(proprio.shape[1]),
        }
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if CONFIRMATION_ROOT.exists():
        raise RuntimeError(
            "independent confirmation root already exists; F0 must precede data"
        )

    development = _json(DEVELOPMENT_REPORT)
    baselines = _json(BASELINE_REPORT)
    statistics = _json(STATISTICS_REPORT)
    if development.get("confirmatory_evidence") is not False:
        raise RuntimeError("development report is not marked development-only")
    if development.get("route_threshold") != ROUTE_THRESHOLD:
        raise RuntimeError("selected route threshold drift")
    if development.get("visual_regularization_C") != VISUAL_REGULARIZATION:
        raise RuntimeError("selected visual regularization drift")
    if development.get("dino_detail_features") != ["ground_mean"]:
        raise RuntimeError("selected DINO pool drift")
    if development.get("detail_only_visual_specialist") is not False:
        raise RuntimeError("selected specialist input drift")

    scale = _load_scale()
    conflict = _load_conflict()
    kino = _fit_kino(scale, conflict)
    baseline_bundle = _fit_baselines(scale, conflict)

    output.mkdir(parents=True, exist_ok=False)
    kino_path = output / "kino_v4.pkl"
    baseline_path = output / "baselines.pkl"
    with kino_path.open("wb") as stream:
        pickle.dump(kino, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with baseline_path.open("wb") as stream:
        pickle.dump(baseline_bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)

    code_paths = [
        ROOT / "kino_vla/eval/conflict_invariant_kino.py",
        ROOT / "scripts/extract_kinofail_t2_dinov2_v4.py",
        ROOT / "scripts/freeze_kinofail_v4_method_f0.py",
        ROOT / "scripts/evaluate_kinofail_v4_fair_baselines.py",
        ROOT / "scripts/analyze_kinofail_v4_paired_statistics.py",
    ]
    evidence_paths = [
        T2_DINO,
        CONFLICT_DINO,
        DEVELOPMENT_REPORT,
        BASELINE_REPORT,
        STATISTICS_REPORT,
    ]
    manifest = {
        "schema_version": "kinofail.kino-v4-method-freeze-f0.v1",
        "status": "frozen_before_independent_confirmation_generation",
        "created_utc": datetime.now(UTC).isoformat(),
        "development_only": True,
        "confirmation_data_present_at_freeze": False,
        "method": {
            "name": "KiNO-v4-DINOv2Large-terrain448",
            "checkpoint_seed": MODEL_SEED,
            "architecture": (
                "relative-proprioception default expert with a learned "
                "proprioceptive conflict gate and T2 visual override"
            ),
            "route_threshold_eta": ROUTE_THRESHOLD,
            "visual_specialist": {
                "inputs": ["CLIP mean", "CLIP final", "DINOv2 ground mean"],
                "classifier": "standardized class-balanced logistic regression",
                "regularization_C": VISUAL_REGULARIZATION,
                "base_visual_included": True,
            },
            "proprioception": (
                "80-D rotation-invariant summary plus 171-D within-window dynamics"
            ),
            "encoder": {
                "model_id": DINO_MODEL_ID,
                "revision": DINO_REVISION,
                "frozen": True,
                "image_size": 448,
                "processor_shortest_edge": 512,
                "input_region": "registered bottom 55% terrain crop",
                "frames": 5,
                "pool": "ground_mean",
            },
            "deployment_inputs_only": [
                "five-frame robot-front RGB sequence",
                "190-D full proprioceptive summary",
                "80-D rotation-invariant proprioceptive summary",
            ],
            "forbidden_deployment_inputs": [
                "scene/domain ID",
                "operator/material/severity ID",
                "battery/cell ID",
                "ground truth or outcome",
            ],
        },
        "baselines": {
            "seed": BASELINE_SEED,
            "training": "battery-specific full development fit",
            "methods": [
                "vision",
                "proprioception",
                "joint_early_fusion",
                "late_fusion",
                "oracle_route_conflict_only",
            ],
            "advantage_to_baselines": (
                "each deployable baseline is fitted independently per battery"
            ),
        },
        "analysis_contract": {
            "primary_metrics": [
                "Scale balanced accuracy",
                "Conflict balanced accuracy",
                "equal-battery macro balanced accuracy",
                "worst-battery balanced accuracy",
            ],
            "primary_comparators": ["joint_early_fusion", "late_fusion"],
            "secondary_comparators": ["vision", "proprioception", "oracle_route"],
            "ordered_gates": [
                "battery-wise noninferiority to late fusion at margin -0.01",
                "Conflict superiority to late fusion at margin 0",
                "worst-battery superiority to late fusion at margin 0",
                "equal-battery macro superiority to late fusion at margin 0",
            ],
            "bootstrap": {
                "draws": 20000,
                "seed": 2026080391,
                "confidence_level": 0.95,
                "clusters": ["scene", "material"],
            },
            "score_once_and_report_regardless_of_outcome": True,
        },
        "independent_confirmation_requirements": {
            "new_scene_sources": True,
            "new_material_source_assets": True,
            "new_random_seed_namespace": True,
            "new_operator_parameter_intervals": True,
            "no_refit_retuning_feature_or_threshold_change": True,
            "model_blind_validity_and_attrition_audit": True,
        },
        "development_selection_evidence": {
            "kino": development["aggregate"],
            "fair_baselines": baselines["cross_battery"],
            "paired_statistics": statistics["comparisons"],
        },
        "artifacts": {
            "kino_checkpoint": str(kino_path.relative_to(ROOT)),
            "kino_checkpoint_sha256": _sha256(kino_path),
            "baseline_checkpoint": str(baseline_path.relative_to(ROOT)),
            "baseline_checkpoint_sha256": _sha256(baseline_path),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in [*code_paths, *evidence_paths]
        },
    }
    manifest_path = output / "freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt = {
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": _sha256(manifest_path),
        "kino_checkpoint_sha256": _sha256(kino_path),
        "baseline_checkpoint_sha256": _sha256(baseline_path),
        "status": manifest["status"],
    }
    (output / "FREEZE_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
