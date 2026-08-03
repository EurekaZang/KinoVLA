#!/usr/bin/env python3
"""Evaluate the frozen O7 reverse-conflict extension with the frozen realistic models."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from kino_vla.eval.realistic_multimodal import predictions_from_probabilities


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("vision_only", "proprio_only", "early_fusion", "structured_bidirectional")
SEEDS = (0, 1, 2, 3, 4)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _rate(rows: list[dict[str, Any]]) -> float | None:
    return float(np.mean([row["prediction"] == row["truth"] for row in rows])) if rows else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-dir", type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v4/a3_reverse/snapshots",
    )
    parser.add_argument(
        "--feature-dir", type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v4/a3_reverse/features",
    )
    parser.add_argument(
        "--base-report", type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v4/a3_publication_report.json",
    )
    parser.add_argument(
        "--training-manifest", type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v4/training_manifest.json",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help=(
            "Root containing scene/seed*/METHOD.pkl. Defaults to the checkpoints directory "
            "beside --training-manifest."
        ),
    )
    parser.add_argument(
        "--output-predictions", type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v4/a3_reverse_predictions.jsonl",
    )
    args = parser.parse_args()
    records_path = args.snapshot_dir / "snapshot_records.jsonl"
    audit_path = args.snapshot_dir / "extraction_audit.json"
    feature_manifest_path = args.feature_dir / "feature_manifest.json"
    features_path = args.feature_dir / "features.npz"
    audit = _json(audit_path)
    feature_manifest = _json(feature_manifest_path)
    training = _json(args.training_manifest)
    checkpoint_root = (
        args.checkpoint_root.resolve()
        if args.checkpoint_root is not None
        else args.training_manifest.resolve().parent / "checkpoints"
    )
    if audit.get("passed") is not True:
        raise RuntimeError("A3 reverse snapshot audit did not pass")
    if feature_manifest.get("output_sha256", {}).get("features") != _sha(features_path):
        raise RuntimeError("A3 reverse feature hash mismatch")
    records = _rows(records_path)
    archive = np.load(features_path, allow_pickle=False)
    if archive["sample_ids"].astype(str).tolist() != [row["sample_id"] for row in records]:
        raise RuntimeError("A3 reverse feature/metadata alignment mismatch")
    visual, proprio = archive["visual"], archive["proprio"]
    checkpoint_hashes = training["checkpoints_sha256"]
    predictions: list[dict[str, Any]] = []
    for seed in SEEDS:
        for method in METHODS:
            checkpoint = checkpoint_root / "scene" / f"seed{seed}" / f"{method}.pkl"
            relative = str(checkpoint.relative_to(ROOT))
            if checkpoint_hashes.get(relative) != _sha(checkpoint):
                raise RuntimeError(f"frozen checkpoint hash mismatch: {relative}")
            with checkpoint.open("rb") as handle:
                model = pickle.load(handle)
            probability = model.predict_proba(visual, proprio)
            predicted = predictions_from_probabilities(probability, model.classes)
            for index, row in enumerate(records):
                predictions.append({
                    "sample_id": row["sample_id"],
                    "physical_episode_id": row["physical_episode_id"],
                    "counterfactual_group_id": row["counterfactual_group_id"],
                    "appearance_intervention_id": row["appearance_intervention_id"],
                    "condition": row["condition"],
                    "domain": row["domain"],
                    "scene_family": row["scene_family"],
                    "target_operator": row["target_operator"],
                    "truth": row["attribution_category"],
                    "prediction": str(predicted[index]),
                    "classes": list(model.classes),
                    "probabilities": probability[index].tolist(),
                    "seed": seed,
                    "method": method,
                    "headline_primary_view": row["appearance_intervention_id"] == "primary",
                    "reverse_conflict_probe": row["condition"] == "nominal_counterfactual",
                })
    args.output_predictions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    results: dict[str, Any] = {}
    primary = [row for row in predictions if row["headline_primary_view"]]
    for method in METHODS:
        results[method] = {}
        for seed in SEEDS:
            subset = [row for row in primary if row["method"] == method and row["seed"] == seed]
            reverse = [row for row in subset if row["reverse_conflict_probe"]]
            control = [row for row in subset if not row["reverse_conflict_probe"]]
            results[method][str(seed)] = {
                "reverse_conflict_nominal_rate": _rate(reverse),
                "hazard_physics_control_rate": _rate(control),
                "per_domain_reverse_rate": {
                    domain: _rate([row for row in reverse if row["domain"] == domain])
                    for domain in ("life", "production", "wild")
                },
                "reverse_scene_clusters": len({row["scene_family"] for row in reverse}),
                "control_scene_clusters": len({row["scene_family"] for row in control}),
            }
    ours = results["structured_bidirectional"]
    reverse_present = (
        len({row["scene_family"] for row in primary if row["reverse_conflict_probe"]}) == 9
        and {row["domain"] for row in primary if row["reverse_conflict_probe"]}
        == {"life", "production", "wild"}
    )
    reverse_pass = all(
        row["reverse_conflict_nominal_rate"] is not None
        and row["reverse_conflict_nominal_rate"] >= 0.9
        for row in ours.values()
    )
    report = _json(args.base_report)
    report["schema_version"] = "kinofail.realistic-a3.v2"
    report["reverse_extension"] = {
        "protocol_id": "kinofail_realistic_a3_reverse_18_v1",
        "snapshot_records_sha256": _sha(records_path),
        "feature_manifest_sha256": _sha(feature_manifest_path),
        "predictions_path": str(
            args.output_predictions.resolve().relative_to(ROOT)
        ),
        "predictions_sha256": _sha(args.output_predictions),
        "results": results,
        "statistical_unit": "registered scene family; synchronized appearance views are not independent physics samples",
        "frozen_model_retrained_on_extension": False,
    }
    for seed in SEEDS:
        report["results"]["structured_bidirectional"][str(seed)]["t3_subcells"]["reverse_conflict"] = ours[str(seed)]["reverse_conflict_nominal_rate"]
    report["acceptance"]["reverse_conflict_realization_present"] = reverse_present
    report["acceptance"]["every_seed_reverse_conflict_ge_0_9"] = reverse_pass
    report["missing_design_guard"] = {
        "reverse_conflict": "closed by separately frozen O7 hazard-looking/nominal-physics extension",
        "ordinary_O7_nominal_relabelled": False,
    }
    report["passed"] = all(report["acceptance"].values())
    report["status"] = "confirmatory_passed" if report["passed"] else "confirmatory_complete"
    report["interpretation"] = (
        "The reverse-conflict design gap is closed with frozen models. Any remaining failed strict "
        "cell is retained as a measured result rather than triggering architecture tuning."
    )
    args.base_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "reverse_present": reverse_present,
        "reverse_pass_every_seed": reverse_pass,
        "ours": ours,
        "overall_passed": report["passed"],
        "status": report["status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
