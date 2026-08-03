#!/usr/bin/env python3
"""Run the frozen O2/O4 construct diagnostic on the realistic scale corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _auc_ci(labels: np.ndarray, scores: np.ndarray, scenes: np.ndarray, *, reps: int, seed: int) -> dict:
    unique = np.asarray(sorted(set(scenes.astype(str))))
    by_scene = {scene: np.flatnonzero(scenes == scene) for scene in unique}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        selected = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([by_scene[str(scene)] for scene in selected])
        draws.append(float(roc_auc_score(labels[indices], scores[indices])))
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "scene_clusters": len(unique),
        "bootstrap_repetitions": reps,
        "bootstrap_seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eval/kinofail_realistic_a1_diagnostic_v1.json")
    parser.add_argument("--snapshot-dir", type=Path, default=ROOT / "outputs/eval/realistic_a0_a7_v4/snapshots")
    parser.add_argument("--feature-dir", type=Path, default=ROOT / "outputs/eval/realistic_a0_a7_v4/features")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/eval/realistic_a0_a7_v4/a1_construct_validity.json")
    args = parser.parse_args()
    config = _json(args.config)
    certificate_path = ROOT / "outputs/eval/realistic_a0_a7_v4/a0_certificate.json"
    certificate = _json(certificate_path)
    if certificate.get("status") != "publication_ready":
        raise RuntimeError("A1 diagnostic requires final A0")
    records_path = args.snapshot_dir / "snapshot_records.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
    feature_path = args.feature_dir / "features.npz"
    feature_manifest_path = args.feature_dir / "feature_manifest.json"
    feature_manifest = _json(feature_manifest_path)
    if feature_manifest["output_sha256"]["features"] != _sha(feature_path):
        raise RuntimeError("feature hash mismatch")
    archive = np.load(feature_path, allow_pickle=False)
    if archive["sample_ids"].astype(str).tolist() != [row["sample_id"] for row in records]:
        raise RuntimeError("feature/metadata alignment mismatch")
    selected = np.asarray([
        row["condition"] == "anomaly"
        and row["appearance_intervention_id"] == "primary"
        and row["target_operator"] in config["operators"]
        for row in records
    ])
    labels = np.asarray([row["target_operator"] == "O4_tether" for row in records])[selected]
    scenes = np.asarray([row["scene_family"] for row in records])[selected]
    feature_blocks = {"vision": archive["visual"][selected], "proprio": archive["proprio"][selected]}
    result_blocks = {}
    for block, values in feature_blocks.items():
        scores = np.zeros(len(values), dtype=np.float64)
        for scene in sorted(set(scenes)):
            test = scenes == scene
            train = ~test
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=0),
            )
            model.fit(values[train], labels[train])
            scores[test] = model.predict_proba(values[test])[:, 1]
        result_blocks[block] = _auc_ci(
            labels, scores, scenes,
            reps=int(config["bootstrap"]["repetitions"]),
            seed=int(config["bootstrap"]["seed"]) + (0 if block == "vision" else 1),
        )
    band = config["acceptance"]["proprio_auc_equivalence_band"]
    checks = {
        "nine_scene_clusters": result_blocks["vision"]["scene_clusters"] >= int(config["acceptance"]["minimum_scene_clusters"]),
        "proprio_ci_inside_equivalence_band": result_blocks["proprio"]["ci95"][0] >= float(band[0]) and result_blocks["proprio"]["ci95"][1] <= float(band[1]),
        "vision_ci_lower_ge_0_8": result_blocks["vision"]["ci95"][0] >= float(config["acceptance"]["vision_auc_cluster_ci_lower"]),
        "fresh_app_deep_reset_matched_o2_o4_design": False,
    }
    result = {
        "schema_version": "kinofail.realistic-a1-construct-validity.v1",
        "status": "partial_complete",
        "dataset_scope": "kinofail_realistic",
        "a0_evidence_bundle_sha256": certificate["a0_evidence_bundle_sha256"],
        "protocol_id": config["protocol_id"],
        "protocol_sha256": _sha(args.config),
        "n_primary_anomaly_episodes": int(selected.sum()),
        "results": result_blocks,
        "acceptance": checks,
        "passed": all(checks.values()),
        "known_design_limit": config["known_design_limit"],
        "interpretation": "This diagnostic measures separability but cannot certify the matched O2/O4 impossibility construct without the predeclared cross-operator matching extension.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "results": result_blocks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
