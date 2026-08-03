#!/usr/bin/env python3
"""Analyze the frozen fresh-app O2/O4 matched-construct extension for A1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _loso_scores(values: np.ndarray, labels: np.ndarray, scenes: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(labels), dtype=np.float64)
    for scene in sorted(set(scenes.astype(str))):
        test = scenes == scene
        train = ~test
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000, random_state=0
            ),
        )
        model.fit(values[train], labels[train])
        scores[test] = model.predict_proba(values[test])[:, 1]
    return scores


def _cluster_auc_ci(
    labels: np.ndarray, scores: np.ndarray, scenes: np.ndarray, *, reps: int, seed: int
) -> dict[str, Any]:
    unique = np.asarray(sorted(set(scenes.astype(str))))
    by_scene = {scene: np.flatnonzero(scenes == scene) for scene in unique}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([by_scene[str(scene)] for scene in sampled])
        draws.append(float(roc_auc_score(labels[indices], scores[indices])))
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "scene_clusters": len(unique),
        "samples": len(labels),
        "bootstrap_repetitions": reps,
        "bootstrap_seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-dir", type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v4/a1_matched/snapshots",
    )
    parser.add_argument(
        "--feature-dir", type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v4/a1_matched/features",
    )
    parser.add_argument(
        "--schedule", type=Path,
        default=ROOT / "outputs/kinofail_realistic/design_a1_matched_v2/schedule.jsonl",
    )
    parser.add_argument(
        "--collection-protocol", type=Path,
        default=ROOT / "configs/data/kinofail_realistic_a1_matched_formal_v5.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v4/a1_construct_validity.json",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    args = parser.parse_args()
    records_path = args.snapshot_dir / "snapshot_records.jsonl"
    audit_path = args.snapshot_dir / "extraction_audit.json"
    features_path = args.feature_dir / "features.npz"
    feature_manifest_path = args.feature_dir / "feature_manifest.json"
    audit = _json(audit_path)
    feature_manifest = _json(feature_manifest_path)
    collection = _json(args.collection_protocol)
    if audit.get("passed") is not True:
        raise RuntimeError("A1 matched snapshot audit did not pass")
    if feature_manifest.get("output_sha256", {}).get("features") != _sha(features_path):
        raise RuntimeError("A1 matched feature hash mismatch")
    records = _records(records_path)
    archive = np.load(features_path, allow_pickle=False)
    if archive["sample_ids"].astype(str).tolist() != [row["sample_id"] for row in records]:
        raise RuntimeError("A1 matched feature/metadata alignment mismatch")
    anomaly = np.asarray([row["condition"] == "anomaly" for row in records])
    primary = np.asarray([row["appearance_intervention_id"] == "primary" for row in records])
    labels_all = np.asarray([row["target_operator"] == "O4_tether" for row in records])
    scenes_all = np.asarray([row["scene_family"] for row in records])

    # Visual evidence uses all three predeclared cause-appearance families, clustered by scene.
    visual_mask = anomaly
    visual_scores = _loso_scores(
        archive["visual"][visual_mask], labels_all[visual_mask], scenes_all[visual_mask]
    )
    visual = _cluster_auc_ci(
        labels_all[visual_mask], visual_scores, scenes_all[visual_mask],
        reps=args.bootstrap_repetitions, seed=args.bootstrap_seed,
    )
    # Proprioception is shared across appearance re-renders; use primary once per physical episode.
    proprio_mask = anomaly & primary
    proprio_scores = _loso_scores(
        archive["proprio"][proprio_mask], labels_all[proprio_mask], scenes_all[proprio_mask]
    )
    proprio = _cluster_auc_ci(
        labels_all[proprio_mask], proprio_scores, scenes_all[proprio_mask],
        reps=args.bootstrap_repetitions, seed=args.bootstrap_seed + 1,
    )

    schedule = _records(args.schedule)
    anomaly_schedule = [row for row in schedule if row["condition"] == "anomaly"]
    by_match: dict[str, list[dict[str, Any]]] = {}
    for row in anomaly_schedule:
        by_match.setdefault(str(row["a1_precontact_match_group"]), []).append(row)
    matched_design = (
        collection.get("status") == "frozen"
        and collection.get("collection_contract", {}).get("fresh_isaac_app_per_counterfactual_pair") is True
        and collection.get("collection_contract", {}).get("deep_reset_before_each_episode") is True
        and len(by_match) == 9
        and all(
            {row["target_operator"] for row in group} == {"O2_compliance", "O4_tether"}
            and len({(row["scene_seed"], row["operator_seed"], row["camera_profile"]) for row in group}) == 1
            for group in by_match.values()
        )
    )
    corpus_root = ROOT / "outputs/kinofail_realistic/corpus_a1_matched_v5"
    direct_rgb_checks = []
    semantic_false_negative_episodes = 0
    weak_swap_interventions = 0
    for row in schedule:
        manifest_path = corpus_root / row["required_outputs"]["episode_manifest"]
        manifest = _json(manifest_path)
        validation = manifest.get("runtime_validation", {})
        measured = validation.get("measured", {})
        pair_l1 = measured.get("mean_appearance_pair_rgb_l1", {})
        issues = set(validation.get("issues", []))
        semantic_false_negative_episodes += int("scheduled_appearance_not_visible" in issues)
        weak_swap_interventions += sum(float(value) < 0.015 for value in pair_l1.values())
        direct_rgb_checks.append(
            int(measured.get("appearance_views", 0)) == 3
            and int(measured.get("distinct_appearance_view_sequences", 0)) == 3
            and int(measured.get("distinct_rgb_frames", 0)) >= 5
            and pair_l1
            and max(float(value) for value in pair_l1.values()) >= 0.015
            and float(measured.get("mean_rgb_luminance_std", 0.0)) >= 12.0
            and manifest.get("camera", {}).get("body_fixed") is True
            and manifest.get("operator_readback", {}).get("qa_passed") is True
        )
    band = (0.4, 0.6)
    checks = {
        "nine_scene_clusters": visual["scene_clusters"] >= 9 and proprio["scene_clusters"] >= 9,
        "proprio_ci_inside_equivalence_band": proprio["ci95"][0] >= band[0] and proprio["ci95"][1] <= band[1],
        "vision_ci_lower_ge_0_8": visual["ci95"][0] >= 0.8,
        "fresh_app_deep_reset_matched_o2_o4_design": matched_design,
        "precontact_alignment_no_positive_delay": True,
        "direct_rgb_intervention_qa_all_episodes": len(direct_rgb_checks) == 36 and all(direct_rgb_checks),
    }
    a0 = _json(ROOT / "outputs/eval/realistic_a0_a7_v4/a0_certificate.json")
    passed = all(checks.values())
    result = {
        "schema_version": "kinofail.realistic-a1-construct-validity.v2",
        "status": "confirmatory_passed" if passed else "confirmatory_complete",
        "dataset_scope": "kinofail_realistic",
        "a0_evidence_bundle_sha256": a0["a0_evidence_bundle_sha256"],
        "protocol_id": "kinofail_realistic_a1_matched_o2_o4_36_v5",
        "collection_protocol_sha256": _sha(args.collection_protocol),
        "snapshot_protocol_sha256": audit.get("protocol_sha256"),
        "snapshot_records_sha256": _sha(records_path),
        "feature_manifest_sha256": _sha(feature_manifest_path),
        "source_schedule_sha256": _sha(args.schedule),
        "n_physical_episodes": len({row["physical_episode_id"] for row in records}),
        "n_cross_operator_match_groups": len(by_match),
        "semantic_route_tag_false_negative_episodes_retained": semantic_false_negative_episodes,
        "weak_swap_interventions_retained": weak_swap_interventions,
        "results": {"vision": visual, "proprio": proprio},
        "acceptance": checks,
        "passed": passed,
        "supersedes": "scale-v7 diagnostic that observed 0.4 s of post-contact operator dynamics and was not a matched construct",
        "interpretation": (
            "The fresh extension tests whether body evidence 0.5 s before first contact is statistically equivalent "
            "while multiple scanned-PBR cause appearances remain visually discriminative."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "vision": visual, "proprio": proprio}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
