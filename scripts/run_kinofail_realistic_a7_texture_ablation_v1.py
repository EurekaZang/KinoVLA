#!/usr/bin/env python3
"""Run the frozen realistic A7 texture-swap augmentation ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.realistic_multimodal import (  # noqa: E402
    StructuredBidirectionalClassifier,
    predictions_from_probabilities,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _jsd(left: np.ndarray, right: np.ndarray) -> float:
    eps = 1.0e-12
    left = np.maximum(np.asarray(left, dtype=np.float64), eps)
    right = np.maximum(np.asarray(right, dtype=np.float64), eps)
    left /= left.sum()
    right /= right.sum()
    middle = 0.5 * (left + right)
    return float(
        0.5 * np.sum(left * np.log(left / middle))
        + 0.5 * np.sum(right * np.log(right / middle))
    )


def _bootstrap(values: dict[str, list[float]], *, repetitions: int, seed: int) -> dict:
    clusters = np.asarray([np.mean(row) for row in values.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        np.mean(rng.choice(clusters, len(clusters), replace=True))
        for _ in range(repetitions)
    ])
    return {
        "unit": "paired counterfactual physics group",
        "clusters": len(clusters),
        "point": float(np.mean(clusters)),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "repetitions": repetitions,
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/eval/kinofail_realistic_a7_texture_ablation_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _json(config_path)
    bound = {
        "records": (ROOT / config["records"], config["records_sha256"]),
        "features": (ROOT / config["features"], config["features_sha256"]),
        "feature_manifest": (
            ROOT / config["feature_manifest"], config["feature_manifest_sha256"]
        ),
        "scene_registry": (
            ROOT / config["scene_registry"], config["scene_registry_sha256"]
        ),
        "model_implementation": (
            ROOT / config["model_implementation"], config["model_implementation_sha256"]
        ),
    }
    mismatches = {
        name: {"expected": expected, "actual": _sha256(path)}
        for name, (path, expected) in bound.items()
        if not path.is_file() or _sha256(path) != expected
    }
    if mismatches:
        raise RuntimeError(f"frozen input mismatch: {mismatches}")
    records = _jsonl(bound["records"][0])
    archive = np.load(bound["features"][0], allow_pickle=False)
    sample_ids = archive["sample_ids"].astype(str)
    if sample_ids.tolist() != [str(row["sample_id"]) for row in records]:
        raise RuntimeError("feature and record order mismatch")
    visual = np.asarray(archive["visual"], dtype=np.float32)
    proprio = np.asarray(archive["proprio"], dtype=np.float32)
    labels = np.asarray([str(row["attribution_category"]) for row in records])
    episodes = np.asarray([str(row["physical_episode_id"]) for row in records])
    groups = np.asarray([str(row["counterfactual_group_id"]) for row in records])
    primary = np.asarray([
        row["appearance_intervention_id"] == "primary" for row in records
    ])
    registry = _json(bound["scene_registry"][0])
    splits = {str(row["scene_id"]): str(row["split"]) for row in registry["scenes"]}
    scene_split = np.asarray([splits[str(row["scene_family"])] for row in records])
    test = scene_split == "test"
    if not test.any() or not np.any(scene_split != "test"):
        raise RuntimeError("empty train or test split")

    variants = {
        "with_texture_swap_augmentation": scene_split != "test",
        "without_texture_swap_augmentation": (scene_split != "test") & primary,
    }
    predictions: list[dict] = []
    summaries: dict[str, dict] = {}
    for variant, fit_mask in variants.items():
        summaries[variant] = {}
        for seed in [int(value) for value in config["training_seeds"]]:
            model = StructuredBidirectionalClassifier.fit(
                visual[fit_mask], proprio[fit_mask], labels[fit_mask],
                episodes[fit_mask], groups[fit_mask], seed=seed,
            )
            probability = model.predict_proba(visual[test], proprio[test])
            predicted = predictions_from_probabilities(probability, model.classes)
            indices = np.flatnonzero(test)
            rows = []
            for local, index in enumerate(indices):
                source = records[int(index)]
                row = {
                    "sample_id": source["sample_id"],
                    "physical_episode_id": source["physical_episode_id"],
                    "counterfactual_group_id": source["counterfactual_group_id"],
                    "appearance_intervention_id": source["appearance_intervention_id"],
                    "scene_family": source["scene_family"],
                    "truth": source["attribution_category"],
                    "prediction": str(predicted[local]),
                    "classes": list(model.classes),
                    "probabilities": probability[local].tolist(),
                    "variant": variant,
                    "seed": seed,
                }
                rows.append(row)
                predictions.append(row)
            headline = [row for row in rows if row["appearance_intervention_id"] == "primary"]
            grouped: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                grouped[row["physical_episode_id"]].append(row)
            episode_values = []
            for episode, views in grouped.items():
                if len(views) != 3:
                    raise RuntimeError(f"test texture group {episode} has {len(views)} views")
                probabilities = [np.asarray(row["probabilities"]) for row in views]
                episode_values.append({
                    "physical_episode_id": episode,
                    "counterfactual_group_id": views[0]["counterfactual_group_id"],
                    "hard_consistent": len({row["prediction"] for row in views}) == 1,
                    "mean_pairwise_jsd": float(np.mean([
                        _jsd(probabilities[i], probabilities[j])
                        for i in range(3) for j in range(i + 1, 3)
                    ])),
                })
            summaries[variant][str(seed)] = {
                "fit_samples": int(np.sum(fit_mask)),
                "fit_physical_episodes": len(set(episodes[fit_mask])),
                "test_primary_samples": len(headline),
                "test_physical_episodes": len(episode_values),
                "primary_balanced_accuracy": float(balanced_accuracy_score(
                    [row["truth"] for row in headline],
                    [row["prediction"] for row in headline],
                )),
                "texture_swap_hard_consistency": float(np.mean([
                    row["hard_consistent"] for row in episode_values
                ])),
                "mean_pairwise_probability_jsd": float(np.mean([
                    row["mean_pairwise_jsd"] for row in episode_values
                ])),
                "episode_values": episode_values,
            }

    paired_statistics = {}
    repetitions = int(config["bootstrap_repetitions"])
    bootstrap_seed = int(config["bootstrap_seed"])
    for seed_index, seed in enumerate([int(value) for value in config["training_seeds"]]):
        with_rows = [
            row for row in predictions
            if row["variant"] == "with_texture_swap_augmentation"
            and row["seed"] == seed and row["appearance_intervention_id"] == "primary"
        ]
        without_rows = {
            row["sample_id"]: row for row in predictions
            if row["variant"] == "without_texture_swap_augmentation"
            and row["seed"] == seed and row["appearance_intervention_id"] == "primary"
        }
        accuracy_delta: dict[str, list[float]] = defaultdict(list)
        for row in with_rows:
            other = without_rows[row["sample_id"]]
            accuracy_delta[row["counterfactual_group_id"]].append(
                float(row["prediction"] == row["truth"])
                - float(other["prediction"] == other["truth"])
            )
        with_episode = {
            row["physical_episode_id"]: row
            for row in summaries["with_texture_swap_augmentation"][str(seed)]["episode_values"]
        }
        without_episode = {
            row["physical_episode_id"]: row
            for row in summaries["without_texture_swap_augmentation"][str(seed)]["episode_values"]
        }
        consistency_delta: dict[str, list[float]] = defaultdict(list)
        jsd_delta: dict[str, list[float]] = defaultdict(list)
        for episode, row in with_episode.items():
            other = without_episode[episode]
            group = row["counterfactual_group_id"]
            consistency_delta[group].append(
                float(row["hard_consistent"]) - float(other["hard_consistent"])
            )
            jsd_delta[group].append(
                float(row["mean_pairwise_jsd"]) - float(other["mean_pairwise_jsd"])
            )
        paired_statistics[str(seed)] = {
            "with_minus_without_primary_accuracy": _bootstrap(
                accuracy_delta, repetitions=repetitions,
                seed=bootstrap_seed + 3 * seed_index,
            ),
            "with_minus_without_hard_consistency": _bootstrap(
                consistency_delta, repetitions=repetitions,
                seed=bootstrap_seed + 3 * seed_index + 1,
            ),
            "with_minus_without_mean_jsd": _bootstrap(
                jsd_delta, repetitions=repetitions,
                seed=bootstrap_seed + 3 * seed_index + 2,
            ),
        }

    # Remove verbose per-episode reducers from the main summary; raw predictions retain full
    # recomputation support and are hash-bound below.
    for variant in summaries.values():
        for seed_summary in variant.values():
            seed_summary.pop("episode_values")
    output_path = ROOT / config["output"]
    predictions_path = output_path.with_name("a7_texture_consistency_predictions.jsonl")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    with_hard = [
        row["texture_swap_hard_consistency"]
        for row in summaries["with_texture_swap_augmentation"].values()
    ]
    result = {
        "schema_version": "kinofail.realistic-a7-texture-ablation-result.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "confirmatory_complete",
        "dataset_scope": "kinofail_realistic",
        "protocol": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha256(config_path)},
        "input_hashes": {name: _sha256(path) for name, (path, _) in bound.items()},
        "summaries": summaries,
        "paired_scene_cluster_statistics": paired_statistics,
        "predictions": {
            "path": str(predictions_path.relative_to(ROOT)),
            "sha256": _sha256(predictions_path),
            "rows": len(predictions),
        },
        "acceptance": {
            "three_training_seeds": len(summaries["with_texture_swap_augmentation"]) == 3,
            "paired_counterfactual_statistics_present": all(
                row["with_minus_without_primary_accuracy"]["clusters"] > 0
                for row in paired_statistics.values()
            ),
            "with_texture_hard_consistency_ge_0_95_every_seed": min(with_hard) >= 0.95,
            "negative_or_null_ablation_result_retained": True,
        },
        "interpretation_policy": (
            "This ablation estimates the effect of synchronized texture-swap training augmentation; "
            "it does not treat appearance views as independent physical samples."
        ),
    }
    result["passed"] = all(result["acceptance"].values())
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(output_path), "passed": result["passed"],
        "summaries": summaries, "paired_statistics": paired_statistics,
    }, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
