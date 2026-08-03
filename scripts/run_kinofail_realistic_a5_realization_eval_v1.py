#!/usr/bin/env python3
"""Evaluate the frozen O8/O9 alternate physical realizations."""

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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _features(records_path: Path, features_path: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    records = _jsonl(records_path)
    archive = np.load(features_path, allow_pickle=False)
    if archive["sample_ids"].astype(str).tolist() != [str(row["sample_id"]) for row in records]:
        raise RuntimeError(f"record/feature order mismatch: {features_path}")
    return records, np.asarray(archive["visual"]), np.asarray(archive["proprio"])


def _bootstrap(values: dict[str, list[float]], *, repetitions: int, seed: int) -> dict:
    scenes = sorted(values)
    means = np.asarray([np.mean(values[scene]) for scene in scenes], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(scenes), size=(repetitions, len(scenes)))
    draws = means[sampled].mean(axis=1)
    return {
        "unit": "scene-family cluster",
        "scene_clusters": len(scenes),
        "physical_samples": sum(len(rows) for rows in values.values()),
        "point": float(np.mean(means)),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "repetitions": repetitions,
        "seed": seed,
        "scene_means": {scene: float(np.mean(values[scene])) for scene in scenes},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/eval/kinofail_realistic_a5_realization_eval_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _json(config_path)
    paths = {key: ROOT / config[key] for key in (
        "base_records", "base_features", "realization_records", "realization_features",
        "realization_schedule",
    )}
    if _sha(paths["realization_schedule"]) != config["realization_schedule_sha256"]:
        raise RuntimeError("realization schedule hash mismatch")
    base_records, base_visual, base_proprio = _features(
        paths["base_records"], paths["base_features"]
    )
    test_records, test_visual, test_proprio = _features(
        paths["realization_records"], paths["realization_features"]
    )
    schedule = _jsonl(paths["realization_schedule"])
    schedule_by_episode = {str(row["episode_id"]): row for row in schedule}
    primary_indices = np.asarray([
        index for index, row in enumerate(test_records)
        if row["appearance_intervention_id"] == "primary"
    ], dtype=int)
    primary_records = [test_records[index] for index in primary_indices]
    groups: dict[str, list[int]] = defaultdict(list)
    for local, row in enumerate(primary_records):
        groups[str(row["counterfactual_group_id"])].append(local)
    valid_groups = {
        group for group, indices in groups.items()
        if len(indices) == 2 and {
            primary_records[index]["condition"] for index in indices
        } == {"anomaly", "nominal_counterfactual"}
    }
    keep_local = np.asarray([
        index for index, row in enumerate(primary_records)
        if row["counterfactual_group_id"] in valid_groups
    ], dtype=int)
    primary_indices = primary_indices[keep_local]
    primary_records = [primary_records[index] for index in keep_local]
    truths = np.asarray([row["attribution_category"] for row in primary_records])
    train_labels = np.asarray([row["attribution_category"] for row in base_records])
    train_episodes = np.asarray([row["physical_episode_id"] for row in base_records])
    train_groups = np.asarray([row["counterfactual_group_id"] for row in base_records])
    test_episodes = {row["physical_episode_id"] for row in primary_records}
    if set(train_episodes) & test_episodes:
        raise RuntimeError("base and alternate-realization physical episode overlap")
    if set(truths) - set(train_labels):
        raise RuntimeError("alternate-realization test contains unseen classes")
    required_ops = set(config["required_operators"])
    observed_ops = {row["target_operator"] for row in primary_records}
    if observed_ops != required_ops:
        raise RuntimeError(f"unexpected realization operators: {observed_ops}")
    for row in primary_records:
        scheduled = schedule_by_episode[row["physical_episode_id"]]
        expected = {
            "O8_invisible_collider": "occluded_low_bar",
            "O9_high_centering": "pallet_edge",
        }[row["target_operator"]]
        if scheduled["physical_realization"] != expected:
            raise RuntimeError("alternate realization schedule mismatch")

    summaries, statistics, predictions = {}, {}, []
    repetitions = int(config["bootstrap_repetitions"])
    for seed_index, seed in enumerate([int(value) for value in config["training_seeds"]]):
        model = StructuredBidirectionalClassifier.fit(
            base_visual, base_proprio, train_labels, train_episodes, train_groups, seed=seed
        )
        probability = model.predict_proba(
            test_visual[primary_indices], test_proprio[primary_indices]
        )
        predicted = predictions_from_probabilities(probability, model.classes)
        correctness: dict[str, list[float]] = defaultdict(list)
        anomaly_correctness: dict[str, list[float]] = defaultdict(list)
        nominal_correctness: dict[str, list[float]] = defaultdict(list)
        for index, row in enumerate(primary_records):
            correct = predicted[index] == truths[index]
            scene = row["scene_family"]
            correctness[scene].append(float(correct))
            target = anomaly_correctness if row["condition"] == "anomaly" else nominal_correctness
            target[scene].append(float(correct))
            scheduled = schedule_by_episode[row["physical_episode_id"]]
            predictions.append({
                "sample_id": row["sample_id"],
                "physical_episode_id": row["physical_episode_id"],
                "counterfactual_group_id": row["counterfactual_group_id"],
                "scene_family": scene,
                "domain": row["domain"],
                "target_operator": row["target_operator"],
                "physical_realization": scheduled["physical_realization"],
                "severity_id": row["severity_id"],
                "condition": row["condition"],
                "truth": row["attribution_category"],
                "prediction": str(predicted[index]),
                "correct": bool(correct),
                "classes": list(model.classes),
                "probabilities": probability[index].tolist(),
                "seed": seed,
            })
        per_operator = {}
        for operator in sorted(required_ops):
            indices = [i for i, row in enumerate(primary_records) if row["target_operator"] == operator]
            anomaly = [i for i in indices if primary_records[i]["condition"] == "anomaly"]
            nominal = [i for i in indices if primary_records[i]["condition"] == "nominal_counterfactual"]
            per_operator[operator] = {
                "samples": len(indices),
                "balanced_accuracy": float(balanced_accuracy_score(truths[indices], predicted[indices])),
                "anomaly_attribution_accuracy": float(np.mean(predicted[anomaly] == truths[anomaly])),
                "nominal_accuracy": float(np.mean(predicted[nominal] == truths[nominal])),
            }
        summaries[str(seed)] = {
            "train_samples": len(base_records),
            "train_physical_episodes": len(set(train_episodes)),
            "test_samples": len(primary_records),
            "test_counterfactual_pairs": len(valid_groups),
            "balanced_accuracy": float(balanced_accuracy_score(truths, predicted)),
            "per_operator": per_operator,
        }
        seed_base = int(config["bootstrap_seed"]) + 3 * seed_index
        statistics[str(seed)] = {
            "overall_accuracy": _bootstrap(
                correctness, repetitions=repetitions, seed=seed_base
            ),
            "anomaly_attribution_accuracy": _bootstrap(
                anomaly_correctness, repetitions=repetitions, seed=seed_base + 1
            ),
            "nominal_accuracy": _bootstrap(
                nominal_correctness, repetitions=repetitions, seed=seed_base + 2
            ),
        }
    predictions_path = ROOT / config["predictions"]
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    scenes = {row["scene_family"] for row in primary_records}
    acceptance = {
        "minimum_valid_pairs": len(valid_groups) >= int(config["minimum_valid_pairs"]),
        "required_scene_families": len(scenes) == int(config["required_scene_families"]),
        "required_operators": observed_ops == required_ops,
        "three_training_seeds": len(summaries) == 3,
        "scene_cluster_bootstrap_present": all(
            row["overall_accuracy"]["scene_clusters"] == len(scenes)
            for row in statistics.values()
        ),
        "no_train_test_physical_episode_overlap": True,
        "negative_ood_results_retained": True,
    }
    result = {
        "schema_version": "kinofail.realistic-a5-physical-realization-result.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "confirmatory_complete",
        "dataset_scope": "kinofail_realistic",
        "protocol": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha(config_path)},
        "input_hashes": {key: _sha(path) for key, path in paths.items()},
        "valid_counterfactual_pairs": len(valid_groups),
        "excluded_scheduled_pairs": 180 - len(valid_groups),
        "scene_families": sorted(scenes),
        "operators": sorted(observed_ops),
        "summaries": summaries,
        "scene_cluster_statistics": statistics,
        "predictions": {
            "path": str(predictions_path.relative_to(ROOT)),
            "sha256": _sha(predictions_path), "rows": len(predictions),
        },
        "acceptance": acceptance,
        "passed": all(acceptance.values()),
        "claim_boundary": config["claim_boundary"],
        "a8_in_scope": False,
    }
    output_path = ROOT / config["output"]
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(output_path), "passed": result["passed"],
        "valid_pairs": len(valid_groups), "excluded_pairs": 180 - len(valid_groups),
        "summaries": summaries, "statistics": statistics,
    }, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
