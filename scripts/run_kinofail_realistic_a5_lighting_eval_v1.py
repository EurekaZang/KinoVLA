#!/usr/bin/env python3
"""Evaluate the frozen held-out lighting axis against paired scale-v7 source states."""

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


def _scene_bootstrap(values: dict[str, list[float]], *, repetitions: int, seed: int) -> dict:
    scenes = sorted(values)
    means = np.asarray([np.mean(values[scene]) for scene in scenes], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(scenes), size=(repetitions, len(scenes)))
    draws = means[indices].mean(axis=1)
    return {
        "unit": "paired scene family",
        "scene_clusters": len(scenes),
        "pair_units": sum(len(value) for value in values.values()),
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
        default=ROOT / "configs/eval/kinofail_realistic_a5_lighting_eval_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _json(config_path)
    paths = {key: ROOT / config[key] for key in (
        "base_records", "base_features", "lighting_records", "lighting_features",
        "lighting_schedule", "model_implementation",
    )}
    if _sha(paths["lighting_schedule"]) != config["lighting_schedule_sha256"]:
        raise RuntimeError("lighting schedule hash mismatch")
    base_records, base_visual, base_proprio = _features(
        paths["base_records"], paths["base_features"]
    )
    dim_records, dim_visual, dim_proprio = _features(
        paths["lighting_records"], paths["lighting_features"]
    )
    schedule = _jsonl(paths["lighting_schedule"])
    source_by_dim_episode = {
        str(row["episode_id"]): str(row["source_scale_v7_episode_id"]) for row in schedule
    }
    schedule_by_dim_episode = {str(row["episode_id"]): row for row in schedule}
    base_index = {str(row["sample_id"]): index for index, row in enumerate(base_records)}
    dim_primary = [
        (index, row) for index, row in enumerate(dim_records)
        if row["appearance_intervention_id"] == "primary"
    ]
    paired = []
    for dim_index, dim_row in dim_primary:
        dim_episode = str(dim_row["physical_episode_id"])
        source_episode = source_by_dim_episode.get(dim_episode)
        if source_episode is None:
            raise RuntimeError(f"lighting snapshot not in frozen schedule: {dim_episode}")
        source_sample = f"{source_episode}__primary"
        if source_sample not in base_index:
            raise RuntimeError(f"source scale-v7 snapshot missing: {source_sample}")
        source_index = base_index[source_sample]
        source_row = base_records[source_index]
        if source_row["severity_id"] != "moderate" or dim_row["severity_id"] != "moderate":
            raise RuntimeError("paired lighting test must use moderate severity")
        if source_row["attribution_category"] != dim_row["attribution_category"]:
            raise RuntimeError("paired lighting truth mismatch")
        scheduled = schedule_by_dim_episode[dim_episode]
        paired.append({
            "dim_index": dim_index,
            "source_index": source_index,
            "dim_episode": dim_episode,
            "source_episode": source_episode,
            "counterfactual_group_id": dim_row["counterfactual_group_id"],
            "scene_family": dim_row["scene_family"],
            "target_operator": dim_row["target_operator"],
            "truth": dim_row["attribution_category"],
            "condition": dim_row["condition"],
            "lighting_profile": scheduled["lighting_profile"],
        })
    complete_groups = defaultdict(list)
    for row in paired:
        complete_groups[row["counterfactual_group_id"]].append(row)
    valid_groups = {
        group for group, rows in complete_groups.items()
        if len(rows) == 2 and {row["condition"] for row in rows} == {
            "anomaly", "nominal_counterfactual"
        }
    }
    paired = [row for row in paired if row["counterfactual_group_id"] in valid_groups]
    train_indices = np.asarray([
        index for index, row in enumerate(base_records) if row["severity_id"] == "severe"
    ], dtype=int)
    train_labels = np.asarray([
        base_records[index]["attribution_category"] for index in train_indices
    ])
    train_episodes = np.asarray([
        base_records[index]["physical_episode_id"] for index in train_indices
    ])
    train_groups = np.asarray([
        base_records[index]["counterfactual_group_id"] for index in train_indices
    ])
    source_indices = np.asarray([row["source_index"] for row in paired], dtype=int)
    dim_indices = np.asarray([row["dim_index"] for row in paired], dtype=int)
    truths = np.asarray([row["truth"] for row in paired])
    if set(train_episodes) & {row["source_episode"] for row in paired}:
        raise RuntimeError("training/test physical episode overlap")
    if set(truths) - set(train_labels):
        raise RuntimeError("lighting test contains unseen classes")

    prediction_rows = []
    summaries = {}
    statistics = {}
    repetitions = int(config["bootstrap_repetitions"])
    for seed_index, seed in enumerate([int(value) for value in config["training_seeds"]]):
        model = StructuredBidirectionalClassifier.fit(
            base_visual[train_indices], base_proprio[train_indices], train_labels,
            train_episodes, train_groups, seed=seed,
        )
        source_probability = model.predict_proba(
            base_visual[source_indices], base_proprio[source_indices]
        )
        dim_probability = model.predict_proba(dim_visual[dim_indices], dim_proprio[dim_indices])
        source_prediction = predictions_from_probabilities(source_probability, model.classes)
        dim_prediction = predictions_from_probabilities(dim_probability, model.classes)
        deltas: dict[str, list[float]] = defaultdict(list)
        agreements: dict[str, list[float]] = defaultdict(list)
        jsds: dict[str, list[float]] = defaultdict(list)
        for index, pair in enumerate(paired):
            source_correct = source_prediction[index] == truths[index]
            dim_correct = dim_prediction[index] == truths[index]
            jsd = _jsd(source_probability[index], dim_probability[index])
            scene = pair["scene_family"]
            deltas[scene].append(float(dim_correct) - float(source_correct))
            agreements[scene].append(float(source_prediction[index] == dim_prediction[index]))
            jsds[scene].append(jsd)
            prediction_rows.append({
                **pair, "seed": seed,
                "classes": list(model.classes),
                "source_prediction": str(source_prediction[index]),
                "dim_prediction": str(dim_prediction[index]),
                "source_probabilities": source_probability[index].tolist(),
                "dim_probabilities": dim_probability[index].tolist(),
                "source_correct": bool(source_correct),
                "dim_correct": bool(dim_correct),
                "prediction_agreement": bool(source_prediction[index] == dim_prediction[index]),
                "probability_jsd": jsd,
            })
        per_operator = {}
        for operator in sorted({row["target_operator"] for row in paired}):
            selected = [i for i, row in enumerate(paired) if row["target_operator"] == operator]
            per_operator[operator] = {
                "n": len(selected),
                "source_accuracy": float(np.mean(source_prediction[selected] == truths[selected])),
                "dim_accuracy": float(np.mean(dim_prediction[selected] == truths[selected])),
                "prediction_agreement": float(np.mean(source_prediction[selected] == dim_prediction[selected])),
            }
        summaries[str(seed)] = {
            "train_samples": len(train_indices),
            "train_physical_episodes": len(set(train_episodes)),
            "paired_test_samples": len(paired),
            "paired_counterfactual_groups": len(valid_groups),
            "source_balanced_accuracy": float(balanced_accuracy_score(truths, source_prediction)),
            "dim_balanced_accuracy": float(balanced_accuracy_score(truths, dim_prediction)),
            "prediction_agreement": float(np.mean(source_prediction == dim_prediction)),
            "mean_probability_jsd": float(np.mean([
                _jsd(source_probability[i], dim_probability[i]) for i in range(len(paired))
            ])),
            "per_operator": per_operator,
        }
        base_seed = int(config["bootstrap_seed"]) + seed_index * 3
        statistics[str(seed)] = {
            "dim_minus_source_accuracy": _scene_bootstrap(
                deltas, repetitions=repetitions, seed=base_seed
            ),
            "prediction_agreement": _scene_bootstrap(
                agreements, repetitions=repetitions, seed=base_seed + 1
            ),
            "probability_jsd": _scene_bootstrap(
                jsds, repetitions=repetitions, seed=base_seed + 2
            ),
        }
    predictions_path = ROOT / config["predictions"]
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    scenes = {row["scene_family"] for row in paired}
    operators = {row["target_operator"] for row in paired}
    acceptance = {
        "minimum_valid_pairs": len(valid_groups) >= int(config["minimum_valid_pairs"]),
        "required_scene_families": len(scenes) == int(config["required_scene_families"]),
        "required_operators": len(operators) == int(config["required_operators"]),
        "three_training_seeds": len(summaries) == 3,
        "paired_scene_cluster_statistics": all(
            row["dim_minus_source_accuracy"]["scene_clusters"] == len(scenes)
            for row in statistics.values()
        ),
        "no_train_test_physical_episode_overlap": True,
        "negative_ood_results_retained": True,
    }
    result = {
        "schema_version": "kinofail.realistic-a5-lighting-heldout-result.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "confirmatory_complete",
        "dataset_scope": "kinofail_realistic",
        "protocol": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha(config_path)},
        "input_hashes": {key: _sha(path) for key, path in paths.items()},
        "valid_counterfactual_pairs": len(valid_groups),
        "excluded_scheduled_pairs": 99 - len(valid_groups),
        "scene_families": sorted(scenes),
        "operators": sorted(operators),
        "summaries": summaries,
        "paired_scene_cluster_statistics": statistics,
        "predictions": {
            "path": str(predictions_path.relative_to(ROOT)),
            "sha256": _sha(predictions_path), "rows": len(prediction_rows),
        },
        "acceptance": acceptance,
        "passed": all(acceptance.values()),
        "claim_boundary": (
            "This is paired synthetic held-out lighting robustness. It does not establish a real-Go2 "
            "lighting-domain claim, and negative or null shifts remain reportable."
        ),
        "a8_in_scope": False,
    }
    output_path = ROOT / config["output"]
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(output_path), "passed": result["passed"],
        "valid_pairs": len(valid_groups), "excluded_pairs": 99 - len(valid_groups),
        "summaries": summaries, "statistics": statistics,
    }, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
