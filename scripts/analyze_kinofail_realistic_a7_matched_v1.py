#!/usr/bin/env python3
"""Evaluate the five matched realistic A7 cells with frozen scale-v8 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import balanced_accuracy_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.realistic_multimodal import (  # noqa: E402
    predictions_from_probabilities,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _bundle(snapshot_dir: Path, feature_dir: Path) -> dict[str, Any]:
    records_path = snapshot_dir / "snapshot_records.jsonl"
    feature_path = feature_dir / "features.npz"
    feature_manifest_path = feature_dir / "feature_manifest.json"
    audit_path = snapshot_dir / "extraction_audit.json"
    audit = _json(audit_path)
    feature_manifest = _json(feature_manifest_path)
    if audit.get("passed") is not True:
        raise RuntimeError(f"snapshot audit failed: {audit_path}")
    if (
        feature_manifest.get("status") != "complete"
        or feature_manifest.get("output_sha256", {}).get("features")
        != _sha256(feature_path)
    ):
        raise RuntimeError(f"feature cache failed integrity: {feature_path}")
    records = _jsonl(records_path)
    archive = np.load(feature_path, allow_pickle=False)
    sample_ids = archive["sample_ids"].astype(str)
    if sample_ids.tolist() != [str(row["sample_id"]) for row in records]:
        raise RuntimeError(f"feature/record order mismatch: {feature_path}")
    return {
        "records": records,
        "visual": np.asarray(archive["visual"], dtype=np.float32),
        "proprio": np.asarray(archive["proprio"], dtype=np.float32),
        "by_id": {str(row["sample_id"]): index for index, row in enumerate(records)},
        "hashes": {
            "records": _sha256(records_path),
            "features": _sha256(feature_path),
            "feature_manifest": _sha256(feature_manifest_path),
            "audit": _sha256(audit_path),
        },
    }


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


def _bootstrap(
    values: Mapping[str, list[float]], *, repetitions: int, seed: int
) -> dict[str, Any]:
    clusters = np.asarray(
        [float(np.mean(row)) for _, row in sorted(values.items())],
        dtype=np.float64,
    )
    if not len(clusters):
        raise RuntimeError("empty A7 scene-cluster bootstrap")
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            float(np.mean(rng.choice(clusters, len(clusters), replace=True)))
            for _ in range(repetitions)
        ],
        dtype=np.float64,
    )
    return {
        "unit": "scene_family",
        "clusters": len(clusters),
        "point": float(np.mean(clusters)),
        "ci95": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
        "repetitions": repetitions,
        "seed": seed,
    }


def _load_models(config: Mapping[str, Any]) -> dict[int, Any]:
    training_path = ROOT / config["training_manifest"]
    training = _json(training_path)
    if training.get("status") != "confirmatory_complete":
        raise RuntimeError("A7 requires complete frozen scale-v8 training")
    checkpoints = training["checkpoints_sha256"]
    models = {}
    for seed in [int(value) for value in config["training_seeds"]]:
        relative = (
            f"{config['checkpoint_root']}/scene/seed{seed}/"
            "structured_bidirectional.pkl"
        )
        path = ROOT / relative
        expected = checkpoints.get(relative)
        if expected is None or _sha256(path) != expected:
            raise RuntimeError(f"A7 checkpoint hash mismatch: {relative}")
        with path.open("rb") as stream:
            models[seed] = pickle.load(stream)
    return models


def _infer(model: Any, bundle: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    probabilities = model.predict_proba(bundle["visual"], bundle["proprio"])
    predictions = predictions_from_probabilities(probabilities, model.classes)
    return {
        str(record["sample_id"]): {
            "record": record,
            "prediction": str(predictions[index]),
            "probabilities": np.asarray(probabilities[index], dtype=np.float64),
            "classes": list(model.classes),
        }
        for index, record in enumerate(bundle["records"])
    }


def _balanced(rows: list[dict[str, Any]]) -> float:
    return float(
        balanced_accuracy_score(
            [str(row["truth"]) for row in rows],
            [str(row["prediction"]) for row in rows],
        )
    )


def _visual_cells(
    bundle: Mapping[str, Any],
    models: Mapping[int, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    arms = list(config["visual_render_arms"])
    base_arm = arms[0]
    by_episode_arm = {
        (str(row["physical_episode_id"]), str(row["appearance_intervention_id"])): str(
            row["sample_id"]
        )
        for row in bundle["records"]
    }
    episodes = sorted({str(row["physical_episode_id"]) for row in bundle["records"]})
    if len(episodes) != 18 or len(bundle["records"]) != 72:
        raise RuntimeError("A7 visual replay is not 9 complete four-arm pairs")
    arm_to_cell = {
        arms[1]: "default_grid_vs_realistic_scene",
        arms[2]: "procedural_texture_vs_scanned_pbr",
        arms[3]: "world_follow_vs_body_fixed_camera",
    }
    output = {
        cell: {
            "status": "confirmatory_complete",
            "base_arm": base_arm,
            "alternative_arm": arm,
            "by_seed": {},
        }
        for arm, cell in arm_to_cell.items()
    }
    repetitions = int(config["bootstrap_repetitions"])
    bootstrap_seed = int(config["bootstrap_seed"])
    for seed_index, (seed, model) in enumerate(sorted(models.items())):
        inference = _infer(model, bundle)
        for arm_index, (arm, cell) in enumerate(arm_to_cell.items()):
            base_rows = []
            alternative_rows = []
            agreement = []
            jsd = []
            accuracy_delta: dict[str, list[float]] = defaultdict(list)
            for episode in episodes:
                base = inference[by_episode_arm[(episode, base_arm)]]
                alternative = inference[by_episode_arm[(episode, arm)]]
                truth = str(base["record"]["attribution_category"])
                if truth != str(alternative["record"]["attribution_category"]):
                    raise RuntimeError("A7 visual truth mismatch across render arms")
                scene = str(base["record"]["scene_family"])
                base_correct = float(base["prediction"] == truth)
                alternative_correct = float(alternative["prediction"] == truth)
                accuracy_delta[scene].append(alternative_correct - base_correct)
                agreement.append(base["prediction"] == alternative["prediction"])
                jsd.append(_jsd(base["probabilities"], alternative["probabilities"]))
                base_rows.append({"truth": truth, "prediction": base["prediction"]})
                alternative_rows.append(
                    {"truth": truth, "prediction": alternative["prediction"]}
                )
            output[cell]["by_seed"][str(seed)] = {
                "physical_episodes": len(episodes),
                "base_balanced_accuracy": _balanced(base_rows),
                "alternative_balanced_accuracy": _balanced(alternative_rows),
                "hard_prediction_agreement": float(np.mean(agreement)),
                "mean_probability_jsd": float(np.mean(jsd)),
                "alternative_minus_base_accuracy": _bootstrap(
                    accuracy_delta,
                    repetitions=repetitions,
                    seed=bootstrap_seed + 10 * seed_index + arm_index,
                ),
            }
    for row in output.values():
        row["complete"] = len(row["by_seed"]) == len(models)
    return output


def _surrogate_cell(
    main: Mapping[str, Any],
    surrogate: Mapping[str, Any],
    models: Mapping[int, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    selected_ids = set(config["surrogate_pair_ids"])
    surrogate_primary = [
        row
        for row in surrogate["records"]
        if row["appearance_intervention_id"] == "primary"
    ]
    if (
        len(surrogate_primary) != 36
        or {str(row["counterfactual_group_id"]) for row in surrogate_primary}
        != selected_ids
    ):
        raise RuntimeError("A7 surrogate primary subset is incomplete")
    sample_ids = [str(row["sample_id"]) for row in surrogate_primary]
    if any(sample_id not in main["by_id"] for sample_id in sample_ids):
        raise RuntimeError("A7 local-physics matched samples are absent from main snapshots")

    repetitions = int(config["bootstrap_repetitions"])
    bootstrap_seed = int(config["bootstrap_seed"])
    by_seed = {}
    for seed_index, (seed, model) in enumerate(sorted(models.items())):
        local_probability = model.predict_proba(
            np.stack([main["visual"][main["by_id"][sample_id]] for sample_id in sample_ids]),
            np.stack([main["proprio"][main["by_id"][sample_id]] for sample_id in sample_ids]),
        )
        surrogate_probability = model.predict_proba(
            np.stack(
                [
                    surrogate["visual"][surrogate["by_id"][sample_id]]
                    for sample_id in sample_ids
                ]
            ),
            np.stack(
                [
                    surrogate["proprio"][surrogate["by_id"][sample_id]]
                    for sample_id in sample_ids
                ]
            ),
        )
        local_prediction = predictions_from_probabilities(
            local_probability, model.classes
        )
        surrogate_prediction = predictions_from_probabilities(
            surrogate_probability, model.classes
        )
        local_rows, surrogate_rows = [], []
        accuracy_delta: dict[str, list[float]] = defaultdict(list)
        agreement, divergences = [], []
        for index, row in enumerate(surrogate_primary):
            truth = str(row["attribution_category"])
            local_correct = float(str(local_prediction[index]) == truth)
            surrogate_correct = float(str(surrogate_prediction[index]) == truth)
            accuracy_delta[str(row["scene_family"])].append(
                surrogate_correct - local_correct
            )
            agreement.append(local_prediction[index] == surrogate_prediction[index])
            divergences.append(
                _jsd(local_probability[index], surrogate_probability[index])
            )
            local_rows.append(
                {"truth": truth, "prediction": str(local_prediction[index])}
            )
            surrogate_rows.append(
                {"truth": truth, "prediction": str(surrogate_prediction[index])}
            )
        by_seed[str(seed)] = {
            "matched_primary_samples": len(sample_ids),
            "local_physics_balanced_accuracy": _balanced(local_rows),
            "legacy_surrogate_balanced_accuracy": _balanced(surrogate_rows),
            "hard_prediction_agreement": float(np.mean(agreement)),
            "mean_probability_jsd": float(np.mean(divergences)),
            "surrogate_minus_local_accuracy": _bootstrap(
                accuracy_delta,
                repetitions=repetitions,
                seed=bootstrap_seed + 100 + seed_index,
            ),
        }
    return {
        "status": "confirmatory_complete",
        "complete": len(by_seed) == len(models),
        "matched_pairs": 18,
        "matched_scenes": 9,
        "operators": ["O2_compliance", "O4_tether"],
        "by_seed": by_seed,
        "interpretation": (
            "This estimates sensitivity to replacing local foot mechanisms with the frozen "
            "legacy trunk-wrench architecture; no parameter-equivalence claim is made."
        ),
    }


def _geometry_physics_cell(
    main: Mapping[str, Any],
    models: Mapping[int, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    o8_records = [
        row for row in main["records"] if row["target_operator"] == "O8_invisible_collider"
    ]
    pair_ids = {str(row["counterfactual_group_id"]) for row in o8_records}
    if len(pair_ids) != 90 or len(o8_records) != 540:
        raise RuntimeError("A7 O8 geometry/physics block is not the full 90-pair corpus")
    repetitions = int(config["bootstrap_repetitions"])
    bootstrap_seed = int(config["bootstrap_seed"])
    by_seed = {}
    for seed_index, (seed, model) in enumerate(sorted(models.items())):
        indices = [main["by_id"][str(row["sample_id"])] for row in o8_records]
        probability = model.predict_proba(
            main["visual"][indices], main["proprio"][indices]
        )
        prediction = predictions_from_probabilities(probability, model.classes)
        inferred = {
            str(row["sample_id"]): {
                "record": row,
                "probabilities": probability[index],
                "prediction": str(prediction[index]),
            }
            for index, row in enumerate(o8_records)
        }
        by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for value in inferred.values():
            by_episode[str(value["record"]["physical_episode_id"])].append(value)
        hard_consistency, texture_jsd = [], []
        for rows in by_episode.values():
            if len(rows) != 3:
                raise RuntimeError("A7 O8 texture arm count mismatch")
            hard_consistency.append(len({row["prediction"] for row in rows}) == 1)
            probs = [np.asarray(row["probabilities"]) for row in rows]
            texture_jsd.append(
                float(
                    np.mean(
                        [
                            _jsd(probs[i], probs[j])
                            for i in range(3)
                            for j in range(i + 1, 3)
                        ]
                    )
                )
            )

        physics_gain: dict[str, list[float]] = defaultdict(list)
        geometry_false_positive = []
        physics_correct = []
        for pair_id in sorted(pair_ids):
            rows = [
                value
                for value in inferred.values()
                if value["record"]["counterfactual_group_id"] == pair_id
                and value["record"]["appearance_intervention_id"] == "primary"
            ]
            by_condition = {
                str(value["record"]["condition"]): value for value in rows
            }
            if set(by_condition) != {"anomaly", "nominal_counterfactual"}:
                raise RuntimeError(f"A7 O8 primary pair incomplete: {pair_id}")
            anomaly = by_condition["anomaly"]
            nominal = by_condition["nominal_counterfactual"]
            anomaly_truth = str(anomaly["record"]["attribution_category"])
            class_index = list(model.classes).index(anomaly_truth)
            physics_gain[str(anomaly["record"]["scene_family"])].append(
                float(anomaly["probabilities"][class_index])
                - float(nominal["probabilities"][class_index])
            )
            geometry_false_positive.append(nominal["prediction"] != "nominal")
            physics_correct.append(anomaly["prediction"] == anomaly_truth)
        by_seed[str(seed)] = {
            "pairs": len(pair_ids),
            "geometry_only_nominal_false_positive_rate": float(
                np.mean(geometry_false_positive)
            ),
            "physics_active_anomaly_accuracy": float(np.mean(physics_correct)),
            "texture_only_hard_consistency": float(np.mean(hard_consistency)),
            "texture_only_mean_probability_jsd": float(np.mean(texture_jsd)),
            "physics_active_minus_geometry_only_anomaly_probability": _bootstrap(
                physics_gain,
                repetitions=repetitions,
                seed=bootstrap_seed + 200 + seed_index,
            ),
        }
    return {
        "status": "confirmatory_complete",
        "complete": len(by_seed) == len(models),
        "matched_pairs": len(pair_ids),
        "arms": {
            "geometry_only": "transparent panel present, collision disabled",
            "physics_active": "same panel present, collision enabled",
            "texture_only": "three synchronized PBR views at fixed physics",
        },
        "by_seed": by_seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/eval/kinofail_realistic_a7_matched_analysis_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _json(config_path)
    bound = {
        "analyzer": Path(__file__).resolve(),
        "design": ROOT / config["design"],
        "snapshot_protocol": ROOT / config["snapshot_protocol"],
    }
    mismatches = {
        key: {"expected": config["input_sha256"][key], "actual": _sha256(path)}
        for key, path in bound.items()
        if not path.is_file() or _sha256(path) != config["input_sha256"][key]
    }
    if mismatches:
        raise RuntimeError(f"A7 analysis frozen-input mismatch: {mismatches}")
    models = _load_models(config)
    main_bundle = _bundle(
        ROOT / config["main_snapshot_dir"], ROOT / config["main_feature_dir"]
    )
    surrogate_bundle = _bundle(
        ROOT / config["surrogate_snapshot_dir"],
        ROOT / config["surrogate_feature_dir"],
    )
    visual_bundle = _bundle(
        ROOT / config["visual_snapshot_dir"], ROOT / config["visual_feature_dir"]
    )
    cells = _visual_cells(visual_bundle, models, config)
    cells["surrogate_trunk_effect_vs_local_physics"] = _surrogate_cell(
        main_bundle, surrogate_bundle, models, config
    )
    cells["texture_vs_geometry_vs_physics_intervention"] = _geometry_physics_cell(
        main_bundle, models, config
    )
    required = [
        "default_grid_vs_realistic_scene",
        "procedural_texture_vs_scanned_pbr",
        "world_follow_vs_body_fixed_camera",
        "surrogate_trunk_effect_vs_local_physics",
        "texture_vs_geometry_vs_physics_intervention",
    ]
    if set(cells) != set(required):
        raise RuntimeError("A7 matched analysis cell mismatch")
    result = {
        "schema_version": "kinofail.realistic-a7-matched-analysis.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "confirmatory_complete",
        "dataset_scope": "kinofail_realistic",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "training_seeds": sorted(models),
        "input_bundle_hashes": {
            "main": main_bundle["hashes"],
            "legacy_surrogate": surrogate_bundle["hashes"],
            "visual_replay": visual_bundle["hashes"],
        },
        "cells": cells,
        "acceptance": {
            "all_five_matched_cells_complete": all(
                cells[cell]["complete"] for cell in required
            ),
            "three_frozen_training_seeds": len(models) == 3,
            "scene_cluster_bootstrap_repetitions": int(
                config["bootstrap_repetitions"]
            )
            == 10000,
            "negative_or_null_results_retained": True,
            "a8_excluded": config.get("a8_in_scope") is False,
        },
        "claim_guard": (
            "A7 reports matched sensitivity and decomposition results, including null or "
            "negative effects. It does not claim real-Go2 sim2real validation."
        ),
    }
    result["passed"] = all(result["acceptance"].values())
    output = ROOT / config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output.relative_to(ROOT)),
                "passed": result["passed"],
                "cells": required,
                "training_seeds": sorted(models),
            },
            indent=2,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
