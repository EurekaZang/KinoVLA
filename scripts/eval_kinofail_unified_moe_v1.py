#!/usr/bin/env python3
"""Retrospective development evaluation of the unified KINO learned router.

This script deliberately labels its outputs as development evidence because
the scale-v8 and C2-v5 test outcomes were already observed before this
architecture was designed.  It establishes the executable model and reporting
contract that can subsequently be frozen for a new scene extension.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.unified_moe import (  # noqa: E402
    EXPERT_NAMES,
    UnifiedEvidenceMoE,
)
from scripts.run_kinofail_realistic_multimodal_v1 import (  # noqa: E402
    _masks,
    _split_labels,
)


LEGACY_BODY_CLASSES = {
    "effort_decay",
    "external_push",
    "high_centering",
    "invisible_obstacle",
    "low_friction",
    "obs_bias",
    "overload",
    "region_collapse",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aligned(
    probabilities: np.ndarray,
    source_classes: list[str],
    target_classes: list[str],
) -> np.ndarray:
    source = {str(label): index for index, label in enumerate(source_classes)}
    values = np.asarray(probabilities, dtype=np.float64)
    out = np.zeros((len(values), len(target_classes)), dtype=np.float64)
    for index, label in enumerate(target_classes):
        if label in source:
            out[:, index] = values[:, source[label]]
    normalizer = out.sum(axis=1, keepdims=True)
    return out / np.where(normalizer <= 0.0, 1.0, normalizer)


def _predict_methods(
    model: UnifiedEvidenceMoE,
    visual: np.ndarray,
    proprio: np.ndarray,
    truth: np.ndarray,
    *,
    random_seed: int,
) -> dict[str, dict[str, Any]]:
    routed = model.predict_with_routes(visual, proprio)
    expert = routed.expert_probabilities
    classes = model.classes
    expert_prediction = expert.argmax(axis=2)
    truth_index = np.asarray([classes.index(str(label)) for label in truth])
    results: dict[str, dict[str, Any]] = {
        "learned_router": {
            "probabilities": routed.probabilities,
            "routes": routed.routes,
            "evidence_routes": routed.evidence_routes,
        },
        "fixed_vision": {
            "probabilities": expert[:, EXPERT_NAMES.index("vision"), :],
            "routes": np.full(len(truth), "vision"),
            "evidence_routes": np.full(len(truth), "vision"),
        },
        "fixed_proprio": {
            "probabilities": expert[:, EXPERT_NAMES.index("proprio"), :],
            "routes": np.full(len(truth), "proprio"),
            "evidence_routes": np.full(len(truth), "proprio"),
        },
        "fixed_joint": {
            "probabilities": expert[:, EXPERT_NAMES.index("joint"), :],
            "routes": np.full(len(truth), "joint"),
            "evidence_routes": np.full(
                len(truth), "cross_modal_interaction"
            ),
        },
        "late_average": {
            "probabilities": expert.mean(axis=1),
            "routes": np.full(len(truth), "late_average"),
            "evidence_routes": np.full(len(truth), "all_modalities"),
        },
    }

    proprio_index = EXPERT_NAMES.index("proprio")
    vision_index = EXPERT_NAMES.index("vision")
    proposal = np.asarray(classes)[expert_prediction[:, proprio_index]]
    use_proprio = np.isin(proposal, sorted(LEGACY_BODY_CLASSES))
    legacy_probability = expert[:, vision_index, :].copy()
    legacy_probability[use_proprio] = expert[
        use_proprio, proprio_index, :
    ]
    results["legacy_class_rule"] = {
        "probabilities": legacy_probability,
        "routes": np.where(use_proprio, "proprio", "vision"),
        "evidence_routes": np.where(use_proprio, "proprio", "vision"),
    }

    rng = np.random.default_rng(int(random_seed))
    random_route = rng.integers(0, len(EXPERT_NAMES), size=len(truth))
    results["random_route"] = {
        "probabilities": expert[np.arange(len(truth)), random_route],
        "routes": np.asarray(EXPERT_NAMES)[random_route],
        "evidence_routes": np.asarray(EXPERT_NAMES)[random_route],
    }

    expert_correct = expert_prediction == truth_index[:, None]
    oracle_route = np.full(len(truth), proprio_index, dtype=np.int64)
    preference = (
        EXPERT_NAMES.index("proprio"),
        EXPERT_NAMES.index("joint"),
        EXPERT_NAMES.index("vision"),
    )
    for row in range(len(truth)):
        correct = [index for index in preference if expert_correct[row, index]]
        if correct:
            oracle_route[row] = correct[0]
        else:
            oracle_route[row] = int(
                np.argmax(expert[row, :, truth_index[row]])
            )
    results["oracle_route"] = {
        "probabilities": expert[np.arange(len(truth)), oracle_route],
        "routes": np.asarray(EXPERT_NAMES)[oracle_route],
        "evidence_routes": np.asarray(EXPERT_NAMES)[oracle_route],
    }
    for result in results.values():
        probability = np.asarray(result["probabilities"], dtype=np.float64)
        result["prediction"] = np.asarray(classes)[
            probability.argmax(axis=1)
        ]
        result["confidence"] = probability.max(axis=1)
    results["learned_router"].update(
        {
            "route_probabilities": routed.route_probabilities,
            "route_scores": routed.route_scores,
            "disagreement": routed.disagreement,
        }
    )
    return results


def _case_level(
    rows: list[dict[str, Any]], group_key: str
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    mean_correctness = []
    strict_correctness = []
    confidence = []
    for group_rows in grouped.values():
        correct = np.asarray(
            [row["prediction"] == row["truth"] for row in group_rows],
            dtype=np.float64,
        )
        mean_correctness.append(float(correct.mean()))
        strict_correctness.append(bool(correct.all()))
        confidence.append(
            float(min(row["confidence"] for row in group_rows))
        )
    return {
        "groups": len(grouped),
        "mean_within_group_accuracy": float(np.mean(mean_correctness)),
        "strict_all_samples_correct": int(sum(strict_correctness)),
        "strict_all_samples_rate": float(np.mean(strict_correctness)),
        "_strict": np.asarray(strict_correctness, dtype=np.float64),
        "_confidence": np.asarray(confidence, dtype=np.float64),
    }


def _selective_curve(
    strict_correctness: np.ndarray,
    confidence: np.ndarray,
    coverage_grid: list[float],
) -> list[dict[str, Any]]:
    order = np.argsort(-np.asarray(confidence), kind="stable")
    total = len(order)
    curve = []
    for requested in coverage_grid:
        selected_count = min(total, max(1, int(math.ceil(requested * total))))
        selected = order[:selected_count]
        curve.append(
            {
                "requested_coverage": float(requested),
                "coverage": float(selected_count / total),
                "selected_groups": selected_count,
                "selective_risk": float(
                    1.0 - strict_correctness[selected].mean()
                ),
            }
        )
    return curve


def _metrics(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    coverage_grid: list[float],
) -> dict[str, Any]:
    truth = [str(row["truth"]) for row in rows]
    prediction = [str(row["prediction"]) for row in rows]
    classes = sorted(set(truth) | set(prediction))
    matrix = confusion_matrix(truth, prediction, labels=classes)
    per_class = {}
    for index, label in enumerate(classes):
        support = int(matrix[index].sum())
        per_class[label] = {
            "support": support,
            "recall": (
                float(matrix[index, index] / support)
                if support
                else None
            ),
        }
    case = _case_level(rows, group_key)
    selective = _selective_curve(
        case.pop("_strict"),
        case.pop("_confidence"),
        coverage_grid,
    )
    scene_values = {}
    for scene in sorted({str(row["scene"]) for row in rows}):
        subset = [row for row in rows if str(row["scene"]) == scene]
        scene_values[scene] = {
            "samples": len(subset),
            "accuracy": float(
                np.mean(
                    [
                        row["prediction"] == row["truth"]
                        for row in subset
                    ]
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    [row["truth"] for row in subset],
                    [row["prediction"] for row in subset],
                )
            ),
        }
    material_values = {}
    for material in sorted({str(row["material"]) for row in rows}):
        subset = [row for row in rows if str(row["material"]) == material]
        material_values[material] = {
            "samples": len(subset),
            "accuracy": float(
                np.mean(
                    [
                        row["prediction"] == row["truth"]
                        for row in subset
                    ]
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    [row["truth"] for row in subset],
                    [row["prediction"] for row in subset],
                )
            ),
        }
    return {
        "samples": len(rows),
        "accuracy": float(
            np.mean(
                [
                    row["prediction"] == row["truth"]
                    for row in rows
                ]
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(truth, prediction)
        ),
        "case_or_pair_level": case,
        "selective_risk": selective,
        "classes": classes,
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class": per_class,
        "per_scene": scene_values,
        "per_material": material_values,
    }


def _load_c2(directory: Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    rows = _jsonl(directory / "records.jsonl")
    with np.load(directory / "features.npz", allow_pickle=False) as archive:
        sample_ids = archive["sample_ids"].astype(str)
        visual = np.asarray(archive["visual"], dtype=np.float32)
        proprio = np.asarray(archive["proprio"], dtype=np.float32)
    expected = [str(row["sample_id"]) for row in rows]
    if sample_ids.tolist() != expected:
        raise RuntimeError(f"misaligned C2 bundle: {directory}")
    if proprio.shape[1] != 80 or visual.shape[1] != 1536:
        raise RuntimeError(f"unexpected C2 feature shape: {directory}")
    return rows, visual, proprio


def _append_prediction_rows(
    destination: list[dict[str, Any]],
    *,
    dataset: str,
    axis: str,
    seed: int,
    metadata: list[dict[str, Any]],
    methods: dict[str, dict[str, Any]],
    classes: list[str],
) -> None:
    for method, result in methods.items():
        probabilities = np.asarray(result["probabilities"])
        for index, source in enumerate(metadata):
            row = {
                "dataset": dataset,
                "axis": axis,
                "seed": int(seed),
                "method": method,
                "sample_id": str(source["sample_id"]),
                "group_id": str(source["group_id"]),
                "scene": str(source["scene"]),
                "material": str(source["material"]),
                "domain": str(source["domain"]),
                "truth": str(source["truth"]),
                "prediction": str(result["prediction"][index]),
                "confidence": float(result["confidence"][index]),
                "classes": list(classes),
                "probabilities": probabilities[index].tolist(),
                "route": str(result["routes"][index]),
                "evidence_route": str(result["evidence_routes"][index]),
            }
            for optional in ("cell", "operator", "condition"):
                if optional in source:
                    row[optional] = source[optional]
            if method == "learned_router":
                row.update(
                    {
                        "route_probabilities": result[
                            "route_probabilities"
                        ][index].tolist(),
                        "route_scores": result["route_scores"][
                            index
                        ].tolist(),
                        "expert_disagreement": bool(
                            result["disagreement"][index]
                        ),
                    }
                )
            destination.append(row)


def _action_report(
    predictions: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    direct_report: dict[str, Any],
    *,
    methods: list[str],
    seeds: list[int],
    rule: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    primary = [
        row
        for row in predictions
        if row["dataset"] == "scale"
        and row["axis"] == "scene_and_material"
    ]
    lookup = {
        (row["method"], int(row["seed"]), row["sample_id"]): row
        for row in primary
    }
    outcomes = {
        str(row["case_id"]): row for row in direct_report["paired_cases"]
    }
    action_rows: list[dict[str, Any]] = []
    report_methods: dict[str, Any] = {}
    realized_outcomes: dict[str, list[dict[str, Any]] | None] = {}
    target = str(rule["target"])
    threshold = float(rule["mean_probability_threshold"])
    minimum_agreement = int(rule["minimum_seed_agreement"])
    for method in methods:
        method_rows = []
        for case in schedule:
            sample_id = str(case["source_sample_id"])
            seed_rows = [
                lookup[(method, seed, sample_id)] for seed in seeds
            ]
            target_indices = [
                row["classes"].index(target) for row in seed_rows
            ]
            target_probability = float(
                np.mean(
                    [
                        row["probabilities"][index]
                        for row, index in zip(
                            seed_rows, target_indices, strict=True
                        )
                    ]
                )
            )
            seed_predictions = [row["prediction"] for row in seed_rows]
            agreement = Counter(seed_predictions).most_common(1)[0][1]
            release = (
                agreement >= minimum_agreement
                and len(set(seed_predictions)) == 1
                and seed_predictions[0] == target
                and target_probability >= threshold
            )
            method_rows.append(
                {
                    "method": method,
                    "case_id": str(case["case_id"]),
                    "source_sample_id": sample_id,
                    "scene": str(case["scene_cluster"]),
                    "domain": str(case["domain"]),
                    "operator": str(case["operator"]),
                    "truth": str(case["truth_attribution"]),
                    "action": (
                        str(rule["release_action"])
                        if release
                        else str(rule["fallback_action"])
                    ),
                    "release": bool(release),
                    "mean_target_probability": target_probability,
                    "seed_predictions": seed_predictions,
                    "seed_agreement": agreement,
                }
            )
        action_rows.extend(method_rows)
        release_rows = [row for row in method_rows if row["release"]]
        measured_reusable = all(
            row["operator"] == "O4_tether" for row in release_rows
        )
        cost_deltas = []
        success_deltas = []
        fall_deltas = []
        if measured_reusable:
            for row in method_rows:
                measured = outcomes[row["case_id"]]
                selected = (
                    measured["selective"]
                    if row["release"]
                    else measured["always_safe"]
                )
                fallback = measured["always_safe"]
                cost_deltas.append(
                    float(selected["terminal_cost"])
                    - float(fallback["terminal_cost"])
                )
                success_deltas.append(
                    float(selected["success"]) - float(fallback["success"])
                )
                fall_deltas.append(
                    float(selected["fell"]) - float(fallback["fell"])
                )
            realized_outcomes[method] = [
                {
                    "case_id": row["case_id"],
                    "scene": row["scene"],
                    "operator": row["operator"],
                    "action": row["action"],
                    "terminal_cost": float(
                        (
                            outcomes[row["case_id"]]["selective"]
                            if row["release"]
                            else outcomes[row["case_id"]]["always_safe"]
                        )["terminal_cost"]
                    ),
                    "success": bool(
                        (
                            outcomes[row["case_id"]]["selective"]
                            if row["release"]
                            else outcomes[row["case_id"]]["always_safe"]
                        )["success"]
                    ),
                    "fell": bool(
                        (
                            outcomes[row["case_id"]]["selective"]
                            if row["release"]
                            else outcomes[row["case_id"]]["always_safe"]
                        )["fell"]
                    ),
                }
                for row in method_rows
            ]
        else:
            realized_outcomes[method] = None
        vector = "".join("1" if row["release"] else "0" for row in method_rows)
        report_methods[method] = {
            "cases": len(method_rows),
            "release_cases": len(release_rows),
            "coverage": float(len(release_rows) / len(method_rows)),
            "release_by_operator": dict(
                sorted(Counter(row["operator"] for row in release_rows).items())
            ),
            "release_precision": (
                float(
                    np.mean(
                        [row["truth"] == target for row in release_rows]
                    )
                )
                if release_rows
                else None
            ),
            "action_vector_sha256": hashlib.sha256(
                vector.encode("ascii")
            ).hexdigest(),
            "measured_outcomes_reusable": measured_reusable,
            "closed_loop_vs_always_fallback": (
                {
                    "mean_terminal_cost_delta": float(
                        np.mean(cost_deltas)
                    ),
                    "mean_success_delta": float(
                        np.mean(success_deltas)
                    ),
                    "mean_fall_delta": float(np.mean(fall_deltas)),
                }
                if measured_reusable
                else None
            ),
        }
    unique_vectors = len(
        {
            value["action_vector_sha256"]
            for value in report_methods.values()
        }
    )
    learned_pairwise: dict[str, Any] = {}
    reference = realized_outcomes.get("learned_router")
    if reference is not None:
        reference_by_case = {
            row["case_id"]: row for row in reference
        }
        learned_actions = {
            row["case_id"]: row["action"] for row in reference
        }
        for method in methods:
            comparator = realized_outcomes.get(method)
            if method == "learned_router" or comparator is None:
                continue
            comparator_by_case = {
                row["case_id"]: row for row in comparator
            }
            comparator_actions = {
                row["case_id"]: row["action"] for row in comparator
            }
            case_ids = sorted(reference_by_case)
            cost_delta = np.asarray(
                [
                    reference_by_case[case_id]["terminal_cost"]
                    - comparator_by_case[case_id]["terminal_cost"]
                    for case_id in case_ids
                ],
                dtype=np.float64,
            )
            success_delta = np.asarray(
                [
                    float(reference_by_case[case_id]["success"])
                    - float(comparator_by_case[case_id]["success"])
                    for case_id in case_ids
                ],
                dtype=np.float64,
            )
            fall_delta = np.asarray(
                [
                    float(reference_by_case[case_id]["fell"])
                    - float(comparator_by_case[case_id]["fell"])
                    for case_id in case_ids
                ],
                dtype=np.float64,
            )
            disagreements = [
                case_id
                for case_id in case_ids
                if learned_actions[case_id]
                != comparator_actions[case_id]
            ]
            learned_pairwise[method] = {
                "comparison": "learned_router minus comparator",
                "cases": len(case_ids),
                "action_disagreement_cases": len(disagreements),
                "action_disagreement_case_ids": disagreements,
                "mean_terminal_cost_delta": float(cost_delta.mean()),
                "mean_success_delta": float(success_delta.mean()),
                "mean_fall_delta": float(fall_delta.mean()),
                "cost_wins_ties_losses": {
                    "wins": int((cost_delta < 0.0).sum()),
                    "ties": int((cost_delta == 0.0).sum()),
                    "losses": int((cost_delta > 0.0).sum()),
                },
            }
    return (
        {
            "schema_version": "kinofail.unified-c4-action-audit.v2",
            "case_count": len(schedule),
            "methods": report_methods,
            "learned_router_pairwise": learned_pairwise,
            "unique_action_vectors": unique_vectors,
            "measured_outcome_reuse_rule": (
                "Exact-prefix measured outcomes are reused only when every "
                "released case is O4; non-O4 release actions were not measured."
            ),
        },
        action_rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/eval/kinofail_unified_moe_v1_development.json",
    )
    parser.add_argument(
        "--skip-loso",
        action="store_true",
        help="Run the four registered axes but omit nine-scene LOSO.",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _json(config_path)
    output = (ROOT / config["output_root"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    records_path = ROOT / config["scale_records"]
    features_path = ROOT / config["scale_features"]
    feature_manifest_path = ROOT / config["scale_feature_manifest"]
    registry_path = ROOT / config["scene_registry"]
    feature_manifest = _json(feature_manifest_path)
    if (
        feature_manifest.get("status") != "complete"
        or feature_manifest["output_sha256"]["features"]
        != _sha256(features_path)
    ):
        raise RuntimeError("unified feature cache is invalid")
    scale_rows = _jsonl(records_path)
    with np.load(features_path, allow_pickle=False) as archive:
        sample_ids = archive["sample_ids"].astype(str)
        visual = np.asarray(archive["visual"], dtype=np.float32)
        proprio = np.asarray(archive["proprio"], dtype=np.float32)
    if sample_ids.tolist() != [
        str(row["sample_id"]) for row in scale_rows
    ]:
        raise RuntimeError("scale features and records are misaligned")
    registry = _json(registry_path)
    scene_split, material_split = _split_labels(scale_rows, registry)
    labels = np.asarray(
        [str(row["attribution_category"]) for row in scale_rows]
    )
    episodes = np.asarray(
        [str(row["physical_episode_id"]) for row in scale_rows]
    )
    groups = np.asarray(
        [str(row["counterfactual_group_id"]) for row in scale_rows]
    )

    c2_dev_dir = ROOT / config["c2_development_features"]
    c2_test_dir = ROOT / config["c2_formal_features"]
    c2_dev_rows, c2_dev_visual, c2_dev_proprio = _load_c2(c2_dev_dir)
    c2_test_rows, c2_test_visual, c2_test_proprio = _load_c2(c2_test_dir)
    c2_dev_labels = np.asarray(
        [str(row["attribution_category"]) for row in c2_dev_rows]
    )
    c2_dev_groups = np.asarray(
        [f"c2::{row['case_id']}" for row in c2_dev_rows]
    )
    c2_dev_episodes = np.asarray(
        [
            f"c2::{row['case_id']}::{row['attribution_category']}"
            for row in c2_dev_rows
        ]
    )
    c2_test_labels = np.asarray(
        [str(row["attribution_category"]) for row in c2_test_rows]
    )
    seeds = [int(value) for value in config["training_seeds"]]
    router_config = config["router"]
    prediction_rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    model_fit_audits: dict[str, Any] = {}

    for axis_index, axis in enumerate(config["heldout_axes"]):
        train, main_test, _ = _masks(
            axis, scale_rows, scene_split, material_split
        )
        test_indices = np.flatnonzero(main_test)
        scale_metadata = [
            {
                "sample_id": scale_rows[index]["sample_id"],
                "group_id": scale_rows[index]["counterfactual_group_id"],
                "scene": scale_rows[index]["scene_family"],
                "material": scale_rows[index]["material_family"],
                "domain": scale_rows[index]["domain"],
                "truth": scale_rows[index]["attribution_category"],
                "operator": scale_rows[index]["target_operator"],
                "condition": scale_rows[index]["condition"],
            }
            for index in test_indices
        ]
        for seed in seeds:
            train_visual = np.concatenate(
                [visual[train], c2_dev_visual], axis=0
            )
            train_proprio = np.concatenate(
                [proprio[train], c2_dev_proprio], axis=0
            )
            train_labels = np.concatenate(
                [labels[train], c2_dev_labels], axis=0
            )
            train_episodes = np.concatenate(
                [episodes[train], c2_dev_episodes], axis=0
            )
            train_groups = np.concatenate(
                [groups[train], c2_dev_groups], axis=0
            )
            train_sources = np.concatenate(
                [
                    np.full(int(train.sum()), "scale", dtype=object),
                    np.full(
                        len(c2_dev_labels), "conflict", dtype=object
                    ),
                ],
                axis=0,
            )
            model = UnifiedEvidenceMoE.fit(
                train_visual,
                train_proprio,
                train_labels,
                train_episodes,
                train_groups,
                train_sources,
                seed=seed,
                minimum_override_precision=float(
                    router_config["minimum_override_precision"]
                ),
                minimum_override_groups=int(
                    router_config["minimum_override_groups"]
                ),
                route_threshold_margin=float(
                    router_config["route_threshold_margin"]
                ),
                deployment_route_threshold=(
                    float(router_config["deployment_route_threshold"])
                    if "deployment_route_threshold" in router_config
                    else None
                ),
            )
            fit_key = f"{axis}/seed{seed}"
            model_fit_audits[fit_key] = model.fit_audit
            scale_methods = _predict_methods(
                model,
                visual[main_test],
                proprio[main_test],
                labels[main_test],
                random_seed=2026072500 + 100 * axis_index + seed,
            )
            _append_prediction_rows(
                prediction_rows,
                dataset="scale",
                axis=axis,
                seed=seed,
                metadata=scale_metadata,
                methods=scale_methods,
                classes=model.classes,
            )

            if axis == config["primary_axis"]:
                c2_metadata = [
                    {
                        "sample_id": row["sample_id"],
                        "group_id": row["case_id"],
                        "scene": row["scene_cluster"],
                        "material": row["material_family"],
                        "domain": row["domain"],
                        "truth": row["attribution_category"],
                        "operator": row["target_operator"],
                        "cell": row["cell"],
                    }
                    for row in c2_test_rows
                ]
                c2_methods = _predict_methods(
                    model,
                    c2_test_visual,
                    c2_test_proprio,
                    c2_test_labels,
                    random_seed=2026072590 + seed,
                )
                _append_prediction_rows(
                    prediction_rows,
                    dataset="c2_conflict",
                    axis=axis,
                    seed=seed,
                    metadata=c2_metadata,
                    methods=c2_methods,
                    classes=model.classes,
                )
                checkpoint_path = (
                    output
                    / "checkpoints"
                    / f"seed{seed}"
                    / "unified_moe.pkl"
                )
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint_path.open("wb") as handle:
                    pickle.dump(
                        model,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                checkpoint_hashes[
                    str(checkpoint_path.relative_to(ROOT))
                ] = _sha256(checkpoint_path)

    predictions_path = output / "predictions.jsonl"
    _write_jsonl(predictions_path, prediction_rows)
    coverage_grid = [
        float(value) for value in config["selective_coverage"]
    ]
    methods = list(config["methods"])
    metrics: dict[str, Any] = {"scale": {}, "c2_conflict": {}}
    for axis in config["heldout_axes"]:
        metrics["scale"][axis] = {}
        for method in methods:
            metrics["scale"][axis][method] = {}
            for seed in seeds:
                subset = [
                    row
                    for row in prediction_rows
                    if row["dataset"] == "scale"
                    and row["axis"] == axis
                    and row["method"] == method
                    and row["seed"] == seed
                ]
                metrics["scale"][axis][method][str(seed)] = _metrics(
                    subset,
                    group_key="group_id",
                    coverage_grid=coverage_grid,
                )
    for method in methods:
        metrics["c2_conflict"][method] = {}
        for seed in seeds:
            subset = [
                row
                for row in prediction_rows
                if row["dataset"] == "c2_conflict"
                and row["method"] == method
                and row["seed"] == seed
            ]
            metrics["c2_conflict"][method][str(seed)] = _metrics(
                subset,
                group_key="group_id",
                coverage_grid=coverage_grid,
            )

    route_audit = {}
    for dataset in ("scale", "c2_conflict"):
        subset = [
            row
            for row in prediction_rows
            if row["dataset"] == dataset
            and row["axis"] == config["primary_axis"]
            and row["method"] == "learned_router"
        ]
        route_audit[dataset] = {
            "samples": len(subset),
            "expert_route_counts": dict(
                sorted(Counter(row["route"] for row in subset).items())
            ),
            "evidence_route_counts": dict(
                sorted(
                    Counter(
                        row["evidence_route"] for row in subset
                    ).items()
                )
            ),
            "expert_disagreement_rate": float(
                np.mean(
                    [row.get("expert_disagreement", False) for row in subset]
                )
            ),
        }
    c2_learned = [
        row
        for row in prediction_rows
        if row["dataset"] == "c2_conflict"
        and row["method"] == "learned_router"
    ]
    expected = {
        "T2_vision_decisive": "vision",
        "T3_proprio_decisive": "proprio",
    }
    route_correct = [
        row["evidence_route"] == expected[row["cell"]]
        for row in c2_learned
    ]
    c2_single_route_lookup = {
        (row["seed"], row["sample_id"], row["method"]): row[
            "prediction"
        ]
        for row in prediction_rows
        if row["dataset"] == "c2_conflict"
        and row["method"] in {"fixed_vision", "fixed_proprio"}
    }
    decision_critical = [
        c2_single_route_lookup[
            (row["seed"], row["sample_id"], "fixed_vision")
        ]
        != c2_single_route_lookup[
            (row["seed"], row["sample_id"], "fixed_proprio")
        ]
        for row in c2_learned
    ]
    route_audit["c2_conflict"]["evidence_route_fidelity"] = {
        "all_samples": {
            "samples": len(route_correct),
            "accuracy": float(np.mean(route_correct)),
            "interpretation": (
                "Descriptive only: when single-route predictions agree, "
                "the selected evidence route is not decision-identifiable."
            ),
        },
        "decision_critical_disagreements": {
            "definition": (
                "The vision and proprioception experts predict different "
                "causes for the same event."
            ),
            "samples": int(sum(decision_critical)),
            "fraction": float(np.mean(decision_critical)),
            "accuracy": float(
                np.mean(
                    [
                        correct
                        for correct, critical in zip(
                            route_correct,
                            decision_critical,
                            strict=True,
                        )
                        if critical
                    ]
                )
            ),
        },
        "per_cell": {
            cell: {
                "all_sample_accuracy": float(
                    np.mean(
                        [
                            row["evidence_route"] == evidence
                            for row in c2_learned
                            if row["cell"] == cell
                        ]
                    )
                ),
                "decision_critical_samples": int(
                    sum(
                        critical
                        for row, critical in zip(
                            c2_learned,
                            decision_critical,
                            strict=True,
                        )
                        if row["cell"] == cell
                    )
                ),
                "decision_critical_accuracy": float(
                    np.mean(
                        [
                            row["evidence_route"] == evidence
                            for row, critical in zip(
                                c2_learned,
                                decision_critical,
                                strict=True,
                            )
                            if row["cell"] == cell and critical
                        ]
                    )
                ),
            }
            for cell, evidence in expected.items()
        },
    }

    primary = config["primary_axis"]
    cross_battery = {}
    for method in methods:
        scale_values = [
            metrics["scale"][primary][method][str(seed)][
                "balanced_accuracy"
            ]
            for seed in seeds
        ]
        c2_values = [
            metrics["c2_conflict"][method][str(seed)][
                "balanced_accuracy"
            ]
            for seed in seeds
        ]
        cross_battery[method] = {
            "scale_scene_and_material_balanced_accuracy": {
                "minimum": float(min(scale_values)),
                "mean": float(np.mean(scale_values)),
                "maximum": float(max(scale_values)),
            },
            "c2_conflict_balanced_accuracy": {
                "minimum": float(min(c2_values)),
                "mean": float(np.mean(c2_values)),
                "maximum": float(max(c2_values)),
            },
            "equal_battery_macro_mean": float(
                0.5 * (np.mean(scale_values) + np.mean(c2_values))
            ),
            "worst_battery_mean": float(
                min(np.mean(scale_values), np.mean(c2_values))
            ),
        }

    schedule_path = ROOT / config["c4_schedule"]
    direct_report_path = ROOT / config["c4_direct_report"]
    action_report, action_rows = _action_report(
        prediction_rows,
        _jsonl(schedule_path),
        _json(direct_report_path),
        methods=methods,
        seeds=seeds,
        rule=config["closed_loop_rule"],
    )
    action_path = output / "c4_action_vectors.jsonl"
    _write_jsonl(action_path, action_rows)

    loso = None
    if not args.skip_loso:
        loso = {}
        primary = np.asarray(
            [
                row["appearance_intervention_id"] == "primary"
                for row in scale_rows
            ]
        )
        scenes = sorted({str(row["scene_family"]) for row in scale_rows})
        for scene_index, held_scene in enumerate(scenes):
            train = np.asarray(
                [
                    str(row["scene_family"]) != held_scene
                    for row in scale_rows
                ]
            )
            test = np.asarray(
                [
                    str(row["scene_family"]) == held_scene
                    for row in scale_rows
                ]
            ) & primary
            loso[held_scene] = {}
            for seed in seeds:
                model = UnifiedEvidenceMoE.fit(
                    np.concatenate(
                        [visual[train], c2_dev_visual], axis=0
                    ),
                    np.concatenate(
                        [proprio[train], c2_dev_proprio], axis=0
                    ),
                    np.concatenate(
                        [labels[train], c2_dev_labels], axis=0
                    ),
                    np.concatenate(
                        [episodes[train], c2_dev_episodes], axis=0
                    ),
                    np.concatenate(
                        [groups[train], c2_dev_groups], axis=0
                    ),
                    np.concatenate(
                        [
                            np.full(
                                int(train.sum()), "scale", dtype=object
                            ),
                            np.full(
                                len(c2_dev_labels),
                                "conflict",
                                dtype=object,
                            ),
                        ],
                        axis=0,
                    ),
                    seed=seed,
                    minimum_override_precision=float(
                        router_config["minimum_override_precision"]
                    ),
                    minimum_override_groups=int(
                        router_config["minimum_override_groups"]
                    ),
                    route_threshold_margin=float(
                        router_config["route_threshold_margin"]
                    ),
                    deployment_route_threshold=(
                        float(
                            router_config[
                                "deployment_route_threshold"
                            ]
                        )
                        if "deployment_route_threshold" in router_config
                        else None
                    ),
                )
                result = _predict_methods(
                    model,
                    visual[test],
                    proprio[test],
                    labels[test],
                    random_seed=2026072600 + 100 * scene_index + seed,
                )
                loso[held_scene][str(seed)] = {
                    method: {
                        "samples": int(test.sum()),
                        "accuracy": float(
                            np.mean(
                                result[method]["prediction"]
                                == labels[test]
                            )
                        ),
                        "balanced_accuracy": float(
                            balanced_accuracy_score(
                                labels[test],
                                result[method]["prediction"],
                            )
                        ),
                    }
                    for method in methods
                }

    report = {
        "schema_version": "kinofail.unified-moe-development-report.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "retrospective_development_complete",
        "confirmatory": False,
        "evidence_boundary": config["evidence_boundary"],
        "single_system_contract": {
            "same_architecture": True,
            "same_primary_checkpoint_for_scale_c2_and_c4": True,
            "shared_input_dimensions": {
                "visual": int(visual.shape[1]),
                "proprio": int(proprio.shape[1]),
            },
            "classes": sorted(set(labels)),
            "learned_route_outputs": [
                "expert route",
                "evidence route",
                "route probabilities",
                "attribution posterior",
            ],
            "closed_loop_action_source": (
                "the same five primary-axis unified checkpoints"
            ),
        },
        "counts": {
            "scale_samples": len(scale_rows),
            "scale_pairs": len(set(groups)),
            "scale_scene_instances": len(
                {row["scene_family"] for row in scale_rows}
            ),
            "scale_material_families": len(
                {row["material_family"] for row in scale_rows}
            ),
            "c2_development_samples": len(c2_dev_rows),
            "c2_development_cases": len(set(c2_dev_groups)),
            "c2_test_samples": len(c2_test_rows),
            "c2_test_cases": len(
                {row["case_id"] for row in c2_test_rows}
            ),
            "c4_cases": len(_jsonl(schedule_path)),
        },
        "cross_battery_summary": cross_battery,
        "metrics": metrics,
        "route_audit": route_audit,
        "c4_action_audit": action_report,
        "nine_scene_loso": loso,
        "fit_audits": model_fit_audits,
        "artifacts": {
            "predictions": str(predictions_path.relative_to(ROOT)),
            "predictions_sha256": _sha256(predictions_path),
            "c4_action_vectors": str(action_path.relative_to(ROOT)),
            "c4_action_vectors_sha256": _sha256(action_path),
            "checkpoints_sha256": checkpoint_hashes,
        },
        "source_sha256": {
            "config": _sha256(config_path),
            "scale_records": _sha256(records_path),
            "scale_features": _sha256(features_path),
            "scale_feature_manifest": _sha256(feature_manifest_path),
            "scene_registry": _sha256(registry_path),
            "c2_development_features": _sha256(
                c2_dev_dir / "features.npz"
            ),
            "c2_test_features": _sha256(c2_test_dir / "features.npz"),
            "c4_schedule": _sha256(schedule_path),
            "c4_direct_report": _sha256(direct_report_path),
            "model_source": _sha256(
                ROOT / "kino_vla/eval/unified_moe.py"
            ),
            "evaluator": _sha256(Path(__file__).resolve()),
        },
    }
    report_path = output / "report.json"
    _write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "cross_battery_summary": cross_battery,
                "route_audit": route_audit,
                "c4_unique_action_vectors": action_report[
                    "unique_action_vectors"
                ],
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
