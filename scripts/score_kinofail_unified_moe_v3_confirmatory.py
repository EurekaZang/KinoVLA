#!/usr/bin/env python3
"""One-shot scorer for the frozen Kino-Fail unified-MoE extension.

The scorer deliberately accepts two sealed inputs:

* a blind JSONL prediction file, which must not contain ground truth; and
* a separate JSONL truth key.

The JSON protocol binds both files by SHA-256, fixes the five checkpoint IDs,
the evaluated methods, expected classes, planned group counts, and the
20,000-replicate bootstrap seed.  The scorer never fits a model or selects an
operating point.  An existing output is never overwritten.

Blind prediction row (one per sample, method, and checkpoint)::

    {
      "sample_id": "...", "method": "learned_router",
      "checkpoint_id": "seed0", "prediction": "adhesion",
      "probabilities": {"adhesion": 0.9, ... all 11 classes ...},
      "evidence_route": "proprio"
    }

Truth-key row (one per planned sample)::

    {
      "sample_id": "...", "group_id": "...", "dataset": "scale",
      "scene": "...", "material": "...", "cluster_material": "...",
      "truth": "adhesion", "valid": true, "operator": "O4_tether",
      "cell": "T2_vision_decisive"
    }

``cluster_material`` is the preregistered physical/material-family cluster.
It must be constant within a matched group.  Appearance-swap samples may have
different ``material`` values while remaining in one physical case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]

ONTOLOGY = (
    "adhesion",
    "compliant_terrain",
    "effort_decay",
    "external_push",
    "high_centering",
    "invisible_obstacle",
    "low_friction",
    "nominal",
    "obs_bias",
    "overload",
    "region_collapse",
)
DATASETS = ("scale", "conflict")
PRIMARY_METHOD = "learned_router"
PRIMARY_COMPARATOR = "late_average"
REQUIRED_ROUTE_METHODS = ("fixed_vision", "fixed_proprio")
CHECKPOINT_IDS = ("seed0", "seed1", "seed2", "seed3", "seed4")
EXPECTED_ROUTE = {
    "T2_vision_decisive": "vision",
    "T3_proprio_decisive": "proprio",
}
BOOTSTRAP_REPLICATES = 20_000
NI_MARGIN = -0.01
WORST_SUPERIORITY_MARGIN = 0.005
ROUTE_FIDELITY_MARGIN = 0.85
MAX_ATTRITION = 0.05


class ConfirmatoryInputError(ValueError):
    """A sealed-input or structural violation."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfirmatoryInputError(f"expected a JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ConfirmatoryInputError(f"expected an object at {path}:{line_number}")
        values.append(value)
    if not values:
        raise ConfirmatoryInputError(f"empty JSONL input: {path}")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve_frozen_path(raw: str, protocol_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    rooted = ROOT / path
    if rooted.exists():
        return rooted
    return protocol_path.parent / path


def _validate_protocol(
    protocol: dict[str, Any],
    protocol_path: Path,
    predictions_path: Path,
    truth_path: Path,
) -> None:
    if tuple(protocol.get("ontology", ())) != ONTOLOGY:
        raise ConfirmatoryInputError(
            "protocol ontology does not equal the frozen 11-class ontology"
        )
    bootstrap = protocol.get("bootstrap", {})
    if int(bootstrap.get("replicates", -1)) != BOOTSTRAP_REPLICATES:
        raise ConfirmatoryInputError("bootstrap replicates must equal 20000")
    if not isinstance(bootstrap.get("seed"), int):
        raise ConfirmatoryInputError("bootstrap seed must be a frozen integer")

    checkpoints = [str(value) for value in protocol.get("checkpoint_ids", [])]
    if tuple(checkpoints) != CHECKPOINT_IDS:
        raise ConfirmatoryInputError("checkpoint IDs must be frozen as seed0 through seed4")
    methods = [str(value) for value in protocol.get("methods", [])]
    required = {
        PRIMARY_METHOD,
        PRIMARY_COMPARATOR,
        *REQUIRED_ROUTE_METHODS,
    }
    if len(methods) != len(set(methods)) or not required.issubset(methods):
        raise ConfirmatoryInputError(
            "methods must be unique and include learned, late, vision, proprio"
        )
    secondary = [str(value) for value in protocol.get("secondary_baselines", [])]
    if (
        len(secondary) != len(set(secondary))
        or any(value not in methods for value in secondary)
        or PRIMARY_METHOD in secondary
        or PRIMARY_COMPARATOR in secondary
    ):
        raise ConfirmatoryInputError("invalid secondary baseline registry")

    expected = protocol.get("expected_classes", {})
    if set(expected) != set(DATASETS):
        raise ConfirmatoryInputError("expected_classes must define scale and conflict")
    for dataset in DATASETS:
        classes = [str(value) for value in expected[dataset]]
        if len(classes) != len(set(classes)) or not set(classes).issubset(ONTOLOGY):
            raise ConfirmatoryInputError(f"invalid expected class list for {dataset}")
    if tuple(expected["scale"]) != ONTOLOGY:
        raise ConfirmatoryInputError("scale must contain the complete frozen 11-class ontology")

    planned = protocol.get("planned_groups", {})
    if set(planned) != set(DATASETS) or any(
        not isinstance(planned[name], int) or int(planned[name]) <= 0 for name in DATASETS
    ):
        raise ConfirmatoryInputError(
            "positive planned group counts are required for both batteries"
        )
    critical = protocol.get("decision_critical_sample_ids")
    if (
        not isinstance(critical, list)
        or not critical
        or len(critical) != len(set(str(value) for value in critical))
    ):
        raise ConfirmatoryInputError(
            "a nonempty unique decision_critical_sample_ids list is required"
        )

    sealed = protocol.get("input_sha256", {})
    expected_prediction_hash = str(sealed.get("blind_predictions", ""))
    expected_truth_hash = str(sealed.get("truth_key", ""))
    if not expected_prediction_hash or not expected_truth_hash:
        raise ConfirmatoryInputError("both sealed input hashes are required")
    actual_prediction_hash = _sha256(predictions_path)
    actual_truth_hash = _sha256(truth_path)
    if actual_prediction_hash != expected_prediction_hash:
        raise ConfirmatoryInputError("blind-prediction SHA-256 does not match the seal")
    if actual_truth_hash != expected_truth_hash:
        raise ConfirmatoryInputError("truth-key SHA-256 does not match the seal")

    for raw_path, expected_hash in protocol.get("frozen_files_sha256", {}).items():
        path = _resolve_frozen_path(str(raw_path), protocol_path)
        if not path.is_file():
            raise ConfirmatoryInputError(f"frozen artifact is missing: {path}")
        if _sha256(path) != str(expected_hash):
            raise ConfirmatoryInputError(f"frozen artifact hash mismatch: {path}")


def _truth_registry(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    by_sample: dict[str, dict[str, Any]] = {}
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        required = (
            "sample_id",
            "group_id",
            "dataset",
            "scene",
            "material",
            "truth",
        )
        missing = [name for name in required if name not in row]
        if missing:
            raise ConfirmatoryInputError(f"truth row is missing fields {missing}")
        sample_id = str(row["sample_id"])
        group_id = str(row["group_id"])
        dataset = str(row["dataset"])
        if sample_id in by_sample:
            raise ConfirmatoryInputError(f"duplicate sample_id: {sample_id}")
        if dataset not in DATASETS:
            raise ConfirmatoryInputError(f"unknown dataset: {dataset}")
        truth = str(row["truth"])
        if truth not in ONTOLOGY:
            raise ConfirmatoryInputError(f"truth is outside ontology: {truth}")
        row["sample_id"] = sample_id
        row["group_id"] = group_id
        row["dataset"] = dataset
        row["scene"] = str(row["scene"])
        row["material"] = str(row["material"])
        row["cluster_material"] = str(row.get("cluster_material", row["material"]))
        row["truth"] = truth
        row["valid"] = bool(row.get("valid", True))
        by_sample[sample_id] = row
        by_group[group_id].append(row)

    group_counts = {name: 0 for name in DATASETS}
    invalid_counts = {name: 0 for name in DATASETS}
    for group_id, group_rows in by_group.items():
        datasets = {row["dataset"] for row in group_rows}
        scenes = {row["scene"] for row in group_rows}
        materials = {row["cluster_material"] for row in group_rows}
        validity = {row["valid"] for row in group_rows}
        if len(datasets) != 1 or len(scenes) != 1 or len(materials) != 1:
            raise ConfirmatoryInputError(f"group metadata is not constant: {group_id}")
        if len(validity) != 1:
            raise ConfirmatoryInputError(f"partial group attrition is forbidden: {group_id}")
        dataset = next(iter(datasets))
        group_counts[dataset] += 1
        if not next(iter(validity)):
            invalid_counts[dataset] += 1

    planned = {name: int(protocol["planned_groups"][name]) for name in DATASETS}
    if group_counts != planned:
        raise ConfirmatoryInputError(f"truth-key group counts {group_counts} != planned {planned}")

    expected = protocol["expected_classes"]
    for dataset in DATASETS:
        observed = {
            row["truth"] for row in by_sample.values() if row["dataset"] == dataset and row["valid"]
        }
        if observed != set(expected[dataset]):
            raise ConfirmatoryInputError(
                f"{dataset} class support {sorted(observed)} does not equal "
                f"the frozen class list {sorted(expected[dataset])}"
            )

    attrition = {
        dataset: {
            "planned_groups": planned[dataset],
            "invalid_groups": invalid_counts[dataset],
            "valid_groups": planned[dataset] - invalid_counts[dataset],
            "rate": float(invalid_counts[dataset] / planned[dataset]),
            "maximum_allowed": MAX_ATTRITION,
            "passed": bool(invalid_counts[dataset] / planned[dataset] <= MAX_ATTRITION),
        }
        for dataset in DATASETS
    }
    attrition["overall"] = {
        "planned_groups": int(sum(planned.values())),
        "invalid_groups": int(sum(invalid_counts.values())),
        "valid_groups": int(sum(planned.values()) - sum(invalid_counts.values())),
        "rate": float(sum(invalid_counts.values()) / sum(planned.values())),
        "maximum_allowed": MAX_ATTRITION,
        "passed": bool(sum(invalid_counts.values()) / sum(planned.values()) <= MAX_ATTRITION),
    }
    return by_sample, by_group, attrition


def _prediction_registry(
    rows: list[dict[str, Any]],
    truth_by_sample: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    methods = [str(value) for value in protocol["methods"]]
    checkpoints = [str(value) for value in protocol["checkpoint_ids"]]
    valid_samples = {sample_id for sample_id, row in truth_by_sample.items() if row["valid"]}
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    forbidden = {"truth", "label", "attribution_category", "expected_route"}
    for source in rows:
        if forbidden.intersection(source):
            raise ConfirmatoryInputError("blind predictions contain a ground-truth field")
        required = (
            "sample_id",
            "method",
            "checkpoint_id",
            "prediction",
            "probabilities",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise ConfirmatoryInputError(f"prediction row is missing fields {missing}")
        sample_id = str(source["sample_id"])
        method = str(source["method"])
        checkpoint = str(source["checkpoint_id"])
        prediction = str(source["prediction"])
        if sample_id not in valid_samples:
            raise ConfirmatoryInputError(
                f"prediction refers to an invalid or unknown sample: {sample_id}"
            )
        if method not in methods or checkpoint not in checkpoints:
            raise ConfirmatoryInputError(f"unregistered method/checkpoint: {method}/{checkpoint}")
        if prediction not in ONTOLOGY:
            raise ConfirmatoryInputError(f"prediction is outside ontology: {prediction}")
        probabilities = source["probabilities"]
        if not isinstance(probabilities, dict) or set(probabilities) != set(ONTOLOGY):
            raise ConfirmatoryInputError(
                "probability map must contain exactly the 11 frozen classes"
            )
        vector = np.asarray(
            [float(probabilities[label]) for label in ONTOLOGY],
            dtype=np.float64,
        )
        if (
            not np.isfinite(vector).all()
            or (vector < 0.0).any()
            or abs(float(vector.sum()) - 1.0) > 1e-6
        ):
            raise ConfirmatoryInputError("invalid probability vector")
        if ONTOLOGY[int(vector.argmax())] != prediction:
            raise ConfirmatoryInputError("prediction does not equal argmax(probabilities)")
        key = (sample_id, method, checkpoint)
        if key in lookup:
            raise ConfirmatoryInputError(f"duplicate prediction row: {key}")
        row = dict(source)
        row["sample_id"] = sample_id
        row["method"] = method
        row["checkpoint_id"] = checkpoint
        row["prediction"] = prediction
        row["_probability_vector"] = vector
        if method == PRIMARY_METHOD:
            route = str(row.get("evidence_route", ""))
            if route not in {
                "vision",
                "proprio",
                "cross_modal_interaction",
                "consensus",
            }:
                raise ConfirmatoryInputError("learned-router row has no valid evidence route")
            row["evidence_route"] = route
        lookup[key] = row

    expected_count = len(valid_samples) * len(methods) * len(checkpoints)
    if len(lookup) != expected_count:
        missing = []
        for sample_id in sorted(valid_samples):
            for method in methods:
                for checkpoint in checkpoints:
                    if (sample_id, method, checkpoint) not in lookup:
                        missing.append((sample_id, method, checkpoint))
                        if len(missing) == 5:
                            break
                if len(missing) == 5:
                    break
            if len(missing) == 5:
                break
        raise ConfirmatoryInputError(
            f"prediction coverage is incomplete; first missing keys: {missing}"
        )
    return lookup


def _stable_tie_key(group_id: str) -> str:
    return hashlib.sha256(group_id.encode("utf-8")).hexdigest()


def _prepare_statistics(
    truth_by_sample: dict[str, dict[str, Any]],
    truth_by_group: dict[str, list[dict[str, Any]]],
    predictions: dict[tuple[str, str, str], dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    methods = [str(value) for value in protocol["methods"]]
    checkpoints = [str(value) for value in protocol["checkpoint_ids"]]
    method_index = {name: index for index, name in enumerate(methods)}
    checkpoint_index = {name: index for index, name in enumerate(checkpoints)}
    class_index = {name: index for index, name in enumerate(ONTOLOGY)}
    dataset_index = {name: index for index, name in enumerate(DATASETS)}

    valid_groups = {group_id: rows for group_id, rows in truth_by_group.items() if rows[0]["valid"]}
    cells = sorted(
        {(rows[0]["scene"], rows[0]["cluster_material"]) for rows in valid_groups.values()}
    )
    cell_index = {value: index for index, value in enumerate(cells)}
    scenes = sorted({value[0] for value in cells})
    materials = sorted({value[1] for value in cells})
    scene_index = {name: index for index, name in enumerate(scenes)}
    material_index = {name: index for index, name in enumerate(materials)}
    cell_scene = np.asarray([scene_index[scene] for scene, _ in cells], dtype=np.int64)
    cell_material = np.asarray([material_index[material] for _, material in cells], dtype=np.int64)

    total = np.zeros((len(cells), len(DATASETS), len(ONTOLOGY)), dtype=np.float64)
    correct = np.zeros(
        (
            len(cells),
            len(DATASETS),
            len(methods),
            len(checkpoints),
            len(ONTOLOGY),
        ),
        dtype=np.float64,
    )
    group_statistics: dict[str, dict[str, Any]] = {}

    for group_id, group_rows in valid_groups.items():
        first = group_rows[0]
        dataset = dataset_index[first["dataset"]]
        cell = cell_index[(first["scene"], first["cluster_material"])]
        group_correct = np.ones((len(methods), len(checkpoints)), dtype=np.float64)
        group_min_confidence = np.ones((len(methods), len(checkpoints)), dtype=np.float64)
        for truth_row in group_rows:
            sample_id = truth_row["sample_id"]
            truth = truth_row["truth"]
            target = class_index[truth]
            total[cell, dataset, target] += 1.0
            for method in methods:
                m = method_index[method]
                for checkpoint in checkpoints:
                    s = checkpoint_index[checkpoint]
                    prediction = predictions[(sample_id, method, checkpoint)]
                    is_correct = prediction["prediction"] == truth
                    correct[cell, dataset, m, s, target] += float(is_correct)
                    group_correct[m, s] *= float(is_correct)
                    confidence = float(prediction["_probability_vector"].max())
                    group_min_confidence[m, s] = min(group_min_confidence[m, s], confidence)
        group_statistics[group_id] = {
            "dataset": first["dataset"],
            "cell": cell,
            "scene": first["scene"],
            "material": first["cluster_material"],
            "error": 1.0 - group_correct.mean(axis=1),
            "confidence": group_min_confidence.mean(axis=1),
        }

    selective_point: dict[str, dict[str, Any]] = {method: {} for method in methods}
    selected_sum = np.zeros((len(cells), len(DATASETS), len(methods)), dtype=np.float64)
    selected_count = np.zeros_like(selected_sum)
    aurc_sum = np.zeros_like(selected_sum)
    aurc_count = np.zeros_like(selected_sum)
    for method in methods:
        m = method_index[method]
        for dataset in DATASETS:
            d = dataset_index[dataset]
            candidates = [
                (group_id, value)
                for group_id, value in group_statistics.items()
                if value["dataset"] == dataset
            ]
            ordered = sorted(
                candidates,
                key=lambda item: (
                    -float(item[1]["confidence"][m]),
                    _stable_tie_key(item[0]),
                ),
            )
            n_groups = len(ordered)
            selected_n = max(1, int(math.ceil(0.75 * n_groups)))
            prefix_errors = np.cumsum([float(value["error"][m]) for _, value in ordered])
            prefix_risk = prefix_errors / np.arange(1, n_groups + 1)
            harmonic_tail = np.cumsum(1.0 / np.arange(n_groups, 0, -1, dtype=np.float64))[::-1]
            for rank, (_, value) in enumerate(ordered):
                cell = int(value["cell"])
                aurc_sum[cell, d, m] += float(value["error"][m]) * harmonic_tail[rank]
                aurc_count[cell, d, m] += 1.0
            selected = ordered[:selected_n]
            for _, value in selected:
                cell = int(value["cell"])
                selected_sum[cell, d, m] += float(value["error"][m])
                selected_count[cell, d, m] += 1.0
            selective_point[method][dataset] = {
                "groups": n_groups,
                "selected_groups": selected_n,
                "coverage": float(selected_n / n_groups),
                "risk_at_75_percent_coverage": float(prefix_risk[selected_n - 1]),
                "aurc_all_prefixes": float(prefix_risk.mean()),
                "confidence_function": (
                    "mean over five checkpoints of the minimum member "
                    "maximum-posterior within each physical group"
                ),
                "group_error_function": (
                    "one minus the mean strict-all-members correctness over "
                    "the five fixed checkpoints"
                ),
            }

    route_sum = np.zeros((len(cells), len(EXPECTED_ROUTE)), dtype=np.float64)
    route_count = np.zeros_like(route_sum)
    route_case_counts: dict[str, dict[str, int]] = {}
    preregistered_critical = {str(value) for value in protocol["decision_critical_sample_ids"]}
    valid_sample_ids = {row["sample_id"] for rows in valid_groups.values() for row in rows}
    unknown_critical = preregistered_critical - valid_sample_ids
    if unknown_critical:
        raise ConfirmatoryInputError(
            "decision-critical registry contains invalid or unknown samples: "
            f"{sorted(unknown_critical)[:5]}"
        )
    for route_index, (cell_name, expected_route) in enumerate(EXPECTED_ROUTE.items()):
        total_cases = 0
        critical_cases = 0
        registered_samples = 0
        for _group_id, group_rows in valid_groups.items():
            matching = [
                row
                for row in group_rows
                if str(row.get("cell", "")) == cell_name
                and row["sample_id"] in preregistered_critical
            ]
            if not matching:
                continue
            total_cases += 1
            registered_samples += len(matching)
            decisions: list[float] = []
            for truth_row in matching:
                sample_id = truth_row["sample_id"]
                for checkpoint in checkpoints:
                    vision = predictions[(sample_id, "fixed_vision", checkpoint)]["prediction"]
                    proprio = predictions[(sample_id, "fixed_proprio", checkpoint)]["prediction"]
                    if vision == proprio:
                        continue
                    route = predictions[(sample_id, PRIMARY_METHOD, checkpoint)]["evidence_route"]
                    decisions.append(float(route == expected_route))
            if not decisions:
                continue
            critical_cases += 1
            first = group_rows[0]
            cell = cell_index[(first["scene"], first["cluster_material"])]
            route_sum[cell, route_index] += float(np.mean(decisions))
            route_count[cell, route_index] += 1.0
        if total_cases == 0 or critical_cases == 0:
            raise ConfirmatoryInputError(f"no decision-critical cases for {cell_name}")
        route_case_counts[cell_name] = {
            "preregistered_samples": registered_samples,
            "preregistered_matched_cases": total_cases,
            "decision_critical_cases": critical_cases,
        }

    return {
        "methods": methods,
        "checkpoints": checkpoints,
        "method_index": method_index,
        "dataset_index": dataset_index,
        "expected_class_indices": {
            dataset: np.asarray(
                [class_index[label] for label in protocol["expected_classes"][dataset]],
                dtype=np.int64,
            )
            for dataset in DATASETS
        },
        "cells": cells,
        "scenes": scenes,
        "materials": materials,
        "cell_scene": cell_scene,
        "cell_material": cell_material,
        "total": total,
        "correct": correct,
        "selective_point": selective_point,
        "selected_sum": selected_sum,
        "selected_count": selected_count,
        "aurc_sum": aurc_sum,
        "aurc_count": aurc_count,
        "route_sum": route_sum,
        "route_count": route_count,
        "route_case_counts": route_case_counts,
        "valid_group_counts": {
            dataset: sum(value["dataset"] == dataset for value in group_statistics.values())
            for dataset in DATASETS
        },
    }


def _balanced_accuracy(
    weighted_total: np.ndarray,
    weighted_correct: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    denominator = weighted_total[..., indices]
    if (denominator <= 0.0).any():
        raise ConfirmatoryInputError("a bootstrap replicate lost a preregistered class")
    numerator = weighted_correct[..., indices]
    return (numerator / denominator[..., None, :]).mean(axis=(-1, -2))


def _bootstrap(
    prepared: dict[str, Any],
    *,
    seed: int,
    replicates: int,
) -> dict[str, np.ndarray]:
    """Paired scene×material pigeonhole multiplier bootstrap.

    Independent Exp(1) multipliers are drawn for scene clusters and material
    clusters; their product weights each preregistered scene-material cell.
    Matched groups and all five checkpoint predictions remain together.
    Results are accumulated in bounded batches, so runtime memory does not grow
    with the number of physical samples.
    """

    total = prepared["total"]
    correct = prepared["correct"]
    selected_sum = prepared["selected_sum"]
    selected_count = prepared["selected_count"]
    aurc_sum = prepared["aurc_sum"]
    aurc_count = prepared["aurc_count"]
    route_sum = prepared["route_sum"]
    route_count = prepared["route_count"]
    cell_scene = prepared["cell_scene"]
    cell_material = prepared["cell_material"]
    methods = prepared["methods"]
    dataset_index = prepared["dataset_index"]

    bacc = np.empty((replicates, len(DATASETS), len(methods)), dtype=np.float64)
    risk75 = np.empty_like(bacc)
    aurc = np.empty_like(bacc)
    route = np.empty((replicates, len(EXPECTED_ROUTE)), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    batch_size = 128
    position = 0
    while position < replicates:
        size = min(batch_size, replicates - position)
        scene_weight = rng.exponential(1.0, size=(size, len(prepared["scenes"])))
        material_weight = rng.exponential(1.0, size=(size, len(prepared["materials"])))
        weight = scene_weight[:, cell_scene] * material_weight[:, cell_material]
        weighted_total = np.einsum("bc,cdk->bdk", weight, total, optimize=True)
        weighted_correct = np.einsum("bc,cdmsk->bdmsk", weight, correct, optimize=True)
        for dataset in DATASETS:
            d = dataset_index[dataset]
            indices = prepared["expected_class_indices"][dataset]
            denominator = weighted_total[:, d, indices]
            if (denominator <= 0.0).any():
                raise ConfirmatoryInputError("crossed bootstrap lost class support")
            numerator = np.take(weighted_correct[:, d], indices, axis=-1)
            recall = numerator / denominator[:, None, None, :]
            bacc[position : position + size, d, :] = recall.mean(axis=(2, 3))

        risk_numerator = np.einsum("bc,cdm->bdm", weight, selected_sum, optimize=True)
        risk_denominator = np.einsum("bc,cdm->bdm", weight, selected_count, optimize=True)
        if (risk_denominator <= 0.0).any():
            raise ConfirmatoryInputError("selective-risk bootstrap has empty coverage")
        risk75[position : position + size] = risk_numerator / risk_denominator
        aurc_numerator = np.einsum("bc,cdm->bdm", weight, aurc_sum, optimize=True)
        aurc_denominator = np.einsum("bc,cdm->bdm", weight, aurc_count, optimize=True)
        if (aurc_denominator <= 0.0).any():
            raise ConfirmatoryInputError("AURC bootstrap has no groups")
        aurc[position : position + size] = aurc_numerator / aurc_denominator

        route_numerator = np.einsum("bc,cr->br", weight, route_sum, optimize=True)
        route_denominator = np.einsum("bc,cr->br", weight, route_count, optimize=True)
        if (route_denominator <= 0.0).any():
            raise ConfirmatoryInputError("route bootstrap has no decision-critical case")
        route[position : position + size] = route_numerator / route_denominator
        position += size
    return {
        "bacc": bacc,
        "risk75": risk75,
        "aurc": aurc,
        "route": route,
    }


def _point_bacc(prepared: dict[str, Any]) -> np.ndarray:
    total = prepared["total"].sum(axis=0)
    correct = prepared["correct"].sum(axis=0)
    result = np.empty((len(DATASETS), len(prepared["methods"])), dtype=np.float64)
    for dataset in DATASETS:
        d = prepared["dataset_index"][dataset]
        indices = prepared["expected_class_indices"][dataset]
        denominator = total[d, indices]
        if (denominator <= 0.0).any():
            raise ConfirmatoryInputError(f"{dataset} is missing a frozen class")
        numerator = np.take(correct[d], indices, axis=-1)
        recall = numerator / denominator[None, None, :]
        result[d] = recall.mean(axis=(1, 2))
    return result


def _interval(values: np.ndarray) -> dict[str, float]:
    return {
        "one_sided_97_5_percent_lower_bound": float(np.quantile(values, 0.025)),
        "two_sided_ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def _holm(comparisons: list[dict[str, Any]], alpha: float = 0.05) -> list[dict[str, Any]]:
    ordered = sorted(comparisons, key=lambda value: value["raw_p_value"])
    count = len(ordered)
    running_adjusted = 0.0
    continue_rejecting = True
    for rank, value in enumerate(ordered):
        multiplier = count - rank
        running_adjusted = max(
            running_adjusted,
            min(1.0, multiplier * float(value["raw_p_value"])),
        )
        value["holm_adjusted_p_value"] = running_adjusted
        threshold = alpha / multiplier
        raw_reject = float(value["raw_p_value"]) <= threshold
        value["holm_reject"] = bool(continue_rejecting and raw_reject)
        if not raw_reject:
            continue_rejecting = False
    return sorted(ordered, key=lambda value: value["method"])


def _analysis_report(
    protocol: dict[str, Any],
    prepared: dict[str, Any],
    bootstrap: dict[str, np.ndarray],
    attrition: dict[str, Any],
) -> dict[str, Any]:
    methods = prepared["methods"]
    m = prepared["method_index"]
    d = prepared["dataset_index"]
    point = _point_bacc(prepared)
    boot = bootstrap["bacc"]
    learned = m[PRIMARY_METHOD]
    late = m[PRIMARY_COMPARATOR]

    delta_point = {
        dataset: float(point[d[dataset], learned] - point[d[dataset], late]) for dataset in DATASETS
    }
    delta_boot = {
        dataset: (boot[:, d[dataset], learned] - boot[:, d[dataset], late]) for dataset in DATASETS
    }
    worst_point = float(min(point[:, learned]) - min(point[:, late]))
    worst_boot = np.min(boot[:, :, learned], axis=1) - np.min(boot[:, :, late], axis=1)
    macro_point = float(point[:, learned].mean() - point[:, late].mean())
    macro_boot = boot[:, :, learned].mean(axis=1) - boot[:, :, late].mean(axis=1)

    ni_results = {
        dataset: {
            "difference_ours_minus_late": delta_point[dataset],
            **_interval(delta_boot[dataset]),
            "margin": NI_MARGIN,
            "raw_pass": bool(np.quantile(delta_boot[dataset], 0.025) > NI_MARGIN),
        }
        for dataset in DATASETS
    }
    gate1 = all(value["raw_pass"] for value in ni_results.values())
    gate2_raw = bool(np.quantile(worst_boot, 0.025) > WORST_SUPERIORITY_MARGIN)
    gate3_raw = bool(np.quantile(macro_boot, 0.025) > 0.0)
    primary_gates = {
        "gate_1_battery_noninferiority": {
            "batteries": ni_results,
            "sequential_pass": gate1,
        },
        "gate_2_worst_battery_superiority": {
            "difference_ours_minus_late": worst_point,
            **_interval(worst_boot),
            "margin": WORST_SUPERIORITY_MARGIN,
            "eligible_after_previous_gate": gate1,
            "raw_pass": gate2_raw,
            "sequential_pass": bool(gate1 and gate2_raw),
        },
        "gate_3_equal_battery_macro_superiority": {
            "difference_ours_minus_late": macro_point,
            **_interval(macro_boot),
            "margin": 0.0,
            "eligible_after_previous_gates": bool(gate1 and gate2_raw),
            "raw_pass": gate3_raw,
            "sequential_pass": bool(gate1 and gate2_raw and gate3_raw),
        },
    }

    route_point = np.divide(
        prepared["route_sum"].sum(axis=0),
        prepared["route_count"].sum(axis=0),
    )
    route_gates = {}
    for index, cell_name in enumerate(EXPECTED_ROUTE):
        values = bootstrap["route"][:, index]
        lower = float(np.quantile(values, 0.025))
        route_gates[cell_name] = {
            **prepared["route_case_counts"][cell_name],
            "case_weighted_fidelity": float(route_point[index]),
            **_interval(values),
            "margin": ROUTE_FIDELITY_MARGIN,
            "passed": bool(lower > ROUTE_FIDELITY_MARGIN),
        }

    selective = prepared["selective_point"]
    risk_boot = bootstrap["risk75"]
    aurc_boot = bootstrap["aurc"]
    for method in methods:
        for dataset in DATASETS:
            selective[method][dataset]["risk_at_75_percent_ci95"] = [
                float(np.quantile(risk_boot[:, d[dataset], m[method]], 0.025)),
                float(np.quantile(risk_boot[:, d[dataset], m[method]], 0.975)),
            ]
            selective[method][dataset]["aurc_ci95"] = [
                float(np.quantile(aurc_boot[:, d[dataset], m[method]], 0.025)),
                float(np.quantile(aurc_boot[:, d[dataset], m[method]], 0.975)),
            ]
        selective[method]["equal_battery_mean"] = {
            "risk_at_75_percent_coverage": float(
                np.mean(
                    [
                        selective[method][dataset]["risk_at_75_percent_coverage"]
                        for dataset in DATASETS
                    ]
                )
            ),
            "aurc_all_prefixes": float(
                np.mean([selective[method][dataset]["aurc_all_prefixes"] for dataset in DATASETS])
            ),
        }
    risk_difference = risk_boot[:, :, learned].mean(axis=1) - risk_boot[:, :, late].mean(axis=1)

    secondary = []
    for baseline in protocol.get("secondary_baselines", []):
        baseline_index = m[str(baseline)]
        values = np.min(boot[:, :, learned], axis=1) - np.min(boot[:, :, baseline_index], axis=1)
        point_difference = float(min(point[:, learned]) - min(point[:, baseline_index]))
        standard_error = float(np.std(values, ddof=1))
        raw_p = (
            0.0
            if standard_error == 0.0 and point_difference > 0.0
            else (
                1.0
                if standard_error == 0.0
                else float(norm.cdf(-point_difference / standard_error))
            )
        )
        secondary.append(
            {
                "method": str(baseline),
                "endpoint": "worst-battery balanced-accuracy difference",
                "difference_ours_minus_baseline": point_difference,
                **_interval(values),
                "raw_p_value": raw_p,
            }
        )

    bacc_report = {
        method: {
            dataset: {
                "five_checkpoint_mean": float(point[d[dataset], m[method]]),
                "crossed_cluster_ci95": [
                    float(np.quantile(boot[:, d[dataset], m[method]], 0.025)),
                    float(np.quantile(boot[:, d[dataset], m[method]], 0.975)),
                ],
            }
            for dataset in DATASETS
        }
        for method in methods
    }

    attrition_passed = all(attrition[name]["passed"] for name in (*DATASETS, "overall"))
    primary_passed = bool(gate1 and gate2_raw and gate3_raw)
    route_passed = all(value["passed"] for value in route_gates.values())
    all_passed = bool(attrition_passed and primary_passed and route_passed)
    return {
        "confirmatory_protocol_valid": bool(attrition_passed),
        "confirmatory": bool(attrition_passed),
        "status": (
            "confirmatory_passed_all_preregistered_gates"
            if all_passed
            else (
                "confirmatory_completed_statistical_gate_failed"
                if attrition_passed
                else "fail_closed_attrition_exceeded"
            )
        ),
        "passed_all_preregistered_gates": all_passed,
        "primary_cross_battery_gates": primary_gates,
        "route_fidelity_gates": route_gates,
        "balanced_accuracy": bacc_report,
        "selective_risk": {
            "methods": selective,
            "equal_battery_75_percent_risk_difference_ours_minus_late": {
                "point": float(
                    selective[PRIMARY_METHOD]["equal_battery_mean"]["risk_at_75_percent_coverage"]
                    - selective[PRIMARY_COMPARATOR]["equal_battery_mean"][
                        "risk_at_75_percent_coverage"
                    ]
                ),
                "ci95": [
                    float(np.quantile(risk_difference, 0.025)),
                    float(np.quantile(risk_difference, 0.975)),
                ],
            },
        },
        "secondary_baseline_holm": _holm(secondary),
        "attrition": attrition,
        "statistical_contract": {
            "bootstrap": (
                "paired scene-by-material crossed-cluster pigeonhole multiplier bootstrap"
            ),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": int(protocol["bootstrap"]["seed"]),
            "training_checkpoint_treatment": (
                "five fixed checkpoints are averaged within each physical "
                "case and are not independent statistical units"
            ),
            "physical_units": {
                "scale": "counterfactual group",
                "conflict": "matched physical case",
                "upper_clusters": ["scene instance", "material family"],
            },
            "one_sided_confidence_level": 0.975,
            "ni_margin": NI_MARGIN,
            "worst_battery_superiority_margin": (WORST_SUPERIORITY_MARGIN),
            "route_fidelity_margin": ROUTE_FIDELITY_MARGIN,
            "holm_familywise_alpha": 0.05,
        },
    }


def _failure_report(
    protocol_path: Path,
    predictions_path: Path,
    truth_path: Path,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": "kinofail.unified-moe-confirmatory-score.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "fail_closed_integrity_or_structure_violation",
        "confirmatory": False,
        "confirmatory_protocol_valid": False,
        "passed_all_preregistered_gates": False,
        "failure_reason": reason,
        "available_hashes": {
            "protocol": (_sha256(protocol_path) if protocol_path.is_file() else None),
            "blind_predictions": (
                _sha256(predictions_path) if predictions_path.is_file() else None
            ),
            "truth_key": (_sha256(truth_path) if truth_path.is_file() else None),
        },
    }


def score_confirmatory(
    protocol_path: Path,
    predictions_path: Path,
    truth_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    predictions_path = predictions_path.resolve()
    truth_path = truth_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)

    try:
        protocol = _json(protocol_path)
        _validate_protocol(protocol, protocol_path, predictions_path, truth_path)
        truth_rows = _jsonl(truth_path)
        prediction_rows = _jsonl(predictions_path)
        truth_by_sample, truth_by_group, attrition = _truth_registry(truth_rows, protocol)
        prediction_lookup = _prediction_registry(prediction_rows, truth_by_sample, protocol)
        prepared = _prepare_statistics(
            truth_by_sample,
            truth_by_group,
            prediction_lookup,
            protocol,
        )
        bootstrap = _bootstrap(
            prepared,
            seed=int(protocol["bootstrap"]["seed"]),
            replicates=BOOTSTRAP_REPLICATES,
        )
        report = _analysis_report(protocol, prepared, bootstrap, attrition)
        report.update(
            {
                "schema_version": ("kinofail.unified-moe-confirmatory-score.v1"),
                "created_utc": datetime.now(UTC).isoformat(),
                "protocol_id": str(protocol.get("protocol_id", "")),
                "counts": {
                    "valid_groups": prepared["valid_group_counts"],
                    "scene_clusters": len(prepared["scenes"]),
                    "material_clusters": len(prepared["materials"]),
                    "fixed_checkpoints": len(prepared["checkpoints"]),
                },
                "source_sha256": {
                    "protocol": _sha256(protocol_path),
                    "blind_predictions": _sha256(predictions_path),
                    "truth_key": _sha256(truth_path),
                    "scorer": _sha256(Path(__file__).resolve()),
                },
            }
        )
    except (ConfirmatoryInputError, FileNotFoundError, json.JSONDecodeError) as error:
        report = _failure_report(protocol_path, predictions_path, truth_path, str(error))
    _write_json(output_path, report)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score the sealed unified-MoE confirmatory extension once."
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--blind-predictions", type=Path, required=True)
    parser.add_argument("--truth-key", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        report = score_confirmatory(
            args.protocol,
            args.blind_predictions,
            args.truth_key,
            args.out,
        )
    except FileExistsError as error:
        print(f"refusing to overwrite one-shot output: {error}", file=sys.stderr)
        return 3
    print(
        json.dumps(
            {
                "status": report["status"],
                "confirmatory": report["confirmatory"],
                "passed_all_preregistered_gates": report["passed_all_preregistered_gates"],
                "output": str(args.out.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["confirmatory_protocol_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
