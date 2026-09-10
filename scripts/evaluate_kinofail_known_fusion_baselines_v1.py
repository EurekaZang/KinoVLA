#!/usr/bin/env python3
"""Train, freeze, and score established multimodal fusion baselines.

The script has two deliberately separate phases.  ``train`` uses development
data only, selects training duration on a held-out development fold, refits on
all development scenes, and writes a sealed checkpoint.  ``score`` verifies
that seal before opening the independent benchmark observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.conflict_invariant_kino import (  # noqa: E402
    invariant_relative_proprio_features,
    physical_group_weights,
    visual_context_features,
)
from kino_vla.eval.known_multimodal_fusions import (  # noqa: E402
    METHODS,
    FusionDimensions,
    build_fusion_model,
)
from scripts.develop_kino_v4_decision_window_invariant_only_v1 import (  # noqa: E402
    _load_conflict,
    _load_scale,
)

SCRIPT = Path(__file__).resolve()
MODEL_SOURCE = ROOT / "kino_vla/eval/known_multimodal_fusions.py"
DEV_DINO = ROOT / "outputs/eval/kino_conflict_dinov2_large_448_terrain_v4_development/features.npz"
CONFIRMATION = ROOT / "outputs/eval/kino_v4_confirmation_v1_f5d"
REFERENCE = ROOT / "outputs/eval/kino_v4_confirmation_v1_f5e_score_once"
DEFAULT_FREEZE = ROOT / "outputs/freeze/kinofail_known_fusion_baselines_f0"
DEFAULT_SCORE = ROOT / "outputs/eval/kinofail_known_fusion_baselines_f1_score_once"
TRAINING_SEEDS = (2026080911, 2026080912, 2026080913)
VALIDATION_FOLD = 4
MAX_EPOCHS = 60
PATIENCE = 7
BATCH_SIZE = 1024
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2026080991


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels).astype(str)
    predictions = np.asarray(predictions).astype(str)
    classes = sorted(set(labels.tolist()))
    recall = recall_score(labels, predictions, labels=classes, average=None)
    return {
        "samples": int(len(labels)),
        "accuracy": float(np.mean(labels == predictions)),
        "balanced_accuracy": float(recall.mean()),
        "per_class_recall": {
            label: float(value) for label, value in zip(classes, recall, strict=True)
        },
        "worst_class_recall": float(recall.min()),
        "worst_class": classes[int(recall.argmin())],
    }


def _fit_normalizer(values: np.ndarray, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected_values = np.asarray(values[selected], dtype=np.float64)
    mean = selected_values.mean(axis=0).astype(np.float32)
    scale = selected_values.std(axis=0).astype(np.float32)
    scale[scale < 1.0e-6] = 1.0
    return mean, scale


def _normalize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    result = (np.asarray(values, dtype=np.float32) - mean) / scale
    np.clip(result, -10.0, 10.0, out=result)
    if not np.isfinite(result).all():
        raise RuntimeError("normalization produced non-finite values")
    return result.astype(np.float32, copy=False)


def _load_development() -> dict[str, dict[str, Any]]:
    scale = _load_scale()
    conflict = _load_conflict()
    with np.load(DEV_DINO, allow_pickle=False) as archive:
        ids = archive["sample_ids"].astype(str)
        detail = np.asarray(archive["visual_ground_mean"], dtype=np.float32)
    if not np.array_equal(ids, conflict["sample_ids"]):
        raise RuntimeError("development DINOv2 features are misaligned")
    return {
        "scale": {
            **scale,
            "visual_fusion": visual_context_features(scale["visual"]),
            "proprio_fusion": invariant_relative_proprio_features(
                scale["invariant"], scale["full"]
            ),
        },
        "conflict": {
            **conflict,
            "visual_fusion": np.concatenate(
                [visual_context_features(conflict["visual"]), detail], axis=1
            ),
            "proprio_fusion": invariant_relative_proprio_features(
                conflict["invariant"], conflict["full"]
            ),
        },
    }


def _load_confirmation() -> dict[str, dict[str, Any]]:
    scale_rows = _jsonl(CONFIRMATION / "scale/records.jsonl")
    with np.load(CONFIRMATION / "scale/features.npz", allow_pickle=False) as archive:
        scale_ids = archive["sample_ids"].astype(str)
        scale_visual = np.asarray(archive["visual"], dtype=np.float32)
        scale_full = np.asarray(archive["full_proprio"], dtype=np.float32)
        scale_invariant = np.asarray(archive["invariant_proprio"], dtype=np.float32)
    conflict_rows = _jsonl(CONFIRMATION / "conflict/base_features/records.jsonl")
    with np.load(
        CONFIRMATION / "conflict/base_features/features.npz", allow_pickle=False
    ) as archive:
        conflict_ids = archive["sample_ids"].astype(str)
        conflict_visual = np.asarray(archive["visual"], dtype=np.float32)
        conflict_full = np.asarray(archive["proprio"], dtype=np.float32)
    with np.load(
        CONFIRMATION / "conflict/invariant_features/features.npz", allow_pickle=False
    ) as archive:
        invariant_ids = archive["sample_ids"].astype(str)
        conflict_invariant = np.asarray(archive["proprio"], dtype=np.float32)
    with np.load(CONFIRMATION / "conflict/dinov2/features.npz", allow_pickle=False) as archive:
        dino_ids = archive["sample_ids"].astype(str)
        conflict_detail = np.asarray(archive["visual_ground_mean"], dtype=np.float32)
    if scale_ids.tolist() != [str(row["sample_id"]) for row in scale_rows]:
        raise RuntimeError("confirmation Scale features are misaligned")
    if (
        conflict_ids.tolist() != [str(row["sample_id"]) for row in conflict_rows]
        or not np.array_equal(conflict_ids, invariant_ids)
        or not np.array_equal(conflict_ids, dino_ids)
    ):
        raise RuntimeError("confirmation Conflict features are misaligned")
    return {
        "scale": {
            "rows": scale_rows,
            "ids": scale_ids,
            "visual_fusion": visual_context_features(scale_visual),
            "proprio_fusion": invariant_relative_proprio_features(
                scale_invariant, scale_full
            ),
            "labels": np.asarray(
                [str(row["attribution_category"]) for row in scale_rows]
            ),
        },
        "conflict": {
            "rows": conflict_rows,
            "ids": conflict_ids,
            "visual_fusion": np.concatenate(
                [visual_context_features(conflict_visual), conflict_detail], axis=1
            ),
            "proprio_fusion": invariant_relative_proprio_features(
                conflict_invariant, conflict_full
            ),
            "labels": np.asarray(
                [str(row["attribution_category"]) for row in conflict_rows]
            ),
        },
    }


def _encode_labels(labels: np.ndarray, classes: list[str]) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(classes)}
    return np.asarray([lookup[str(label)] for label in labels], dtype=np.int64)


def _training_weights(labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    group_weight = physical_group_weights(groups)
    class_mass = {
        label: float(group_weight[labels == label].sum()) for label in sorted(set(labels))
    }
    total = float(sum(class_mass.values()))
    class_factor = {
        label: total / (len(class_mass) * mass) for label, mass in class_mass.items()
    }
    weights = group_weight * np.asarray([class_factor[str(label)] for label in labels])
    weights /= weights.mean()
    return weights.astype(np.float32)


def _group_predictions(
    probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (group, truth) in enumerate(zip(groups, labels, strict=True)):
        grouped[(str(group), str(truth))].append(index)
    truth_values: list[str] = []
    prediction_values: list[str] = []
    for (_, truth), indices in sorted(grouped.items()):
        truth_values.append(truth)
        prediction_values.append(classes[int(probabilities[indices].mean(axis=0).argmax())])
    return np.asarray(truth_values), np.asarray(prediction_values)


def _balanced_group_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
) -> float:
    truth, prediction = _group_predictions(probabilities, labels, groups, classes)
    return float(_metrics(truth, prediction)["balanced_accuracy"])


def _predict(
    model: nn.Module,
    visual: torch.Tensor,
    proprio: torch.Tensor,
    *,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    results: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(visual), 4096):
            visual_batch = visual[start : start + 4096].to(device)
            proprio_batch = proprio[start : start + 4096].to(device)
            logits = model(visual_batch, proprio_batch)
            results.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(results, axis=0)


def _train_model(
    method: str,
    dims: FusionDimensions,
    visual: torch.Tensor,
    proprio: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    selected: np.ndarray,
    *,
    seed: int,
    epochs: int,
    device: torch.device,
    validation: np.ndarray | None = None,
    validation_labels: np.ndarray | None = None,
    validation_groups: np.ndarray | None = None,
    classes: list[str] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_fusion_model(method, dims).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    selected_indices = torch.as_tensor(np.flatnonzero(selected), dtype=torch.long)
    dataset = TensorDataset(
        visual[selected_indices],
        proprio[selected_indices],
        targets[selected_indices],
        weights[selected_indices],
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -np.inf
    best_epoch = epochs
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        weight_sum = 0.0
        for visual_batch, proprio_batch, target_batch, weight_batch in loader:
            visual_batch = visual_batch.to(device, non_blocking=True)
            proprio_batch = proprio_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            weight_batch = weight_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(visual_batch, proprio_batch)
            per_item = nn.functional.cross_entropy(logits, target_batch, reduction="none")
            loss = (per_item * weight_batch).sum() / weight_batch.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss_sum += float((per_item * weight_batch).sum().detach().cpu())
            weight_sum += float(weight_batch.sum().detach().cpu())
        record = {"epoch": float(epoch), "loss": loss_sum / weight_sum}
        if validation is not None:
            if validation_labels is None or validation_groups is None or classes is None:
                raise ValueError("validation metadata is incomplete")
            probability = _predict(model, visual[validation], proprio[validation], device=device)
            score = _balanced_group_score(
                probability, validation_labels, validation_groups, classes
            )
            record["validation_balanced_accuracy"] = score
            if score > best_score + 1.0e-4:
                best_score = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
        history.append(record)
        if validation is not None and stale >= PATIENCE:
            break
    if validation is not None:
        if best_state is None:
            raise RuntimeError("validation failed to select a checkpoint")
        model.load_state_dict(best_state)
    return model, {
        "epochs_run": len(history),
        "selected_epoch": int(best_epoch),
        "best_validation_balanced_accuracy": (
            None if validation is None else float(best_score)
        ),
        "final_loss": float(history[-1]["loss"]),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def train_and_freeze(output: Path) -> int:
    if output.exists():
        raise FileExistsError(output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    development = _load_development()
    bundle: dict[str, Any] = {
        "schema_version": "kinofail.known-fusion-checkpoint.v1",
        "methods": list(METHODS),
        "training_seeds": list(TRAINING_SEEDS),
        "datasets": {},
    }
    training_report: dict[str, Any] = {}
    for battery, source in development.items():
        labels = np.asarray(source["labels"]).astype(str)
        groups = np.asarray(source["groups"]).astype(str)
        classes = sorted(set(labels.tolist()))
        encoded = _encode_labels(labels, classes)
        train_mask = np.asarray(source["folds"]) != VALIDATION_FOLD
        validation_mask = ~train_mask
        train_visual_mean, train_visual_scale = _fit_normalizer(
            source["visual_fusion"], train_mask
        )
        train_proprio_mean, train_proprio_scale = _fit_normalizer(
            source["proprio_fusion"], train_mask
        )
        validation_visual = torch.from_numpy(
            _normalize(source["visual_fusion"], train_visual_mean, train_visual_scale)
        )
        validation_proprio = torch.from_numpy(
            _normalize(source["proprio_fusion"], train_proprio_mean, train_proprio_scale)
        )
        targets = torch.from_numpy(encoded)
        sample_weights = torch.from_numpy(_training_weights(labels, groups))
        dims = FusionDimensions(
            visual=validation_visual.shape[1],
            proprio=validation_proprio.shape[1],
            classes=len(classes),
        )
        selected_epochs: dict[str, int] = {}
        validation_reports: dict[str, Any] = {}
        for method_index, method in enumerate(METHODS):
            print(
                json.dumps(
                    {
                        "phase": "duration_selection",
                        "battery": battery,
                        "method": method,
                    }
                ),
                flush=True,
            )
            _, report = _train_model(
                method,
                dims,
                validation_visual,
                validation_proprio,
                targets,
                sample_weights,
                train_mask,
                seed=TRAINING_SEEDS[0] + 100 * method_index,
                epochs=MAX_EPOCHS,
                device=device,
                validation=validation_mask,
                validation_labels=labels[validation_mask],
                validation_groups=groups[validation_mask],
                classes=classes,
            )
            selected_epochs[method] = int(report["selected_epoch"])
            validation_reports[method] = report
            print(
                json.dumps(
                    {"battery": battery, "method": method, **report}, sort_keys=True
                ),
                flush=True,
            )
        del validation_visual, validation_proprio
        if device.type == "cuda":
            torch.cuda.empty_cache()
        visual_mean, visual_scale = _fit_normalizer(
            source["visual_fusion"], np.ones(len(labels), dtype=bool)
        )
        proprio_mean, proprio_scale = _fit_normalizer(
            source["proprio_fusion"], np.ones(len(labels), dtype=bool)
        )
        visual = torch.from_numpy(_normalize(source["visual_fusion"], visual_mean, visual_scale))
        proprio = torch.from_numpy(
            _normalize(source["proprio_fusion"], proprio_mean, proprio_scale)
        )
        all_rows = np.ones(len(labels), dtype=bool)
        method_states: dict[str, list[dict[str, torch.Tensor]]] = {}
        refit_reports: dict[str, Any] = {}
        for method_index, method in enumerate(METHODS):
            method_states[method] = []
            refit_reports[method] = []
            for seed_index, seed in enumerate(TRAINING_SEEDS):
                effective_seed = seed + 100 * method_index
                print(
                    json.dumps(
                        {
                            "phase": "full_development_refit",
                            "battery": battery,
                            "method": method,
                            "seed": effective_seed,
                            "epochs": selected_epochs[method],
                        }
                    ),
                    flush=True,
                )
                model, report = _train_model(
                    method,
                    dims,
                    visual,
                    proprio,
                    targets,
                    sample_weights,
                    all_rows,
                    seed=effective_seed,
                    epochs=selected_epochs[method],
                    device=device,
                )
                method_states[method].append(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                )
                refit_reports[method].append(
                    {"seed_index": seed_index, "seed": effective_seed, **report}
                )
        bundle["datasets"][battery] = {
            "classes": classes,
            "dimensions": {
                "visual": dims.visual,
                "proprio": dims.proprio,
                "classes": dims.classes,
            },
            "normalizer": {
                "visual_mean": visual_mean,
                "visual_scale": visual_scale,
                "proprio_mean": proprio_mean,
                "proprio_scale": proprio_scale,
            },
            "states": method_states,
        }
        training_report[battery] = {
            "samples": len(labels),
            "physical_groups": len(set(groups.tolist())),
            "classes": classes,
            "duration_selection": validation_reports,
            "selected_epochs": selected_epochs,
            "full_refits": refit_reports,
        }
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "known_fusion_baselines.pt"
    torch.save(bundle, checkpoint)
    manifest = {
        "schema_version": "kinofail.known-fusion-freeze.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_before_known_fusion_confirmation_inference",
        "scope": "post-study benchmark baseline expansion",
        "confirmation_labels_predictions_or_scores_used_for_selection": False,
        "methods": {
            "concat_mlp": "capacity-matched neural concatenation control",
            "tfn": "Tensor Fusion Network augmented outer product",
            "lmf": "Low-rank Multimodal Fusion, rank 4",
            "gmu": "Gated Multimodal Unit",
            "embracenet": "EmbraceNet coordinate-wise stochastic fusion",
            "moddrop": "ModDrop with modality-drop probability 0.2",
        },
        "shared_protocol": {
            "input": "frozen CLIP+DINOv2 visual and registered decision-window proprio descriptors",
            "battery_specific_fit": True,
            "class_and_physical_group_balancing": True,
            "validation_fold": VALIDATION_FOLD,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "ensemble_seeds": list(TRAINING_SEEDS),
        },
        "training_report": training_report,
        "artifacts": {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": _sha256(checkpoint),
        },
        "source_sha256": {
            "script": _sha256(SCRIPT),
            "model_source": _sha256(MODEL_SOURCE),
            "development_dino": _sha256(DEV_DINO),
        },
    }
    manifest_path = output / "freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["artifacts"], indent=2), flush=True)
    return 0


def _aggregate_confirmation(
    source: dict[str, Any],
    probabilities: dict[str, np.ndarray],
    classes: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(source["rows"]):
        unit = str(
            row.get("physical_episode_id")
            or f"{row['case_id']}::{row['attribution_category']}"
        )
        grouped[(unit, str(row["attribution_category"]))].append(index)
    rows: list[dict[str, Any]] = []
    for (unit, truth), indices in sorted(grouped.items()):
        if len(indices) != 3:
            raise RuntimeError(f"physical unit does not contain three views: {unit}")
        reference = source["rows"][indices[0]]
        rows.append(
            {
                "unit_id": unit,
                "truth": truth,
                "scene": str(
                    reference.get("scene_cluster") or reference.get("scene_family")
                ),
                "material": str(reference["cluster_material"]),
                "cell": str(reference.get("cell", "Scale")),
                "predictions": {
                    method: classes[int(value[indices].mean(axis=0).argmax())]
                    for method, value in probabilities.items()
                },
            }
        )
    return rows


def _bootstrap(rows: dict[str, list[dict[str, Any]]], methods: list[str]) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws: dict[str, np.ndarray] = {}
    for battery, battery_rows in rows.items():
        scenes = sorted({str(row["scene"]) for row in battery_rows})
        materials = sorted({str(row["material"]) for row in battery_rows})
        classes = sorted({str(row["truth"]) for row in battery_rows})
        scene_index = {value: index for index, value in enumerate(scenes)}
        material_index = {value: index for index, value in enumerate(materials)}
        class_index = {value: index for index, value in enumerate(classes)}
        totals = np.zeros((len(scenes), len(materials), len(classes)), dtype=np.float64)
        correct = np.zeros((len(methods), *totals.shape), dtype=np.float64)
        for row in battery_rows:
            index = (
                scene_index[str(row["scene"])],
                material_index[str(row["material"])],
                class_index[str(row["truth"])],
            )
            totals[index] += 1.0
            for method_index, method in enumerate(methods):
                correct[method_index][index] += float(
                    str(row["predictions"][method]) == str(row["truth"])
                )
        values = np.empty((BOOTSTRAP_DRAWS, len(methods)), dtype=np.float64)
        for draw in range(BOOTSTRAP_DRAWS):
            scene_count = np.bincount(
                rng.integers(0, len(scenes), size=len(scenes)), minlength=len(scenes)
            )
            material_count = np.bincount(
                rng.integers(0, len(materials), size=len(materials)),
                minlength=len(materials),
            )
            weights = scene_count[:, None] * material_count[None, :]
            class_total = np.einsum("sm,smc->c", weights, totals)
            class_correct = np.einsum("sm,ksmc->kc", weights, correct)
            recall = class_correct / np.where(
                class_total[None, :] <= 0.0, np.nan, class_total[None, :]
            )
            values[draw] = np.nanmean(recall, axis=1)
        draws[battery] = values

    def interval(values: np.ndarray) -> list[float]:
        return [
            float(np.nanquantile(values, 0.025)),
            float(np.nanquantile(values, 0.975)),
        ]

    index = {method: position for position, method in enumerate(methods)}
    scale = draws["scale"]
    conflict = draws["conflict"]
    macro = 0.5 * (scale + conflict)
    worst = np.minimum(scale, conflict)
    metrics = {
        method: {
            "scale_ci95": interval(scale[:, position]),
            "conflict_ci95": interval(conflict[:, position]),
            "macro_ci95": interval(macro[:, position]),
            "worst_ci95": interval(worst[:, position]),
        }
        for method, position in index.items()
    }
    early = index["joint_early_fusion"]
    paired_deltas = {}
    for method in METHODS:
        position = index[method]
        paired_deltas[method] = {
            "scale": interval(scale[:, position] - scale[:, early]),
            "conflict": interval(conflict[:, position] - conflict[:, early]),
            "macro": interval(macro[:, position] - macro[:, early]),
            "worst": interval(worst[:, position] - worst[:, early]),
        }
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "clusters": ["scene", "material"],
        "metrics": metrics,
        "paired_ci95_minus_joint_early_fusion": paired_deltas,
    }


def score_once(freeze: Path, output: Path) -> int:
    if output.exists():
        raise FileExistsError(output)
    manifest_path = freeze / "freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checkpoint_path = ROOT / manifest["artifacts"]["checkpoint"]
    if (
        manifest.get("status") != "frozen_before_known_fusion_confirmation_inference"
        or _sha256(checkpoint_path) != manifest["artifacts"]["checkpoint_sha256"]
        or _sha256(SCRIPT) != manifest["source_sha256"]["script"]
        or _sha256(MODEL_SOURCE) != manifest["source_sha256"]["model_source"]
    ):
        raise RuntimeError("known-fusion freeze seal failed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    confirmation = _load_confirmation()
    new_unit_rows: dict[str, list[dict[str, Any]]] = {}
    view_metrics: dict[str, Any] = {}
    for battery, source in confirmation.items():
        saved = bundle["datasets"][battery]
        classes = [str(value) for value in saved["classes"]]
        dims = FusionDimensions(**saved["dimensions"])
        normalizer = saved["normalizer"]
        visual = torch.from_numpy(
            _normalize(
                source["visual_fusion"],
                normalizer["visual_mean"],
                normalizer["visual_scale"],
            )
        )
        proprio = torch.from_numpy(
            _normalize(
                source["proprio_fusion"],
                normalizer["proprio_mean"],
                normalizer["proprio_scale"],
            )
        )
        probabilities: dict[str, np.ndarray] = {}
        for method in METHODS:
            seed_probabilities = []
            for state in saved["states"][method]:
                model = build_fusion_model(method, dims)
                model.load_state_dict(state)
                model.to(device)
                seed_probabilities.append(_predict(model, visual, proprio, device=device))
            probabilities[method] = np.mean(seed_probabilities, axis=0)
        view_metrics[battery] = {
            method: _metrics(
                source["labels"], np.asarray(classes)[probability.argmax(axis=1)]
            )
            for method, probability in probabilities.items()
        }
        new_unit_rows[battery] = _aggregate_confirmation(source, probabilities, classes)
    reference_rows = _jsonl(REFERENCE / "physical_units.jsonl")
    reference_lookup = {
        (str(row["battery"]), str(row["unit_id"])): row for row in reference_rows
    }
    combined_rows: dict[str, list[dict[str, Any]]] = {}
    for battery, rows in new_unit_rows.items():
        combined_rows[battery] = []
        for row in rows:
            reference = reference_lookup[(battery, str(row["unit_id"]))]
            if str(reference["truth"]) != str(row["truth"]):
                raise RuntimeError("reference/new truth mismatch")
            combined_rows[battery].append(
                {
                    **row,
                    "predictions": {
                        **reference["predictions"],
                        **row["predictions"],
                    },
                }
            )
    physical_metrics: dict[str, dict[str, Any]] = defaultdict(dict)
    per_cell: dict[str, dict[str, Any]] = {}
    all_methods = sorted(combined_rows["scale"][0]["predictions"])
    for battery, rows in combined_rows.items():
        labels = np.asarray([str(row["truth"]) for row in rows])
        for method in all_methods:
            predictions = np.asarray([str(row["predictions"][method]) for row in rows])
            physical_metrics[battery][method] = _metrics(labels, predictions)
    for cell in ("T2_vision_decisive", "T3_proprio_decisive"):
        selected = [row for row in combined_rows["conflict"] if row["cell"] == cell]
        labels = np.asarray([str(row["truth"]) for row in selected])
        per_cell[cell] = {
            method: _metrics(
                labels,
                np.asarray([str(row["predictions"][method]) for row in selected]),
            )
            for method in all_methods
        }
    cross_battery = {}
    for method in all_methods:
        scale_score = float(physical_metrics["scale"][method]["balanced_accuracy"])
        conflict_score = float(physical_metrics["conflict"][method]["balanced_accuracy"])
        cross_battery[method] = {
            "scale": scale_score,
            "t2": float(per_cell["T2_vision_decisive"][method]["balanced_accuracy"]),
            "t3": float(per_cell["T3_proprio_decisive"][method]["balanced_accuracy"]),
            "conflict": conflict_score,
            "macro": 0.5 * (scale_score + conflict_score),
            "worst": min(scale_score, conflict_score),
        }
    bootstrap = _bootstrap(combined_rows, all_methods)
    output.mkdir(parents=True, exist_ok=False)
    units_path = output / "physical_units.jsonl"
    units_path.write_text(
        "".join(
            json.dumps({"battery": battery, **row}, sort_keys=True) + "\n"
            for battery, rows in combined_rows.items()
            for row in rows
        )
    )
    report = {
        "schema_version": "kinofail.known-fusion-score-once.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "known_fusion_baselines_scored_once",
        "scope": "post-study benchmark baseline expansion",
        "optimization_uses_confirmation_labels": False,
        "view_level_secondary_metrics": view_metrics,
        "physical_unit_primary_metrics": physical_metrics,
        "per_conflict_cell_physical_unit": per_cell,
        "cross_battery_physical_unit": cross_battery,
        "physical_unit_crossed_cluster_bootstrap": bootstrap,
        "source_sha256": {
            "freeze_manifest": _sha256(manifest_path),
            "checkpoint": _sha256(checkpoint_path),
            "reference_physical_units": _sha256(REFERENCE / "physical_units.jsonl"),
        },
        "artifacts": {
            "physical_units": str(units_path.relative_to(ROOT)),
            "physical_units_sha256": _sha256(units_path),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(cross_battery, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("train", "score"))
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_SCORE)
    args = parser.parse_args()
    if args.phase == "train":
        return train_and_freeze(args.freeze.resolve())
    return score_once(args.freeze.resolve(), args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
