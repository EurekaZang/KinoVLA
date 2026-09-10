#!/usr/bin/env python3
"""Develop, freeze, and score one deployable 11-class GMU on KiNO-Fail.

Unlike the earlier battery-specific benchmark, this program builds one input
schema, one 11-class posterior, and one set of GMU weights.  The same frozen
three-seed ensemble is used for Scale, T2, T3, Conflict, and primary-view
deployment inference.  Battery identity is used only to construct development
weights and report metrics; it is never provided to the model.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
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
    visual_context_features,
)
from kino_vla.eval.known_multimodal_fusions import (  # noqa: E402
    ConcatMLP,
    EmbraceNet,
    FusionDimensions,
    GatedMultimodalUnit,
    LowRankMultimodalFusion,
    ModDropFusion,
    TensorFusionNetwork,
)
from scripts.evaluate_kinofail_known_fusion_baselines_v1 import (  # noqa: E402
    _load_development,
)


ALL191 = Path("/data/eureka/kinovla_outputs/kino_v4_all191_v1")
DEV_SCALE_DINO = ROOT / "outputs/eval/kinofail_single_gmu_dinov2_scale_development_v1/features.npz"
DEV_CONFLICT_DINO = ROOT / "outputs/eval/kino_conflict_dinov2_large_448_terrain_v4_development/features.npz"
DEV_SCALE_V6 = ROOT / "outputs/eval/kinofail_single_gmu_scale_development_v1/features.npz"
DEV_SCALE_V6_RECORDS = ROOT / "outputs/eval/realistic_a0_a7_v6/snapshots/snapshot_records.jsonl"
DEFAULT_DEVELOPMENT = ROOT / "outputs/eval/kinofail_single_gmu_v1_development"
DEFAULT_FREEZE = ROOT / "outputs/freeze/kinofail_single_gmu_v1"
DEFAULT_SCORE = ROOT / "outputs/eval/kinofail_single_gmu_v1_all191"
TRAINING_SEEDS = (2026082411, 2026082412, 2026082413)
CANONICAL_CLASSES = (
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
UNIFIED_METHODS = (
    "vision",
    "proprioception",
    "early_fusion",
    "late_fusion",
    "concat_mlp",
    "tfn",
    "lmf",
    "gmu",
    "embracenet",
    "moddrop",
)


@dataclass(frozen=True)
class TrainingConfig:
    visual_mode: str
    conflict_mass: float
    epochs: int
    method: str = "gmu"
    hidden: int = 128
    pretrain_epochs: int = 0
    pretrain_learning_rate: float = 1.0e-3
    normalizer_scope: str = "combined"
    distillation_weight: float = 0.0
    distillation_temperature: float = 2.0
    conflict_aux_weight: float = 0.0
    batch_size: int = 2048
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def visual_features(values: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if mode == "context":
        return visual_context_features(values)
    if mode == "full":
        if values.ndim != 2 or values.shape[1] != 1536:
            raise ValueError(f"full visual mode expects 1536 values, got {values.shape}")
        return values
    raise ValueError(f"unknown visual mode: {mode}")


def aligned_dino(
    path: Path, sample_ids: np.ndarray, *, final_only: bool = False
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["sample_ids"].astype(str)
        values = np.asarray(archive["visual_ground_mean"], dtype=np.float32)
    if not np.array_equal(ids, np.asarray(sample_ids).astype(str)):
        raise RuntimeError(f"DINOv2 features are misaligned: {path}")
    if final_only and values.shape[1] == 2048:
        values = values[:, 1024:]
    if final_only and values.shape[1] != 1024:
        raise RuntimeError(f"final-frame DINOv2 descriptor must be 1024-D: {path}")
    return values


def load_development(mode: str) -> dict[str, dict[str, Any]]:
    source = _load_development()
    result: dict[str, dict[str, Any]] = {}
    for battery in ("scale", "conflict"):
        item = source[battery]
        if battery == "scale" and mode in ("dino", "dino_final"):
            rows = jsonl(DEV_SCALE_V6_RECORDS)
            with np.load(DEV_SCALE_V6, allow_pickle=False) as archive:
                scale_ids = archive["sample_ids"].astype(str)
                scale_visual = np.asarray(archive["visual"], dtype=np.float32)
                scale_full = np.asarray(archive["full_proprio"], dtype=np.float32)
                scale_invariant = np.asarray(
                    archive["invariant_proprio"], dtype=np.float32
                )
            if scale_ids.tolist() != [str(row["sample_id"]) for row in rows]:
                raise RuntimeError("retained Scale development records are misaligned")
            scene_names = sorted({str(row["scene_family"]) for row in rows})
            scene_fold = {
                scene: index % 5 for index, scene in enumerate(scene_names)
            }
            item = {
                "sample_ids": scale_ids,
                "visual": scale_visual,
                "proprio_fusion": invariant_relative_proprio_features(
                    scale_invariant, scale_full
                ),
                "labels": np.asarray(
                    [str(row["attribution_category"]) for row in rows]
                ),
                "groups": np.asarray(
                    [str(row["counterfactual_group_id"]) for row in rows]
                ),
                "scenes": np.asarray([str(row["scene_family"]) for row in rows]),
                "folds": np.asarray(
                    [scene_fold[str(row["scene_family"])] for row in rows],
                    dtype=np.int64,
                ),
            }
        if mode in ("dino", "dino_final"):
            dino_path = DEV_SCALE_DINO if battery == "scale" else DEV_CONFLICT_DINO
            visual = np.concatenate(
                [
                    visual_context_features(item["visual"]),
                    aligned_dino(
                        dino_path,
                        item["sample_ids"],
                        final_only=mode == "dino_final",
                    ),
                ],
                axis=1,
            )
        else:
            visual = visual_features(item["visual"], mode)
        cells = (
            np.asarray(["Scale"] * len(item["labels"]))
            if battery == "scale"
            else np.asarray(item["cells"]).astype(str)
        )
        result[battery] = {
            "visual": visual,
            "proprio": np.asarray(
                item.get("proprio_fusion", item.get("proprio")), dtype=np.float32
            ),
            "labels": np.asarray(item["labels"]).astype(str),
            "groups": np.asarray(item["groups"]).astype(str),
            "scenes": np.asarray(item["scenes"]).astype(str),
            "folds": np.asarray(item["folds"], dtype=np.int64),
            "cells": cells,
        }
    return result


def load_all191(root: Path, mode: str) -> dict[str, dict[str, Any]]:
    scale_rows = jsonl(root / "scale/records.jsonl")
    with np.load(root / "scale/features.npz", allow_pickle=False) as archive:
        scale_ids = archive["sample_ids"].astype(str)
        scale_visual = np.asarray(archive["visual"], dtype=np.float32)
        scale_full = np.asarray(archive["full_proprio"], dtype=np.float32)
        scale_invariant = np.asarray(archive["invariant_proprio"], dtype=np.float32)
    conflict_rows = jsonl(root / "conflict/base_features/records.jsonl")
    with np.load(root / "conflict/base_features/features.npz", allow_pickle=False) as archive:
        conflict_ids = archive["sample_ids"].astype(str)
        conflict_visual = np.asarray(archive["visual"], dtype=np.float32)
        conflict_full = np.asarray(archive["proprio"], dtype=np.float32)
    with np.load(
        root / "conflict/invariant_features/features.npz", allow_pickle=False
    ) as archive:
        invariant_ids = archive["sample_ids"].astype(str)
        conflict_invariant = np.asarray(archive["proprio"], dtype=np.float32)
    if scale_ids.tolist() != [str(row["sample_id"]) for row in scale_rows]:
        raise RuntimeError("all-191 Scale records and features are misaligned")
    if (
        conflict_ids.tolist() != [str(row["sample_id"]) for row in conflict_rows]
        or not np.array_equal(conflict_ids, invariant_ids)
    ):
        raise RuntimeError("all-191 Conflict feature blocks are misaligned")
    if mode in ("dino", "dino_final"):
        scale_visual_final = np.concatenate(
            [
                visual_context_features(scale_visual),
                aligned_dino(
                    root / "scale/dinov2/features.npz",
                    scale_ids,
                    final_only=mode == "dino_final",
                ),
            ],
            axis=1,
        )
        conflict_visual_final = np.concatenate(
            [
                visual_context_features(conflict_visual),
                aligned_dino(
                    root / "conflict/dinov2/features.npz",
                    conflict_ids,
                    final_only=mode == "dino_final",
                ),
            ],
            axis=1,
        )
    else:
        scale_visual_final = visual_features(scale_visual, mode)
        conflict_visual_final = visual_features(conflict_visual, mode)
    return {
        "scale": {
            "rows": scale_rows,
            "ids": scale_ids,
            "visual": scale_visual_final,
            "proprio": invariant_relative_proprio_features(
                scale_invariant, scale_full
            ),
            "labels": np.asarray(
                [str(row["attribution_category"]) for row in scale_rows]
            ),
        },
        "conflict": {
            "rows": conflict_rows,
            "ids": conflict_ids,
            "visual": conflict_visual_final,
            "proprio": invariant_relative_proprio_features(
                conflict_invariant, conflict_full
            ),
            "labels": np.asarray(
                [str(row["attribution_category"]) for row in conflict_rows]
            ),
        },
    }


def combine(
    data: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    scale = data["scale"]
    conflict = data["conflict"]
    return {
        "visual": np.concatenate([scale["visual"], conflict["visual"]]),
        "proprio": np.concatenate([scale["proprio"], conflict["proprio"]]),
        "labels": np.concatenate([scale["labels"], conflict["labels"]]).astype(str),
        "groups": np.concatenate([scale["groups"], conflict["groups"]]).astype(str),
        "scenes": np.concatenate([scale["scenes"], conflict["scenes"]]).astype(str),
        "folds": np.concatenate([scale["folds"], conflict["folds"]]).astype(np.int64),
        "cells": np.concatenate([scale["cells"], conflict["cells"]]).astype(str),
        "batteries": np.asarray(
            ["scale"] * len(scale["labels"]) + ["conflict"] * len(conflict["labels"])
        ),
    }


def fit_normalizer(values: np.ndarray, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    subset = np.asarray(values[selected], dtype=np.float64)
    mean = subset.mean(axis=0).astype(np.float32)
    scale = subset.std(axis=0).astype(np.float32)
    scale[scale < 1.0e-6] = 1.0
    return mean, scale


def normalize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    result = (np.asarray(values, dtype=np.float32) - mean) / scale
    np.clip(result, -10.0, 10.0, out=result)
    if not np.isfinite(result).all():
        raise RuntimeError("normalization produced non-finite values")
    return result.astype(np.float32, copy=False)


def training_weights(
    labels: np.ndarray,
    groups: np.ndarray,
    batteries: np.ndarray,
    selected: np.ndarray,
    conflict_mass: float,
) -> np.ndarray:
    """Equalize physical groups and classes within each battery.

    Scale and Conflict receive total masses ``1-conflict_mass`` and
    ``conflict_mass``.  Within a battery, every registered class receives equal
    mass; within a class, every physical group receives equal mass.
    """
    masses = {"scale": 1.0 - conflict_mass, "conflict": conflict_mass}
    result = np.zeros(len(labels), dtype=np.float64)
    for battery in ("scale", "conflict"):
        battery_mask = selected & (batteries == battery)
        classes = sorted(set(labels[battery_mask].tolist()))
        for label in classes:
            cell = battery_mask & (labels == label)
            unique, counts = np.unique(groups[cell], return_counts=True)
            group_size = dict(zip(unique.tolist(), counts.tolist(), strict=True))
            indices = np.flatnonzero(cell)
            values = np.asarray(
                [1.0 / group_size[str(groups[index])] for index in indices],
                dtype=np.float64,
            )
            values *= masses[battery] / len(classes) / values.sum()
            result[indices] = values
    result[selected] *= float(selected.sum()) / result[selected].sum()
    return result.astype(np.float32)


def encode(labels: np.ndarray) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(CANONICAL_CLASSES)}
    unknown = sorted(set(labels.tolist()) - set(lookup))
    if unknown:
        raise ValueError(f"labels outside the canonical ontology: {unknown}")
    return np.asarray([lookup[str(label)] for label in labels], dtype=np.int64)


class SingleModalityMLP(nn.Module):
    def __init__(self, input_dim: int, classes: int, hidden: int, modality: str) -> None:
        super().__init__()
        self.modality = modality
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, classes),
        )

    def forward(self, visual: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.network(visual if self.modality == "vision" else proprio)


class EarlyFusionMLP(nn.Module):
    def __init__(self, dims: FusionDimensions, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dims.visual + dims.proprio, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, dims.classes),
        )

    def forward(self, visual: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([visual, proprio], dim=1))


class LatePosteriorFusion(nn.Module):
    def __init__(self, dims: FusionDimensions, hidden: int) -> None:
        super().__init__()
        self.visual = SingleModalityMLP(dims.visual, dims.classes, hidden, "vision")
        self.proprio = SingleModalityMLP(dims.proprio, dims.classes, hidden, "proprioception")

    def forward(self, visual: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        visual_probability = torch.softmax(self.visual(visual, proprio), dim=1)
        proprio_probability = torch.softmax(self.proprio(visual, proprio), dim=1)
        return torch.log(0.5 * (visual_probability + proprio_probability) + 1.0e-8)


def build_model(visual_dim: int, config: TrainingConfig) -> nn.Module:
    dims = FusionDimensions(
        visual=visual_dim,
        proprio=251,
        classes=len(CANONICAL_CLASSES),
    )
    if config.method == "vision":
        return SingleModalityMLP(dims.visual, dims.classes, config.hidden, "vision")
    if config.method == "proprioception":
        return SingleModalityMLP(
            dims.proprio, dims.classes, config.hidden, "proprioception"
        )
    if config.method == "early_fusion":
        return EarlyFusionMLP(dims, config.hidden)
    if config.method == "late_fusion":
        return LatePosteriorFusion(dims, config.hidden)
    if config.method == "concat_mlp":
        return ConcatMLP(dims, hidden=config.hidden)
    if config.method == "tfn":
        return TensorFusionNetwork(dims, hidden=config.hidden)
    if config.method == "lmf":
        return LowRankMultimodalFusion(dims, fusion_dim=config.hidden)
    if config.method == "gmu":
        return GatedMultimodalUnit(dims, hidden=config.hidden)
    if config.method == "embracenet":
        return EmbraceNet(dims, hidden=config.hidden)
    if config.method == "moddrop":
        return ModDropFusion(dims, hidden=config.hidden)
    raise ValueError(f"unknown unified method: {config.method}")


def predict(
    model: nn.Module,
    visual: np.ndarray,
    proprio: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(visual), 4096):
            logits = model(
                torch.from_numpy(visual[start : start + 4096]).to(device),
                torch.from_numpy(proprio[start : start + 4096]).to(device),
            )
            output.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(output)


def train(
    data: dict[str, np.ndarray],
    selected: np.ndarray,
    config: TrainingConfig,
    seed: int,
    normalizer: dict[str, np.ndarray],
) -> tuple[nn.Module, dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    visual = normalize(data["visual"], normalizer["visual_mean"], normalizer["visual_scale"])
    proprio = normalize(
        data["proprio"], normalizer["proprio_mean"], normalizer["proprio_scale"]
    )
    targets = encode(data["labels"])
    model = build_model(visual.shape[1], config).to(device)

    def fit_stage(
        stage_mask: np.ndarray,
        *,
        conflict_mass: float,
        epochs: int,
        learning_rate: float,
        stage_seed: int,
        teacher: nn.Module | None = None,
    ) -> float:
        weights = training_weights(
            data["labels"],
            data["groups"],
            data["batteries"],
            stage_mask,
            conflict_mass,
        )
        indices = torch.as_tensor(np.flatnonzero(stage_mask), dtype=torch.long)
        dataset = TensorDataset(
            torch.from_numpy(visual)[indices],
            torch.from_numpy(proprio)[indices],
            torch.from_numpy(targets)[indices],
            torch.from_numpy(weights)[indices],
            torch.from_numpy((data["batteries"] == "scale").astype(np.bool_))[
                indices
            ],
        )
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(stage_seed),
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=config.weight_decay,
        )
        final_loss = float("nan")
        for _ in range(epochs):
            model.train()
            numerator = 0.0
            denominator = 0.0
            for (
                visual_batch,
                proprio_batch,
                target_batch,
                weight_batch,
                scale_batch,
            ) in loader:
                visual_batch = visual_batch.to(device, non_blocking=True)
                proprio_batch = proprio_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)
                weight_batch = weight_batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(visual_batch, proprio_batch)
                per_item = nn.functional.cross_entropy(
                    logits, target_batch, reduction="none"
                )
                loss = (per_item * weight_batch).sum() / weight_batch.sum()
                if teacher is not None and bool(scale_batch.any()):
                    temperature = config.distillation_temperature
                    with torch.no_grad():
                        teacher_probability = torch.softmax(
                            teacher(
                                visual_batch[scale_batch], proprio_batch[scale_batch]
                            )
                            / temperature,
                            dim=1,
                        )
                    distillation = nn.functional.kl_div(
                        torch.log_softmax(
                            logits[scale_batch] / temperature, dim=1
                        ),
                        teacher_probability,
                        reduction="batchmean",
                    ) * (temperature**2)
                    loss = loss + config.distillation_weight * distillation
                conflict_batch = ~scale_batch
                if config.conflict_aux_weight > 0.0 and bool(conflict_batch.any()):
                    conflict_indices = torch.as_tensor(
                        [
                            CANONICAL_CLASSES.index("adhesion"),
                            CANONICAL_CLASSES.index("compliant_terrain"),
                            CANONICAL_CLASSES.index("invisible_obstacle"),
                            CANONICAL_CLASSES.index("low_friction"),
                        ],
                        dtype=torch.long,
                        device=device,
                    )
                    local_target_lookup = torch.full(
                        (len(CANONICAL_CLASSES),), -1, dtype=torch.long, device=device
                    )
                    local_target_lookup[conflict_indices] = torch.arange(
                        len(conflict_indices), device=device
                    )
                    local_target = local_target_lookup[target_batch[conflict_batch]]
                    if bool((local_target < 0).any()):
                        raise RuntimeError("Conflict auxiliary loss saw a non-conflict cause")
                    auxiliary = nn.functional.cross_entropy(
                        logits[conflict_batch][:, conflict_indices], local_target
                    )
                    loss = loss + config.conflict_aux_weight * auxiliary
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                numerator += float((per_item * weight_batch).sum().detach().cpu())
                denominator += float(weight_batch.sum().detach().cpu())
            final_loss = numerator / denominator
        return final_loss

    audit: dict[str, float] = {}
    teacher: nn.Module | None = None
    if config.pretrain_epochs > 0:
        pretrain_mask = selected & (data["batteries"] == "scale")
        audit["pretrain_final_loss"] = fit_stage(
            pretrain_mask,
            conflict_mass=0.0,
            epochs=config.pretrain_epochs,
            learning_rate=config.pretrain_learning_rate,
            stage_seed=seed,
        )
        if config.distillation_weight > 0.0:
            teacher = copy.deepcopy(model).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
    audit["final_loss"] = fit_stage(
        selected,
        conflict_mass=config.conflict_mass,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        stage_seed=seed + 1000,
        teacher=teacher,
    )
    return model, audit


def physical_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    indices = np.flatnonzero(selected)
    keys = np.char.add(np.char.add(groups[indices], "::"), labels[indices])
    unique, inverse = np.unique(keys, return_inverse=True)
    aggregate = np.zeros((len(unique), len(CANONICAL_CLASSES)), dtype=np.float64)
    counts = np.bincount(inverse)
    np.add.at(aggregate, inverse, probabilities[indices])
    aggregate /= counts[:, None]
    first = np.full(len(unique), -1, dtype=np.int64)
    first[inverse] = np.arange(len(inverse))
    truth = labels[indices[first]]
    prediction = np.asarray(CANONICAL_CLASSES)[aggregate.argmax(axis=1)]
    classes = sorted(set(truth.tolist()))
    recall = recall_score(truth, prediction, labels=classes, average=None)
    return {
        "physical_units": len(unique),
        "balanced_accuracy": float(recall.mean()),
        "accuracy": float(np.mean(truth == prediction)),
        "per_class_recall": {
            label: float(value) for label, value in zip(classes, recall, strict=True)
        },
    }


def development(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    rows: list[dict[str, Any]] = []
    for visual_mode in args.visual_modes.split(","):
        source = combine(load_development(visual_mode.strip()))
        for method in args.methods:
            for hidden in args.hidden:
                for pretrain_epochs in args.pretrain_epochs:
                    for conflict_mass in args.conflict_masses:
                        for epochs in args.epochs:
                            config = TrainingConfig(
                                visual_mode=visual_mode.strip(),
                                conflict_mass=float(conflict_mass),
                                epochs=int(epochs),
                                method=str(method),
                                hidden=int(hidden),
                                pretrain_epochs=int(pretrain_epochs),
                                pretrain_learning_rate=float(
                                    args.pretrain_learning_rate
                                ),
                                normalizer_scope=args.normalizer_scope,
                                learning_rate=float(args.learning_rate),
                                distillation_weight=float(args.distillation_weight),
                                distillation_temperature=float(
                                    args.distillation_temperature
                                ),
                                conflict_aux_weight=float(args.conflict_aux_weight),
                            )
                            for fold in range(5):
                                train_mask = source["folds"] != fold
                                test_mask = ~train_mask
                                normalizer_mask = train_mask
                                if config.normalizer_scope == "scale":
                                    normalizer_mask = train_mask & (
                                        source["batteries"] == "scale"
                                    )
                                visual_mean, visual_scale = fit_normalizer(
                                    source["visual"], normalizer_mask
                                )
                                proprio_mean, proprio_scale = fit_normalizer(
                                    source["proprio"], normalizer_mask
                                )
                                normalizer = {
                                    "visual_mean": visual_mean,
                                    "visual_scale": visual_scale,
                                    "proprio_mean": proprio_mean,
                                    "proprio_scale": proprio_scale,
                                }
                                model, audit = train(
                                    source,
                                    train_mask,
                                    config,
                                    seed=2026082400 + fold,
                                    normalizer=normalizer,
                                )
                                device = next(model.parameters()).device
                                probability = predict(
                                    model,
                                    normalize(
                                        source["visual"], visual_mean, visual_scale
                                    )[test_mask],
                                    normalize(
                                        source["proprio"], proprio_mean, proprio_scale
                                    )[test_mask],
                                    device,
                                )
                                local_labels = source["labels"][test_mask]
                                local_groups = source["groups"][test_mask]
                                local_batteries = source["batteries"][test_mask]
                                local_cells = source["cells"][test_mask]
                                scale = physical_metrics(
                                    probability,
                                    local_labels,
                                    local_groups,
                                    local_batteries == "scale",
                                )
                                conflict = physical_metrics(
                                    probability,
                                    local_labels,
                                    local_groups,
                                    local_batteries == "conflict",
                                )
                                t2 = physical_metrics(
                                    probability,
                                    local_labels,
                                    local_groups,
                                    local_cells == "T2_vision_decisive",
                                )
                                t3 = physical_metrics(
                                    probability,
                                    local_labels,
                                    local_groups,
                                    local_cells == "T3_proprio_decisive",
                                )
                                row = {
                                    "config": asdict(config),
                                    "fold": fold,
                                    "scale": scale["balanced_accuracy"],
                                    "conflict": conflict["balanced_accuracy"],
                                    "t2": t2["balanced_accuracy"],
                                    "t3": t3["balanced_accuracy"],
                                    "macro": 0.5
                                    * (
                                        scale["balanced_accuracy"]
                                        + conflict["balanced_accuracy"]
                                    ),
                                    "worst": min(
                                        scale["balanced_accuracy"],
                                        conflict["balanced_accuracy"],
                                    ),
                                    **audit,
                                }
                                rows.append(row)
                                print(json.dumps(row, sort_keys=True), flush=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[json.dumps(row["config"], sort_keys=True)].append(row)
    summaries = []
    for key, values in grouped.items():
        summaries.append(
            {
                "config": json.loads(key),
                **{
                    metric: float(np.mean([row[metric] for row in values]))
                    for metric in ("scale", "t2", "t3", "conflict", "macro", "worst")
                },
                "folds": len(values),
            }
        )
    summaries.sort(key=lambda row: (row["worst"], row["macro"]), reverse=True)
    report = {
        "schema_version": "kinofail.single-gmu-development.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "selection_rule": "maximize mean five-fold worst(Scale, Conflict), then Macro",
        "model_identity": "one 11-class GMU per fold; no battery input or battery-specific head",
        "summaries": summaries,
        "fold_results": rows,
    }
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": summaries[0]}, indent=2), flush=True)
    return 0


def freeze(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    config = TrainingConfig(
        visual_mode=args.visual_mode,
        conflict_mass=args.conflict_mass,
        epochs=args.epochs,
        method=args.method,
        hidden=args.hidden,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_learning_rate=args.pretrain_learning_rate,
        normalizer_scope=args.normalizer_scope,
        learning_rate=args.learning_rate,
        distillation_weight=args.distillation_weight,
        distillation_temperature=args.distillation_temperature,
        conflict_aux_weight=args.conflict_aux_weight,
    )
    source = combine(load_development(config.visual_mode))
    selected = np.ones(len(source["labels"]), dtype=bool)
    normalizer_mask = selected
    if config.normalizer_scope == "scale":
        normalizer_mask = source["batteries"] == "scale"
    visual_mean, visual_scale = fit_normalizer(source["visual"], normalizer_mask)
    proprio_mean, proprio_scale = fit_normalizer(source["proprio"], normalizer_mask)
    normalizer = {
        "visual_mean": visual_mean,
        "visual_scale": visual_scale,
        "proprio_mean": proprio_mean,
        "proprio_scale": proprio_scale,
    }
    states = []
    audits = []
    for seed in TRAINING_SEEDS:
        model, audit = train(source, selected, config, seed, normalizer)
        states.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
        audits.append({"seed": seed, **audit})
    output.mkdir(parents=True)
    checkpoint = output / "single_gmu.pt"
    torch.save(
        {
            "schema_version": "kinofail.single-gmu-checkpoint.v1",
            "config": asdict(config),
            "classes": list(CANONICAL_CLASSES),
            "dimensions": {
                "visual": int(source["visual"].shape[1]),
                "proprio": int(source["proprio"].shape[1]),
                "classes": len(CANONICAL_CLASSES),
            },
            "normalizer": normalizer,
            "states": states,
        },
        checkpoint,
    )
    manifest = {
        "schema_version": "kinofail.single-gmu-freeze.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "model_identity": (
            "one input schema, one 11-class output space, and one three-seed "
            "GMU ensemble for every diagnostic battery and deployment analysis"
        ),
        "battery_metadata_available_to_model": False,
        "battery_specific_heads": False,
        "config": asdict(config),
        "classes": list(CANONICAL_CLASSES),
        "development_counts": {
            "views": len(source["labels"]),
            "scale_views": int(np.sum(source["batteries"] == "scale")),
            "conflict_views": int(np.sum(source["batteries"] == "conflict")),
            "physical_groups": len(set(source["groups"].tolist())),
        },
        "training": audits,
        "checkpoint": str(checkpoint),
    }
    (output / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


def aggregate_all191(
    battery: str,
    source: dict[str, Any],
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(source["rows"]):
        if battery == "scale":
            unit = str(row["physical_episode_id"])
        else:
            unit = f"{row['case_id']}::{row['attribution_category']}"
        grouped[(unit, str(row["attribution_category"]))].append(index)
    rows = []
    for (unit, truth), indices in sorted(grouped.items()):
        if len(indices) != 3:
            raise RuntimeError(f"physical unit must have three renders: {unit}")
        probability = probabilities[indices].mean(axis=0)
        reference = source["rows"][indices[0]]
        rows.append(
            {
                "battery": battery,
                "unit_id": unit,
                "truth": truth,
                "prediction": CANONICAL_CLASSES[int(probability.argmax())],
                "confidence": float(probability.max()),
                "scene": str(reference["scene_cluster"]),
                "material": str(reference["cluster_material"]),
                "cell": str(reference.get("cell", "Scale")),
            }
        )
    return rows


def metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = np.asarray([str(row["truth"]) for row in rows])
    prediction = np.asarray([str(row["prediction"]) for row in rows])
    classes = sorted(set(truth.tolist()))
    recall = recall_score(truth, prediction, labels=classes, average=None)
    return {
        "physical_units": len(rows),
        "accuracy": float(np.mean(truth == prediction)),
        "balanced_accuracy": float(recall.mean()),
        "per_class_recall": {
            label: float(value) for label, value in zip(classes, recall, strict=True)
        },
    }


def score(args: argparse.Namespace) -> int:
    freeze_root = args.freeze.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    bundle = torch.load(freeze_root / "single_gmu.pt", map_location="cpu", weights_only=False)
    config = TrainingConfig(**bundle["config"])
    source = load_all191(args.all191.resolve(), config.visual_mode)
    normalizer = bundle["normalizer"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unit_rows: dict[str, list[dict[str, Any]]] = {}
    for battery in ("scale", "conflict"):
        visual = normalize(
            source[battery]["visual"],
            normalizer["visual_mean"],
            normalizer["visual_scale"],
        )
        proprio = normalize(
            source[battery]["proprio"],
            normalizer["proprio_mean"],
            normalizer["proprio_scale"],
        )
        seed_probability = []
        for state in bundle["states"]:
            model = build_model(visual.shape[1], config)
            model.load_state_dict(state)
            model.to(device)
            seed_probability.append(predict(model, visual, proprio, device))
        probability = np.mean(seed_probability, axis=0)
        unit_rows[battery] = aggregate_all191(battery, source[battery], probability)
    scale_metrics = metrics_from_rows(unit_rows["scale"])
    conflict_metrics = metrics_from_rows(unit_rows["conflict"])
    t2_metrics = metrics_from_rows(
        [row for row in unit_rows["conflict"] if row["cell"] == "T2_vision_decisive"]
    )
    t3_metrics = metrics_from_rows(
        [row for row in unit_rows["conflict"] if row["cell"] == "T3_proprio_decisive"]
    )
    cross = {
        "scale": scale_metrics["balanced_accuracy"],
        "t2": t2_metrics["balanced_accuracy"],
        "t3": t3_metrics["balanced_accuracy"],
        "conflict": conflict_metrics["balanced_accuracy"],
        "macro": 0.5
        * (scale_metrics["balanced_accuracy"] + conflict_metrics["balanced_accuracy"]),
        "worst": min(
            scale_metrics["balanced_accuracy"], conflict_metrics["balanced_accuracy"]
        ),
    }
    report = {
        "schema_version": "kinofail.single-gmu-all191-score.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "model_identity": (
            "the exact same frozen three-seed 11-class GMU ensemble is scored "
            "on Scale, T2, T3, and Conflict"
        ),
        "battery_metadata_available_to_model": False,
        "battery_specific_heads": False,
        "config": asdict(config),
        "metrics": {
            "scale": scale_metrics,
            "t2": t2_metrics,
            "t3": t3_metrics,
            "conflict": conflict_metrics,
            "cross_battery": cross,
        },
    }
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "physical_units.jsonl").open("w") as stream:
        for battery in ("scale", "conflict"):
            for row in unit_rows[battery]:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


def score_action(args: argparse.Namespace) -> int:
    freeze_root = args.freeze.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    with args.action_case_csv.resolve().open(newline="", encoding="utf-8") as stream:
        case_rows = list(csv.DictReader(stream))
    groups = {str(row["case_id"]).rsplit("__", 1)[-1] for row in case_rows}
    if len(case_rows) != 277 or len(groups) != 277:
        raise RuntimeError("action study must contain 277 unique physical groups")
    bundle = torch.load(
        freeze_root / "single_gmu.pt", map_location="cpu", weights_only=False
    )
    config = TrainingConfig(**bundle["config"])
    scale = load_all191(args.all191.resolve(), config.visual_mode)["scale"]
    indices = np.asarray(
        [
            index
            for index, row in enumerate(scale["rows"])
            if str(row["counterfactual_group_id"]) in groups
            and str(row["condition"]) == "anomaly"
            and str(row["appearance_intervention_id"]) == "primary"
        ],
        dtype=np.int64,
    )
    if len(indices) != 277:
        raise RuntimeError(f"expected 277 primary action inputs, found {len(indices)}")
    normalizer = bundle["normalizer"]
    visual = normalize(
        scale["visual"][indices],
        normalizer["visual_mean"],
        normalizer["visual_scale"],
    )
    proprio = normalize(
        scale["proprio"][indices],
        normalizer["proprio_mean"],
        normalizer["proprio_scale"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probabilities = []
    for state in bundle["states"]:
        model = build_model(visual.shape[1], config)
        model.load_state_dict(state)
        model.to(device)
        probabilities.append(predict(model, visual, proprio, device))
    probability = np.mean(probabilities, axis=0)
    prediction_rows = []
    for local, source_index in enumerate(indices.tolist()):
        source = scale["rows"][source_index]
        values = probability[local]
        prediction_rows.append(
            {
                "counterfactual_group_id": str(source["counterfactual_group_id"]),
                "sample_id": str(source["sample_id"]),
                "scene": str(source["scene_cluster"]),
                "material": str(source["cluster_material"]),
                "truth": str(source["attribution_category"]),
                "prediction": CANONICAL_CLASSES[int(values.argmax())],
                "confidence": float(values.max()),
                "class_probability": {
                    label: float(value)
                    for label, value in zip(
                        CANONICAL_CLASSES, values.tolist(), strict=True
                    )
                },
            }
        )
    if {row["counterfactual_group_id"] for row in prediction_rows} != groups:
        raise RuntimeError("action prediction join is not one-to-one")
    output.mkdir(parents=True)
    with (output / "primary_view_predictions.jsonl").open("w") as stream:
        for row in prediction_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "schema_version": "kinofail.single-gmu-action-predictions.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "model_identity": "same frozen 11-class GMU used by every diagnostic battery",
        "appearance_view": "primary robot-front view",
        "cases": len(prediction_rows),
        "exact_cause_accuracy": float(
            np.mean(
                [row["prediction"] == row["truth"] for row in prediction_rows]
            )
        ),
        "config": asdict(config),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="phase", required=True)
    develop_parser = sub.add_parser("develop")
    develop_parser.add_argument("--output", type=Path, default=DEFAULT_DEVELOPMENT)
    develop_parser.add_argument("--methods", nargs="+", choices=UNIFIED_METHODS, default=("gmu",))
    develop_parser.add_argument("--visual-modes", default="context,full")
    develop_parser.add_argument(
        "--conflict-masses", type=float, nargs="+", default=(0.55, 0.70, 0.85)
    )
    develop_parser.add_argument("--epochs", type=int, nargs="+", default=(12, 16, 20))
    develop_parser.add_argument("--hidden", type=int, nargs="+", default=(128,))
    develop_parser.add_argument("--pretrain-epochs", type=int, nargs="+", default=(0,))
    develop_parser.add_argument("--pretrain-learning-rate", type=float, default=1.0e-3)
    develop_parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    develop_parser.add_argument("--distillation-weight", type=float, default=0.0)
    develop_parser.add_argument("--distillation-temperature", type=float, default=2.0)
    develop_parser.add_argument("--conflict-aux-weight", type=float, default=0.0)
    develop_parser.add_argument(
        "--normalizer-scope", choices=("combined", "scale"), default="combined"
    )
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--output", type=Path, default=DEFAULT_FREEZE)
    freeze_parser.add_argument("--method", choices=UNIFIED_METHODS, default="gmu")
    freeze_parser.add_argument(
        "--visual-mode",
        choices=("context", "full", "dino", "dino_final"),
        required=True,
    )
    freeze_parser.add_argument("--conflict-mass", type=float, required=True)
    freeze_parser.add_argument("--epochs", type=int, required=True)
    freeze_parser.add_argument("--hidden", type=int, default=128)
    freeze_parser.add_argument("--pretrain-epochs", type=int, default=0)
    freeze_parser.add_argument("--pretrain-learning-rate", type=float, default=1.0e-3)
    freeze_parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    freeze_parser.add_argument("--distillation-weight", type=float, default=0.0)
    freeze_parser.add_argument("--distillation-temperature", type=float, default=2.0)
    freeze_parser.add_argument("--conflict-aux-weight", type=float, default=0.0)
    freeze_parser.add_argument(
        "--normalizer-scope", choices=("combined", "scale"), default="combined"
    )
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    score_parser.add_argument("--all191", type=Path, default=ALL191)
    score_parser.add_argument("--output", type=Path, default=DEFAULT_SCORE)
    action_parser = sub.add_parser("action")
    action_parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    action_parser.add_argument("--all191", type=Path, default=ALL191)
    action_parser.add_argument("--action-case-csv", type=Path, required=True)
    action_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "develop":
        return development(args)
    if args.phase == "freeze":
        return freeze(args)
    if args.phase == "score":
        return score(args)
    return score_action(args)


if __name__ == "__main__":
    raise SystemExit(main())
