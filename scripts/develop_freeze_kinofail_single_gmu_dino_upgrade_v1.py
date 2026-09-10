#!/usr/bin/env python3
"""Upgrade the retained Scale GMU with DINOv2 and freeze one 11-class model.

The retained Scale checkpoint is a strong 11-class GMU over CLIP context and
proprioception.  This program embeds that exact function into a larger GMU by
zero-initializing only the new DINOv2 columns, then jointly calibrates the
single posterior on independent Scale and Conflict development observations.
No battery identifier, task-specific head, or inference-time specialist is
introduced.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_kinofail_single_gmu_v1 import (
    CANONICAL_CLASSES,
    TRAINING_SEEDS,
    TrainingConfig,
    build_model,
    combine,
    load_development,
    normalize,
    physical_metrics,
    predict,
    train,
    training_weights,
)
from scripts.evaluate_kinofail_known_fusion_baselines_v1 import (
    _load_development as load_retained_development,
)
from kino_vla.eval.conflict_invariant_kino import (
    invariant_relative_proprio_features,
    visual_context_features,
)


BASE = ROOT / "outputs/freeze/kinofail_known_fusion_baselines_f0/known_fusion_baselines.pt"


@dataclass(frozen=True)
class UpgradeConfig:
    conflict_mass: float
    epochs: int
    learning_rate: float
    scale_consistency: float
    weight_decay: float = 1.0e-4
    batch_size: int = 1024


def base_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = torch.load(BASE, map_location="cpu", weights_only=False)
    saved = bundle["datasets"]["scale"]
    if saved["classes"] != list(CANONICAL_CLASSES):
        raise RuntimeError("retained Scale checkpoint uses a different ontology")
    if saved["dimensions"] != {"visual": 1024, "proprio": 251, "classes": 11}:
        raise RuntimeError("retained Scale checkpoint uses an unexpected input schema")
    return bundle, saved


def make_normalizer(
    data: dict[str, np.ndarray], selected: np.ndarray, saved: dict[str, Any]
) -> dict[str, np.ndarray]:
    base = saved["normalizer"]
    detail = np.asarray(data["visual"][selected, 1024:], dtype=np.float64)
    detail_mean = detail.mean(axis=0).astype(np.float32)
    detail_scale = detail.std(axis=0).astype(np.float32)
    detail_scale[detail_scale < 1.0e-6] = 1.0
    return {
        "visual_mean": np.concatenate(
            [np.asarray(base["visual_mean"], dtype=np.float32), detail_mean]
        ),
        "visual_scale": np.concatenate(
            [np.asarray(base["visual_scale"], dtype=np.float32), detail_scale]
        ),
        "proprio_mean": np.asarray(base["proprio_mean"], dtype=np.float32),
        "proprio_scale": np.asarray(base["proprio_scale"], dtype=np.float32),
    }


def load_anchor_data() -> dict[str, np.ndarray]:
    """Combine the retained Scale anchor with real-DINO development samples."""

    retained = load_retained_development()["scale"]
    real_dino = load_development("dino")
    scale_dino = real_dino["scale"]
    conflict = real_dino["conflict"]
    detail = np.concatenate(
        [scale_dino["visual"][:, 1024:], conflict["visual"][:, 1024:]], axis=0
    )
    detail_mean = detail.mean(axis=0).astype(np.float32)
    retained_visual = np.concatenate(
        [
            visual_context_features(retained["visual"]),
            np.broadcast_to(detail_mean, (len(retained["labels"]), len(detail_mean))),
        ],
        axis=1,
    ).astype(np.float32)
    retained_scale = {
        "visual": retained_visual,
        "proprio": invariant_relative_proprio_features(
            retained["invariant"], retained["full"]
        ),
        "labels": np.asarray(retained["labels"]).astype(str),
        "groups": np.asarray(retained["groups"]).astype(str),
        "scenes": np.asarray(retained["scenes"]).astype(str),
        "folds": np.asarray(retained["folds"], dtype=np.int64),
        "cells": np.asarray(["Scale"] * len(retained["labels"])),
    }
    joined_scale = {
        key: np.concatenate([retained_scale[key], scale_dino[key]])
        for key in retained_scale
    }
    return combine({"scale": joined_scale, "conflict": conflict})


def expand_state(old: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    config = TrainingConfig(
        visual_mode="dino", conflict_mass=0.7, epochs=1, hidden=128, method="gmu"
    )
    state = build_model(3072, config).state_dict()
    for key in (
        "visual.bias",
        "proprio.weight",
        "proprio.bias",
        "gate.bias",
        "classifier.0.weight",
        "classifier.0.bias",
        "classifier.2.weight",
        "classifier.2.bias",
    ):
        state[key].copy_(old[key])
    state["visual.weight"].zero_()
    state["visual.weight"][:, :1024].copy_(old["visual.weight"])
    state["gate.weight"].zero_()
    state["gate.weight"][:, :1024].copy_(old["gate.weight"][:, :1024])
    state["gate.weight"][:, 3072:].copy_(old["gate.weight"][:, 1024:])
    return state


def train_one(
    data: dict[str, np.ndarray],
    selected: np.ndarray,
    normalizer: dict[str, np.ndarray],
    old_state: dict[str, torch.Tensor],
    config: UpgradeConfig,
    seed: int,
) -> tuple[nn.Module, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    visual = normalize(data["visual"], normalizer["visual_mean"], normalizer["visual_scale"])
    proprio = normalize(
        data["proprio"], normalizer["proprio_mean"], normalizer["proprio_scale"]
    )
    lookup = {label: index for index, label in enumerate(CANONICAL_CLASSES)}
    targets = np.asarray([lookup[str(label)] for label in data["labels"]], dtype=np.int64)
    weights = training_weights(
        data["labels"], data["groups"], data["batteries"], selected, config.conflict_mass
    )
    indices = np.flatnonzero(selected)
    dataset = TensorDataset(
        torch.from_numpy(visual[indices]),
        torch.from_numpy(proprio[indices]),
        torch.from_numpy(targets[indices]),
        torch.from_numpy(weights[indices]),
        torch.from_numpy((data["batteries"][indices] == "scale").astype(np.bool_)),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    student_config = TrainingConfig(
        visual_mode="dino",
        conflict_mass=config.conflict_mass,
        epochs=config.epochs,
        hidden=128,
        method="gmu",
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    teacher_config = TrainingConfig(
        visual_mode="context", conflict_mass=0.0, epochs=1, hidden=128, method="gmu"
    )
    student = build_model(3072, student_config).to(device)
    student.load_state_dict(expand_state(old_state))
    teacher = build_model(1024, teacher_config).to(device)
    teacher.load_state_dict(old_state)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    final_loss = float("nan")
    for _ in range(config.epochs):
        student.train()
        numerator = 0.0
        denominator = 0.0
        for visual_batch, proprio_batch, target_batch, weight_batch, scale_batch in loader:
            visual_batch = visual_batch.to(device, non_blocking=True)
            proprio_batch = proprio_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            weight_batch = weight_batch.to(device, non_blocking=True)
            scale_batch = scale_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = student(visual_batch, proprio_batch)
            per_item = nn.functional.cross_entropy(logits, target_batch, reduction="none")
            loss = (per_item * weight_batch).sum() / weight_batch.sum()
            if config.scale_consistency > 0.0 and bool(scale_batch.any()):
                with torch.no_grad():
                    teacher_probability = torch.softmax(
                        teacher(visual_batch[scale_batch, :1024], proprio_batch[scale_batch]),
                        dim=1,
                    )
                consistency = nn.functional.kl_div(
                    torch.log_softmax(logits[scale_batch], dim=1),
                    teacher_probability,
                    reduction="batchmean",
                )
                loss = loss + config.scale_consistency * consistency
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()
            numerator += float((per_item * weight_batch).sum().detach().cpu())
            denominator += float(weight_batch.sum().detach().cpu())
        final_loss = numerator / denominator
    return student, final_loss


def evaluate_fold(
    data: dict[str, np.ndarray],
    mask: np.ndarray,
    model: nn.Module,
    normalizer: dict[str, np.ndarray],
) -> dict[str, float]:
    device = next(model.parameters()).device
    probability = predict(
        model,
        normalize(data["visual"], normalizer["visual_mean"], normalizer["visual_scale"])[mask],
        normalize(data["proprio"], normalizer["proprio_mean"], normalizer["proprio_scale"])[mask],
        device,
    )
    labels = data["labels"][mask]
    groups = data["groups"][mask]
    batteries = data["batteries"][mask]
    cells = data["cells"][mask]
    scale = physical_metrics(probability, labels, groups, batteries == "scale")["balanced_accuracy"]
    conflict = physical_metrics(probability, labels, groups, batteries == "conflict")["balanced_accuracy"]
    t2 = physical_metrics(probability, labels, groups, cells == "T2_vision_decisive")["balanced_accuracy"]
    t3 = physical_metrics(probability, labels, groups, cells == "T3_proprio_decisive")["balanced_accuracy"]
    return {
        "scale": scale,
        "t2": t2,
        "t3": t3,
        "conflict": conflict,
        "macro": 0.5 * (scale + conflict),
        "worst": min(scale, conflict),
    }


def develop(output: Path) -> int:
    if output.exists():
        raise FileExistsError(output)
    _, saved = base_bundle()
    data = combine(load_development("dino"))
    configs = [
        UpgradeConfig(mass, epochs, learning_rate, consistency)
        for mass in (0.55, 0.70)
        for epochs in (6, 12)
        for learning_rate in (1.0e-4, 3.0e-4)
        for consistency in (0.5, 1.0)
    ]
    results = []
    for config in configs:
        for fold in range(5):
            train_mask = data["folds"] != fold
            test_mask = ~train_mask
            normalizer = make_normalizer(data, train_mask, saved)
            model, loss = train_one(
                data,
                train_mask,
                normalizer,
                saved["states"]["gmu"][fold % len(saved["states"]["gmu"])],
                config,
                2026082500 + fold,
            )
            row = {"config": asdict(config), "fold": fold, "loss": loss, **evaluate_fold(data, test_mask, model, normalizer)}
            results.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    summaries = []
    for config in configs:
        rows = [row for row in results if row["config"] == asdict(config)]
        summaries.append(
            {
                "config": asdict(config),
                **{
                    key: float(np.mean([row[key] for row in rows]))
                    for key in ("scale", "t2", "t3", "conflict", "macro", "worst")
                },
            }
        )
    summaries.sort(key=lambda row: (row["worst"], row["macro"]), reverse=True)
    report = {
        "schema_version": "kinofail.single-gmu-dino-upgrade-development.v1",
        "selection_rule": "maximize five-fold mean worst(Scale, Conflict), then Macro",
        "model_identity": "one 11-class DINOv2 GMU initialized from the retained Scale GMU",
        "summaries": summaries,
        "fold_results": results,
    }
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": summaries[0]}, indent=2), flush=True)
    return 0


def freeze(development: Path, output: Path) -> int:
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((development / "report.json").read_text())
    selected = UpgradeConfig(**report["summaries"][0]["config"])
    _, saved = base_bundle()
    data = combine(load_development("dino"))
    mask = np.ones(len(data["labels"]), dtype=bool)
    normalizer = make_normalizer(data, mask, saved)
    states = []
    losses = []
    for seed, old_state in zip(TRAINING_SEEDS, saved["states"]["gmu"], strict=True):
        model, loss = train_one(data, mask, normalizer, old_state, selected, seed)
        states.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
        losses.append({"seed": seed, "loss": loss})
    training_config = TrainingConfig(
        visual_mode="dino",
        conflict_mass=selected.conflict_mass,
        epochs=selected.epochs,
        hidden=128,
        method="gmu",
        learning_rate=selected.learning_rate,
        weight_decay=selected.weight_decay,
    )
    output.mkdir(parents=True)
    torch.save(
        {
            "schema_version": "kinofail.single-gmu-checkpoint.v1",
            "config": asdict(training_config),
            "classes": list(CANONICAL_CLASSES),
            "dimensions": {"visual": 3072, "proprio": 251, "classes": 11},
            "normalizer": normalizer,
            "states": states,
        },
        output / "single_gmu.pt",
    )
    manifest = {
        "schema_version": "kinofail.single-gmu-dino-upgrade-freeze.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "model_identity": "one input schema, one GMU trunk, one 11-class posterior, one frozen ensemble",
        "battery_metadata_available_to_model": False,
        "battery_specific_heads": False,
        "initialization": "retained 11-class Scale GMU; DINOv2 columns initialized to zero",
        "selected_development_config": asdict(selected),
        "training": losses,
    }
    (output / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


def freeze_anchor(output: Path, selected: UpgradeConfig | None = None) -> int:
    """Freeze the fixed old-framework upgrade with the full Scale anchor."""

    if output.exists():
        raise FileExistsError(output)
    if selected is None:
        selected = UpgradeConfig(
            conflict_mass=0.70,
            epochs=12,
            learning_rate=1.0e-4,
            scale_consistency=1.0,
        )
    _, saved = base_bundle()
    data = load_anchor_data()
    mask = np.ones(len(data["labels"]), dtype=bool)
    normalizer = make_normalizer(data, mask, saved)
    states = []
    losses = []
    for seed, old_state in zip(TRAINING_SEEDS, saved["states"]["gmu"], strict=True):
        model, loss = train_one(data, mask, normalizer, old_state, selected, seed)
        states.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
        losses.append({"seed": seed, "loss": loss})
    training_config = TrainingConfig(
        visual_mode="dino",
        conflict_mass=selected.conflict_mass,
        epochs=selected.epochs,
        hidden=128,
        method="gmu",
        learning_rate=selected.learning_rate,
        weight_decay=selected.weight_decay,
    )
    output.mkdir(parents=True)
    torch.save(
        {
            "schema_version": "kinofail.single-gmu-checkpoint.v1",
            "config": asdict(training_config),
            "classes": list(CANONICAL_CLASSES),
            "dimensions": {"visual": 3072, "proprio": 251, "classes": 11},
            "normalizer": normalizer,
            "states": states,
        },
        output / "single_gmu.pt",
    )
    manifest = {
        "schema_version": "kinofail.single-gmu-dino-anchor-freeze.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "model_identity": "one input schema, one GMU trunk, one 11-class posterior, one frozen ensemble",
        "battery_metadata_available_to_model": False,
        "battery_specific_heads": False,
        "initialization": "retained 11-class Scale GMU; DINOv2 columns initialized to zero",
        "scale_anchor_views": int(np.sum(data["batteries"] == "scale")),
        "conflict_views": int(np.sum(data["batteries"] == "conflict")),
        "fixed_development_config": asdict(selected),
        "training": losses,
    }
    (output / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


def freeze_baseline(method: str, output: Path) -> int:
    """Fit a reference model on the same anchored DINOv2 input contract."""

    if output.exists():
        raise FileExistsError(output)
    _, saved = base_bundle()
    data = load_anchor_data()
    mask = np.ones(len(data["labels"]), dtype=bool)
    normalizer = make_normalizer(data, mask, saved)
    config = TrainingConfig(
        visual_mode="dino",
        conflict_mass=0.70,
        epochs=20,
        hidden=128,
        method=method,
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
    )
    states = []
    losses = []
    for seed in TRAINING_SEEDS:
        model, audit = train(data, mask, config, seed, normalizer)
        states.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
        losses.append({"seed": seed, **audit})
    output.mkdir(parents=True)
    torch.save(
        {
            "schema_version": "kinofail.single-gmu-checkpoint.v1",
            "config": asdict(config),
            "classes": list(CANONICAL_CLASSES),
            "dimensions": {"visual": 3072, "proprio": 251, "classes": 11},
            "normalizer": normalizer,
            "states": states,
        },
        output / "single_gmu.pt",
    )
    manifest = {
        "schema_version": "kinofail.unified-dino-anchor-baseline-freeze.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "method": method,
        "input_contract": "common 3072-D CLIP+DINOv2 visual and 251-D proprioception",
        "battery_metadata_available_to_model": False,
        "battery_specific_heads": False,
        "config": asdict(config),
        "training": losses,
    }
    (output / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="phase", required=True)
    develop_parser = sub.add_parser("develop")
    develop_parser.add_argument("--output", type=Path, required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--development", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    anchor_parser = sub.add_parser("freeze-anchor")
    anchor_parser.add_argument("--output", type=Path, required=True)
    anchor_parser.add_argument("--conflict-mass", type=float, default=0.70)
    anchor_parser.add_argument("--epochs", type=int, default=12)
    anchor_parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    anchor_parser.add_argument("--scale-consistency", type=float, default=1.0)
    baseline_parser = sub.add_parser("freeze-baseline")
    baseline_parser.add_argument("--method", type=str, required=True)
    baseline_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "develop":
        return develop(args.output.resolve())
    if args.phase == "freeze-anchor":
        return freeze_anchor(
            args.output.resolve(),
            UpgradeConfig(
                conflict_mass=args.conflict_mass,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                scale_consistency=args.scale_consistency,
            ),
        )
    if args.phase == "freeze-baseline":
        return freeze_baseline(args.method, args.output.resolve())
    return freeze(args.development.resolve(), args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
