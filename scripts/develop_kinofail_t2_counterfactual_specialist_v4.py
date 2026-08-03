#!/usr/bin/env python3
"""Develop a counterfactual visual specialist for Kino-Fail T2.

Each T2 case contains an adhesion/compliant pair rendered in the same scene,
material, camera, and appearance view.  The specialist uses these pairings only
during training: a ranking loss suppresses shared scene appearance, while the
deployed classifier still consumes a single five-frame RGB descriptor.

All reported values are five-fold scene-disjoint development estimates on F42.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.conflict_invariant_kino import (  # noqa: E402
    physical_group_weights,
    visual_context_features,
)
from scripts.develop_kinofail_conflict_invariant_kino_v4 import (  # noqa: E402
    _load_conflict,
)


@dataclass(frozen=True)
class PairIndex:
    adhesion: np.ndarray
    compliant: np.ndarray
    scene: np.ndarray


def _pairs(rows: list[dict[str, object]], selected: np.ndarray) -> PairIndex:
    buckets: dict[tuple[str, str], dict[str, int]] = {}
    for index in np.flatnonzero(selected):
        row = rows[int(index)]
        key = (str(row["case_id"]), str(row["appearance_view_id"]))
        buckets.setdefault(key, {})[str(row["attribution_category"])] = int(index)
    if not buckets or any(
        set(pair) != {"adhesion", "compliant_terrain"} for pair in buckets.values()
    ):
        raise RuntimeError("T2 counterfactual pairs are incomplete")
    keys = sorted(buckets)
    return PairIndex(
        adhesion=np.asarray([buckets[key]["adhesion"] for key in keys]),
        compliant=np.asarray(
            [buckets[key]["compliant_terrain"] for key in keys]
        ),
        scene=np.asarray(
            [str(rows[buckets[key]["adhesion"]]["scene_cluster"]) for key in keys]
        ),
    )


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(
        np.mean(
            [
                np.mean(predictions[labels == label] == label)
                for label in ("adhesion", "compliant_terrain")
            ]
        )
    )


def _fit_counterfactual_linear(
    values: np.ndarray,
    pairs: PairIndex,
    *,
    ranking_weight: float,
    centering_weight: float,
    group_dro_eta: float,
    weight_decay: float,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit a single-image linear classifier with paired training losses."""

    torch.manual_seed(int(seed))
    mean = values[np.concatenate([pairs.adhesion, pairs.compliant])].mean(
        axis=0, dtype=np.float64
    ).astype(np.float32)
    scale = values[np.concatenate([pairs.adhesion, pairs.compliant])].std(
        axis=0, dtype=np.float64
    ).astype(np.float32)
    scale = np.maximum(scale, 1.0e-4)
    standardized = (values - mean) / scale
    adhesion = torch.from_numpy(standardized[pairs.adhesion]).to(device)
    compliant = torch.from_numpy(standardized[pairs.compliant]).to(device)
    unique_scenes, scene_index = np.unique(pairs.scene, return_inverse=True)
    scene_index_tensor = torch.from_numpy(scene_index).to(device)
    scene_weights = torch.full(
        (len(unique_scenes),), 1.0 / len(unique_scenes), device=device
    )
    linear = torch.nn.Linear(values.shape[1], 1).to(device)
    torch.nn.init.zeros_(linear.weight)
    torch.nn.init.zeros_(linear.bias)
    optimizer = torch.optim.AdamW(
        linear.parameters(), lr=0.03, weight_decay=float(weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(epochs))
    )
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        adhesion_logit = linear(adhesion).squeeze(1)
        compliant_logit = linear(compliant).squeeze(1)
        pair_loss = (
            F.softplus(-adhesion_logit)
            + F.softplus(compliant_logit)
            + float(ranking_weight)
            * F.softplus(1.0 - (adhesion_logit - compliant_logit))
            + float(centering_weight)
            * torch.square(0.5 * (adhesion_logit + compliant_logit))
        )
        scene_losses = torch.stack(
            [pair_loss[scene_index_tensor == index].mean() for index in range(len(unique_scenes))]
        )
        if group_dro_eta > 0.0:
            with torch.no_grad():
                scene_weights *= torch.exp(float(group_dro_eta) * scene_losses.detach())
                scene_weights /= scene_weights.sum()
        loss = torch.sum(scene_weights * scene_losses)
        loss.backward()
        optimizer.step()
        scheduler.step()
    weight = linear.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    bias = float(linear.bias.detach().cpu().item())
    return weight, bias, mean, scale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    conflict = _load_conflict()
    t2 = conflict["vision_decisive"]
    values = visual_context_features(conflict["visual"])
    labels = conflict["labels"]
    configurations = [
        {
            "name": "paired_bce",
            "ranking_weight": 0.0,
            "centering_weight": 0.0,
            "group_dro_eta": 0.0,
            "weight_decay": 1.0e-3,
        },
        {
            "name": "paired_rank_0.25",
            "ranking_weight": 0.25,
            "centering_weight": 0.0,
            "group_dro_eta": 0.0,
            "weight_decay": 1.0e-3,
        },
        {
            "name": "paired_rank_1",
            "ranking_weight": 1.0,
            "centering_weight": 0.0,
            "group_dro_eta": 0.0,
            "weight_decay": 1.0e-3,
        },
        {
            "name": "paired_rank_1_center_0.01",
            "ranking_weight": 1.0,
            "centering_weight": 0.01,
            "group_dro_eta": 0.0,
            "weight_decay": 1.0e-3,
        },
        {
            "name": "paired_rank_1_center_0.1",
            "ranking_weight": 1.0,
            "centering_weight": 0.1,
            "group_dro_eta": 0.0,
            "weight_decay": 1.0e-3,
        },
        {
            "name": "paired_rank_1_groupdro",
            "ranking_weight": 1.0,
            "centering_weight": 0.0,
            "group_dro_eta": 0.01,
            "weight_decay": 1.0e-3,
        },
        {
            "name": "paired_rank_1_center_groupdro",
            "ranking_weight": 1.0,
            "centering_weight": 0.01,
            "group_dro_eta": 0.01,
            "weight_decay": 1.0e-3,
        },
    ]
    report: dict[str, object] = {
        "status": "five_fold_scene_disjoint_development",
        "device": str(device),
        "epochs": int(args.epochs),
        "configurations": {},
    }

    # Reproduce the current specialist in the same script as a stable reference.
    reference_predictions = np.empty(len(labels), dtype=object)
    for fold in range(5):
        train = t2 & (conflict["folds"] != fold)
        test = t2 & (conflict["folds"] == fold)
        classifier = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=2026080300 + fold,
                    ),
                ),
            ]
        )
        classifier.fit(
            values[train],
            labels[train],
            classifier__sample_weight=physical_group_weights(conflict["groups"][train]),
        )
        reference_predictions[test] = classifier.predict(values[test])
    report["reference_logistic"] = {
        "balanced_accuracy": _balanced_accuracy(labels[t2], reference_predictions[t2]),
        "per_fold": [
            _balanced_accuracy(
                labels[t2 & (conflict["folds"] == fold)],
                reference_predictions[t2 & (conflict["folds"] == fold)],
            )
            for fold in range(5)
        ],
    }
    print(json.dumps({"reference_logistic": report["reference_logistic"]}), flush=True)

    for configuration in configurations:
        predictions = np.empty(len(labels), dtype=object)
        fold_values: list[float] = []
        for fold in range(5):
            train = t2 & (conflict["folds"] != fold)
            test = t2 & (conflict["folds"] == fold)
            pairs = _pairs(conflict["rows"], train)
            weight, bias, mean, scale = _fit_counterfactual_linear(
                values,
                pairs,
                ranking_weight=float(configuration["ranking_weight"]),
                centering_weight=float(configuration["centering_weight"]),
                group_dro_eta=float(configuration["group_dro_eta"]),
                weight_decay=float(configuration["weight_decay"]),
                epochs=int(args.epochs),
                seed=2026080300 + fold,
                device=device,
            )
            score = ((values[test] - mean) / scale) @ weight + bias
            predictions[test] = np.where(
                score >= 0.0, "adhesion", "compliant_terrain"
            )
            fold_values.append(_balanced_accuracy(labels[test], predictions[test]))
        result = {
            **configuration,
            "balanced_accuracy": _balanced_accuracy(labels[t2], predictions[t2]),
            "per_fold": fold_values,
            "worst_fold": float(min(fold_values)),
        }
        report["configurations"][str(configuration["name"])] = result
        print(json.dumps({str(configuration["name"]): result}), flush=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
