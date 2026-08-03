#!/usr/bin/env python3
"""Nested scene-disjoint screen of a nonlinear RGB-only T2 specialist.

The outer fold is used exactly once for evaluation.  One of the remaining
scene folds selects the epoch count, after which the model is re-fit on all
four outer-training folds.  Camera profile and scene identifiers are never
model inputs; the optional camera GroupDRO objective uses calibration groups
only while fitting to discourage a weak-view failure mode.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.conflict_invariant_kino import visual_context_features  # noqa: E402
from scripts.develop_kinofail_conflict_invariant_kino_v4 import (  # noqa: E402
    _load_conflict,
)


DEFAULT_DINO = (
    ROOT / "outputs/eval/kino_t2_dinov2_448_full_v4_development/features.npz"
)


class Specialist(nn.Module):
    def __init__(self, dimension: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(
        np.mean(
            [np.mean(predictions[labels == value] == value) for value in (0, 1)]
        )
    )


def _fit(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_groups: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    *,
    hidden: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int | None,
    group_dro_eta: float,
    select_best: bool,
    seed: int,
) -> tuple[Specialist, int, float]:
    _seed(seed)
    device = torch.device("cuda")
    model = Specialist(train_x.shape[1], hidden, dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    dataset = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_y),
        torch.from_numpy(train_groups),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    group_count = int(train_groups.max()) + 1
    group_weights = torch.full((group_count,), 1.0 / group_count, device=device)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -np.inf
    best_epoch = 0
    stale = 0
    eval_tensor = torch.from_numpy(eval_x).to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y, batch_groups in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_groups = batch_groups.to(device, non_blocking=True)
            losses = nn.functional.cross_entropy(
                model(batch_x), batch_y, reduction="none"
            )
            if group_dro_eta > 0.0:
                group_losses = torch.zeros(group_count, device=device)
                present = torch.zeros(group_count, dtype=torch.bool, device=device)
                for group in range(group_count):
                    selected = batch_groups == group
                    if selected.any():
                        group_losses[group] = losses[selected].mean()
                        present[group] = True
                with torch.no_grad():
                    group_weights[present] *= torch.exp(
                        group_dro_eta * group_losses[present].detach()
                    )
                    group_weights /= group_weights.sum()
                loss = (group_weights[present] * group_losses[present]).sum()
                loss /= group_weights[present].sum().clamp_min(1.0e-8)
            else:
                loss = losses.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if select_best:
            model.eval()
            with torch.inference_mode():
                prediction = model(eval_tensor).argmax(dim=1).cpu().numpy()
            score = _balanced_accuracy(eval_y, prediction)
            if score > best_score + 1.0e-8:
                best_score = score
                best_epoch = epoch
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if patience is not None and stale >= patience:
                break
    if select_best and best_state is not None:
        model.load_state_dict(best_state)
    if not select_best:
        best_epoch = epochs
        best_score = float("nan")
    return model, best_epoch, float(best_score)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino", type=Path, default=DEFAULT_DINO)
    parser.add_argument(
        "--feature",
        choices=(
            "ground_mean",
            "route_mean",
            "cls_ground_mean",
            "cls_ground_mean_max",
            "clip_ground_mean",
        ),
        default="clip_ground_mean",
    )
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--group-dro-eta", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    conflict = _load_conflict()
    selected = conflict["vision_decisive"]
    with np.load(args.dino.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(
            archive["sample_ids"].astype(str), conflict["sample_ids"][selected]
        ):
            raise RuntimeError("DINO feature cache does not align with T2 records")
        cls = np.asarray(archive["visual_cls"], dtype=np.float32)
        ground = np.asarray(archive["visual_ground_mean"], dtype=np.float32)
        ground_max = np.asarray(archive["visual_ground_max"], dtype=np.float32)
        route = np.asarray(archive["visual_route_mean"], dtype=np.float32)
    clip = visual_context_features(conflict["visual"][selected])
    features = {
        "ground_mean": ground,
        "route_mean": route,
        "cls_ground_mean": np.concatenate([cls, ground], axis=1),
        "cls_ground_mean_max": np.concatenate(
            [cls, ground, ground_max], axis=1
        ),
        "clip_ground_mean": np.concatenate([clip, ground], axis=1),
    }[args.feature].astype(np.float32)
    labels_text = conflict["labels"][selected]
    labels = (labels_text == "compliant_terrain").astype(np.int64)
    folds = conflict["folds"][selected]
    rows = np.asarray(conflict["rows"], dtype=object)[selected]
    profiles = sorted({str(row["camera_profile"]) for row in rows})
    profile_index = {value: index for index, value in enumerate(profiles)}
    groups = np.asarray(
        [2 * profile_index[str(row["camera_profile"])] + int(label) for row, label in zip(rows, labels, strict=True)],
        dtype=np.int64,
    )

    predictions = np.empty(len(labels), dtype=np.int64)
    folds_report: list[dict[str, object]] = []
    for outer in range(5):
        inner = (outer + 1) % 5
        tune_train = (folds != outer) & (folds != inner)
        tune_valid = folds == inner
        outer_train = folds != outer
        outer_test = folds == outer
        tune_scaler = StandardScaler().fit(features[tune_train])
        _, best_epoch, validation_score = _fit(
            tune_scaler.transform(features[tune_train]).astype(np.float32),
            labels[tune_train],
            groups[tune_train],
            tune_scaler.transform(features[tune_valid]).astype(np.float32),
            labels[tune_valid],
            hidden=int(args.hidden),
            dropout=float(args.dropout),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            epochs=int(args.max_epochs),
            patience=int(args.patience),
            group_dro_eta=float(args.group_dro_eta),
            select_best=True,
            seed=2026080300 + outer,
        )
        scaler = StandardScaler().fit(features[outer_train])
        # Re-fit on every allowed outer-training scene using the epoch count
        # selected without consulting the outer test fold.
        model, _, _ = _fit(
            scaler.transform(features[outer_train]).astype(np.float32),
            labels[outer_train],
            groups[outer_train],
            scaler.transform(features[tune_valid]).astype(np.float32),
            labels[tune_valid],
            hidden=int(args.hidden),
            dropout=float(args.dropout),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            epochs=max(1, best_epoch),
            patience=None,
            group_dro_eta=float(args.group_dro_eta),
            select_best=False,
            seed=2026081300 + outer,
        )
        model.eval()
        with torch.inference_mode():
            test_x = torch.from_numpy(
                scaler.transform(features[outer_test]).astype(np.float32)
            ).cuda()
            predictions[outer_test] = model(test_x).argmax(dim=1).cpu().numpy()
        fold_score = _balanced_accuracy(labels[outer_test], predictions[outer_test])
        row = {
            "outer_fold": outer,
            "inner_validation_fold": inner,
            "selected_epochs": best_epoch,
            "inner_balanced_accuracy": validation_score,
            "outer_balanced_accuracy": fold_score,
        }
        folds_report.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    report = {
        "schema_version": "kinofail.t2-mlp-v4-development.v1",
        "status": "nested_scene_disjoint_development_complete",
        "confirmatory_evidence": False,
        "deployment_metadata": [],
        "training_only_groups": (
            ["camera_profile", "class"] if args.group_dro_eta > 0.0 else []
        ),
        "dino": str(args.dino.resolve()),
        "feature": args.feature,
        "dimension": int(features.shape[1]),
        "hyperparameters": {
            "hidden": int(args.hidden),
            "dropout": float(args.dropout),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "group_dro_eta": float(args.group_dro_eta),
        },
        "folds": folds_report,
        "balanced_accuracy": _balanced_accuracy(labels, predictions),
        "worst_fold": float(min(row["outer_balanced_accuracy"] for row in folds_report)),
        "per_class_recall": {
            "adhesion": float(np.mean(predictions[labels == 0] == 0)),
            "compliant_terrain": float(np.mean(predictions[labels == 1] == 1)),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
