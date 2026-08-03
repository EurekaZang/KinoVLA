#!/usr/bin/env python3
"""Scene-disjoint last-block DINOv2 fine-tuning for the T2 specialist.

This is a development screen.  The outer scene fold is never used for epoch
selection; the next scene fold is the validation fold and the other three
folds are used for fitting.  Only RGB pixels are accepted by the deployed
model.  Scene, camera, material, operator, and case identifiers remain outside
the forward signature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2
from transformers import AutoModel


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.develop_kinofail_conflict_invariant_kino_v4 import (  # noqa: E402
    _load_conflict,
)


MODEL_ID = "facebook/dinov2-base"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


class T2Frames(Dataset[tuple[torch.Tensor, int, int]]):
    def __init__(
        self,
        rows: list[dict[str, object]],
        labels: np.ndarray,
        indices: np.ndarray,
        *,
        image_size: int,
        training: bool,
    ) -> None:
        self.rows = rows
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        shortest = int(round(float(image_size) * 256.0 / 224.0))
        transforms: list[nn.Module] = [
            v2.Resize(shortest, interpolation=InterpolationMode.BICUBIC),
            v2.CenterCrop(image_size),
        ]
        if training:
            transforms.extend(
                [
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomApply(
                        [v2.ColorJitter(0.12, 0.12, 0.08, 0.03)], p=0.5
                    ),
                ]
            )
        transforms.extend(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self.transform = v2.Compose(transforms)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int, int]:
        index = int(self.indices[item])
        row = self.rows[index]
        source = Path(str(row["source_case_dir"]))
        path = source / str(row["rgb_paths"][-1])
        with Image.open(path) as image:
            frame = image.convert("RGB")
            crop = dict(row["rgb_crop"])
            x_fraction = crop["x_fraction"]
            x0 = int(round(float(x_fraction[0]) * frame.width))
            x1 = int(round(float(x_fraction[1]) * frame.width))
            y0 = int(crop["pixel_y_start"])
            frame = frame.crop((x0, y0, x1, frame.height))
            tensor = self.transform(frame)
        return tensor, int(self.labels[index]), index


class FineTunedSpecialist(nn.Module):
    def __init__(self, *, unfreeze_blocks: int) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            MODEL_ID, local_files_only=True
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for block in self.backbone.encoder.layer[-unfreeze_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for parameter in self.backbone.layernorm.parameters():
            parameter.requires_grad_(True)
        hidden = int(self.backbone.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(3 * hidden),
            nn.Dropout(0.1),
            nn.Linear(3 * hidden, 2),
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(pixel_values=pixels).last_hidden_state
        cls = tokens[:, 0]
        patches = tokens[:, 1:]
        grid = int(round(float(patches.shape[1]) ** 0.5))
        patch_grid = patches.reshape(len(pixels), grid, grid, patches.shape[-1])
        ground = patch_grid[:, int(round(0.40 * grid)) :, :, :]
        descriptor = torch.cat(
            [cls, ground.mean(dim=(1, 2)), ground.amax(dim=(1, 2))], dim=1
        )
        return self.classifier(descriptor)


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(
        np.mean(
            [np.mean(predictions[labels == value] == value) for value in (0, 1)]
        )
    )


def _evaluate(
    model: FineTunedSpecialist,
    loader: DataLoader,
    labels: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    predictions = np.empty(len(labels), dtype=np.int64)
    seen = np.zeros(len(labels), dtype=bool)
    probabilities = np.empty(len(labels), dtype=np.float32)
    with torch.inference_mode():
        for pixels, _targets, indices in loader:
            pixels = pixels.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(pixels)
            probability = logits.softmax(dim=1)[:, 1].float().cpu().numpy()
            index = indices.numpy()
            probabilities[index] = probability
            predictions[index] = (probability >= 0.5).astype(np.int64)
            seen[index] = True
    selected = np.flatnonzero(seen)
    return (
        _balanced_accuracy(labels[selected], predictions[selected]),
        predictions,
        probabilities,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--backbone-lr", type=float, default=1.0e-5)
    parser.add_argument("--head-lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.outer_fold not in range(5):
        raise ValueError("outer fold must be in [0, 4]")
    if args.image_size <= 0 or args.image_size % 14 != 0:
        raise ValueError("image size must be a positive multiple of 14")
    if args.unfreeze_blocks not in range(1, 13):
        raise ValueError("unfreeze blocks must be in [1, 12]")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    _seed(2026080300 + args.outer_fold)
    conflict = _load_conflict()
    selected = conflict["vision_decisive"]
    rows = [
        dict(row)
        for row, keep in zip(conflict["rows"], selected, strict=True)
        if keep
    ]
    labels_text = conflict["labels"][selected]
    labels = (labels_text == "compliant_terrain").astype(np.int64)
    folds = conflict["folds"][selected]
    outer = int(args.outer_fold)
    inner = (outer + 1) % 5
    indices = {
        "train": np.flatnonzero((folds != outer) & (folds != inner)),
        "validation": np.flatnonzero(folds == inner),
        "test": np.flatnonzero(folds == outer),
    }
    datasets = {
        split: T2Frames(
            rows,
            labels,
            values,
            image_size=int(args.image_size),
            training=split == "train",
        )
        for split, values in indices.items()
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=split == "train",
            num_workers=int(args.workers),
            pin_memory=True,
            persistent_workers=int(args.workers) > 0,
            drop_last=False,
        )
        for split, dataset in datasets.items()
    }
    model = FineTunedSpecialist(
        unfreeze_blocks=int(args.unfreeze_blocks)
    ).cuda()
    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("backbone") and parameter.requires_grad
    ]
    head_parameters = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": float(args.backbone_lr)},
            {"params": head_parameters, "lr": float(args.head_lr)},
        ],
        weight_decay=float(args.weight_decay),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = -np.inf
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        for pixels, targets, _indices in loaders["train"]:
            pixels = pixels.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = nn.functional.cross_entropy(model(pixels), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation, _, _ = _evaluate(model, loaders["validation"], labels)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_balanced_accuracy": validation,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if validation > best_validation + 1.0e-8:
            best_validation = validation
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(args.patience):
            break
    if best_state is None:
        raise RuntimeError("fine-tuning did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_score, test_prediction, test_probability = _evaluate(
        model, loaders["test"], labels
    )

    output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output / "checkpoint.pt"
    torch.save(
        {
            "state_dict": best_state,
            "model_id": MODEL_ID,
            "image_size": int(args.image_size),
            "unfreeze_blocks": int(args.unfreeze_blocks),
        },
        checkpoint_path,
    )
    prediction_path = output / "predictions.npz"
    test = indices["test"]
    np.savez_compressed(
        prediction_path,
        sample_ids=conflict["sample_ids"][selected][test],
        truth=labels_text[test],
        prediction=np.where(
            test_prediction[test] == 1, "compliant_terrain", "adhesion"
        ),
        compliant_probability=test_probability[test],
    )
    report = {
        "schema_version": "kinofail.t2-dinov2-finetune-v4-development.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "single_outer_fold_development_complete",
        "confirmatory_evidence": False,
        "outer_fold": outer,
        "validation_fold": inner,
        "fit_folds": sorted(set(range(5)) - {outer, inner}),
        "counts": {name: len(value) for name, value in indices.items()},
        "deployment_inputs": ["fixed lower-55-percent robot-front RGB crop"],
        "forbidden_deployment_inputs": [
            "scene/camera/material/operator/case ID",
            "battery/cell ID",
            "ground truth or outcome",
        ],
        "model": {
            "encoder": MODEL_ID,
            "image_size": int(args.image_size),
            "unfreeze_blocks": int(args.unfreeze_blocks),
            "descriptor": ["CLS", "ground patch mean", "ground patch max"],
            "backbone_lr": float(args.backbone_lr),
            "head_lr": float(args.head_lr),
            "weight_decay": float(args.weight_decay),
        },
        "selection": {
            "best_epoch": best_epoch,
            "validation_balanced_accuracy": best_validation,
            "history": history,
        },
        "test_balanced_accuracy": test_score,
        "test_per_class_recall": {
            "adhesion": float(
                np.mean(test_prediction[test][labels[test] == 0] == 0)
            ),
            "compliant_terrain": float(
                np.mean(test_prediction[test][labels[test] == 1] == 1)
            ),
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "predictions": str(prediction_path),
            "predictions_sha256": _sha256(prediction_path),
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
