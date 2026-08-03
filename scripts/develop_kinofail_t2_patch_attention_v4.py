#!/usr/bin/env python3
"""Five-fold development of a frozen-DINO patch-attention T2 specialist."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.conflict_invariant_kino import physical_group_weights  # noqa: E402
from scripts.develop_kinofail_conflict_invariant_kino_v4 import (  # noqa: E402
    _load_conflict,
)


CACHE = Path(
    "/data/eureka/KinoVLA/outputs/eval/kino_t2_dinov2_patches_v4_development"
)


class GatedPatchAttention(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        *,
        attention_dimension: int = 96,
        value_dimension: int = 128,
        heads: int = 4,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(input_dimension)
        self.attention_tanh = torch.nn.Linear(input_dimension, attention_dimension)
        self.attention_sigmoid = torch.nn.Linear(input_dimension, attention_dimension)
        self.attention_out = torch.nn.Linear(attention_dimension, heads)
        self.value = torch.nn.Linear(input_dimension, value_dimension)
        self.cls_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dimension),
            torch.nn.Linear(input_dimension, value_dimension),
            torch.nn.GELU(),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm((heads + 1) * value_dimension),
            torch.nn.Dropout(dropout),
            torch.nn.Linear((heads + 1) * value_dimension, value_dimension),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(value_dimension, 1),
        )

    def forward(self, patches: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        tokens = self.norm(patches)
        attention = self.attention_out(
            torch.tanh(self.attention_tanh(tokens))
            * torch.sigmoid(self.attention_sigmoid(tokens))
        )
        attention = torch.softmax(attention.transpose(1, 2), dim=2)
        values = self.value(tokens)
        pooled = torch.matmul(attention, values).flatten(1)
        global_context = self.cls_projection(cls)
        return self.classifier(torch.cat([pooled, global_context], dim=1)).squeeze(1)


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(
        np.mean(
            [
                np.mean(predictions[labels == value] == value)
                for value in (0, 1)
            ]
        )
    )


def _predict(
    model: GatedPatchAttention,
    patches: np.ndarray,
    cls: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            patch_tensor = torch.from_numpy(np.asarray(patches[batch])).to(
                "cuda", dtype=torch.float32, non_blocking=True
            )
            cls_tensor = torch.from_numpy(np.asarray(cls[batch])).to(
                "cuda", dtype=torch.float32, non_blocking=True
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                score = model(patch_tensor, cls_tensor)
            scores.append(score.float().cpu().numpy())
    return np.concatenate(scores)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--token-dropout", type=float, default=0.10)
    args = parser.parse_args()

    conflict = _load_conflict()
    t2 = conflict["vision_decisive"]
    sample_ids = np.load(CACHE / "sample_ids.npy").astype(str)
    if not np.array_equal(sample_ids, conflict["sample_ids"][t2]):
        raise RuntimeError("patch cache does not align with T2 records")
    print("loading 3.3 GiB patch cache into RAM", flush=True)
    patches = np.asarray(np.load(CACHE / "final_patches.npy", mmap_mode="r")).copy()
    cls = np.asarray(np.load(CACHE / "final_cls.npy", mmap_mode="r")).copy()
    labels = (conflict["labels"][t2] == "adhesion").astype(np.int64)
    folds = conflict["folds"][t2]
    groups = conflict["groups"][t2]
    predictions = np.empty(len(labels), dtype=np.int64)
    scores = np.empty(len(labels), dtype=np.float32)
    fold_reports: list[dict[str, object]] = []
    torch.backends.cuda.matmul.allow_tf32 = True

    for fold in range(5):
        seed = 2026080300 + fold
        torch.manual_seed(seed)
        np_rng = np.random.default_rng(seed)
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        weights = physical_group_weights(groups[train]).astype(np.float32)
        weights /= weights.mean()
        model = GatedPatchAttention(int(patches.shape[2])).to("cuda")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(args.epochs))
        )
        for epoch in range(int(args.epochs)):
            model.train()
            order = np_rng.permutation(len(train))
            total_loss = 0.0
            for start in range(0, len(train), int(args.batch_size)):
                locations = order[start : start + int(args.batch_size)]
                batch = train[locations]
                patch_tensor = torch.from_numpy(np.asarray(patches[batch])).to(
                    "cuda", dtype=torch.float32, non_blocking=True
                )
                cls_tensor = torch.from_numpy(np.asarray(cls[batch])).to(
                    "cuda", dtype=torch.float32, non_blocking=True
                )
                target = torch.from_numpy(labels[batch].astype(np.float32)).to(
                    "cuda", non_blocking=True
                )
                sample_weight = torch.from_numpy(weights[locations]).to(
                    "cuda", non_blocking=True
                )
                if args.token_dropout > 0.0:
                    keep = torch.rand(
                        patch_tensor.shape[:2], device="cuda"
                    ) >= float(args.token_dropout)
                    patch_tensor = patch_tensor * keep.unsqueeze(2)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logit = model(patch_tensor, cls_tensor)
                    losses = F.binary_cross_entropy_with_logits(
                        logit, target, reduction="none"
                    )
                    loss = torch.mean(losses * sample_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.detach()) * len(batch)
            scheduler.step()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    json.dumps(
                        {
                            "fold": fold,
                            "epoch": epoch + 1,
                            "train_loss": total_loss / len(train),
                        }
                    ),
                    flush=True,
                )
        fold_scores = _predict(
            model,
            patches,
            cls,
            test,
            batch_size=max(int(args.batch_size), 128),
        )
        fold_predictions = (fold_scores >= 0.0).astype(np.int64)
        scores[test] = fold_scores
        predictions[test] = fold_predictions
        result = {
            "fold": fold,
            "balanced_accuracy": _balanced_accuracy(
                labels[test], fold_predictions
            ),
        }
        fold_reports.append(result)
        print(json.dumps(result), flush=True)

    report = {
        "status": "five_fold_scene_disjoint_development",
        "architecture": "frozen DINOv2 final-frame gated patch attention",
        "hyperparameters": vars(args),
        "balanced_accuracy": _balanced_accuracy(labels, predictions),
        "per_fold": fold_reports,
        "worst_fold": min(
            float(row["balanced_accuracy"]) for row in fold_reports
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
