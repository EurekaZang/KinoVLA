#!/usr/bin/env python3
"""Scene-disjoint diagnostic for non-privileged KiNO route features.

This script is development-only.  It compares route gates without changing the
attribution experts, so the architecture decision can be based on held-out
scene behavior rather than training fit.  Every candidate consumes only the
same RGB descriptor and proprioceptive summaries available at deployment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.conflict_invariant_kino import (  # noqa: E402
    invariant_relative_proprio_features,
    physical_group_weights,
    visual_context_features,
)
from scripts.develop_kinofail_conflict_invariant_kino_v4 import (  # noqa: E402
    _load_conflict,
    _load_scale,
)


def visual_summary_features(visual: np.ndarray) -> np.ndarray:
    """Return fixed low-dimensional statistics of the three CLIP blocks."""

    values = np.asarray(visual, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] % 3 != 0:
        raise ValueError("visual descriptor must contain three equal blocks")
    blocks = values.reshape(len(values), 3, values.shape[1] // 3)
    summaries = [
        blocks.mean(axis=2),
        blocks.std(axis=2),
        blocks.min(axis=2),
        blocks.max(axis=2),
        np.quantile(blocks, 0.25, axis=2),
        np.quantile(blocks, 0.75, axis=2),
        np.linalg.norm(blocks, axis=2),
    ]
    for left, right in ((0, 1), (0, 2), (1, 2)):
        delta = blocks[:, left] - blocks[:, right]
        denominator = np.maximum(
            np.linalg.norm(blocks[:, left], axis=1)
            * np.linalg.norm(blocks[:, right], axis=1),
            1.0e-8,
        )
        summaries.extend(
            [
                np.linalg.norm(delta, axis=1, keepdims=True),
                np.mean(np.abs(delta), axis=1, keepdims=True),
                np.max(np.abs(delta), axis=1, keepdims=True),
                (
                    np.sum(blocks[:, left] * blocks[:, right], axis=1)
                    / denominator
                )[:, None],
            ]
        )
    output = np.concatenate(summaries, axis=1).astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("visual summary produced non-finite values")
    return output


def _positive_probability(model: ExtraTreesClassifier, values: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(values)
    positive = int(np.flatnonzero(model.classes_ == 1)[0])
    return probabilities[:, positive]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=256)
    args = parser.parse_args()

    scale = _load_scale()
    conflict = _load_conflict()
    datasets = (scale, conflict)
    for source in datasets:
        source["proprio_route"] = invariant_relative_proprio_features(
            source["invariant"], source["full"]
        )
        source["visual_context"] = visual_context_features(source["visual"])
        source["visual_summary"] = visual_summary_features(source["visual"])

    variants = {
        "proprio": lambda source: source["proprio_route"],
        "proprio+visual_summary": lambda source: np.concatenate(
            [source["proprio_route"], source["visual_summary"]], axis=1
        ),
        "visual_context": lambda source: source["visual_context"],
        "proprio+visual_context": lambda source: np.concatenate(
            [source["proprio_route"], source["visual_context"]], axis=1
        ),
    }
    report: dict[str, object] = {
        "status": "development_only",
        "folding": "scene suffix modulo five",
        "trees": int(args.trees),
        "variants": {},
    }
    thresholds = np.linspace(0.20, 0.80, 25)
    for variant, transform in variants.items():
        rows: list[dict[str, object]] = []
        for fold in range(5):
            scale_train = scale["folds"] != fold
            conflict_train = conflict["folds"] != fold
            train_values = np.concatenate(
                [
                    transform(scale)[scale_train],
                    transform(conflict)[conflict_train],
                ],
                axis=0,
            )
            train_targets = np.concatenate(
                [
                    np.zeros(int(scale_train.sum()), dtype=np.int8),
                    conflict["vision_decisive"][conflict_train].astype(np.int8),
                ]
            )
            train_groups = np.concatenate(
                [scale["groups"][scale_train], conflict["groups"][conflict_train]]
            )
            gate = ExtraTreesClassifier(
                n_estimators=int(args.trees),
                max_features="sqrt",
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=2026080300 + fold,
                n_jobs=-1,
            ).fit(
                train_values,
                train_targets,
                sample_weight=physical_group_weights(train_groups),
            )
            for dataset_name, source in (("scale", scale), ("conflict", conflict)):
                test = source["folds"] == fold
                scores = _positive_probability(gate, transform(source)[test])
                expected = (
                    np.zeros(int(test.sum()), dtype=bool)
                    if dataset_name == "scale"
                    else source["vision_decisive"][test]
                )
                cells = (
                    np.full(int(test.sum()), "scale", dtype=object)
                    if dataset_name == "scale"
                    else source["cells"][test]
                )
                for index, score in enumerate(scores):
                    rows.append(
                        {
                            "fold": fold,
                            "dataset": dataset_name,
                            "cell": str(cells[index]),
                            "expected_vision": bool(expected[index]),
                            "score": float(score),
                        }
                    )
        scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
        expected = np.asarray([row["expected_vision"] for row in rows], dtype=bool)
        dataset = np.asarray([row["dataset"] for row in rows])
        cells = np.asarray([row["cell"] for row in rows])
        sweep: list[dict[str, object]] = []
        for threshold in thresholds:
            predicted = scores >= threshold
            scale_mask = dataset == "scale"
            t2_mask = cells == "T2_vision_decisive"
            t3_mask = cells == "T3_proprio_decisive"
            scale_specificity = float(np.mean(~predicted[scale_mask]))
            t2_sensitivity = float(np.mean(predicted[t2_mask]))
            t3_specificity = float(np.mean(~predicted[t3_mask]))
            battery_macro = float(
                np.mean(
                    [
                        scale_specificity,
                        0.5 * (t2_sensitivity + t3_specificity),
                    ]
                )
            )
            minimax = float(
                min(scale_specificity, t2_sensitivity, t3_specificity)
            )
            sweep.append(
                {
                    "threshold": float(threshold),
                    "scale_specificity": scale_specificity,
                    "t2_sensitivity": t2_sensitivity,
                    "t3_specificity": t3_specificity,
                    "battery_macro": battery_macro,
                    "minimax": minimax,
                    "scale_false_vision": int(predicted[scale_mask].sum()),
                    "t2_missed": int((~predicted[t2_mask]).sum()),
                    "t3_false_vision": int(predicted[t3_mask].sum()),
                }
            )
        best_macro = max(sweep, key=lambda row: (row["battery_macro"], row["minimax"]))
        best_minimax = max(sweep, key=lambda row: (row["minimax"], row["battery_macro"]))
        at_half = min(sweep, key=lambda row: abs(row["threshold"] - 0.5))
        report["variants"][variant] = {
            "dimensions": int(transform(scale).shape[1]),
            "threshold_0_5": at_half,
            "best_battery_macro": best_macro,
            "best_minimax": best_minimax,
        }
        print(json.dumps({variant: report["variants"][variant]}, sort_keys=True), flush=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
