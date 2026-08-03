#!/usr/bin/env python3
"""Scene-disjoint screen of learned non-privileged KiNO route gates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.conflict_invariant_kino import (  # noqa: E402
    invariant_relative_proprio_features,
    physical_group_weights,
)
from scripts.develop_kinofail_conflict_invariant_kino_v4 import (  # noqa: E402
    _load_conflict,
    _load_scale,
)


def main() -> int:
    scale = _load_scale()
    conflict = _load_conflict()
    for source in (scale, conflict):
        source["route_features"] = invariant_relative_proprio_features(
            source["invariant"], source["full"]
        )
    configurations = [
        {"max_leaf_nodes": 15, "l2_regularization": 0.1},
        {"max_leaf_nodes": 15, "l2_regularization": 1.0},
        {"max_leaf_nodes": 31, "l2_regularization": 0.1},
        {"max_leaf_nodes": 31, "l2_regularization": 1.0},
    ]
    thresholds = np.linspace(0.05, 0.50, 46)
    report: dict[str, object] = {
        "status": "five_fold_scene_disjoint_development",
        "classifiers": {},
    }
    for configuration in configurations:
        rows: list[tuple[str, str, bool, float]] = []
        for fold in range(5):
            scale_train = scale["folds"] != fold
            conflict_train = conflict["folds"] != fold
            values = np.concatenate(
                [
                    scale["route_features"][scale_train],
                    conflict["route_features"][conflict_train],
                ]
            )
            targets = np.concatenate(
                [
                    np.zeros(int(scale_train.sum()), dtype=np.int8),
                    conflict["vision_decisive"][conflict_train].astype(np.int8),
                ]
            )
            groups = np.concatenate(
                [scale["groups"][scale_train], conflict["groups"][conflict_train]]
            )
            model = HistGradientBoostingClassifier(
                learning_rate=0.10,
                max_iter=200,
                max_leaf_nodes=int(configuration["max_leaf_nodes"]),
                min_samples_leaf=20,
                l2_regularization=float(configuration["l2_regularization"]),
                class_weight="balanced",
                early_stopping=False,
                random_state=2026080300 + fold,
            ).fit(values, targets, sample_weight=physical_group_weights(groups))
            positive = int(np.flatnonzero(model.classes_ == 1)[0])
            for dataset_name, source in (("scale", scale), ("conflict", conflict)):
                test = source["folds"] == fold
                scores = model.predict_proba(source["route_features"][test])[:, positive]
                cells = (
                    np.full(int(test.sum()), "scale", dtype=object)
                    if dataset_name == "scale"
                    else source["cells"][test]
                )
                expected = (
                    np.zeros(int(test.sum()), dtype=bool)
                    if dataset_name == "scale"
                    else source["vision_decisive"][test]
                )
                rows.extend(
                    (dataset_name, str(cell), bool(target), float(score))
                    for cell, target, score in zip(cells, expected, scores, strict=True)
                )
        dataset = np.asarray([row[0] for row in rows])
        cells = np.asarray([row[1] for row in rows])
        expected = np.asarray([row[2] for row in rows])
        scores = np.asarray([row[3] for row in rows])
        sweep = []
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
            sweep.append(
                {
                    "threshold": float(threshold),
                    "scale_specificity": scale_specificity,
                    "t2_sensitivity": t2_sensitivity,
                    "t3_specificity": t3_specificity,
                    "battery_macro": battery_macro,
                    "minimax": float(
                        min(scale_specificity, t2_sensitivity, t3_specificity)
                    ),
                    "scale_false_vision": int(predicted[scale_mask].sum()),
                    "t2_missed": int((~predicted[t2_mask]).sum()),
                    "t3_false_vision": int(predicted[t3_mask].sum()),
                }
            )
        name = (
            f"hist_leaf{configuration['max_leaf_nodes']}_"
            f"l2{configuration['l2_regularization']}"
        )
        report["classifiers"][name] = {
            "best_macro": max(
                sweep, key=lambda row: (row["battery_macro"], row["minimax"])
            ),
            "best_minimax": max(
                sweep, key=lambda row: (row["minimax"], row["battery_macro"])
            ),
        }
        print(json.dumps({name: report["classifiers"][name]}), flush=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
