#!/usr/bin/env python3
"""Five-fold scene-disjoint screen of T2 visual feature combinations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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


ROI = ROOT / "outputs/eval/kino_t2_roi_clip_v4_development/features.npz"


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(
        np.mean(
            [
                np.mean(predictions[labels == label] == label)
                for label in ("adhesion", "compliant_terrain")
            ]
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c-values", default="0.01,0.03,0.1,0.3")
    args = parser.parse_args()
    c_values = [float(value) for value in args.c_values.split(",")]
    conflict = _load_conflict()
    t2 = conflict["vision_decisive"]
    t2_ids = conflict["sample_ids"][t2]
    with np.load(ROI, allow_pickle=False) as archive:
        roi_ids = archive["sample_ids"].astype(str)
        if not np.array_equal(roi_ids, t2_ids):
            raise RuntimeError("ROI feature cache does not align with T2 records")
        ground = np.asarray(archive["visual_ground"], dtype=np.float32)
        route = np.asarray(archive["visual_route"], dtype=np.float32)
    global_visual = conflict["visual"][t2]
    geometry = conflict["local_visual"][t2]
    context = {
        "global": visual_context_features(global_visual),
        "ground": visual_context_features(ground),
        "route": visual_context_features(route),
    }
    feature_sets = {
        "global": context["global"],
        "ground": context["ground"],
        "route": context["route"],
        "global+ground": np.concatenate(
            [context["global"], context["ground"]], axis=1
        ),
        "global+route": np.concatenate(
            [context["global"], context["route"]], axis=1
        ),
        "ground+route": np.concatenate(
            [context["ground"], context["route"]], axis=1
        ),
        "global+ground+route": np.concatenate(
            [context["global"], context["ground"], context["route"]], axis=1
        ),
        "geometry": geometry,
        "global+geometry": np.concatenate(
            [context["global"], geometry], axis=1
        ),
        "ground+geometry": np.concatenate(
            [context["ground"], geometry], axis=1
        ),
        "route+geometry": np.concatenate([context["route"], geometry], axis=1),
    }
    labels = conflict["labels"][t2]
    groups = conflict["groups"][t2]
    folds = conflict["folds"][t2]
    report: dict[str, object] = {
        "status": "five_fold_scene_disjoint_development",
        "c_values": c_values,
        "features": {},
    }
    for name, values in feature_sets.items():
        candidates: list[dict[str, object]] = []
        for c_value in c_values:
            predictions = np.empty(len(labels), dtype=object)
            per_fold: list[float] = []
            for fold in range(5):
                train = folds != fold
                test = ~train
                model = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "classifier",
                            LogisticRegression(
                                C=float(c_value),
                                class_weight="balanced",
                                max_iter=2_000,
                                random_state=2026080300 + fold,
                            ),
                        ),
                    ]
                )
                model.fit(
                    values[train],
                    labels[train],
                    classifier__sample_weight=physical_group_weights(groups[train]),
                )
                predictions[test] = model.predict(values[test])
                per_fold.append(_balanced_accuracy(labels[test], predictions[test]))
            candidates.append(
                {
                    "C": float(c_value),
                    "balanced_accuracy": _balanced_accuracy(labels, predictions),
                    "per_fold": per_fold,
                    "worst_fold": float(min(per_fold)),
                }
            )
        best = max(
            candidates,
            key=lambda row: (row["balanced_accuracy"], row["worst_fold"]),
        )
        report["features"][name] = {
            "dimensions": int(values.shape[1]),
            "best": best,
            "candidates": candidates,
        }
        print(json.dumps({name: report["features"][name]}, sort_keys=True), flush=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
