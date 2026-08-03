#!/usr/bin/env python3
"""Five-fold scene-disjoint screen of frozen DINOv2 T2 descriptors."""

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


DINO = ROOT / "outputs/eval/kino_t2_dinov2_v4_development/features.npz"


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
    parser.add_argument("--c-values", default="0.003,0.01,0.03,0.1,0.3")
    parser.add_argument("--dino", type=Path, default=DINO)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    c_values = [float(value) for value in args.c_values.split(",")]
    conflict = _load_conflict()
    t2 = conflict["vision_decisive"]
    dino_path = args.dino.resolve()
    with np.load(dino_path, allow_pickle=False) as archive:
        if not np.array_equal(
            archive["sample_ids"].astype(str), conflict["sample_ids"][t2]
        ):
            raise RuntimeError("DINO feature cache does not align with T2 records")
        dino = {
            name: np.asarray(archive[f"visual_{name}"], dtype=np.float32)
            for name in (
                "cls",
                "patch_mean",
                "ground_mean",
                "ground_max",
                "route_mean",
                "route_max",
            )
        }
    clip = visual_context_features(conflict["visual"][t2])
    feature_sets = dict(dino)
    feature_sets.update(
        {
            "ground_mean+ground_max": np.concatenate(
                [dino["ground_mean"], dino["ground_max"]], axis=1
            ),
            "route_mean+route_max": np.concatenate(
                [dino["route_mean"], dino["route_max"]], axis=1
            ),
            "cls+ground_mean": np.concatenate(
                [dino["cls"], dino["ground_mean"]], axis=1
            ),
            "cls+ground_max": np.concatenate(
                [dino["cls"], dino["ground_max"]], axis=1
            ),
            "cls+ground_mean+ground_max": np.concatenate(
                [dino["cls"], dino["ground_mean"], dino["ground_max"]], axis=1
            ),
            "clip+cls": np.concatenate([clip, dino["cls"]], axis=1),
            "clip+ground_mean": np.concatenate(
                [clip, dino["ground_mean"]], axis=1
            ),
            "clip+ground_max": np.concatenate(
                [clip, dino["ground_max"]], axis=1
            ),
        }
    )
    labels = conflict["labels"][t2]
    groups = conflict["groups"][t2]
    folds = conflict["folds"][t2]
    report: dict[str, object] = {
        "status": "five_fold_scene_disjoint_development",
        "dino_features": str(dino_path),
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
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
