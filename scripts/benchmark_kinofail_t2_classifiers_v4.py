#!/usr/bin/env python3
"""Scene-disjoint classifier screen on the best CLIP+DINO T2 descriptor."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


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
    conflict = _load_conflict()
    t2 = conflict["vision_decisive"]
    with np.load(DINO, allow_pickle=False) as archive:
        if not np.array_equal(
            archive["sample_ids"].astype(str), conflict["sample_ids"][t2]
        ):
            raise RuntimeError("DINO feature cache does not align with T2 records")
        dino_cls = np.asarray(archive["visual_cls"], dtype=np.float32)
    values = np.concatenate(
        [visual_context_features(conflict["visual"][t2]), dino_cls], axis=1
    )
    labels = conflict["labels"][t2]
    groups = conflict["groups"][t2]
    folds = conflict["folds"][t2]
    candidates: list[tuple[str, object, str | None]] = [
        (
            "logistic_c0.01",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=0.01,
                            class_weight="balanced",
                            max_iter=2_000,
                        ),
                    ),
                ]
            ),
            "classifier__sample_weight",
        ),
    ]
    for c_value in (0.0003, 0.001, 0.003, 0.01, 0.03):
        candidates.append(
            (
                f"linear_svc_c{c_value}",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "classifier",
                            LinearSVC(
                                C=float(c_value),
                                class_weight="balanced",
                                dual="auto",
                                max_iter=10_000,
                            ),
                        ),
                    ]
                ),
                "classifier__sample_weight",
            )
        )
    for max_features in ("sqrt", 0.2):
        for leaf in (2, 4, 8):
            candidates.append(
                (
                    f"extra_trees_mf{max_features}_leaf{leaf}",
                    ExtraTreesClassifier(
                        n_estimators=512,
                        max_features=max_features,
                        min_samples_leaf=leaf,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                    "sample_weight",
                )
            )

    report: dict[str, object] = {
        "status": "five_fold_scene_disjoint_development",
        "features": "CLIP mean+final plus DINOv2 CLS mean+final",
        "dimensions": int(values.shape[1]),
        "classifiers": {},
    }
    for name, estimator, weight_argument in candidates:
        predictions = np.empty(len(labels), dtype=object)
        per_fold: list[float] = []
        for fold in range(5):
            train = folds != fold
            test = ~train
            estimator.set_params(
                **{
                    key: 2026080300 + fold
                    for key in estimator.get_params()
                    if key.endswith("random_state")
                }
            )
            fit_kwargs = (
                {weight_argument: physical_group_weights(groups[train])}
                if weight_argument is not None
                else {}
            )
            estimator.fit(values[train], labels[train], **fit_kwargs)
            predictions[test] = estimator.predict(values[test])
            per_fold.append(_balanced_accuracy(labels[test], predictions[test]))
        result = {
            "balanced_accuracy": _balanced_accuracy(labels, predictions),
            "per_fold": per_fold,
            "worst_fold": float(min(per_fold)),
        }
        report["classifiers"][name] = result
        print(json.dumps({name: result}, sort_keys=True), flush=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
