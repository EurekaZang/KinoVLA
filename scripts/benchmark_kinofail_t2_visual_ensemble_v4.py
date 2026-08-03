#!/usr/bin/env python3
"""Scene-disjoint ensemble screen for frozen visual T2 specialists."""

from __future__ import annotations

import itertools
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


TERRAIN = ROOT / "outputs/eval/kino_t2_dinov2_448_terrain_v4_development/features.npz"
FULL = ROOT / "outputs/eval/kino_t2_dinov2_448_full_v4_development/features.npz"
ORIGINAL = ROOT / "outputs/eval/kino_t2_dinov2_v4_development/features.npz"
OUTPUT = ROOT / "outputs/eval/kino_t2_visual_ensemble_v4_development/report.json"


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(
        np.mean(
            [
                np.mean(predictions[labels == label] == label)
                for label in ("adhesion", "compliant_terrain")
            ]
        )
    )


def _load(path: Path, ids: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["sample_ids"].astype(str), ids):
            raise RuntimeError(f"feature cache does not align: {path}")
        return {
            name.removeprefix("visual_"): np.asarray(archive[name], dtype=np.float32)
            for name in archive.files
            if name.startswith("visual_")
        }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    conflict = _load_conflict()
    selected = conflict["vision_decisive"]
    ids = conflict["sample_ids"][selected]
    terrain = _load(TERRAIN, ids)
    full = _load(FULL, ids)
    original = _load(ORIGINAL, ids)
    clip = visual_context_features(conflict["visual"][selected])
    features = {
        "terrain_all": np.concatenate(
            [terrain["cls"], terrain["ground_mean"], terrain["ground_max"]],
            axis=1,
        ),
        "terrain_ground_max": terrain["ground_max"],
        "terrain_clip_ground_mean": np.concatenate(
            [clip, terrain["ground_mean"]], axis=1
        ),
        "full_clip_ground_mean": np.concatenate(
            [clip, full["ground_mean"]], axis=1
        ),
        "original_clip_cls": np.concatenate([clip, original["cls"]], axis=1),
    }
    regularization = {
        "terrain_all": 0.01,
        "terrain_ground_max": 0.01,
        "terrain_clip_ground_mean": 0.01,
        "full_clip_ground_mean": 0.01,
        "original_clip_cls": 0.01,
    }
    labels = conflict["labels"][selected]
    groups = conflict["groups"][selected]
    folds = conflict["folds"][selected]
    positive = "compliant_terrain"
    probabilities: dict[str, np.ndarray] = {}
    individual: dict[str, object] = {}
    for name, values in features.items():
        output = np.empty(len(labels), dtype=np.float64)
        per_fold = []
        for fold in range(5):
            train = folds != fold
            test = ~train
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=regularization[name],
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
            class_index = list(model.classes_).index(positive)
            output[test] = model.predict_proba(values[test])[:, class_index]
            fold_prediction = np.where(
                output[test] >= 0.5, positive, "adhesion"
            )
            per_fold.append(_balanced_accuracy(labels[test], fold_prediction))
        prediction = np.where(output >= 0.5, positive, "adhesion")
        probabilities[name] = output
        individual[name] = {
            "balanced_accuracy": _balanced_accuracy(labels, prediction),
            "per_fold": per_fold,
            "worst_fold": float(min(per_fold)),
        }
        print(json.dumps({name: individual[name]}, sort_keys=True), flush=True)

    ensembles = []
    names = sorted(probabilities)
    for size in range(2, len(names) + 1):
        for members in itertools.combinations(names, size):
            probability = np.mean([probabilities[name] for name in members], axis=0)
            prediction = np.where(probability >= 0.5, positive, "adhesion")
            per_fold = [
                _balanced_accuracy(labels[folds == fold], prediction[folds == fold])
                for fold in range(5)
            ]
            ensembles.append(
                {
                    "members": list(members),
                    "balanced_accuracy": _balanced_accuracy(labels, prediction),
                    "per_fold": per_fold,
                    "worst_fold": float(min(per_fold)),
                }
            )
    ensembles.sort(
        key=lambda row: (row["balanced_accuracy"], row["worst_fold"]),
        reverse=True,
    )
    report = {
        "schema_version": "kinofail.t2-visual-ensemble-v4-development.v1",
        "status": "five_fold_scene_disjoint_development_complete",
        "confirmatory_evidence": False,
        "individual": individual,
        "best_ensembles": ensembles[:10],
        "selection_note": (
            "Ensemble membership is a development choice and requires a new "
            "frozen confirmation set."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
