#!/usr/bin/env python3
"""Development-screen and formally evaluate structured bidirectional routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "vision_only",
    "proprio_only",
    "early_fusion",
    "late_fusion",
    "structured_evidence_router",
)
BASELINES = METHODS[:-1]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _weights(case_ids: np.ndarray) -> np.ndarray:
    values, counts = np.unique(case_ids.astype(str), return_counts=True)
    lookup = dict(zip(values.tolist(), counts.tolist(), strict=True))
    return np.asarray([1.0 / lookup[value] for value in case_ids], dtype=np.float64)


def _pipeline(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=2026072412,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _family_gate() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=256,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=2026072412,
        n_jobs=-1,
    )


def _model_classes(model: Any) -> list[str]:
    if hasattr(model, "named_steps"):
        values = model.named_steps["classifier"].classes_
    else:
        values = model.classes_
    return [str(value) for value in values]


def _aligned(model: Any, values: np.ndarray, classes: list[str]) -> np.ndarray:
    raw = np.asarray(model.predict_proba(values), dtype=np.float64)
    source = {label: index for index, label in enumerate(_model_classes(model))}
    out = np.zeros((len(values), len(classes)), dtype=np.float64)
    for index, label in enumerate(classes):
        if label in source:
            out[:, index] = raw[:, source[label]]
    denominator = out.sum(axis=1, keepdims=True)
    return out / np.where(denominator <= 0.0, 1.0, denominator)


class FlatModel:
    def __init__(self, model: Any, feature_key: str, classes: list[str]):
        self.model = model
        self.feature_key = feature_key
        self.classes = classes

    def predict_proba(
        self, visual: np.ndarray, proprio: np.ndarray
    ) -> np.ndarray:
        values = {
            "vision": visual,
            "proprio": proprio,
            "joint": np.concatenate([visual, proprio], axis=1),
        }[self.feature_key]
        return _aligned(self.model, values, self.classes)


class LateFusionModel:
    def __init__(self, vision: FlatModel, proprio: FlatModel):
        self.vision = vision
        self.proprio = proprio
        self.classes = vision.classes

    def predict_proba(
        self, visual: np.ndarray, proprio: np.ndarray
    ) -> np.ndarray:
        return 0.5 * (
            self.vision.predict_proba(visual, proprio)
            + self.proprio.predict_proba(visual, proprio)
        )


class CentroidProbabilityExpert:
    """Shrinkage nearest-centroid expert with distance-derived probabilities."""

    def __init__(self, shrink_threshold: float = 0.1):
        self.scale = StandardScaler()
        self.model = NearestCentroid(shrink_threshold=shrink_threshold)
        self.classes_: np.ndarray | None = None

    def fit(self, values: np.ndarray, labels: np.ndarray) -> "CentroidProbabilityExpert":
        transformed = self.scale.fit_transform(values)
        self.model.fit(transformed, labels)
        self.classes_ = np.asarray(self.model.classes_)
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("centroid expert has not been fitted")
        transformed = self.scale.transform(values)
        distance = np.square(
            transformed[:, None, :] - self.model.centroids_[None, :, :]
        ).mean(axis=2)
        logits = -distance
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        return probability / probability.sum(axis=1, keepdims=True)


class StructuredEvidenceRouter:
    """Observable family gate plus direction-specific evidence experts."""

    def __init__(
        self,
        family: Any,
        t2_visual: Any,
        t3_proprio: Any,
        classes: list[str],
    ):
        self.family = family
        self.t2_visual = t2_visual
        self.t3_proprio = t3_proprio
        self.classes = classes

    @staticmethod
    def _binary_probability(model: Any, values: np.ndarray, label: str) -> np.ndarray:
        raw = np.asarray(model.predict_proba(values), dtype=np.float64)
        classes = _model_classes(model)
        return raw[:, classes.index(label)]

    def predict_proba(
        self, visual: np.ndarray, proprio: np.ndarray
    ) -> np.ndarray:
        joint = np.concatenate([visual, proprio], axis=1)
        p_t2 = self._binary_probability(
            self.family, joint, "T2_vision_decisive"
        )
        p_adhesion = self._binary_probability(
            self.t2_visual, visual, "adhesion"
        )
        p_obstacle = self._binary_probability(
            self.t3_proprio, proprio, "invisible_obstacle"
        )
        out = np.zeros((len(visual), len(self.classes)), dtype=np.float64)
        index = {label: i for i, label in enumerate(self.classes)}
        out[:, index["adhesion"]] = p_t2 * p_adhesion
        out[:, index["compliant_terrain"]] = p_t2 * (1.0 - p_adhesion)
        out[:, index["invisible_obstacle"]] = (1.0 - p_t2) * p_obstacle
        out[:, index["low_friction"]] = (1.0 - p_t2) * (1.0 - p_obstacle)
        return out / out.sum(axis=1, keepdims=True)


def _fit(
    method: str,
    c_value: float,
    visual: np.ndarray,
    proprio: np.ndarray,
    labels: np.ndarray,
    cells: np.ndarray,
    case_ids: np.ndarray,
) -> Any:
    weights = _weights(case_ids)
    classes = sorted(set(labels.astype(str)))
    if method in {"vision_only", "proprio_only", "early_fusion"}:
        feature_key = {
            "vision_only": "vision",
            "proprio_only": "proprio",
            "early_fusion": "joint",
        }[method]
        values = {
            "vision": visual,
            "proprio": proprio,
            "joint": np.concatenate([visual, proprio], axis=1),
        }[feature_key]
        model = _pipeline(c_value)
        model.fit(values, labels, classifier__sample_weight=weights)
        return FlatModel(model, feature_key, classes)
    if method == "late_fusion":
        vision = _fit(
            "vision_only",
            c_value,
            visual,
            proprio,
            labels,
            cells,
            case_ids,
        )
        proprio_model = _fit(
            "proprio_only",
            c_value,
            visual,
            proprio,
            labels,
            cells,
            case_ids,
        )
        return LateFusionModel(vision, proprio_model)
    if method == "structured_evidence_router":
        family = _family_gate()
        family.fit(
            np.concatenate([visual, proprio], axis=1),
            cells,
            sample_weight=weights,
        )
        t2 = cells == "T2_vision_decisive"
        t3 = cells == "T3_proprio_decisive"
        t2_visual = CentroidProbabilityExpert(shrink_threshold=0.1).fit(
            visual[t2], labels[t2]
        )
        t3_proprio = _pipeline(c_value)
        t3_proprio.fit(
            proprio[t3],
            labels[t3],
            classifier__sample_weight=_weights(case_ids[t3]),
        )
        return StructuredEvidenceRouter(
            family, t2_visual, t3_proprio, classes
        )
    raise ValueError(method)


def _metrics(
    rows: list[dict[str, Any]],
    truth: np.ndarray,
    probability: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    prediction = np.asarray(
        [classes[index] for index in probability.argmax(axis=1)]
    )
    cells = np.asarray([str(row["cell"]) for row in rows])
    domains = np.asarray([str(row["domain"]) for row in rows])
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(str(row["case_id"]), str(row["target_operator"]))].append(index)
    texture_values = []
    for key, indices in sorted(groups.items()):
        if len(indices) != 3:
            raise RuntimeError(f"C2 appearance group is not three views: {key}")
        texture_values.append(bool(np.all(prediction[indices] == prediction[indices][0])))
    return {
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "accuracy": float(np.mean(prediction == truth)),
        "per_cell_accuracy": {
            cell: float(np.mean(prediction[cells == cell] == truth[cells == cell]))
            for cell in sorted(set(cells))
        },
        "per_domain_accuracy": {
            domain: float(
                np.mean(prediction[domains == domain] == truth[domains == domain])
            )
            for domain in sorted(set(domains))
        },
        "texture_swap_hard_consistency": float(np.mean(texture_values)),
        "prediction": prediction,
    }


def _hierarchical_bootstrap_delta(
    rows: list[dict[str, Any]],
    truth: np.ndarray,
    ours: np.ndarray,
    baseline: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    by_scene_case: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row, target, left, right in zip(
        rows, truth, ours, baseline, strict=True
    ):
        by_scene_case[str(row["scene_cluster"])][str(row["case_id"])].append(
            float(left == target) - float(right == target)
        )
    scenes = sorted(by_scene_case)
    case_values = {
        scene: {
            case: float(np.mean(values))
            for case, values in by_scene_case[scene].items()
        }
        for scene in scenes
    }
    flat = [
        value for scene in scenes for value in case_values[scene].values()
    ]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        sampled_scenes = rng.choice(scenes, size=len(scenes), replace=True)
        values = []
        for scene in sampled_scenes:
            source = np.asarray(list(case_values[str(scene)].values()))
            values.extend(
                rng.choice(source, size=len(source), replace=True).tolist()
            )
        samples.append(float(np.mean(values)))
    wins = sum(value > 0.0 for value in flat)
    losses = sum(value < 0.0 for value in flat)
    return {
        "unit": "scene cluster, with matched case resampling nested within scene",
        "scene_clusters": len(scenes),
        "matched_cases": len(flat),
        "estimate": float(np.mean(flat)),
        "ci95": [
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        ],
        "wins_ties_losses": [wins, len(flat) - wins - losses, losses],
        "paired_exact_sign_p_two_sided": (
            float(binomtest(wins, wins + losses, 0.5).pvalue)
            if wins + losses
            else 1.0
        ),
        "draws": draws,
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("development", "formal"), required=True
    )
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    feature_dir = args.features.resolve()
    output = args.output.resolve()
    manifest_path = feature_dir / "feature_manifest.json"
    records_path = feature_dir / "records.jsonl"
    features_path = feature_dir / "features.npz"
    manifest = _json(manifest_path)
    if manifest.get("passed") is not True:
        raise RuntimeError("C2 feature manifest is not complete")
    if (
        _sha(records_path) != manifest["output_sha256"]["records"]
        or _sha(features_path) != manifest["output_sha256"]["features"]
    ):
        raise RuntimeError("C2 feature provenance mismatch")
    rows = _jsonl(records_path)
    archive = np.load(features_path, allow_pickle=False)
    if archive["sample_ids"].astype(str).tolist() != [
        str(row["sample_id"]) for row in rows
    ]:
        raise RuntimeError("C2 features and records are misaligned")
    visual = np.asarray(archive["visual"], dtype=np.float32)
    proprio = np.asarray(archive["proprio"], dtype=np.float32)
    labels = np.asarray([str(row["attribution_category"]) for row in rows])
    cells = np.asarray([str(row["cell"]) for row in rows])
    case_ids = np.asarray([str(row["case_id"]) for row in rows])
    splits = np.asarray([str(row["split"]) for row in rows])
    classes = sorted(set(labels))
    if args.mode == "development":
        fit = splits == "train"
        evaluate = splits == "val"
        c_values = [0.01, 0.1, 1.0, 10.0]
        selected_c: dict[str, float] = {}
        development_results: dict[str, list[dict[str, Any]]] = {}
        for method in METHODS:
            candidates = []
            for c_value in c_values:
                model = _fit(
                    method,
                    c_value,
                    visual[fit],
                    proprio[fit],
                    labels[fit],
                    cells[fit],
                    case_ids[fit],
                )
                probability = model.predict_proba(
                    visual[evaluate], proprio[evaluate]
                )
                subset = [
                    row for row, keep in zip(rows, evaluate, strict=True) if keep
                ]
                metric = _metrics(
                    subset, labels[evaluate], probability, model.classes
                )
                candidates.append(
                    {
                        "C": c_value,
                        "balanced_accuracy": metric["balanced_accuracy"],
                        "per_cell_accuracy": metric["per_cell_accuracy"],
                        "texture_swap_hard_consistency": metric[
                            "texture_swap_hard_consistency"
                        ],
                    }
                )
            chosen = sorted(
                candidates,
                key=lambda row: (
                    -row["balanced_accuracy"],
                    -min(row["per_cell_accuracy"].values()),
                    row["C"],
                ),
            )[0]
            selected_c[method] = float(chosen["C"])
            development_results[method] = candidates
        best_baseline = sorted(
            BASELINES,
            key=lambda method: (
                -max(
                    row["balanced_accuracy"]
                    for row in development_results[method]
                    if row["C"] == selected_c[method]
                ),
                method,
            ),
        )[0]
        report = {
            "schema_version": "kinofail.realistic-c2-development.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "development_complete_test_unopened",
            "test_outcomes_used": False,
            "fit_split": "train",
            "evaluation_split": "val",
            "candidate_C": c_values,
            "results": development_results,
            "selected_C": selected_c,
            "best_baseline_method": best_baseline,
            "feature_manifest_sha256": _sha(manifest_path),
            "evaluator_sha256": _sha(Path(__file__).resolve()),
        }
        output.mkdir(parents=True, exist_ok=True)
        path = output / "development_report.json"
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.protocol is None:
        raise ValueError("formal C2 evaluation requires --protocol")
    protocol_path = args.protocol.resolve()
    protocol = _json(protocol_path)
    for key, actual in (
        ("feature_manifest_sha256", _sha(manifest_path)),
        ("evaluator_sha256", _sha(Path(__file__).resolve())),
    ):
        if protocol.get(key) != actual:
            raise RuntimeError(f"C2 formal protocol {key} mismatch")
    fit = splits != "test"
    evaluate = splits == "test"
    test_rows = [row for row, keep in zip(rows, evaluate, strict=True) if keep]
    results: dict[str, dict[str, Any]] = {}
    models: dict[str, Any] = {}
    probabilities: dict[str, np.ndarray] = {}
    for method in METHODS:
        model = _fit(
            method,
            float(protocol["selected_C"][method]),
            visual[fit],
            proprio[fit],
            labels[fit],
            cells[fit],
            case_ids[fit],
        )
        probability = model.predict_proba(visual[evaluate], proprio[evaluate])
        metric = _metrics(test_rows, labels[evaluate], probability, model.classes)
        probabilities[method] = probability
        models[method] = model
        results[method] = {
            key: value
            for key, value in metric.items()
            if key != "prediction"
        }
        results[method]["prediction"] = metric["prediction"]
    best_baseline = str(protocol["best_baseline_method"])
    ours_prediction = results["structured_evidence_router"]["prediction"]
    baseline_prediction = results[best_baseline]["prediction"]
    delta = _hierarchical_bootstrap_delta(
        test_rows,
        labels[evaluate],
        ours_prediction,
        baseline_prediction,
        draws=int(protocol["bootstrap"]["draws"]),
        seed=int(protocol["bootstrap"]["seed"]),
    )
    acceptance = protocol["acceptance"]
    ours = results["structured_evidence_router"]
    checks = {
        "minimum_balanced_accuracy": (
            ours["balanced_accuracy"]
            >= float(acceptance["minimum_balanced_accuracy"])
        ),
        "minimum_worst_direction_accuracy": (
            min(ours["per_cell_accuracy"].values())
            >= float(acceptance["minimum_worst_direction_accuracy"])
        ),
        "minimum_texture_swap_hard_consistency": (
            ours["texture_swap_hard_consistency"]
            >= float(acceptance["minimum_texture_swap_hard_consistency"])
        ),
        "beats_vision_only_point": (
            ours["balanced_accuracy"] > results["vision_only"]["balanced_accuracy"]
        ),
        "beats_proprio_only_point": (
            ours["balanced_accuracy"] > results["proprio_only"]["balanced_accuracy"]
        ),
        "ours_minus_best_unstructured_ci_lower_gt_0": delta["ci95"][0] > 0.0,
        "three_test_scene_clusters": (
            len({row["scene_cluster"] for row in test_rows}) == 3
        ),
        "three_test_domains": len({row["domain"] for row in test_rows}) == 3,
    }
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.jsonl"
    prediction_rows = []
    for index, row in enumerate(test_rows):
        prediction_rows.append(
            {
                "sample_id": row["sample_id"],
                "case_id": row["case_id"],
                "scene_cluster": row["scene_cluster"],
                "domain": row["domain"],
                "cell": row["cell"],
                "truth": labels[evaluate][index],
                **{
                    f"{method}_prediction": str(results[method]["prediction"][index])
                    for method in METHODS
                },
            }
        )
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    checkpoint_hashes = {}
    for method, model in models.items():
        path = output / f"{method}.pkl"
        with path.open("wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        checkpoint_hashes[method] = _sha(path)
    serializable_results = {
        method: {
            key: (
                value.tolist() if isinstance(value, np.ndarray) else value
            )
            for key, value in metric.items()
            if key != "prediction"
        }
        for method, metric in results.items()
    }
    report = {
        "schema_version": "kinofail.realistic-c2-formal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha(protocol_path),
        "passed": all(checks.values()),
        "checks": checks,
        "test_counts": {
            "samples": int(evaluate.sum()),
            "matched_cases": len({row["case_id"] for row in test_rows}),
            "scene_clusters": len({row["scene_cluster"] for row in test_rows}),
            "domains": len({row["domain"] for row in test_rows}),
            "directions": 2,
            "classes": 4,
        },
        "results": serializable_results,
        "best_baseline_selected_on_validation": best_baseline,
        "ours_minus_best_unstructured": delta,
        "artifacts": {
            "predictions_sha256": _sha(prediction_path),
            "checkpoint_sha256": checkpoint_hashes,
            "feature_manifest_sha256": _sha(manifest_path),
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": checks,
                "results": serializable_results,
                "best_baseline": best_baseline,
                "delta": delta,
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
