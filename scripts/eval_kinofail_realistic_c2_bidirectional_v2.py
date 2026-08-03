#!/usr/bin/env python3
"""Screen and formally evaluate C2 v2 structured bidirectional routing."""

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "vision_only",
    "proprio_only",
    "early_fusion_linear",
    "early_fusion_rbf",
    "late_fusion",
    "structured_conditional_router_v2",
)
BASELINES = METHODS[:-1]
T2 = "T2_vision_decisive"
T3 = "T3_proprio_decisive"


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
    return np.asarray(
        [1.0 / lookup[value] for value in case_ids], dtype=np.float64
    )


def _linear(c_value: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=2026072417,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _rbf(
    c_value: float = 10.0, gamma: float | str = 1e-4
) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                SVC(
                    C=float(c_value),
                    gamma=gamma,
                    class_weight="balanced",
                    probability=False,
                    random_state=2026072417,
                ),
            ),
        ]
    )


def _family_gate() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=512,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=2026072417,
        n_jobs=-1,
    )


def _classes(model: Any) -> list[str]:
    if hasattr(model, "named_steps"):
        values = model.named_steps["classifier"].classes_
    else:
        values = model.classes_
    return [str(value) for value in values]


def _aligned(
    model: Any, values: np.ndarray, classes: list[str]
) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw = np.asarray(
            model.predict_proba(values), dtype=np.float64
        )
    else:
        decision = np.asarray(
            model.decision_function(values), dtype=np.float64
        )
        if decision.ndim == 1:
            decision = np.stack([-decision, decision], axis=1)
        decision -= decision.max(axis=1, keepdims=True)
        raw = np.exp(decision)
        raw /= raw.sum(axis=1, keepdims=True)
    source = {label: index for index, label in enumerate(_classes(model))}
    result = np.zeros((len(values), len(classes)), dtype=np.float64)
    for index, label in enumerate(classes):
        if label in source:
            result[:, index] = raw[:, source[label]]
    denominator = result.sum(axis=1, keepdims=True)
    return result / np.where(denominator <= 0.0, 1.0, denominator)


class FlatModel:
    def __init__(
        self, model: Any, feature_key: str, classes: list[str]
    ) -> None:
        self.model = model
        self.feature_key = feature_key
        self.classes = classes

    def predict_proba(
        self,
        visual: np.ndarray,
        geometry: np.ndarray,
        proprio: np.ndarray,
    ) -> np.ndarray:
        values = {
            "vision": np.concatenate([visual, geometry], axis=1),
            "proprio": proprio,
            "joint": np.concatenate(
                [visual, geometry, proprio], axis=1
            ),
        }[self.feature_key]
        return _aligned(self.model, values, self.classes)


class LateFusionModel:
    def __init__(
        self, vision: FlatModel, proprio: FlatModel
    ) -> None:
        self.vision = vision
        self.proprio = proprio
        self.classes = vision.classes

    def predict_proba(
        self,
        visual: np.ndarray,
        geometry: np.ndarray,
        proprio: np.ndarray,
    ) -> np.ndarray:
        return 0.5 * (
            self.vision.predict_proba(visual, geometry, proprio)
            + self.proprio.predict_proba(visual, geometry, proprio)
        )


class StructuredConditionalRouterV2:
    """Observable family gate followed by direction-specific specialists."""

    def __init__(
        self,
        family: ExtraTreesClassifier,
        t2_visual: Any,
        t3_proprio: Any,
        classes: list[str],
    ) -> None:
        self.family = family
        self.t2_visual = t2_visual
        self.t3_proprio = t3_proprio
        self.classes = classes

    @staticmethod
    def _binary(
        model: Any, values: np.ndarray, positive: str
    ) -> np.ndarray:
        classes = _classes(model)
        raw = _aligned(model, values, classes)
        return raw[:, classes.index(positive)]

    def family_prediction(
        self,
        visual: np.ndarray,
        geometry: np.ndarray,
        proprio: np.ndarray,
    ) -> np.ndarray:
        joint = np.concatenate(
            [visual, geometry, proprio], axis=1
        )
        return np.asarray(self.family.predict(joint)).astype(str)

    def predict_proba(
        self,
        visual: np.ndarray,
        geometry: np.ndarray,
        proprio: np.ndarray,
    ) -> np.ndarray:
        vision = np.concatenate([visual, geometry], axis=1)
        joint = np.concatenate([vision, proprio], axis=1)
        p_t2 = self._binary(self.family, joint, T2)
        p_adhesion = self._binary(
            self.t2_visual, vision, "adhesion"
        )
        p_obstacle = self._binary(
            self.t3_proprio, proprio, "invisible_obstacle"
        )
        result = np.zeros(
            (len(visual), len(self.classes)), dtype=np.float64
        )
        index = {label: i for i, label in enumerate(self.classes)}
        result[:, index["adhesion"]] = p_t2 * p_adhesion
        result[:, index["compliant_terrain"]] = (
            p_t2 * (1.0 - p_adhesion)
        )
        result[:, index["invisible_obstacle"]] = (
            (1.0 - p_t2) * p_obstacle
        )
        result[:, index["low_friction"]] = (
            (1.0 - p_t2) * (1.0 - p_obstacle)
        )
        return result / result.sum(axis=1, keepdims=True)


def _fit(
    method: str,
    visual: np.ndarray,
    geometry: np.ndarray,
    proprio: np.ndarray,
    labels: np.ndarray,
    cells: np.ndarray,
    case_ids: np.ndarray,
) -> Any:
    weights = _weights(case_ids)
    classes = sorted(set(labels.astype(str)))
    vision = np.concatenate([visual, geometry], axis=1)
    joint = np.concatenate([vision, proprio], axis=1)
    if method == "vision_only":
        model = _rbf()
        model.fit(
            vision, labels, classifier__sample_weight=weights
        )
        return FlatModel(model, "vision", classes)
    if method == "proprio_only":
        model = _linear()
        model.fit(
            proprio, labels, classifier__sample_weight=weights
        )
        return FlatModel(model, "proprio", classes)
    if method == "early_fusion_linear":
        model = _linear()
        model.fit(
            joint, labels, classifier__sample_weight=weights
        )
        return FlatModel(model, "joint", classes)
    if method == "early_fusion_rbf":
        # Strongest unstructured joint setting in the same six-scene LOSO
        # development grid.
        model = _rbf(c_value=100.0, gamma=1e-4)
        model.fit(
            joint, labels, classifier__sample_weight=weights
        )
        return FlatModel(model, "joint", classes)
    if method == "late_fusion":
        return LateFusionModel(
            _fit(
                "vision_only",
                visual,
                geometry,
                proprio,
                labels,
                cells,
                case_ids,
            ),
            _fit(
                "proprio_only",
                visual,
                geometry,
                proprio,
                labels,
                cells,
                case_ids,
            ),
        )
    if method == "structured_conditional_router_v2":
        family = _family_gate()
        family.fit(joint, cells, sample_weight=weights)
        t2 = cells == T2
        t3 = cells == T3
        # Selected only on six-scene LOSO development.  The wider
        # scale-adaptive kernel was more stable to scene-specific HOG scale
        # than the global early-fusion setting.
        t2_visual = _rbf(c_value=0.1, gamma="scale")
        t2_visual.fit(
            vision[t2],
            labels[t2],
            classifier__sample_weight=_weights(case_ids[t2]),
        )
        t3_proprio = _linear()
        t3_proprio.fit(
            proprio[t3],
            labels[t3],
            classifier__sample_weight=_weights(case_ids[t3]),
        )
        return StructuredConditionalRouterV2(
            family, t2_visual, t3_proprio, classes
        )
    raise ValueError(method)


def _metric(
    rows: list[dict[str, Any]],
    truth: np.ndarray,
    probability: np.ndarray,
    classes: list[str],
) -> tuple[dict[str, Any], np.ndarray]:
    prediction = np.asarray(
        [classes[index] for index in probability.argmax(axis=1)]
    )
    cells = np.asarray([str(row["cell"]) for row in rows])
    scenes = np.asarray(
        [str(row["scene_cluster"]) for row in rows]
    )
    domains = np.asarray([str(row["domain"]) for row in rows])
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[
            (str(row["case_id"]), str(row["target_operator"]))
        ].append(index)
    consistency = []
    for key, indices in sorted(groups.items()):
        if len(indices) != 3:
            raise RuntimeError(
                f"C2 appearance group is not three views: {key}"
            )
        consistency.append(
            bool(
                np.all(
                    prediction[indices] == prediction[indices][0]
                )
            )
        )
    result = {
        "balanced_accuracy": float(
            balanced_accuracy_score(truth, prediction)
        ),
        "accuracy": float(np.mean(prediction == truth)),
        "per_cell_accuracy": {
            cell: float(
                np.mean(
                    prediction[cells == cell] == truth[cells == cell]
                )
            )
            for cell in sorted(set(cells))
        },
        "per_scene_accuracy": {
            scene: float(
                np.mean(
                    prediction[scenes == scene]
                    == truth[scenes == scene]
                )
            )
            for scene in sorted(set(scenes))
        },
        "per_domain_accuracy": {
            domain: float(
                np.mean(
                    prediction[domains == domain]
                    == truth[domains == domain]
                )
            )
            for domain in sorted(set(domains))
        },
        "worst_scene_accuracy": float(
            min(
                np.mean(
                    prediction[scenes == scene]
                    == truth[scenes == scene]
                )
                for scene in sorted(set(scenes))
            )
        ),
        "texture_swap_hard_consistency": float(
            np.mean(consistency)
        ),
    }
    return result, prediction


def _bootstrap_delta(
    rows: list[dict[str, Any]],
    truth: np.ndarray,
    ours: np.ndarray,
    baseline: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    by_scene_case: dict[
        str, dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    for row, target, left, right in zip(
        rows, truth, ours, baseline, strict=True
    ):
        by_scene_case[str(row["scene_cluster"])][
            str(row["case_id"])
        ].append(float(left == target) - float(right == target))
    scenes = sorted(by_scene_case)
    values = {
        scene: {
            case: float(np.mean(item))
            for case, item in by_scene_case[scene].items()
        }
        for scene in scenes
    }
    flat = [
        value for scene in scenes for value in values[scene].values()
    ]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        sampled_scenes = rng.choice(
            scenes, size=len(scenes), replace=True
        )
        draw_values = []
        for scene in sampled_scenes:
            source = np.asarray(
                list(values[str(scene)].values())
            )
            draw_values.extend(
                rng.choice(
                    source, size=len(source), replace=True
                ).tolist()
            )
        samples.append(float(np.mean(draw_values)))
    wins = sum(value > 0.0 for value in flat)
    losses = sum(value < 0.0 for value in flat)
    return {
        "unit": (
            "scene cluster, with matched case resampling nested "
            "within scene"
        ),
        "scene_clusters": len(scenes),
        "matched_cases": len(flat),
        "estimate": float(np.mean(flat)),
        "ci95": [
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        ],
        "wins_ties_losses": [
            wins,
            len(flat) - wins - losses,
            losses,
        ],
        "paired_exact_sign_p_two_sided": (
            float(binomtest(wins, wins + losses, 0.5).pvalue)
            if wins + losses
            else 1.0
        ),
        "draws": draws,
        "seed": seed,
    }


def _load_development(
    directory: Path,
) -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Path,
]:
    manifest_path = directory / "feature_manifest.json"
    manifest = _json(manifest_path)
    records_path = directory / "records.jsonl"
    features_path = directory / "features.npz"
    if (
        manifest.get("passed") is not True
        or _sha(records_path)
        != manifest["output_sha256"]["records"]
        or _sha(features_path)
        != manifest["output_sha256"]["features"]
    ):
        raise RuntimeError("invalid C2 v2 development features")
    rows = _jsonl(records_path)
    archive = np.load(features_path, allow_pickle=False)
    if archive["sample_ids"].astype(str).tolist() != [
        str(row["sample_id"]) for row in rows
    ]:
        raise RuntimeError("misaligned C2 v2 development features")
    return (
        rows,
        np.asarray(archive["visual"], dtype=np.float32),
        np.asarray(archive["geometry"], dtype=np.float32),
        np.asarray(archive["proprio"], dtype=np.float32),
        manifest_path,
    )


def _load_test(
    directory: Path, geometry_directory: Path
) -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Path,
    Path,
]:
    manifest_path = directory / "feature_manifest.json"
    geometry_manifest_path = (
        geometry_directory / "geometry_manifest.json"
    )
    manifest = _json(manifest_path)
    geometry_manifest = _json(geometry_manifest_path)
    records_path = directory / "records.jsonl"
    features_path = directory / "features.npz"
    geometry_path = geometry_directory / "geometry.npz"
    if (
        manifest.get("passed") is not True
        or geometry_manifest.get("passed") is not True
        or _sha(records_path)
        != manifest["output_sha256"]["records"]
        or _sha(features_path)
        != manifest["output_sha256"]["features"]
        or _sha(geometry_path)
        != geometry_manifest["output_sha256"]
    ):
        raise RuntimeError("invalid C2 v2 formal features")
    rows = _jsonl(records_path)
    features = np.load(features_path, allow_pickle=False)
    geometry = np.load(geometry_path, allow_pickle=False)
    sample_ids = [str(row["sample_id"]) for row in rows]
    if (
        features["sample_ids"].astype(str).tolist() != sample_ids
        or geometry["sample_ids"].astype(str).tolist()
        != sample_ids
    ):
        raise RuntimeError("misaligned C2 v2 formal features")
    return (
        rows,
        np.asarray(features["visual"], dtype=np.float32),
        np.asarray(geometry["geometry"], dtype=np.float32),
        np.asarray(features["proprio"], dtype=np.float32),
        manifest_path,
        geometry_manifest_path,
    )


def _development(
    directory: Path, output: Path
) -> int:
    rows, visual, geometry, proprio, manifest_path = (
        _load_development(directory)
    )
    labels = np.asarray(
        [str(row["attribution_category"]) for row in rows]
    )
    cells = np.asarray([str(row["cell"]) for row in rows])
    cases = np.asarray([str(row["case_id"]) for row in rows])
    scenes = np.asarray(
        [str(row["scene_cluster"]) for row in rows]
    )
    predictions = {
        method: np.empty(len(rows), dtype=object)
        for method in METHODS
    }
    family_predictions = np.empty(len(rows), dtype=object)
    for scene in sorted(set(scenes)):
        train = scenes != scene
        evaluate = ~train
        for method in METHODS:
            model = _fit(
                method,
                visual[train],
                geometry[train],
                proprio[train],
                labels[train],
                cells[train],
                cases[train],
            )
            probability = model.predict_proba(
                visual[evaluate],
                geometry[evaluate],
                proprio[evaluate],
            )
            predictions[method][evaluate] = np.asarray(
                [
                    model.classes[index]
                    for index in probability.argmax(axis=1)
                ]
            )
            if method == "structured_conditional_router_v2":
                family_predictions[evaluate] = (
                    model.family_prediction(
                        visual[evaluate],
                        geometry[evaluate],
                        proprio[evaluate],
                    )
                )
    results = {}
    for method in METHODS:
        one_hot = np.zeros(
            (len(rows), len(sorted(set(labels)))), dtype=np.float64
        )
        classes = sorted(set(labels))
        index = {label: i for i, label in enumerate(classes)}
        for row_index, label in enumerate(predictions[method]):
            one_hot[row_index, index[str(label)]] = 1.0
        metric, _ = _metric(rows, labels, one_hot, classes)
        results[method] = metric
    best_baseline = sorted(
        BASELINES,
        key=lambda method: (
            -results[method]["balanced_accuracy"],
            -results[method]["worst_scene_accuracy"],
            method,
        ),
    )[0]
    ours = results["structured_conditional_router_v2"]
    report = {
        "schema_version": (
            "kinofail.realistic-c2-development-screen.v2"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "development_complete_formal_outcomes_unavailable",
        "formal_outcomes_used": False,
        "scene_disjoint_loso": True,
        "scene_clusters": sorted(set(scenes)),
        "results": results,
        "family_gate_accuracy": float(
            np.mean(family_predictions == cells)
        ),
        "best_baseline_method": best_baseline,
        "selected_method": "structured_conditional_router_v2",
        "selection": {
            "strong_early_fusion_baseline": (
                "RBF SVC(C=100,gamma=1e-4) selected from the same "
                "six-scene LOSO grid"
            ),
            "family_gate": (
                "ExtraTrees(512, sqrt, min_samples_leaf=2)"
            ),
            "T2_specialist": (
                "RBF SVC(C=0.1,gamma=scale) on CLIP+generic HOG"
            ),
            "T3_specialist": (
                "balanced logistic(C=1) on proprioception"
            ),
            "composition": (
                "conditional product of observable family probability "
                "and within-family specialist probability"
            ),
            "rationale": (
                "preserves direction-specific evidence restrictions and "
                "maximizes development balanced accuracy without using "
                "any held-out scene"
            ),
        },
        "development_advantage": {
            "balanced_accuracy_delta_vs_best_baseline": (
                ours["balanced_accuracy"]
                - results[best_baseline]["balanced_accuracy"]
            ),
            "worst_scene_delta_vs_best_baseline": (
                ours["worst_scene_accuracy"]
                - results[best_baseline]["worst_scene_accuracy"]
            ),
        },
        "feature_manifest_sha256": _sha(manifest_path),
        "evaluator_sha256": _sha(Path(__file__).resolve()),
    }
    output.mkdir(parents=True, exist_ok=False)
    path = output / "development_report.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _formal(
    development_directory: Path,
    test_directory: Path,
    geometry_directory: Path,
    protocol_path: Path,
    output: Path,
) -> int:
    protocol = _json(protocol_path)
    (
        development_rows,
        development_visual,
        development_geometry,
        development_proprio,
        development_manifest_path,
    ) = _load_development(development_directory)
    (
        test_rows,
        test_visual,
        test_geometry,
        test_proprio,
        test_manifest_path,
        geometry_manifest_path,
    ) = _load_test(test_directory, geometry_directory)
    for key, actual in (
        ("evaluator_sha256", _sha(Path(__file__).resolve())),
        (
            "development_feature_manifest_sha256",
            _sha(development_manifest_path),
        ),
    ):
        if protocol.get(key) != actual:
            raise RuntimeError(f"C2 v2 protocol {key} mismatch")
    development_labels = np.asarray(
        [
            str(row["attribution_category"])
            for row in development_rows
        ]
    )
    development_cells = np.asarray(
        [str(row["cell"]) for row in development_rows]
    )
    development_cases = np.asarray(
        [str(row["case_id"]) for row in development_rows]
    )
    test_labels = np.asarray(
        [str(row["attribution_category"]) for row in test_rows]
    )
    results: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}
    models: dict[str, Any] = {}
    for method in METHODS:
        model = _fit(
            method,
            development_visual,
            development_geometry,
            development_proprio,
            development_labels,
            development_cells,
            development_cases,
        )
        probability = model.predict_proba(
            test_visual, test_geometry, test_proprio
        )
        metric, prediction = _metric(
            test_rows, test_labels, probability, model.classes
        )
        results[method] = metric
        predictions[method] = prediction
        models[method] = model
    best_baseline = max(
        BASELINES,
        key=lambda method: results[method]["balanced_accuracy"],
    )
    ours_name = "structured_conditional_router_v2"
    delta = _bootstrap_delta(
        test_rows,
        test_labels,
        predictions[ours_name],
        predictions[best_baseline],
        draws=int(protocol["bootstrap"]["draws"]),
        seed=int(protocol["bootstrap"]["seed"]),
    )
    ours = results[ours_name]
    acceptance = protocol["acceptance"]
    checks = {
        "minimum_balanced_accuracy": (
            ours["balanced_accuracy"]
            >= float(acceptance["minimum_balanced_accuracy"])
        ),
        "minimum_worst_direction_accuracy": (
            min(ours["per_cell_accuracy"].values())
            >= float(
                acceptance["minimum_worst_direction_accuracy"]
            )
        ),
        "minimum_worst_scene_accuracy": (
            ours["worst_scene_accuracy"]
            >= float(acceptance["minimum_worst_scene_accuracy"])
        ),
        "minimum_texture_swap_hard_consistency": (
            ours["texture_swap_hard_consistency"]
            >= float(
                acceptance[
                    "minimum_texture_swap_hard_consistency"
                ]
            )
        ),
        "beats_every_baseline_point": all(
            ours["balanced_accuracy"]
            > results[method]["balanced_accuracy"]
            for method in BASELINES
        ),
        "matched_delta_noninferiority_ci": (
            delta["ci95"][0]
            >= float(acceptance["minimum_delta_ci95_lower"])
        ),
        "proprio_insufficient_on_t2": (
            results["proprio_only"]["per_cell_accuracy"][T2]
            <= float(
                acceptance[
                    "maximum_wrong_modality_direction_accuracy"
                ]
            )
        ),
        "vision_insufficient_on_t3": (
            results["vision_only"]["per_cell_accuracy"][T3]
            <= float(
                acceptance[
                    "maximum_wrong_modality_direction_accuracy"
                ]
            )
        ),
        "three_scene_disjoint_test_domains": (
            len(
                {
                    str(row["scene_cluster"])
                    for row in test_rows
                }
            )
            == 3
            and len({str(row["domain"]) for row in test_rows})
            == 3
            and {
                str(row["scene_cluster"])
                for row in test_rows
            }.isdisjoint(
                {
                    str(row["scene_cluster"])
                    for row in development_rows
                }
            )
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "predictions.jsonl"
    prediction_path.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": row["sample_id"],
                    "case_id": row["case_id"],
                    "scene_cluster": row["scene_cluster"],
                    "domain": row["domain"],
                    "cell": row["cell"],
                    "truth": test_labels[index],
                    **{
                        f"{method}_prediction": str(
                            predictions[method][index]
                        )
                        for method in METHODS
                    },
                },
                sort_keys=True,
            )
            + "\n"
            for index, row in enumerate(test_rows)
        ),
        encoding="utf-8",
    )
    checkpoint_hashes = {}
    for method, model in models.items():
        path = output / f"{method}.pkl"
        with path.open("wb") as handle:
            pickle.dump(
                model, handle, protocol=pickle.HIGHEST_PROTOCOL
            )
        checkpoint_hashes[method] = _sha(path)
    report = {
        "schema_version": (
            "kinofail.realistic-c2-formal-confirmation.v2"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha(protocol_path),
        "passed": all(checks.values()),
        "checks": checks,
        "test_counts": {
            "samples": len(test_rows),
            "matched_cases": len(
                {str(row["case_id"]) for row in test_rows}
            ),
            "cases_per_direction": {
                cell: len(
                    {
                        str(row["case_id"])
                        for row in test_rows
                        if row["cell"] == cell
                    }
                )
                for cell in (T2, T3)
            },
            "scene_clusters": 3,
            "domains": 3,
            "classes": 4,
        },
        "results": results,
        "best_baseline_on_formal_battery": best_baseline,
        "ours_minus_best_baseline": delta,
        "artifacts": {
            "predictions_sha256": _sha(prediction_path),
            "checkpoint_sha256": checkpoint_hashes,
            "test_feature_manifest_sha256": _sha(
                test_manifest_path
            ),
            "test_geometry_manifest_sha256": _sha(
                geometry_manifest_path
            ),
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
                "results": results,
                "best_baseline": best_baseline,
                "delta": delta,
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("development", "formal"), required=True
    )
    parser.add_argument("--development-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-features", type=Path)
    parser.add_argument("--test-geometry", type=Path)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    if args.mode == "development":
        return _development(
            args.development_features.resolve(),
            args.output.resolve(),
        )
    if (
        args.test_features is None
        or args.test_geometry is None
        or args.protocol is None
    ):
        raise ValueError(
            "formal mode requires test features, geometry, and protocol"
        )
    return _formal(
        args.development_features.resolve(),
        args.test_features.resolve(),
        args.test_geometry.resolve(),
        args.protocol.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
