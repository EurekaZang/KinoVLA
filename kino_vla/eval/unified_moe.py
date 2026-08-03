"""Unified, auditable mixture-of-experts for Kino-Fail attribution.

The same model consumes the shared five-frame visual descriptor and the
rotation-invariant 21-sample proprioceptive descriptor for both the
bidirectional-conflict battery and the complete anomaly taxonomy.  Three
ordinary experts (vision, proprioception, and joint early fusion) are trained
on the same labels.  A learned router is then trained only from grouped
out-of-fold expert predictions to decide when a non-default expert is more
reliable than the proprioceptive default.

The router exposes its selected evidence route and route probabilities for
every prediction.  Scene, operator, material, severity, split, truth, and
outcome metadata are never deployment inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from kino_vla.eval.realistic_multimodal import (
    ObservableClassifier,
    _aligned_probabilities,
    _probability_entropy,
    _probability_margin,
    physical_episode_weights,
)


EXPERT_NAMES: tuple[str, ...] = ("vision", "proprio", "joint")
DEFAULT_EXPERT = "proprio"
ROUTED_EXPERTS: tuple[str, ...] = ("vision", "joint")

FORBIDDEN_DEPLOYMENT_INPUTS: tuple[str, ...] = (
    "scene",
    "scene_family",
    "domain",
    "operator",
    "operator_id",
    "target_operator",
    "material",
    "material_family",
    "severity",
    "appearance_id",
    "split",
    "truth",
    "ground_truth",
    "outcome",
    "cost",
    "privileged_telemetry",
)


def _router_estimator(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=256,
        max_features="sqrt",
        min_samples_leaf=6,
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=-1,
    )


@dataclass(frozen=True)
class _ConstantRouter:
    """Predict a training-fold prior when a route fold has one-sided support."""

    positive_probability: float

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray([False, True])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = float(np.clip(self.positive_probability, 0.0, 1.0))
        return np.tile([1.0 - positive, positive], (len(features), 1))


def _aligned(
    model: ObservableClassifier,
    visual: np.ndarray,
    proprio: np.ndarray,
    classes: list[str],
) -> np.ndarray:
    return _reindex(
        model.predict_proba(visual, proprio),
        model.classes,
        classes,
    )


def _reindex(
    probabilities: np.ndarray,
    source_classes: Iterable[str],
    target_classes: Iterable[str],
) -> np.ndarray:
    source = {str(label): index for index, label in enumerate(source_classes)}
    targets = [str(label) for label in target_classes]
    values = np.asarray(probabilities, dtype=np.float64)
    out = np.zeros((len(values), len(targets)), dtype=np.float64)
    for index, label in enumerate(targets):
        if label in source:
            out[:, index] = values[:, source[label]]
    normalizer = out.sum(axis=1, keepdims=True)
    return out / np.where(normalizer <= 0.0, 1.0, normalizer)


def _meta_features(expert_probabilities: np.ndarray) -> np.ndarray:
    """Observable router inputs derived only from expert posteriors."""

    values = np.asarray(expert_probabilities, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != len(EXPERT_NAMES):
        raise ValueError(
            "expert probabilities must have shape (N, 3, K), "
            f"received {values.shape}"
        )
    diagnostics: list[np.ndarray] = []
    for index in range(len(EXPERT_NAMES)):
        probability = values[:, index, :]
        diagnostics.extend(
            [
                probability.max(axis=1),
                _probability_margin(probability),
                _probability_entropy(probability),
            ]
        )
    for left in range(len(EXPERT_NAMES)):
        for right in range(left + 1, len(EXPERT_NAMES)):
            diagnostics.extend(
                [
                    np.abs(values[:, left, :] - values[:, right, :]).sum(axis=1),
                    (
                        values[:, left, :].argmax(axis=1)
                        == values[:, right, :].argmax(axis=1)
                    ).astype(np.float64),
                ]
            )
    return np.concatenate(
        [
            values.reshape(len(values), -1),
            np.stack(diagnostics, axis=1),
        ],
        axis=1,
    )


def _positive_probability(model: Any, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    lookup = {str(label): index for index, label in enumerate(model.classes_)}
    positive = lookup.get("True", lookup.get("1"))
    if positive is None:
        return np.zeros(len(features), dtype=np.float64)
    return probabilities[:, positive]


def _logit(value: np.ndarray | float) -> np.ndarray:
    bounded = np.clip(np.asarray(value, dtype=np.float64), 1.0e-6, 1.0 - 1.0e-6)
    return np.log(bounded / (1.0 - bounded))


def _truth_support_balanced_accuracy(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Macro recall over classes present in the requested training stratum."""

    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    classes = np.unique(truth)
    return float(
        np.mean(
            [
                np.mean(prediction[truth == label] == label)
                for label in classes
            ]
        )
    )


@dataclass(frozen=True)
class RoutedPrediction:
    """Full prediction record required for attribution and downstream action."""

    probabilities: np.ndarray
    expert_probabilities: np.ndarray
    route_probabilities: np.ndarray
    routes: np.ndarray
    evidence_routes: np.ndarray
    route_scores: np.ndarray
    disagreement: np.ndarray


@dataclass
class UnifiedEvidenceMoE:
    """Top-1 learned MoE with an explicit evidence-route interface."""

    experts: dict[str, ObservableClassifier]
    routers: dict[str, Any]
    route_thresholds: dict[str, float]
    classes: list[str]
    fit_audit: dict[str, Any]

    @staticmethod
    def _fit_experts(
        visual: np.ndarray,
        proprio: np.ndarray,
        labels: np.ndarray,
        episode_ids: np.ndarray,
        *,
        seed: int,
    ) -> dict[str, ObservableClassifier]:
        return {
            "vision": ObservableClassifier.fit(
                visual,
                proprio,
                labels,
                episode_ids,
                feature_key="vision",
                seed=seed + 11,
            ),
            "proprio": ObservableClassifier.fit(
                visual,
                proprio,
                labels,
                episode_ids,
                feature_key="proprio",
                seed=seed + 23,
            ),
            "joint": ObservableClassifier.fit(
                visual,
                proprio,
                labels,
                episode_ids,
                feature_key="early_fusion",
                seed=seed + 37,
            ),
        }

    @staticmethod
    def _expert_probabilities(
        experts: dict[str, ObservableClassifier],
        visual: np.ndarray,
        proprio: np.ndarray,
        classes: list[str],
    ) -> np.ndarray:
        return np.stack(
            [
                _aligned(experts[name], visual, proprio, classes)
                for name in EXPERT_NAMES
            ],
            axis=1,
        )

    @staticmethod
    def _select_routes(
        expert_probabilities: np.ndarray,
        route_scores: np.ndarray,
        thresholds: dict[str, float],
    ) -> np.ndarray:
        expert_prediction = expert_probabilities.argmax(axis=2)
        default_index = EXPERT_NAMES.index(DEFAULT_EXPERT)
        selected = np.full(len(expert_probabilities), default_index, dtype=np.int64)
        best_excess = np.zeros(len(expert_probabilities), dtype=np.float64)
        for score_index, name in enumerate(ROUTED_EXPERTS):
            candidate_index = EXPERT_NAMES.index(name)
            threshold = float(thresholds[name])
            score = route_scores[:, score_index]
            disagreement = (
                expert_prediction[:, candidate_index]
                != expert_prediction[:, default_index]
            )
            excess = score - threshold
            use = disagreement & (excess >= 0.0) & (excess > best_excess)
            selected[use] = candidate_index
            best_excess[use] = excess[use]
        return selected

    @classmethod
    def fit(
        cls,
        visual: np.ndarray,
        proprio: np.ndarray,
        labels: np.ndarray,
        episode_ids: np.ndarray,
        group_ids: np.ndarray,
        training_source_ids: np.ndarray | None = None,
        *,
        seed: int,
        minimum_override_precision: float = 0.80,
        minimum_override_groups: int = 6,
        route_threshold_margin: float = 0.0,
        deployment_route_threshold: float | None = None,
    ) -> "UnifiedEvidenceMoE":
        visual = np.asarray(visual, dtype=np.float32)
        proprio = np.asarray(proprio, dtype=np.float32)
        labels = np.asarray(labels).astype(str)
        episode_ids = np.asarray(episode_ids).astype(str)
        group_ids = np.asarray(group_ids).astype(str)
        if training_source_ids is None:
            training_source_ids = np.full(
                len(labels), "training_corpus", dtype=str
            )
        else:
            training_source_ids = np.asarray(
                training_source_ids
            ).astype(str)
        if not (
            len(visual)
            == len(proprio)
            == len(labels)
            == len(episode_ids)
            == len(group_ids)
            == len(training_source_ids)
        ):
            raise ValueError("training arrays must have equal length")
        if not np.isfinite(visual).all() or not np.isfinite(proprio).all():
            raise ValueError("training observables must be finite")
        if route_threshold_margin < 0.0:
            raise ValueError("route threshold margin must be non-negative")
        if (
            deployment_route_threshold is not None
            and not 0.0 <= deployment_route_threshold <= 1.0
        ):
            raise ValueError(
                "deployment route threshold must lie in [0, 1]"
            )
        classes = sorted(set(labels))
        class_group_counts = [
            len(set(group_ids[labels == label])) for label in classes
        ]
        n_splits = min(3, min(class_group_counts))
        if n_splits < 2:
            raise ValueError("each class needs at least two independent groups")

        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(seed),
        )
        oof = np.zeros(
            (len(labels), len(EXPERT_NAMES), len(classes)),
            dtype=np.float64,
        )
        fold_ids = np.full(len(labels), -1, dtype=np.int64)
        folds = list(splitter.split(visual, labels, groups=group_ids))
        for fold, (fit_index, hold_index) in enumerate(folds):
            experts = cls._fit_experts(
                visual[fit_index],
                proprio[fit_index],
                labels[fit_index],
                episode_ids[fit_index],
                seed=int(seed) + 1009 * (fold + 1),
            )
            oof[hold_index] = cls._expert_probabilities(
                experts,
                visual[hold_index],
                proprio[hold_index],
                classes,
            )
            fold_ids[hold_index] = fold
        if (fold_ids < 0).any():
            raise RuntimeError("expert cross-fitting left unassigned rows")

        meta = _meta_features(oof)
        truth_index = np.asarray([classes.index(label) for label in labels])
        expert_prediction = oof.argmax(axis=2)
        expert_correct = expert_prediction == truth_index[:, None]
        default_index = EXPERT_NAMES.index(DEFAULT_EXPERT)
        router_models: dict[str, Any] = {}
        crossfit_scores = np.zeros(
            (len(labels), len(ROUTED_EXPERTS)),
            dtype=np.float64,
        )
        router_audit: dict[str, Any] = {}
        for score_index, name in enumerate(ROUTED_EXPERTS):
            candidate_index = EXPERT_NAMES.index(name)
            informative = (
                expert_correct[:, candidate_index]
                != expert_correct[:, default_index]
            )
            target = expert_correct[:, candidate_index]
            for fold, (fit_index, hold_index) in enumerate(folds):
                router_fit = fit_index[informative[fit_index]]
                if (
                    len(router_fit) < 10
                    or len(set(target[router_fit])) < 2
                ):
                    prior = (
                        float(target[router_fit].mean())
                        if len(router_fit)
                        else 0.0
                    )
                    router = _ConstantRouter(prior)
                else:
                    router = _router_estimator(
                        int(seed) + 7001 + 101 * fold + score_index
                    ).fit(
                        meta[router_fit],
                        target[router_fit],
                        sample_weight=physical_episode_weights(
                            episode_ids[router_fit]
                        ),
                    )
                crossfit_scores[hold_index, score_index] = (
                    _positive_probability(router, meta[hold_index])
                )
            if (
                informative.sum() < 10
                or len(set(target[informative])) < 2
            ):
                router_models[name] = _ConstantRouter(
                    float(target[informative].mean())
                    if informative.any()
                    else 0.0
                )
            else:
                router_models[name] = _router_estimator(
                    int(seed) + 8009 + score_index
                ).fit(
                    meta[informative],
                    target[informative],
                    sample_weight=physical_episode_weights(
                        episode_ids[informative]
                    ),
                )
            router_audit[name] = {
                "informative_samples": int(informative.sum()),
                "candidate_better_samples": int(target[informative].sum()),
                "informative_groups": len(set(group_ids[informative])),
            }

        candidates = (
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            0.975,
            0.99,
            2.0,
        )
        threshold_audit: list[dict[str, Any]] = []
        default_prediction = expert_prediction[:, default_index]
        default_balanced_accuracy = float(
            balanced_accuracy_score(truth_index, default_prediction)
        )
        source_names = sorted(set(training_source_ids))
        default_source_balanced_accuracy = {
            source: _truth_support_balanced_accuracy(
                truth_index[training_source_ids == source],
                default_prediction[training_source_ids == source],
            )
            for source in source_names
        }
        default_equal_source_macro = float(
            np.mean(list(default_source_balanced_accuracy.values()))
        )
        default_worst_source = float(
            min(default_source_balanced_accuracy.values())
        )
        for vision_threshold in candidates:
            for joint_threshold in candidates:
                thresholds = {
                    "vision": float(vision_threshold),
                    "joint": float(joint_threshold),
                }
                routes = cls._select_routes(oof, crossfit_scores, thresholds)
                routed_prediction = expert_prediction[
                    np.arange(len(labels)), routes
                ]
                accepted = True
                route_details: dict[str, Any] = {}
                for name in ROUTED_EXPERTS:
                    index = EXPERT_NAMES.index(name)
                    override = routes == index
                    groups = len(set(group_ids[override]))
                    precision = (
                        float(expert_correct[override, index].mean())
                        if override.any()
                        else 0.0
                    )
                    if override.any():
                        accepted = accepted and (
                            groups >= int(minimum_override_groups)
                            and precision >= float(minimum_override_precision)
                        )
                    route_details[name] = {
                        "override_samples": int(override.sum()),
                        "override_groups": groups,
                        "override_precision": precision,
                    }
                balanced_accuracy = float(
                    balanced_accuracy_score(truth_index, routed_prediction)
                )
                source_balanced_accuracy = {
                    source: _truth_support_balanced_accuracy(
                        truth_index[training_source_ids == source],
                        routed_prediction[
                            training_source_ids == source
                        ],
                    )
                    for source in source_names
                }
                equal_source_macro = float(
                    np.mean(list(source_balanced_accuracy.values()))
                )
                worst_source = float(
                    min(source_balanced_accuracy.values())
                )
                accepted = accepted and (
                    equal_source_macro >= default_equal_source_macro
                    and worst_source >= default_worst_source
                )
                threshold_audit.append(
                    {
                        "thresholds": thresholds,
                        "balanced_accuracy": balanced_accuracy,
                        "source_balanced_accuracy": (
                            source_balanced_accuracy
                        ),
                        "equal_source_macro_balanced_accuracy": (
                            equal_source_macro
                        ),
                        "worst_source_balanced_accuracy": worst_source,
                        "accepted": bool(accepted),
                        "routes": route_details,
                    }
                )
        accepted_candidates = [
            row for row in threshold_audit if row["accepted"]
        ]
        if not accepted_candidates:
            selected = {
                "thresholds": {"vision": 2.0, "joint": 2.0},
                "balanced_accuracy": default_balanced_accuracy,
                "source_balanced_accuracy": (
                    default_source_balanced_accuracy
                ),
                "equal_source_macro_balanced_accuracy": (
                    default_equal_source_macro
                ),
                "worst_source_balanced_accuracy": default_worst_source,
                "accepted": True,
                "fallback_to_default": True,
                "routes": {
                    name: {
                        "override_samples": 0,
                        "override_groups": 0,
                        "override_precision": 0.0,
                    }
                    for name in ROUTED_EXPERTS
                },
            }
        else:
            selected = max(
                accepted_candidates,
                key=lambda row: (
                    row["worst_source_balanced_accuracy"],
                    row["equal_source_macro_balanced_accuracy"],
                    row["balanced_accuracy"],
                    sum(
                        row["routes"][name]["override_precision"]
                        for name in ROUTED_EXPERTS
                    ),
                    -sum(
                        row["routes"][name]["override_samples"]
                        for name in ROUTED_EXPERTS
                    ),
                    sum(row["thresholds"].values()),
                ),
            )

        final_experts = cls._fit_experts(
            visual,
            proprio,
            labels,
            episode_ids,
            seed=int(seed),
        )
        if deployment_route_threshold is None:
            deployed_thresholds = {
                name: (
                    2.0
                    if float(selected["thresholds"][name]) > 1.0
                    else min(
                        0.99,
                        float(selected["thresholds"][name])
                        + float(route_threshold_margin),
                    )
                )
                for name in ROUTED_EXPERTS
            }
            deployment_rule = (
                "per-seed source-minimax OOF threshold plus frozen margin"
            )
        else:
            deployed_thresholds = {
                name: float(deployment_route_threshold)
                for name in ROUTED_EXPERTS
            }
            deployment_rule = (
                "shared development-frozen route-score threshold"
            )
        return cls(
            experts=final_experts,
            routers=router_models,
            route_thresholds=deployed_thresholds,
            classes=classes,
            fit_audit={
                "architecture": "group-cross-fitted top-1 learned mixture-of-experts",
                "expert_names": list(EXPERT_NAMES),
                "default_expert": DEFAULT_EXPERT,
                "router_inputs": "expert posteriors and posterior-disagreement diagnostics",
                "selection_data": "grouped out-of-fold training predictions only",
                "n_splits": n_splits,
                "minimum_override_precision": float(
                    minimum_override_precision
                ),
                "minimum_override_groups": int(minimum_override_groups),
                "route_threshold_margin": float(route_threshold_margin),
                "deployment_route_threshold": (
                    None
                    if deployment_route_threshold is None
                    else float(deployment_route_threshold)
                ),
                "deployment_operating_point_rule": deployment_rule,
                "default_oof_balanced_accuracy": default_balanced_accuracy,
                "training_source_ids_are_deployment_inputs": False,
                "training_sources": source_names,
                "default_source_balanced_accuracy": (
                    default_source_balanced_accuracy
                ),
                "default_equal_source_macro_balanced_accuracy": (
                    default_equal_source_macro
                ),
                "default_worst_source_balanced_accuracy": (
                    default_worst_source
                ),
                "router_training": router_audit,
                "selected_crossfit_operating_point": selected,
                "deployed_thresholds": deployed_thresholds,
                "test_truth_or_outcomes_used": False,
                "forbidden_deployment_inputs": list(
                    FORBIDDEN_DEPLOYMENT_INPUTS
                ),
            },
        )

    def predict_with_routes(
        self,
        visual: np.ndarray,
        proprio: np.ndarray,
    ) -> RoutedPrediction:
        visual = np.asarray(visual, dtype=np.float32)
        proprio = np.asarray(proprio, dtype=np.float32)
        expert_probabilities = self._expert_probabilities(
            self.experts,
            visual,
            proprio,
            self.classes,
        )
        meta = _meta_features(expert_probabilities)
        route_scores = np.stack(
            [
                _positive_probability(self.routers[name], meta)
                for name in ROUTED_EXPERTS
            ],
            axis=1,
        )
        selected = self._select_routes(
            expert_probabilities,
            route_scores,
            self.route_thresholds,
        )
        probabilities = expert_probabilities[
            np.arange(len(expert_probabilities)), selected
        ]

        utilities = np.zeros(
            (len(expert_probabilities), len(EXPERT_NAMES)),
            dtype=np.float64,
        )
        for score_index, name in enumerate(ROUTED_EXPERTS):
            expert_index = EXPERT_NAMES.index(name)
            threshold = self.route_thresholds[name]
            if threshold > 1.0:
                utilities[:, expert_index] = -20.0
            else:
                utilities[:, expert_index] = (
                    _logit(route_scores[:, score_index])
                    - _logit(threshold)
                )
        utilities -= utilities.max(axis=1, keepdims=True)
        route_probabilities = np.exp(utilities)
        route_probabilities /= route_probabilities.sum(
            axis=1, keepdims=True
        )
        expert_prediction = expert_probabilities.argmax(axis=2)
        evidence_routes = np.asarray(EXPERT_NAMES, dtype=object)[
            selected
        ]
        joint_index = EXPERT_NAMES.index("joint")
        vision_index = EXPERT_NAMES.index("vision")
        proprio_index = EXPERT_NAMES.index("proprio")
        for row in np.flatnonzero(selected == joint_index):
            joint_prediction = expert_prediction[row, joint_index]
            agrees_vision = (
                joint_prediction == expert_prediction[row, vision_index]
            )
            agrees_proprio = (
                joint_prediction == expert_prediction[row, proprio_index]
            )
            if agrees_vision and agrees_proprio:
                evidence_routes[row] = "consensus"
            elif agrees_vision:
                evidence_routes[row] = "vision"
            elif agrees_proprio:
                evidence_routes[row] = "proprio"
            else:
                evidence_routes[row] = "cross_modal_interaction"
        disagreement = (
            np.apply_along_axis(lambda row: len(set(row.tolist())) > 1, 1, expert_prediction)
        )
        return RoutedPrediction(
            probabilities=probabilities,
            expert_probabilities=expert_probabilities,
            route_probabilities=route_probabilities,
            routes=np.asarray(EXPERT_NAMES, dtype=str)[selected],
            evidence_routes=evidence_routes.astype(str),
            route_scores=route_scores,
            disagreement=disagreement.astype(bool),
        )

    def predict_proba(
        self,
        visual: np.ndarray,
        proprio: np.ndarray,
    ) -> np.ndarray:
        return self.predict_with_routes(visual, proprio).probabilities
