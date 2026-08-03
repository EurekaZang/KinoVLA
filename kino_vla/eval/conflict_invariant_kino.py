"""Conflict-aware, domain-invariant routing for Kino-Fail attribution.

The model has one conservative default and one narrowly scoped override:

* a proprioceptive expert consumes rotation-invariant statistics together with
  within-window relative dynamics; and
* a visual specialist is invoked only when an observable gate identifies the
  vision-decisive conflict pattern.

Scene, domain, operator, material, battery, and ground-truth identifiers are
used for grouped development splits or auxiliary route supervision only.  They
are never accepted by :meth:`predict_with_routes`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


T2_CLASSES: tuple[str, str] = ("adhesion", "compliant_terrain")


def physical_group_weights(group_ids: np.ndarray) -> np.ndarray:
    """Give every physical group total weight one."""

    values = np.asarray(group_ids).astype(str)
    unique, counts = np.unique(values, return_counts=True)
    lookup = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    return np.asarray([1.0 / lookup[value] for value in values], dtype=np.float64)


def relative_proprio_features(summary: np.ndarray) -> np.ndarray:
    """Turn a 10-statistic proprio summary into within-window dynamics.

    The input follows ``proprio_summary``'s block order:
    mean, std, min, max, q25, q75, first, last, delta, and slope.  Absolute
    response magnitude is retained by signed-log features, while ratios are
    normalized by each sample's own temporal variation.  This removes the
    global amplitude dependence that caused the F42 body-feature shift.
    """

    values = np.asarray(summary, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] % 10 != 0:
        raise ValueError(
            "proprio summary must have shape (N, 10 * channels), "
            f"received {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("proprio summary contains non-finite values")
    channels = values.shape[1] // 10
    blocks = values.reshape(len(values), 10, channels)
    mean, std, minimum, maximum, q25, q75, first, last, delta, slope = (
        blocks[:, index] for index in range(10)
    )
    scale = np.maximum(std, 1.0e-3)

    def signed_log(block: np.ndarray) -> np.ndarray:
        return np.sign(block) * np.log1p(np.abs(block))

    relative = np.concatenate(
        [
            signed_log(delta),
            signed_log(20.0 * slope),
            signed_log(maximum - minimum),
            signed_log(q75 - q25),
            np.clip(delta / scale, -50.0, 50.0),
            np.clip((mean - first) / scale, -50.0, 50.0),
            np.clip((last - first) / scale, -50.0, 50.0),
            np.clip((maximum - first) / scale, -50.0, 50.0),
            np.clip((minimum - first) / scale, -50.0, 50.0),
        ],
        axis=1,
    ).astype(np.float32)
    if not np.isfinite(relative).all():
        raise RuntimeError("relative proprioception produced non-finite values")
    return relative


def invariant_relative_proprio_features(
    invariant_summary: np.ndarray,
    full_summary: np.ndarray,
) -> np.ndarray:
    """Combine the 80-D invariant descriptor and relative full-body dynamics."""

    invariant = np.asarray(invariant_summary, dtype=np.float32)
    relative = relative_proprio_features(full_summary)
    if invariant.ndim != 2 or len(invariant) != len(relative):
        raise ValueError("invariant and full proprio summaries must align")
    if not np.isfinite(invariant).all():
        raise ValueError("invariant proprio summary contains non-finite values")
    return np.concatenate([invariant, relative], axis=1).astype(np.float32)


def visual_context_features(visual: np.ndarray) -> np.ndarray:
    """Keep the mean and final CLIP blocks, excluding unstable frame delta.

    Kino-Fail's visual descriptor concatenates ``mean``, ``final``, and
    ``final-minus-first`` blocks.  The delta block is dominated by view motion
    in T2; scene-disjoint development therefore uses the two semantic context
    blocks.  The rule is dimensional and contains no scene or label metadata.
    """

    values = np.asarray(visual, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] % 3 != 0:
        raise ValueError(
            "visual descriptor must contain three equal temporal blocks, "
            f"received {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("visual features contain non-finite values")
    block = values.shape[1] // 3
    return values[:, : 2 * block]


def _aligned_probabilities(
    model: Any,
    values: np.ndarray,
    target_classes: list[str],
) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(values), dtype=np.float64)
    source = {str(label): index for index, label in enumerate(model.classes_)}
    output = np.zeros((len(values), len(target_classes)), dtype=np.float64)
    for index, label in enumerate(target_classes):
        if label in source:
            output[:, index] = probabilities[:, source[label]]
    normalizer = output.sum(axis=1, keepdims=True)
    return output / np.where(normalizer <= 0.0, 1.0, normalizer)


def _positive_probability(model: Any, values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(values), dtype=np.float64)
    lookup = {str(label): index for index, label in enumerate(model.classes_)}
    index = lookup.get("True", lookup.get("1"))
    if index is None:
        return np.zeros(len(values), dtype=np.float64)
    return probabilities[:, index]


def _proprio_estimator(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=512,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=-1,
    )


def _route_estimator(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=512,
        max_features="sqrt",
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=-1,
    )


def _vision_estimator(seed: int, *, regularization: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(regularization),
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=int(seed),
                ),
            ),
        ]
    )


@dataclass(frozen=True)
class ConflictInvariantPrediction:
    """Auditable prediction, default expert, and selected evidence route."""

    probabilities: np.ndarray
    proprio_probabilities: np.ndarray
    visual_t2_probabilities: np.ndarray
    vision_decisive_probability: np.ndarray
    routes: np.ndarray


@dataclass
class ConflictInvariantKiNO:
    """Proprio-default KiNO with a learned vision-decisive override."""

    proprio_expert: Any
    visual_t2_specialist: Any
    conflict_gate: Any
    classes: list[str]
    route_threshold: float
    detail_visual_dimension: int
    include_base_visual_in_specialist: bool
    fit_audit: dict[str, Any]

    @classmethod
    def fit(
        cls,
        visual: np.ndarray,
        full_proprio: np.ndarray,
        invariant_proprio: np.ndarray,
        labels: np.ndarray,
        group_ids: np.ndarray,
        vision_decisive: np.ndarray,
        *,
        seed: int,
        route_threshold: float = 0.11,
        detail_visual: np.ndarray | None = None,
        visual_regularization: float = 0.03,
        include_base_visual_in_specialist: bool = True,
    ) -> "ConflictInvariantKiNO":
        visual = np.asarray(visual, dtype=np.float32)
        full_proprio = np.asarray(full_proprio, dtype=np.float32)
        invariant_proprio = np.asarray(invariant_proprio, dtype=np.float32)
        labels = np.asarray(labels).astype(str)
        group_ids = np.asarray(group_ids).astype(str)
        vision_decisive = np.asarray(vision_decisive, dtype=bool)
        if not (
            len(visual)
            == len(full_proprio)
            == len(invariant_proprio)
            == len(labels)
            == len(group_ids)
            == len(vision_decisive)
        ):
            raise ValueError("all training arrays must have equal length")
        visual_context = visual_context_features(visual)
        if detail_visual is None:
            detail = np.empty((len(visual), 0), dtype=np.float32)
        else:
            detail = np.asarray(detail_visual, dtype=np.float32)
            if detail.ndim != 2 or len(detail) != len(visual):
                raise ValueError("detail visual features must align with RGB features")
            if not np.isfinite(detail).all():
                raise ValueError("detail visual features contain non-finite values")
        if not include_base_visual_in_specialist and detail.shape[1] == 0:
            raise ValueError(
                "detail-only visual specialist requires detail visual features"
            )
        visual_specialist = (
            np.concatenate([visual_context, detail], axis=1)
            if include_base_visual_in_specialist
            else detail
        )
        if not 0.0 <= float(route_threshold) <= 1.0:
            raise ValueError("route threshold must lie in [0, 1]")
        proprio = invariant_relative_proprio_features(
            invariant_proprio, full_proprio
        )
        classes = sorted(set(labels.tolist()))
        proprio_fit = ~vision_decisive
        if len(set(labels[proprio_fit])) < 2:
            raise ValueError("proprio expert needs at least two classes")
        if set(labels[vision_decisive]) != set(T2_CLASSES):
            raise ValueError(
                "vision-decisive supervision must contain exactly the two T2 classes"
            )
        if len(set(vision_decisive.tolist())) != 2:
            raise ValueError("conflict gate needs both route targets")

        proprio_expert = _proprio_estimator(seed + 11).fit(
            proprio[proprio_fit],
            labels[proprio_fit],
            sample_weight=physical_group_weights(group_ids[proprio_fit]),
        )
        visual_t2_specialist = _vision_estimator(
            seed + 23, regularization=float(visual_regularization)
        )
        visual_t2_specialist.fit(
            visual_specialist[vision_decisive],
            labels[vision_decisive],
            classifier__sample_weight=physical_group_weights(
                group_ids[vision_decisive]
            ),
        )
        conflict_gate = _route_estimator(seed + 37).fit(
            proprio,
            vision_decisive,
            sample_weight=physical_group_weights(group_ids),
        )
        return cls(
            proprio_expert=proprio_expert,
            visual_t2_specialist=visual_t2_specialist,
            conflict_gate=conflict_gate,
            classes=classes,
            route_threshold=float(route_threshold),
            detail_visual_dimension=int(detail.shape[1]),
            include_base_visual_in_specialist=bool(
                include_base_visual_in_specialist
            ),
            fit_audit={
                "architecture": (
                    "relative-proprio default with supervised vision-decisive override"
                ),
                "classes": classes,
                "proprio_feature_dimension": int(proprio.shape[1]),
                "visual_feature_dimension": int(visual.shape[1]),
                "visual_context_dimension": int(visual_context.shape[1]),
                "visual_context_blocks": ["mean", "final"],
                "base_visual_in_specialist": bool(
                    include_base_visual_in_specialist
                ),
                "detail_visual_dimension": int(detail.shape[1]),
                "visual_specialist_dimension": int(visual_specialist.shape[1]),
                "visual_regularization_C": float(visual_regularization),
                "proprio_training_samples": int(proprio_fit.sum()),
                "vision_specialist_training_samples": int(vision_decisive.sum()),
                "route_training_samples": len(labels),
                "route_threshold": float(route_threshold),
                "route_supervision_used_only_during_fit": True,
                "deployment_inputs": [
                    "five-frame visual descriptor",
                    *(
                        ["frozen fine-grained visual descriptor"]
                        if detail.shape[1] > 0
                        else []
                    ),
                    "190-D full proprio summary",
                    "80-D rotation-invariant proprio summary",
                ],
                "forbidden_deployment_inputs": [
                    "scene/domain ID",
                    "operator/material/severity ID",
                    "battery/cell ID",
                    "ground truth or outcome",
                ],
            },
        )

    def predict_with_routes(
        self,
        visual: np.ndarray,
        full_proprio: np.ndarray,
        invariant_proprio: np.ndarray,
        detail_visual: np.ndarray | None = None,
    ) -> ConflictInvariantPrediction:
        visual = np.asarray(visual, dtype=np.float32)
        visual_context = visual_context_features(visual)
        if detail_visual is None:
            detail = np.empty((len(visual), 0), dtype=np.float32)
        else:
            detail = np.asarray(detail_visual, dtype=np.float32)
        if detail.ndim != 2 or len(detail) != len(visual):
            raise ValueError("detail visual features must align with RGB features")
        if detail.shape[1] != self.detail_visual_dimension:
            raise ValueError(
                "detail visual dimension differs from fit: "
                f"{detail.shape[1]} != {self.detail_visual_dimension}"
            )
        visual_specialist = (
            np.concatenate([visual_context, detail], axis=1)
            if self.include_base_visual_in_specialist
            else detail
        )
        proprio = invariant_relative_proprio_features(
            invariant_proprio, full_proprio
        )
        if visual.ndim != 2 or len(visual) != len(proprio):
            raise ValueError("visual and proprio features must align")
        proprio_probability = _aligned_probabilities(
            self.proprio_expert, proprio, self.classes
        )
        t2_probability = _aligned_probabilities(
            self.visual_t2_specialist,
            visual_specialist,
            list(T2_CLASSES),
        )
        route_score = _positive_probability(self.conflict_gate, proprio)
        use_vision = route_score >= self.route_threshold
        output = proprio_probability.copy()
        if use_vision.any():
            output[use_vision] = 0.0
            target = {label: index for index, label in enumerate(self.classes)}
            for index, label in enumerate(T2_CLASSES):
                output[use_vision, target[label]] = t2_probability[
                    use_vision, index
                ]
        routes = np.where(use_vision, "vision", "proprio")
        return ConflictInvariantPrediction(
            probabilities=output,
            proprio_probabilities=proprio_probability,
            visual_t2_probabilities=t2_probability,
            vision_decisive_probability=route_score,
            routes=routes,
        )

    def predict_proba(
        self,
        visual: np.ndarray,
        full_proprio: np.ndarray,
        invariant_proprio: np.ndarray,
        detail_visual: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.predict_with_routes(
            visual, full_proprio, invariant_proprio, detail_visual
        ).probabilities
