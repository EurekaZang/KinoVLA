"""Frozen low-capacity models for realistic Kino-Fail A2/A3/A5.

The models consume only model-visible RGB and proprioception.  Scene, operator, severity,
appearance IDs, and privileged simulator telemetry are metadata used for splitting and clustered
statistics, never deployment inputs.  Appearance swaps from one physical episode receive weights
that sum to one so they cannot inflate the effective training sample size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold


def proprio_summary(window: np.ndarray) -> np.ndarray:
    """Fixed temporal summary, including direction as well as distributional statistics."""

    value = np.asarray(window, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 2 or not np.isfinite(value).all():
        raise ValueError(f"expected finite (T,F) proprioception with T>=2, got {value.shape}")
    time = np.arange(value.shape[0], dtype=np.float64)
    centered = time - time.mean()
    slope = (centered[:, None] * value).sum(axis=0) / np.square(centered).sum()
    return np.concatenate(
        [
            value.mean(axis=0),
            value.std(axis=0),
            value.min(axis=0),
            value.max(axis=0),
            np.quantile(value, 0.25, axis=0),
            np.quantile(value, 0.75, axis=0),
            value[0],
            value[-1],
            value[-1] - value[0],
            slope,
        ]
    ).astype(np.float32)


def temporal_clip_summary(frame_embeddings: np.ndarray) -> np.ndarray:
    """Mean/last/delta CLIP summary from the frozen five-frame body-fixed sequence."""

    value = np.asarray(frame_embeddings, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 2 or not np.isfinite(value).all():
        raise ValueError(f"expected finite (T,D) frame embeddings with T>=2, got {value.shape}")
    return np.concatenate([value.mean(axis=0), value[-1], value[-1] - value[0]]).astype(
        np.float32
    )


def physical_episode_weights(ids: np.ndarray) -> np.ndarray:
    """Each physical episode contributes total weight one, irrespective of appearance views."""

    values = np.asarray(ids).astype(str)
    unique, counts = np.unique(values, return_counts=True)
    count_by_id = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    return np.asarray([1.0 / count_by_id[value] for value in values], dtype=np.float64)


def _estimator(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=192,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=-1,
    )


def _aligned_probabilities(model: Any, values: np.ndarray, classes: list[str]) -> np.ndarray:
    raw = np.asarray(model.predict_proba(values), dtype=np.float64)
    out = np.zeros((len(values), len(classes)), dtype=np.float64)
    lookup = {str(value): index for index, value in enumerate(model.classes_)}
    for index, label in enumerate(classes):
        if label in lookup:
            out[:, index] = raw[:, lookup[label]]
    normalizer = out.sum(axis=1, keepdims=True)
    return out / np.where(normalizer <= 0.0, 1.0, normalizer)


def _reindex_probabilities(
    probabilities: np.ndarray,
    source_classes: list[str],
    target_classes: list[str],
) -> np.ndarray:
    """Reindex an already-normalized probability matrix by class name."""

    source = {str(label): index for index, label in enumerate(source_classes)}
    out = np.zeros((len(probabilities), len(target_classes)), dtype=np.float64)
    for index, label in enumerate(target_classes):
        if label in source:
            out[:, index] = np.asarray(probabilities)[:, source[label]]
    normalizer = out.sum(axis=1, keepdims=True)
    return out / np.where(normalizer <= 0.0, 1.0, normalizer)


def _probability_margin(probabilities: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(probabilities, dtype=np.float64), axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _probability_entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(probabilities, dtype=np.float64), 1.0e-12)
    return -(values * np.log(values)).sum(axis=1)


@dataclass
class ObservableClassifier:
    """One ordinary closed-set baseline over a predeclared observable feature block."""

    model: Any
    classes: list[str]
    feature_key: str

    @classmethod
    def fit(
        cls,
        visual: np.ndarray,
        proprio: np.ndarray,
        labels: np.ndarray,
        episode_ids: np.ndarray,
        *,
        feature_key: str,
        seed: int,
    ) -> "ObservableClassifier":
        blocks = {
            "vision": np.asarray(visual),
            "proprio": np.asarray(proprio),
            "early_fusion": np.concatenate([visual, proprio], axis=1),
        }
        if feature_key not in blocks:
            raise ValueError(f"unknown feature block: {feature_key}")
        model = _estimator(seed)
        model.fit(
            blocks[feature_key],
            np.asarray(labels).astype(str),
            sample_weight=physical_episode_weights(episode_ids),
        )
        return cls(model=model, classes=sorted(set(np.asarray(labels).astype(str))), feature_key=feature_key)

    def predict_proba(self, visual: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        blocks = {
            "vision": np.asarray(visual),
            "proprio": np.asarray(proprio),
            "early_fusion": np.concatenate([visual, proprio], axis=1),
        }
        return _aligned_probabilities(self.model, blocks[self.feature_key], self.classes)


@dataclass
class StructuredBidirectionalClassifier:
    """Nominal gate plus leakage-safe stacking of vision/proprio/joint anomaly experts."""

    gate: Any
    vision_expert: Any
    proprio_expert: Any
    joint_expert: Any
    meta: Any
    anomaly_classes: list[str]
    classes: list[str]

    @staticmethod
    def _expert_meta_features(
        vision: np.ndarray, proprio: np.ndarray, joint: np.ndarray
    ) -> np.ndarray:
        eps = 1.0e-9
        blocks = [vision, proprio, joint]
        diagnostics = []
        for block in blocks:
            sorted_prob = np.sort(block, axis=1)
            entropy = -(block * np.log(np.maximum(block, eps))).sum(axis=1)
            diagnostics.extend(
                [block.max(axis=1), sorted_prob[:, -1] - sorted_prob[:, -2], entropy]
            )
        diagnostics.extend(
            [
                np.abs(vision - proprio).sum(axis=1),
                np.abs(vision - joint).sum(axis=1),
                np.abs(proprio - joint).sum(axis=1),
            ]
        )
        return np.concatenate([*blocks, np.stack(diagnostics, axis=1)], axis=1)

    @classmethod
    def fit(
        cls,
        visual: np.ndarray,
        proprio: np.ndarray,
        labels: np.ndarray,
        episode_ids: np.ndarray,
        group_ids: np.ndarray,
        *,
        seed: int,
    ) -> "StructuredBidirectionalClassifier":
        visual = np.asarray(visual, dtype=np.float32)
        proprio = np.asarray(proprio, dtype=np.float32)
        labels = np.asarray(labels).astype(str)
        episode_ids = np.asarray(episode_ids).astype(str)
        group_ids = np.asarray(group_ids).astype(str)
        joint = np.concatenate([visual, proprio], axis=1)
        weights = physical_episode_weights(episode_ids)

        gate = _estimator(seed).fit(joint, labels != "nominal", sample_weight=weights)
        anomaly = labels != "nominal"
        if len(set(labels[anomaly])) < 2:
            raise ValueError("structured model requires at least two anomaly categories")
        av, ap, aj = visual[anomaly], proprio[anomaly], joint[anomaly]
        ay, ae, ag = labels[anomaly], episode_ids[anomaly], group_ids[anomaly]
        aw = physical_episode_weights(ae)
        anomaly_classes = sorted(set(ay))

        class_group_counts = [len(set(ag[ay == label])) for label in anomaly_classes]
        n_splits = min(3, min(class_group_counts))
        if n_splits < 2:
            raise ValueError("each anomaly class needs at least two independent groups")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_v = np.zeros((len(ay), len(anomaly_classes)), dtype=np.float64)
        oof_p = np.zeros_like(oof_v)
        oof_j = np.zeros_like(oof_v)
        for fold, (fit_idx, hold_idx) in enumerate(splitter.split(av, ay, groups=ag)):
            for values, target in ((av, oof_v), (ap, oof_p), (aj, oof_j)):
                model = _estimator(seed + 100 * (fold + 1) + len(values[0]))
                model.fit(values[fit_idx], ay[fit_idx], sample_weight=aw[fit_idx])
                target[hold_idx] = _aligned_probabilities(
                    model, values[hold_idx], anomaly_classes
                )
        meta_x = cls._expert_meta_features(oof_v, oof_p, oof_j)
        meta = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        ).fit(meta_x, ay, sample_weight=aw)

        vision_expert = _estimator(seed + 1001).fit(av, ay, sample_weight=aw)
        proprio_expert = _estimator(seed + 2001).fit(ap, ay, sample_weight=aw)
        joint_expert = _estimator(seed + 3001).fit(aj, ay, sample_weight=aw)
        return cls(
            gate=gate,
            vision_expert=vision_expert,
            proprio_expert=proprio_expert,
            joint_expert=joint_expert,
            meta=meta,
            anomaly_classes=anomaly_classes,
            classes=["nominal", *anomaly_classes],
        )

    def predict_proba(self, visual: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        visual = np.asarray(visual, dtype=np.float32)
        proprio = np.asarray(proprio, dtype=np.float32)
        joint = np.concatenate([visual, proprio], axis=1)
        # sklearn stringifies booleans differently across versions; read the positive class directly.
        gate_raw = np.asarray(self.gate.predict_proba(joint), dtype=np.float64)
        gate_lookup = {str(value): index for index, value in enumerate(self.gate.classes_)}
        p_anomaly = gate_raw[:, gate_lookup.get("True", gate_lookup.get("1", 1))]
        pv = _aligned_probabilities(self.vision_expert, visual, self.anomaly_classes)
        pp = _aligned_probabilities(self.proprio_expert, proprio, self.anomaly_classes)
        pj = _aligned_probabilities(self.joint_expert, joint, self.anomaly_classes)
        meta_x = self._expert_meta_features(pv, pp, pj)
        anomaly_prob = _aligned_probabilities(self.meta, meta_x, self.anomaly_classes)
        out = np.zeros((len(visual), len(self.classes)), dtype=np.float64)
        out[:, 0] = 1.0 - p_anomaly
        out[:, 1:] = p_anomaly[:, None] * anomaly_prob
        return out / out.sum(axis=1, keepdims=True)


@dataclass
class ConflictGuardedBidirectionalClassifier:
    """Proprio-preserving multimodal model with OOF-certified selective overrides.

    The strong proprioception model is the default expert.  The structured
    bidirectional candidate may override it only when an out-of-fold router,
    trained without held-out test data, predicts that the candidate is more
    reliable.  If no threshold improves grouped development predictions at the
    frozen precision floor, the model becomes exactly the proprio baseline.
    """

    proprio_default: ObservableClassifier
    structured_candidate: StructuredBidirectionalClassifier
    router: Any | None
    router_threshold: float
    classes: list[str]
    fit_audit: dict[str, Any]

    @staticmethod
    def _router_features(
        default: np.ndarray,
        candidate: np.ndarray,
    ) -> np.ndarray:
        default = np.asarray(default, dtype=np.float64)
        candidate = np.asarray(candidate, dtype=np.float64)
        default_pred = default.argmax(axis=1)
        candidate_pred = candidate.argmax(axis=1)
        diagnostics = np.stack(
            [
                default.max(axis=1),
                candidate.max(axis=1),
                _probability_margin(default),
                _probability_margin(candidate),
                _probability_entropy(default),
                _probability_entropy(candidate),
                np.abs(default - candidate).sum(axis=1),
                (default_pred == candidate_pred).astype(np.float64),
            ],
            axis=1,
        )
        return np.concatenate([default, candidate, diagnostics], axis=1)

    @staticmethod
    def _routed_probabilities(
        default: np.ndarray,
        candidate: np.ndarray,
        router_scores: np.ndarray,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        default = np.asarray(default, dtype=np.float64)
        candidate = np.asarray(candidate, dtype=np.float64)
        disagree = default.argmax(axis=1) != candidate.argmax(axis=1)
        override = disagree & (np.asarray(router_scores) >= float(threshold))
        out = default.copy()
        out[override] = candidate[override]
        return out, override

    @classmethod
    def fit(
        cls,
        visual: np.ndarray,
        proprio: np.ndarray,
        labels: np.ndarray,
        episode_ids: np.ndarray,
        group_ids: np.ndarray,
        *,
        seed: int,
        minimum_override_precision: float = 0.80,
        minimum_override_groups: int = 6,
    ) -> "ConflictGuardedBidirectionalClassifier":
        visual = np.asarray(visual, dtype=np.float32)
        proprio = np.asarray(proprio, dtype=np.float32)
        labels = np.asarray(labels).astype(str)
        episode_ids = np.asarray(episode_ids).astype(str)
        group_ids = np.asarray(group_ids).astype(str)
        classes = sorted(set(labels))
        class_group_counts = [
            len(set(group_ids[labels == label])) for label in classes
        ]
        n_splits = min(3, min(class_group_counts))
        if n_splits < 2:
            raise ValueError(
                "conflict-guarded model needs at least two independent groups per class"
            )

        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=int(seed)
        )
        oof_default = np.zeros((len(labels), len(classes)), dtype=np.float64)
        oof_candidate = np.zeros_like(oof_default)
        for fold, (fit_idx, hold_idx) in enumerate(
            splitter.split(visual, labels, groups=group_ids)
        ):
            default = ObservableClassifier.fit(
                visual[fit_idx],
                proprio[fit_idx],
                labels[fit_idx],
                episode_ids[fit_idx],
                feature_key="proprio",
                seed=int(seed) + 101 * (fold + 1),
            )
            candidate = StructuredBidirectionalClassifier.fit(
                visual[fit_idx],
                proprio[fit_idx],
                labels[fit_idx],
                episode_ids[fit_idx],
                group_ids[fit_idx],
                seed=int(seed) + 1009 * (fold + 1),
            )
            oof_default[hold_idx] = _reindex_probabilities(
                default.predict_proba(visual[hold_idx], proprio[hold_idx]),
                default.classes,
                classes,
            )
            oof_candidate[hold_idx] = _reindex_probabilities(
                candidate.predict_proba(visual[hold_idx], proprio[hold_idx]),
                candidate.classes,
                classes,
            )

        truth_index = np.asarray([classes.index(label) for label in labels])
        default_pred = oof_default.argmax(axis=1)
        candidate_pred = oof_candidate.argmax(axis=1)
        default_correct = default_pred == truth_index
        candidate_correct = candidate_pred == truth_index
        informative = default_correct != candidate_correct
        router = None
        router_scores = np.zeros(len(labels), dtype=np.float64)
        router_train_count = int(informative.sum())
        if router_train_count >= 10 and len(set(candidate_correct[informative])) == 2:
            router = LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=2000,
                random_state=int(seed),
            ).fit(
                cls._router_features(oof_default, oof_candidate)[informative],
                candidate_correct[informative],
                sample_weight=physical_episode_weights(episode_ids[informative]),
            )
            router_scores = np.asarray(
                router.predict_proba(
                    cls._router_features(oof_default, oof_candidate)
                )[:, list(router.classes_).index(True)],
                dtype=np.float64,
            )

        base_balanced_accuracy = float(
            np.mean(
                [
                    np.mean(default_pred[labels == label] == truth_index[labels == label])
                    for label in classes
                ]
            )
        )
        candidates: list[dict[str, Any]] = []
        for threshold in np.asarray(
            [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]
        ):
            routed, override = cls._routed_probabilities(
                oof_default, oof_candidate, router_scores, float(threshold)
            )
            routed_pred = routed.argmax(axis=1)
            override_groups = len(set(group_ids[override]))
            override_precision = (
                float(np.mean(candidate_correct[override])) if override.any() else 0.0
            )
            balanced_accuracy = float(
                np.mean(
                    [
                        np.mean(
                            routed_pred[labels == label] == truth_index[labels == label]
                        )
                        for label in classes
                    ]
                )
            )
            accepted = (
                override_groups >= int(minimum_override_groups)
                and override_precision >= float(minimum_override_precision)
                and balanced_accuracy >= base_balanced_accuracy
            )
            candidates.append(
                {
                    "threshold": float(threshold),
                    "override_sample_count": int(override.sum()),
                    "override_group_count": override_groups,
                    "override_precision": override_precision,
                    "balanced_accuracy": balanced_accuracy,
                    "accepted": accepted,
                }
            )
        accepted = [row for row in candidates if row["accepted"]]
        if accepted:
            selected = max(
                accepted,
                key=lambda row: (
                    row["balanced_accuracy"],
                    row["override_precision"],
                    -row["override_sample_count"],
                    row["threshold"],
                ),
            )
            threshold = float(selected["threshold"])
        else:
            selected = {
                "threshold": 2.0,
                "override_sample_count": 0,
                "override_group_count": 0,
                "override_precision": 0.0,
                "balanced_accuracy": base_balanced_accuracy,
                "accepted": True,
                "fallback_to_exact_proprio_default": True,
            }
            threshold = 2.0

        final_default = ObservableClassifier.fit(
            visual,
            proprio,
            labels,
            episode_ids,
            feature_key="proprio",
            seed=int(seed),
        )
        final_candidate = StructuredBidirectionalClassifier.fit(
            visual,
            proprio,
            labels,
            episode_ids,
            group_ids,
            seed=int(seed),
        )
        return cls(
            proprio_default=final_default,
            structured_candidate=final_candidate,
            router=router,
            router_threshold=threshold,
            classes=classes,
            fit_audit={
                "selection_data": "grouped out-of-fold training predictions only",
                "n_splits": n_splits,
                "router_train_count": router_train_count,
                "minimum_override_precision": float(minimum_override_precision),
                "minimum_override_groups": int(minimum_override_groups),
                "proprio_default_oof_balanced_accuracy": base_balanced_accuracy,
                "threshold_candidates": candidates,
                "selected": selected,
                "test_outcomes_used": False,
            },
        )

    def predict_proba(
        self, visual: np.ndarray, proprio: np.ndarray
    ) -> np.ndarray:
        default = _reindex_probabilities(
            self.proprio_default.predict_proba(visual, proprio),
            self.proprio_default.classes,
            self.classes,
        )
        candidate = _reindex_probabilities(
            self.structured_candidate.predict_proba(visual, proprio),
            self.structured_candidate.classes,
            self.classes,
        )
        if self.router is None or self.router_threshold > 1.0:
            return default
        scores = np.asarray(
            self.router.predict_proba(
                self._router_features(default, candidate)
            )[:, list(self.router.classes_).index(True)],
            dtype=np.float64,
        )
        routed, _ = self._routed_probabilities(
            default, candidate, scores, self.router_threshold
        )
        return routed


def predictions_from_probabilities(probabilities: np.ndarray, classes: list[str]) -> np.ndarray:
    return np.asarray(classes, dtype=str)[np.argmax(probabilities, axis=1)]
