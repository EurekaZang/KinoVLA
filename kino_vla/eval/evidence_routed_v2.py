"""Low-capacity structured evidence router for the A3 successor v2.

The first successor exposed a confirmatory failure caused by routing novel appearances through a
high-capacity VLA category expert.  V2 limits each decision to auditable observables: temporal
proprioception summaries, train-fitted RGB material-prototype distances, and colour chroma.  It
uses no scenario, operator, appearance id, taxonomy cell, truth, or privileged physics at
deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kino_vla.eval.material_support import (
    dominant_chromatic_rgb,
    matched_material_pair_logits,
    material_evidence_features,
)

FORBIDDEN_DEPLOYMENT_INPUTS: tuple[str, ...] = (
    "operator",
    "operator_name",
    "scenario",
    "scenario_id",
    "appearance_id",
    "appearance_split",
    "taxonomy_cell",
    "truth",
    "theta",
    "privileged_theta",
)


def proprio_summary(window: np.ndarray) -> np.ndarray:
    """Fixed distributional summary of an observable temporal proprioception window."""
    value = np.asarray(window, dtype=np.float64)
    if value.ndim != 2 or not value.shape[0] or not np.isfinite(value).all():
        raise ValueError(f"expected finite (T,F) proprioception, got {value.shape}")
    return np.concatenate(
        [
            value.mean(axis=0),
            value.std(axis=0),
            value.min(axis=0),
            value.max(axis=0),
            np.quantile(value, 0.25, axis=0),
            np.quantile(value, 0.75, axis=0),
        ]
    ).astype(np.float32)


@dataclass(frozen=True)
class StructuredFeatures:
    proprio: np.ndarray
    material: np.ndarray
    joint: np.ndarray
    material_rgb: np.ndarray


def observable_features(
    rgb: np.ndarray,
    proprio_window: np.ndarray,
    material_models: dict[str, dict[str, Any]],
    material_classes: tuple[str, ...],
) -> StructuredFeatures:
    material_rgb = dominant_chromatic_rgb(rgb)
    material = material_evidence_features(
        material_rgb, material_models, classes=material_classes
    )
    proprio = proprio_summary(proprio_window)
    return StructuredFeatures(
        proprio=proprio,
        material=material,
        joint=np.concatenate([proprio, material]).astype(np.float32),
        material_rgb=material_rgb.astype(np.float32),
    )


class StructuredEvidencePolicyV2:
    """Two-stage continue gate plus proprio-routed category experts."""

    def __init__(
        self,
        *,
        intervention_model: Any,
        regime_model: Any,
        category_model: Any,
        material_models: dict[str, dict[str, Any]],
        material_classes: list[str],
        categories: list[str],
        continue_threshold: float,
        material_pair_logit_scale: float,
    ) -> None:
        self.intervention_model = intervention_model
        self.regime_model = regime_model
        self.category_model = category_model
        self.material_models = material_models
        self.material_classes = tuple(material_classes)
        self.categories = list(categories)
        self.continue_threshold = float(continue_threshold)
        self.material_pair_logit_scale = float(material_pair_logit_scale)

    @staticmethod
    def _positive_probability(model: Any, values: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(model.predict_proba(values), dtype=np.float64)
        classes = list(model.classes_)
        if True not in classes and 1 not in classes:
            return np.zeros(len(values), dtype=np.float64)
        target = classes.index(True) if True in classes else classes.index(1)
        return probabilities[:, target]

    def predict_features(
        self, features: list[StructuredFeatures]
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        joint = np.stack([feature.joint for feature in features])
        proprio = np.stack([feature.proprio for feature in features])
        p_intervene = self._positive_probability(self.intervention_model, joint)
        p_material = self._positive_probability(self.regime_model, proprio)
        structured = self.category_model.predict(joint).astype(str)
        predictions = []
        for index, feature in enumerate(features):
            if p_intervene[index] < self.continue_threshold:
                predictions.append("nominal")
            elif p_material[index] >= 0.5:
                logits = matched_material_pair_logits(
                    feature.material_rgb,
                    self.material_models,
                    self.categories,
                    scale=self.material_pair_logit_scale,
                )
                predictions.append(self.categories[int(np.argmax(logits))])
            else:
                predictions.append(str(structured[index]))
        return predictions, p_intervene, p_material

    def predict_arrays(
        self, rgbs: list[np.ndarray], proprio_windows: list[np.ndarray]
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        if len(rgbs) != len(proprio_windows):
            raise ValueError("RGB and proprioception batches must have equal length")
        features = [
            observable_features(
                rgb,
                proprio,
                self.material_models,
                self.material_classes,
            )
            for rgb, proprio in zip(rgbs, proprio_windows, strict=True)
        ]
        return self.predict_features(features)
