from __future__ import annotations

import numpy as np

from kino_vla.eval.realistic_multimodal import (
    ObservableClassifier,
    StructuredBidirectionalClassifier,
    physical_episode_weights,
    predictions_from_probabilities,
    proprio_summary,
    temporal_clip_summary,
)


def _toy() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(4)
    labels = []
    episode_ids = []
    groups = []
    visual = []
    proprio = []
    for class_index, label in enumerate(["nominal", "low_friction", "external_push"]):
        for group_index in range(6):
            group = f"g_{class_index}_{group_index}"
            for view in range(3):
                labels.append(label)
                episode_ids.append(f"{group}_{label}")
                groups.append(group)
                visual.append(rng.normal(class_index * 3.0, 0.1, 5))
                proprio.append(rng.normal(class_index * -2.0, 0.1, 4))
    return (
        np.asarray(visual, dtype=np.float32),
        np.asarray(proprio, dtype=np.float32),
        np.asarray(labels),
        np.asarray(episode_ids),
        np.asarray(groups),
    )


def test_summaries_and_episode_weighting() -> None:
    window = np.arange(24, dtype=np.float32).reshape(6, 4)
    assert proprio_summary(window).shape == (40,)
    assert temporal_clip_summary(window).shape == (12,)
    weights = physical_episode_weights(np.asarray(["a", "a", "a", "b"]))
    assert np.allclose(weights, [1 / 3, 1 / 3, 1 / 3, 1])


def test_realistic_models_fit_and_predict() -> None:
    visual, proprio, labels, episodes, groups = _toy()
    baseline = ObservableClassifier.fit(
        visual, proprio, labels, episodes, feature_key="early_fusion", seed=1
    )
    base_prob = baseline.predict_proba(visual, proprio)
    assert np.mean(predictions_from_probabilities(base_prob, baseline.classes) == labels) > 0.95

    ours = StructuredBidirectionalClassifier.fit(
        visual, proprio, labels, episodes, groups, seed=1
    )
    prob = ours.predict_proba(visual, proprio)
    assert prob.shape == (len(labels), 3)
    assert np.allclose(prob.sum(axis=1), 1.0)
    assert np.mean(predictions_from_probabilities(prob, ours.classes) == labels) > 0.95
