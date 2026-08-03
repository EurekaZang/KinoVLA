from __future__ import annotations

import numpy as np

from kino_vla.eval.unified_moe import (
    EXPERT_NAMES,
    UnifiedEvidenceMoE,
)


def _toy_data(seed: int = 4):
    rng = np.random.default_rng(seed)
    labels = np.asarray(["a", "b", "c"] * 24)
    groups = np.asarray([f"group_{index // 2:03d}" for index in range(len(labels))])
    episodes = np.asarray([f"episode_{index:03d}" for index in range(len(labels))])
    class_index = np.asarray([{"a": 0, "b": 1, "c": 2}[label] for label in labels])
    visual = rng.normal(size=(len(labels), 8)).astype(np.float32)
    proprio = rng.normal(size=(len(labels), 6)).astype(np.float32)
    visual[:, 0] += 2.0 * (class_index == 0)
    visual[:, 1] += 2.0 * (class_index == 1)
    proprio[:, 0] += 2.0 * (class_index == 1)
    proprio[:, 1] += 2.0 * (class_index == 2)
    return visual, proprio, labels, episodes, groups


def test_unified_moe_exposes_auditable_routes():
    visual, proprio, labels, episodes, groups = _toy_data()
    model = UnifiedEvidenceMoE.fit(
        visual,
        proprio,
        labels,
        episodes,
        groups,
        seed=9,
        minimum_override_precision=0.0,
        minimum_override_groups=1,
    )
    result = model.predict_with_routes(visual[:7], proprio[:7])
    assert result.probabilities.shape == (7, 3)
    assert result.expert_probabilities.shape == (7, 3, 3)
    assert result.route_probabilities.shape == (7, 3)
    assert result.route_scores.shape == (7, 2)
    assert result.routes.shape == (7,)
    assert result.evidence_routes.shape == (7,)
    assert set(result.routes) <= set(EXPERT_NAMES)
    assert set(result.evidence_routes) <= {
        "vision",
        "proprio",
        "consensus",
        "cross_modal_interaction",
    }
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert np.allclose(result.route_probabilities.sum(axis=1), 1.0)
    assert model.fit_audit["test_truth_or_outcomes_used"] is False


def test_predict_proba_matches_routed_interface():
    visual, proprio, labels, episodes, groups = _toy_data(8)
    model = UnifiedEvidenceMoE.fit(
        visual,
        proprio,
        labels,
        episodes,
        groups,
        seed=2,
        minimum_override_precision=0.0,
        minimum_override_groups=1,
    )
    routed = model.predict_with_routes(visual[:5], proprio[:5])
    assert np.allclose(
        model.predict_proba(visual[:5], proprio[:5]),
        routed.probabilities,
    )
