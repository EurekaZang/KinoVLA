from __future__ import annotations

import inspect

import numpy as np
import pytest

from kino_vla.eval.conflict_invariant_kino import (
    ConflictInvariantKiNO,
    invariant_relative_proprio_features,
    relative_proprio_features,
    visual_context_features,
)


def _summary(rows: int, offset: float) -> np.ndarray:
    rng = np.random.default_rng(11 + int(offset * 10))
    blocks = rng.normal(offset, 0.1, size=(rows, 10, 19)).astype(np.float32)
    blocks[:, 1] = np.abs(blocks[:, 1]) + 0.1
    blocks[:, 8] = offset
    blocks[:, 9] = 0.1 * offset
    return blocks.reshape(rows, 190)


def test_relative_proprio_features_are_finite_and_shape_checked() -> None:
    summary = _summary(7, 1.0)
    relative = relative_proprio_features(summary)
    assert relative.shape == (7, 171)
    assert np.isfinite(relative).all()
    combined = invariant_relative_proprio_features(
        np.ones((7, 80), dtype=np.float32), summary
    )
    assert combined.shape == (7, 251)
    visual = np.arange(84, dtype=np.float32).reshape(7, 12)
    assert np.array_equal(visual_context_features(visual), visual[:, :8])


def test_deployment_signature_has_no_privileged_metadata() -> None:
    parameters = set(
        inspect.signature(ConflictInvariantKiNO.predict_with_routes).parameters
    )
    assert parameters == {
        "self",
        "visual",
        "full_proprio",
        "invariant_proprio",
        "detail_visual",
    }


def test_conflict_router_selects_visual_pattern_and_normalizes_output() -> None:
    rng = np.random.default_rng(23)
    samples = 48
    visual = rng.normal(size=(samples, 12)).astype(np.float32)
    full = np.concatenate(
        [_summary(samples // 2, 0.2), _summary(samples // 2, 2.0)], axis=0
    )
    invariant = np.concatenate(
        [
            np.full((samples // 2, 80), 0.2, dtype=np.float32),
            np.full((samples // 2, 80), 2.0, dtype=np.float32),
        ],
        axis=0,
    )
    labels = np.asarray(
        ["adhesion", "compliant_terrain"] * (samples // 4)
        + ["low_friction", "invisible_obstacle"] * (samples // 4)
    )
    vision_decisive = np.arange(samples) < samples // 2
    visual[vision_decisive & (labels == "adhesion"), 0] -= 5.0
    visual[vision_decisive & (labels == "compliant_terrain"), 0] += 5.0
    groups = np.asarray([f"g{index // 2}" for index in range(samples)])
    model = ConflictInvariantKiNO.fit(
        visual,
        full,
        invariant,
        labels,
        groups,
        vision_decisive,
        seed=5,
        route_threshold=0.5,
        detail_visual=np.ones((samples, 6), dtype=np.float32),
    )
    prediction = model.predict_with_routes(
        visual,
        full,
        invariant,
        np.ones((samples, 6), dtype=np.float32),
    )
    assert prediction.probabilities.shape == (samples, 4)
    assert np.allclose(prediction.probabilities.sum(axis=1), 1.0)
    assert (prediction.routes[vision_decisive] == "vision").mean() > 0.9
    assert (prediction.routes[~vision_decisive] == "proprio").mean() > 0.9


def test_detail_only_specialist_requires_and_reuses_detail_features() -> None:
    rng = np.random.default_rng(31)
    samples = 48
    visual = rng.normal(size=(samples, 12)).astype(np.float32)
    detail = rng.normal(size=(samples, 8)).astype(np.float32)
    full = np.concatenate(
        [_summary(samples // 2, 0.2), _summary(samples // 2, 2.0)], axis=0
    )
    invariant = np.concatenate(
        [
            np.full((samples // 2, 80), 0.2, dtype=np.float32),
            np.full((samples // 2, 80), 2.0, dtype=np.float32),
        ],
        axis=0,
    )
    labels = np.asarray(
        ["adhesion", "compliant_terrain"] * (samples // 4)
        + ["low_friction", "invisible_obstacle"] * (samples // 4)
    )
    vision_decisive = np.arange(samples) < samples // 2
    detail[vision_decisive & (labels == "adhesion"), 0] -= 5.0
    detail[vision_decisive & (labels == "compliant_terrain"), 0] += 5.0
    groups = np.asarray([f"g{index // 2}" for index in range(samples)])
    model = ConflictInvariantKiNO.fit(
        visual,
        full,
        invariant,
        labels,
        groups,
        vision_decisive,
        seed=7,
        route_threshold=0.5,
        detail_visual=detail,
        include_base_visual_in_specialist=False,
    )
    prediction = model.predict_with_routes(visual, full, invariant, detail)
    assert model.fit_audit["base_visual_in_specialist"] is False
    assert np.allclose(prediction.probabilities.sum(axis=1), 1.0)
    with pytest.raises(ValueError, match="detail-only"):
        ConflictInvariantKiNO.fit(
            visual,
            full,
            invariant,
            labels,
            groups,
            vision_decisive,
            seed=7,
            include_base_visual_in_specialist=False,
        )
