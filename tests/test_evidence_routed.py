from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kino_vla.eval.evidence_routed import (  # noqa: E402
    EXPERT_NAMES,
    EvidenceRoutedNetwork,
    category_params,
    predict_category,
    regime_target,
)


def test_network_shapes_and_router_simplex():
    model = EvidenceRoutedNetwork.build(
        visual_dim=16, proprio_dim=11, n_categories=8, hidden=24
    )
    out = model(
        torch.randn(5, 16),
        torch.randn(5, 25, 11),
        torch.randn(5, 8),
        torch.randn(5, 8),
    )
    assert out["category_logits"].shape == (5, 8)
    assert out["intervention_logit"].shape == (5,)
    assert out["router_weights"].shape == (5, len(EXPERT_NAMES))
    torch.testing.assert_close(out["router_weights"].sum(dim=-1), torch.ones(5))
    assert out["theta"].shape == (5, 4)


def test_explicit_continue_gate_overrides_category_logits():
    categories = ["adhesion", "nominal", "overload"]
    outputs = {
        "category_logits": torch.tensor([[9.0, -2.0, 0.0], [0.0, 9.0, 1.0]]),
        "intervention_logit": torch.tensor([-8.0, 8.0]),
        "router_weights": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]] * 2),
    }
    pred, p, _ = predict_category(outputs, categories, continue_threshold=0.5)
    assert pred == ["nominal", "overload"]
    assert p[0] < 0.5 < p[1]


def test_regime_targets_are_training_only_evidence_roles():
    assert EXPERT_NAMES[regime_target("T2", "other", "adhesion")] == "material_pair"
    assert EXPERT_NAMES[regime_target("T4", "other", "overload")] == "joint"
    assert EXPERT_NAMES[regime_target("T3", "O8", "invisible_obstacle")] == "conflict_vla"
    assert EXPERT_NAMES[regime_target("T3", "reverse", "nominal")] == "conflict_vla"
    assert EXPERT_NAMES[regime_target("T5", "other", "nominal")] == "joint"


def test_structured_params_include_continue_and_fine_proprio_actions():
    assert category_params("nominal", "continue") == {}
    assert category_params("effort_decay", "Switch_Gait")["mode"] == "crawl"
    assert "payload" in category_params("overload", "Hold_and_Request")["reason"]


def test_observable_model_has_no_forbidden_metadata_argument():
    import inspect

    model = EvidenceRoutedNetwork.build(
        visual_dim=4, proprio_dim=2, n_categories=3, hidden=8, material_feature_dim=4
    )
    assert list(inspect.signature(model.forward).parameters) == [
        "visual",
        "proprio",
        "material_logits",
        "conflict_logits",
    ]


def test_forward_is_finite():
    model = EvidenceRoutedNetwork.build(
        visual_dim=4, proprio_dim=2, n_categories=3, hidden=8, material_feature_dim=4
    )
    out = model(
        torch.from_numpy(np.zeros((2, 4), dtype=np.float32)),
        torch.from_numpy(np.zeros((2, 5, 2), dtype=np.float32)),
        torch.from_numpy(np.zeros((2, 3), dtype=np.float32)),
        torch.from_numpy(np.zeros((2, 3), dtype=np.float32)),
    )
    assert all(torch.isfinite(value).all() for value in out.values())
