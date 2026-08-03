from __future__ import annotations

import pytest

from scripts.analyze_kinofail_reconfirmation_a6_v7 import anomaly_only


def test_nominal_accuracy_cannot_mask_anomaly_operator_accuracy() -> None:
    rows = [
        {"condition": "nominal_counterfactual", "correct": 1.0},
        {"condition": "anomaly", "correct": 0.0},
    ]
    selected = anomaly_only(rows)
    assert len(selected) == 1
    assert selected[0]["condition"] == "anomaly"
    assert selected[0]["correct"] == 0.0


def test_both_counterfactual_conditions_are_required() -> None:
    with pytest.raises(RuntimeError, match="unexpected A6 conditions"):
        anomaly_only([{"condition": "anomaly", "correct": 1.0}])
