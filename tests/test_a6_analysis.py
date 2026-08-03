"""Regression tests for A6 boundary and abstention estimands."""

from __future__ import annotations

from scripts.a6_eval import _aggregate_decisions, _base_boundary


def test_base_boundary_reports_grid_bracket_not_false_midpoint_precision():
    base = {
        "theta_star": 0.2754,
        "rows": [
            {"floor": 0.4, "base_reach": 1.0, "n": 6},
            {"floor": 0.3, "base_reach": 1.0, "n": 6},
            {"floor": 0.25, "base_reach": 0.0, "n": 6},
            {"floor": 0.2, "base_reach": 0.0, "n": 6},
        ],
    }
    result = _base_boundary(base)
    assert result["primary_identified_bracket"] == [0.25, 0.3]
    assert result["bracket_width"] == 0.05
    assert result["logistic_midpoint_descriptive"] == 0.2754


def test_aggregate_keeps_nominal_attribution_separate_from_action():
    rows = [
        {
            "floor": 0.4,
            "parsed": True,
            "attribution_abstain": True,
            "action_intervene": True,
            "action_nonintervene": False,
            "attribution": "nominal",
            "primitive": "Set_Constraint",
        },
        {
            "floor": 0.4,
            "parsed": True,
            "attribution_abstain": False,
            "action_intervene": False,
            "action_nonintervene": True,
            "attribution": "low_friction",
            "primitive": "continue",
        },
    ]
    result = _aggregate_decisions(rows, "floor")[0]
    assert result["p_attribution_abstain"] == 0.5
    assert result["p_action_intervene"] == 0.5
    assert result["p_action_nonintervene"] == 0.5
