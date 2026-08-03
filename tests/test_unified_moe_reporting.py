from __future__ import annotations

import numpy as np

from scripts.eval_kinofail_unified_moe_v1 import (
    _action_report,
    _case_level,
    _selective_curve,
)


def test_case_level_counts_strict_pairs():
    rows = [
        {
            "group_id": "a",
            "prediction": "x",
            "truth": "x",
            "confidence": 0.9,
        },
        {
            "group_id": "a",
            "prediction": "y",
            "truth": "y",
            "confidence": 0.8,
        },
        {
            "group_id": "b",
            "prediction": "x",
            "truth": "z",
            "confidence": 0.7,
        },
        {
            "group_id": "b",
            "prediction": "y",
            "truth": "y",
            "confidence": 0.6,
        },
    ]
    result = _case_level(rows, "group_id")
    assert result["groups"] == 2
    assert result["mean_within_group_accuracy"] == 0.75
    assert result["strict_all_samples_correct"] == 1
    assert result["strict_all_samples_rate"] == 0.5
    assert result["_confidence"].tolist() == [0.8, 0.6]


def test_selective_curve_uses_fixed_group_coverage():
    curve = _selective_curve(
        np.asarray([1.0, 0.0, 1.0, 0.0]),
        np.asarray([0.9, 0.8, 0.7, 0.6]),
        [0.5, 1.0],
    )
    assert curve[0]["selected_groups"] == 2
    assert curve[0]["selective_risk"] == 0.5
    assert curve[1]["selected_groups"] == 4
    assert curve[1]["selective_risk"] == 0.5


def test_action_report_compares_measured_method_vectors():
    predictions = []
    for method, sample_predictions in {
        "learned_router": {"s_o4": "adhesion", "s_o2": "compliant_terrain"},
        "late_average": {"s_o4": "compliant_terrain", "s_o2": "compliant_terrain"},
    }.items():
        for sample_id, prediction in sample_predictions.items():
            predictions.append(
                {
                    "dataset": "scale",
                    "axis": "scene_and_material",
                    "method": method,
                    "seed": 0,
                    "sample_id": sample_id,
                    "classes": ["adhesion", "compliant_terrain"],
                    "probabilities": (
                        [0.9, 0.1]
                        if prediction == "adhesion"
                        else [0.1, 0.9]
                    ),
                    "prediction": prediction,
                }
            )
    schedule = [
        {
            "case_id": "o4",
            "source_sample_id": "s_o4",
            "scene_cluster": "scene",
            "domain": "life",
            "operator": "O4_tether",
            "truth_attribution": "adhesion",
        },
        {
            "case_id": "o2",
            "source_sample_id": "s_o2",
            "scene_cluster": "scene",
            "domain": "life",
            "operator": "O2_compliance",
            "truth_attribution": "compliant_terrain",
        },
    ]
    direct = {
        "paired_cases": [
            {
                "case_id": "o4",
                "selective": {
                    "terminal_cost": 1.0,
                    "success": True,
                    "fell": False,
                },
                "always_safe": {
                    "terminal_cost": 31.0,
                    "success": False,
                    "fell": False,
                },
            },
            {
                "case_id": "o2",
                "selective": {
                    "terminal_cost": 5.0,
                    "success": False,
                    "fell": False,
                },
                "always_safe": {
                    "terminal_cost": 5.0,
                    "success": False,
                    "fell": False,
                },
            },
        ]
    }
    report, _ = _action_report(
        predictions,
        schedule,
        direct,
        methods=["learned_router", "late_average"],
        seeds=[0],
        rule={
            "target": "adhesion",
            "mean_probability_threshold": 0.5,
            "minimum_seed_agreement": 1,
            "release_action": "backstep_release",
            "fallback_action": "hold_and_request",
        },
    )
    comparison = report["learned_router_pairwise"]["late_average"]
    assert comparison["action_disagreement_cases"] == 1
    assert comparison["mean_terminal_cost_delta"] == -15.0
    assert comparison["mean_success_delta"] == 0.5
    assert comparison["cost_wins_ties_losses"] == {
        "wins": 1,
        "ties": 1,
        "losses": 0,
    }
