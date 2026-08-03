"""Pure-logic tests for A5.5 held-out selective prediction."""

from __future__ import annotations

import pytest

from scripts.a5_5_risk_coverage import (
    _actual_label,
    evaluate_protocol,
    risk_coverage_curve,
)


def _point(sid: str, split: str, appearance: str, score: float, agent_label: str) -> dict:
    return {
        "sample_id": sid,
        "scenario": "matched_O4_twophase",
        "appearance_id": appearance,
        "appearance_split": split,
        "cluster_id": f"matched_O4_twophase|{appearance}",
        "score": score,
        "agent_label": agent_label,
        "safe_label": "backstep_detour",
        "continue_label": "continue",
        "attr_ok": agent_label == "backstep_detour",
    }


def test_constant_score_has_only_endpoint_coverages():
    rows = [
        _point("a", "train", "amber", 0.1, "backstep_detour"),
        _point("b", "train", "yellow", 0.1, "high_step"),
    ]
    costs = {
        ("matched_O4_twophase", "backstep_detour"): 1.0,
        ("matched_O4_twophase", "high_step"): 4.0,
        ("matched_O4_twophase", "continue"): 3.0,
    }
    curve = risk_coverage_curve(rows, costs)
    assert [p["coverage"] for p in curve] == [0.0, 1.0]


def test_heldout_protocol_marks_constant_score_non_identifiable():
    rows = [
        _point("cal1", "train", "amber", 0.1, "backstep_detour"),
        _point("cal2", "train", "yellow", 0.1, "high_step"),
        _point("test1", "test", "gray", 0.1, "backstep_detour"),
        _point("test2", "test", "checker", 0.1, "high_step"),
    ]
    costs = {
        ("matched_O4_twophase", "backstep_detour"): 1.0,
        ("matched_O4_twophase", "high_step"): 4.0,
        ("matched_O4_twophase", "continue"): 3.0,
    }
    episodes = {key: [value] * 4 for key, value in costs.items()}
    result = evaluate_protocol(
        "t2_pair", rows, costs, episodes, bootstrap_reps=20, bootstrap_seed=7
    )
    assert result["score_support"]["selective_identifiable"] is False
    assert result["selected_min_cost"]["test"]["coverage"] in {0.0, 1.0}


def test_actual_action_rejects_legacy_gait_without_mode():
    with pytest.raises(ValueError, match="lost Switch_Gait mode"):
        _actual_label(
            {
                "sid": "legacy",
                "primitive": "Switch_Gait",
                "primitive_params": {},
                "action_params_observed": False,
            }
        )
