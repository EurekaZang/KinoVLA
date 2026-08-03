"""Tests for paired A4 composition bootstrap plumbing."""

from __future__ import annotations

from kino_vla.eval.a4_consequence import LABELS, Outcome
from scripts.a4_analyze import bootstrap_compositions


def _out(label: str, success: bool) -> Outcome:
    return Outcome(
        scenario="matched_O4_twophase",
        label=label,
        seed=0,
        reached=success,
        fell=False,
        immobilized=not success,
        catapult=False,
        final_dist_m=0.0,
        sim_time_s=10.0,
        peak_omega=0.0,
        n_semantic_interventions=1,
        left_hazard=success,
        success=success,
    )


def _row(sid: str, appearance: str, primitive: str, params: dict | None = None) -> dict:
    return {
        "sid": sid,
        "cell": "T2",
        "truth": "adhesion",
        "t3_sub": "other",
        "appearance_id": appearance,
        "attribution": "adhesion" if primitive == "Backstep" else "compliant_terrain",
        "primitive": primitive,
        "primitive_params": params or {},
        "action_params_observed": True,
    }


def test_paired_bootstrap_preserves_agent_ordering():
    per_items = {
        "B5_conflict": [_row("a", "amber", "Backstep"), _row("b", "gray", "Backstep")],
        "B1": [
            _row("a", "amber", "Switch_Gait", {"mode": "high_step"}),
            _row("b", "gray", "Switch_Gait", {"mode": "high_step"}),
        ],
    }
    outcomes = [_out(label, label == "backstep_detour") for label in LABELS]
    result = bootstrap_compositions(
        per_items,
        outcomes,
        ["matched_O4_twophase"],
        {"matched_O4_twophase": "backstep_detour"},
        reps=30,
        seed=5,
    )
    paired = result["by_source"]["actual_action"]["paired_differences"]["B1"]
    delta = paired["B5_conflict_minus_B1_ers_mean_cost"]
    assert delta["ci95"][1] < 0.0
