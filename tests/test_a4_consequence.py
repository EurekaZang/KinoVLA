"""Unit tests for the A4.3 consequence-analysis core (pure logic; no Isaac).

Validates the C3 decision-theoretic machinery against SYNTHETIC outcomes that encode the predicted
antisymmetries — high_step on adhesion is catastrophic, backstep on mud is benign, continue on
nominal wins, intervention strands — so the safe-default crossover and ERS/Regret are exercised
end-to-end before the real-Go2 matrix is run. (The surrogate never produces A4 results; these tests
are import-smoke + logic scaffolding only, per CLAUDE.md §0.)
"""

from __future__ import annotations

import os
import sys

import pytest

# allow ``pytest tests/`` from repo root without an install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kino_vla.eval.a4_consequence import (  # noqa: E402
    LABELS,
    M_matrix,
    Outcome,
    agent_label_distribution,
    compose_agent,
    physical_cost,
    primitive_to_label,
    safe_default_crossover,
)


def _out(
    s: str,
    lab: str,
    *,
    reached=False,
    fell=False,
    immob=False,
    cat=False,
    t=10.0,
    peak=0.5,
    final=0.5,
    success=None,
) -> Outcome:
    succ = success if success is not None else (reached and not fell)
    return Outcome(s, lab, 0, reached, fell, immob, cat, final, t, peak, 1, False, succ)


def test_physical_cost_orders_catastrophic_above_benign():
    fall = _out("s", "l", fell=True)
    cat = _out("s", "l", cat=True)
    immob = _out("s", "l", immob=True)
    stranded = _out("s", "l", reached=False)
    slow = _out("s", "l", reached=True, t=30.0)
    clean = _out("s", "l", reached=True, t=10.0)
    costs = [physical_cost(o) for o in (clean, slow, stranded, immob, cat, fall)]
    assert costs == sorted(costs), "cost levels must be monotone ordinal"
    assert physical_cost(fall) > physical_cost(cat) > physical_cost(immob) > physical_cost(stranded)
    assert physical_cost(slow) > physical_cost(clean)


def test_m_matrix_diagonal_dominates_with_antisymmetry():
    # The T2 pair: matched_O4 (adhesion) and matched_O2 (mud). Canonical: O4→backstep, O2→high_step.
    # high_step on O4 is CATASTROPHIC; backstep on O2 is benign-slow (the off-diagonal).
    outs: list[Outcome] = []
    # O4 adhesion: backstep (canonical) succeeds; high_step catapults; continue strands.
    for _ in range(10):
        outs.append(_out("O4_adhesion", "backstep_detour", reached=True, success=True))
        outs.append(_out("O4_adhesion", "high_step", cat=True, success=False))
        outs.append(_out("O4_adhesion", "continue", reached=False, success=False))
    # O2 mud: high_step (canonical) succeeds clean; backstep succeeds-but-slow (benign off-diag).
    for _ in range(10):
        outs.append(_out("O2_mud", "high_step", reached=True, success=True))
        outs.append(_out("O2_mud", "backstep_detour", reached=True, t=30.0, success=True))
    m = M_matrix(outs, ["O4_adhesion", "O2_mud"], LABELS)
    assert m["O4_adhesion"]["backstep_detour"]["success_rate"] == 1.0
    assert m["O4_adhesion"]["high_step"]["catapult_rate"] == 1.0
    assert m["O4_adhesion"]["high_step"]["success_rate"] == 0.0
    assert m["O2_mud"]["high_step"]["success_rate"] == 1.0
    # antisymmetry: high_step@O4 cost ≫ backstep@O2 cost (catastrophic vs benign)
    assert m["O4_adhesion"]["high_step"]["mean_cost"] > m["O2_mud"]["backstep_detour"]["mean_cost"]


def test_safe_default_crossover_conservative_when_asymmetric():
    # The C3 "Safe" content: when high_step@adhesion is catastrophic and backstep@mud is benign,
    # the conservative recovery (backstep) is the safe default for any non-negligible P(adhesion).
    # Pass actions=(backstep_detour, high_step) so p = P(adhesion); off[high]=5 ≫ off[back]=1.
    diag = {"backstep_detour": 0.0, "high_step": 0.0}  # both correct actions are clean
    off = {
        "backstep_detour": 1.0,
        "high_step": 5.0,
    }  # high_step@adhesion catapult ≫ backstep@mud slow
    res = safe_default_crossover(diag, off, ("backstep_detour", "high_step"))
    assert res["p_star"] < 0.3, "crossover must be small when mis-attribution costs are asymmetric"
    assert (
        res["safe_default_above_pstar"] == "backstep_detour"
    )  # conservative wins for p_adhesion > p*
    assert res["cost_asymmetry_ratio"] == 5.0
    # inverse asymmetry: backstep@adhesion-now-benign ⇒ high_step becomes the conservative default
    res2 = safe_default_crossover(
        {"high_step": 0.0, "backstep_detour": 0.0},
        {"high_step": 1.0, "backstep_detour": 5.0},
        ("high_step", "backstep_detour"),
    )
    assert res2["p_star"] < 0.3
    assert res2["safe_default_above_pstar"] == "high_step"


def test_primitive_to_label_map_covers_registry_vocabulary():
    assert primitive_to_label("Backstep") == "backstep_detour"
    assert primitive_to_label("Set_Constraint") == "slow_low"
    assert primitive_to_label("Hold_and_Request") == "hold_request"
    assert primitive_to_label("Update_Topology") == "detour_replan"
    assert primitive_to_label("Switch_Gait", {"mode": "high_step"}) == "high_step"
    assert primitive_to_label("Switch_Gait", {"mode": "crawl"}) == "crawl"
    assert primitive_to_label("Switch_Gait", {}) == "high_step"
    assert primitive_to_label("Adjust_Posture") == "slow_low"
    assert primitive_to_label("continue") == "continue"
    assert primitive_to_label("Whatever") == "continue"


def test_compose_agent_regret_zero_for_perfect_attributor():
    # A perfect attributor always emits the canonical label ⇒ regret 0 and the canonical cost.
    outs: list[Outcome] = []
    canon = {"O4_adhesion": "backstep_detour", "O2_mud": "high_step"}
    for _ in range(10):
        outs.append(_out("O4_adhesion", "backstep_detour", reached=True, success=True))
        outs.append(_out("O4_adhesion", "high_step", cat=True))
        outs.append(_out("O2_mud", "high_step", reached=True, success=True))
        outs.append(_out("O2_mud", "backstep_detour", reached=True, t=30.0, success=True))
    m = M_matrix(outs, ["O4_adhesion", "O2_mud"], LABELS)
    perf = {s: {canon[s]: 1.0} for s in canon}
    comp = compose_agent(perf, m, ["O4_adhesion", "O2_mud"], canon)
    assert comp.regret_mean == 0.0
    assert comp.argmin_label_per_scenario["O4_adhesion"] == "backstep_detour"
    # a catastrophically-wrong attributor (always high_step) has large regret on adhesion
    wrong = {s: {"high_step": 1.0} for s in canon}
    comp2 = compose_agent(wrong, m, ["O4_adhesion", "O2_mud"], canon)
    assert comp2.regret_per_scenario["O4_adhesion"] > 0.0
    assert (
        comp2.ers_per_scenario["O4_adhesion"] > comp.ers_per_scenario["O4_adhesion"]
    )  # higher cost


def test_agent_label_distribution_from_per_item():
    per_item = [
        {"scenario": "O4_adhesion", "primitive": "Backstep"},
        {"scenario": "O4_adhesion", "primitive": "Backstep"},
        {
            "scenario": "O4_adhesion",
            "primitive": "Switch_Gait",
            "primitive_params": {"mode": "high_step"},
        },
        {
            "scenario": "O2_mud",
            "primitive": "Switch_Gait",
            "primitive_params": {"mode": "high_step"},
        },
    ]
    d = agent_label_distribution(per_item, "O4_adhesion")
    assert d["backstep_detour"] == pytest.approx(2 / 3, abs=1e-3)
    assert d["high_step"] == pytest.approx(1 / 3, abs=1e-3)
    d2 = agent_label_distribution(per_item, "O2_mud")
    assert d2["high_step"] == 1.0
