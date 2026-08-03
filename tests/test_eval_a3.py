"""Regression tests for A3's abstention truth and publication-row schema."""

from __future__ import annotations

from types import SimpleNamespace

from kino_vla.data.schema import CoTAnnotation, RecoveryPrimitive
from kino_vla.vla.output import ParsedDecision
from scripts.eval_a3 import Snap, score_agent, upgrade_cached_rows


def _t5_snap() -> Snap:
    return Snap(
        sid="a0corpus_O1_A_nominal_clear_frost_s500",
        cell="T5",
        t3_sub="other",
        truth="nominal",
        operator="O1_mu_field",
        admissible=frozenset({"continue"}),
        appearance_id="clear_frost",
        appearance_split="train",
        pair_id="",
        mu=None,
        snapshot=SimpleNamespace(),
    )


def test_upgrade_cached_t5_uses_abstention_truth_without_inventing_params():
    legacy = [
        {
            "sid": _t5_snap().sid,
            "cell": "T5",
            "truth": "low_friction",
            "attribution": "low_friction",
            "primitive": "Switch_Gait",
            "attr_ok": True,
            "parsed": True,
        }
    ]
    row = upgrade_cached_rows(legacy, [_t5_snap()])[0]
    assert row["truth"] == "nominal"
    assert row["attr_ok"] is False
    assert row["prim_feasible"] is False
    assert row["action_params_observed"] is False
    assert row["primitive_params"] == {}


def test_score_agent_persists_exact_primitive_params_and_joint_metric():
    snap = Snap(
        sid="a0corpus_O1_ice_sheet_s500",
        cell="T1",
        t3_sub="other",
        truth="low_friction",
        operator="O1_mu_field",
        admissible=frozenset({"Set_Constraint"}),
        appearance_id="ice_sheet",
        appearance_split="train",
        pair_id="",
        mu=None,
        snapshot=SimpleNamespace(),
    )
    ann = CoTAnnotation(
        thought="slippery",
        attribution="low_friction",
        primitive=RecoveryPrimitive("Set_Constraint", {"max_speed": 0.3, "stiffness": 0.5}),
        attribution_raw="low_friction",
        raw_text="",
    )
    policy = SimpleNamespace(
        decide=lambda _snapshot: ParsedDecision(ok=True, raw_text="", annotation=ann)
    )
    row = score_agent(policy, [snap])[0]
    assert row["primitive_params"] == {"max_speed": 0.3, "stiffness": 0.5}
    assert row["action_params_observed"] is True
    assert row["attr_ok"] is True
    assert row["prim_feasible"] is True
    assert row["joint_ok"] is True
