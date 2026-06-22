"""Closed-loop B-class scenarios + safe-halt metric + the B2 rule-FSM baseline (Gap-3 / §2.5)."""

from __future__ import annotations

from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.utils.config import load_config
from kino_vla.vla import scenarios as S
from kino_vla.vla.planner import StubVlaPolicy
from kino_vla.vla.rollout import run_vla_rollout


def test_all_scenarios_is_the_full_bclass_set():
    scns = S.all_scenarios(y=0.0)
    names = [s.name for s in scns]
    assert names == [
        "O1_mu_field",
        "O3_collapse",
        "O8_invisible_collider",
        "O2_compliance",
        "O4_tether",
        "O5_payload",
        "O10_effort_decay",
    ]
    appr = {s.name: s.appearance_class for s in scns}
    assert appr["O2_compliance"] == "brown_mud"
    assert appr["O4_tether"] == "yellow_adhesive"
    assert appr["O5_payload"] == "solid_ground"  # global op, plain ground


def test_o5_is_safe_halt_others_reach():
    """O5 overload: a deliberate non-falling Hold is the correct (and credited) outcome."""
    by_name = {s.name: s for s in S.all_scenarios()}
    assert by_name["O5_payload"].success_mode == "safe_halt"
    assert all(s.success_mode == "reach" for n, s in by_name.items() if n != "O5_payload")


def test_distinct_lanes_keep_scenarios_disjoint():
    a = S.o2_compliance(y=0.0)
    b = S.o2_compliance(y=6.0)
    assert a.start_xy[1] == 0.0 and b.start_xy[1] == 6.0
    assert a.goal_xy[1] == 0.0 and b.goal_xy[1] == 6.0


def test_fsm_baseline_runs_closed_loop_without_vla():
    """B2 = the cause-blind rule-FSM; it has no VLA decisions/snapshot, but run_closed_loop must
    still produce a valid RolloutResult (the spec §12 baseline contrast)."""
    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    # surrogate-faithful scenario (O3); policy is ignored under fsm_baseline.
    r = run_vla_rollout(S.o3_collapse(0.0), None, seed=0, backend="surrogate", fsm_baseline=True)
    assert r.first_attribution is None  # the FSM attributes nothing (cause-blind)
    assert r.decisions == []
    assert isinstance(r.success, bool)
    # the VLA path on the same scenario DOES attribute (the contrast).
    rv = run_vla_rollout(S.o3_collapse(0.0), StubVlaPolicy(pcfg, tax), seed=0, backend="surrogate")
    assert rv.first_attribution is not None
