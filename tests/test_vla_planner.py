"""VLA closed-loop planner + rollout sampler on the surrogate (CI; spec §5/§11)."""

from __future__ import annotations

import pytest

from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.utils.config import load_config
from kino_vla.vla import scenarios as S
from kino_vla.vla.planner import StubVlaPolicy
from kino_vla.vla.rollout import run_vla_rollout, sample_rollouts


@pytest.fixture(scope="module")
def cfg():
    return load_config("data/hindsight.yaml")


@pytest.fixture(scope="module")
def tax(cfg):
    return FailureTaxonomy(cfg)


def test_planner_is_recovery_policy_drop_in(cfg, tax):
    """The planner implements on_event/step and runs the standard monitor→planner→shield loop."""
    r = run_vla_rollout(S.o3_collapse(), StubVlaPolicy(cfg, tax), seed=0, backend="surrogate")
    assert r.n_rounds >= 1  # the monitor fired and the planner reflected
    assert r.first_attribution == "region_collapse"  # the stub oracle attributes correctly
    assert r.first_primitive in {"Update_Topology", "Backstep", "Replan_Waypoint"}


def test_correct_attribution_succeeds_wrong_fails(cfg, tax):
    """The headline §5 claim, on the surrogate: a wrong attribution drives the opposite (failing)
    recovery. O3 thin-ice — escape+detour (correct) reaches; 'uniform ice ⇒ slow & continue'
    (the sibling mis-attribution) crosses the give-way and falls."""
    correct = run_vla_rollout(S.o3_collapse(), StubVlaPolicy(cfg, tax), seed=0, backend="surrogate")
    wrong = run_vla_rollout(
        S.o3_collapse(), StubVlaPolicy(cfg, tax, error_mode="sibling"), seed=0, backend="surrogate"
    )
    assert correct.success and correct.reached and not correct.fell
    assert not wrong.success  # the wrong strategy physically fails
    assert wrong.first_attribution == "low_friction"  # mis-attributed to the sibling


def test_invisible_collider_detour_reaches(cfg, tax):
    r = run_vla_rollout(S.o8_invisible(), StubVlaPolicy(cfg, tax), seed=0, backend="surrogate")
    assert r.success and r.first_attribution == "invisible_obstacle"


def test_sample_rollouts_varied_outcomes(cfg, tax):
    """Sampling correct vs wrong policies at one node yields both successes and failures."""
    scn = S.o3_collapse()

    def factory(temp: float):
        # temp encodes the policy here (CI surrogate has no real sampling): >0 ⇒ the wrong sibling.
        return StubVlaPolicy(cfg, tax, error_mode="sibling" if temp > 0 else None)

    results = sample_rollouts(scn, factory, temperatures=[0.0, 1.0], seed=0, backend="surrogate")
    assert any(r.success for r in results) and any(not r.success for r in results)
