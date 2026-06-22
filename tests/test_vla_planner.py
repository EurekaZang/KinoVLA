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


def test_correct_vs_wrong_attribution_drive_opposite_recoveries(cfg, tax):
    """The headline §5 claim at the DECISION level: a correct vs a sibling-wrong attribution pick
    OPPOSITE recoveries (escape vs push-through). The closed-loop OUTCOME divergence (correct routes
    around + survives, wrong topples) is a VLA-route-around capability measured on the REAL Isaac
    stack (§0); the surrogate stub has no VLA nav, so a closed-loop outcome here is not meaningful
    (both fall without a route-around). O3 thin-ice: correct ⇒ Update_Topology (escape), sibling ⇒
    Set_Constraint (slow & continue across the give-way)."""
    correct = run_vla_rollout(S.o3_collapse(), StubVlaPolicy(cfg, tax), seed=0, backend="surrogate")
    wrong = run_vla_rollout(
        S.o3_collapse(), StubVlaPolicy(cfg, tax, error_mode="sibling"), seed=0, backend="surrogate"
    )
    assert correct.first_attribution == "region_collapse"
    assert correct.first_primitive == "Update_Topology"  # escape (mark untraversable + back out)
    assert wrong.first_attribution == "low_friction"  # mis-attributed to the sibling
    assert wrong.first_primitive == "Set_Constraint"  # the opposite push-through (slow & continue)


def test_invisible_collider_attributed(cfg, tax):
    """O8 invisible collider: the stub attributes it and picks an escape primitive. The route-around
    to the goal is a VLA-nav capability verified on the real stack (not the surrogate stub)."""
    r = run_vla_rollout(S.o8_invisible(), StubVlaPolicy(cfg, tax), seed=0, backend="surrogate")
    assert r.first_attribution == "invisible_obstacle"
    assert r.first_primitive in {"Update_Topology", "Backstep"}


def test_sample_rollouts_varied_decisions(cfg, tax):
    """Sampling correct vs wrong policies at one node yields DIFFERENT recovery decisions (the DPO
    preference signal). The physical-outcome divergence is the real-stack dichotomy, not the
    surrogate (which has no VLA route-around)."""
    scn = S.o3_collapse()

    def factory(temp: float):
        # temp encodes the policy here (CI surrogate has no real sampling): >0 ⇒ the wrong sibling.
        return StubVlaPolicy(cfg, tax, error_mode="sibling" if temp > 0 else None)

    results = sample_rollouts(scn, factory, temperatures=[0.0, 1.0], seed=0, backend="surrogate")
    attrs = {r.first_attribution for r in results}
    assert "region_collapse" in attrs and "low_friction" in attrs  # correct vs sibling mis-attr


def test_probe_defers_reflect_until_window_refilled(cfg, tax):
    """With an active probe (Gap-3 #34c), the planner runs the decel→accel maneuver and only
    captures the snapshot / calls the VLA AFTER the probe has refilled the 500 ms window."""
    import numpy as np

    from kino_vla.monitor.reflex import ActiveProbe
    from kino_vla.monitor.rule_monitor import MonitorEvent
    from kino_vla.sim.types import Obs
    from kino_vla.vla.output import ParsedDecision
    from kino_vla.vla.planner import VlaPlanner

    decided: list = []

    class SpyPolicy:
        def decide(self, snapshot, map_note=""):
            decided.append(snapshot)
            return ParsedDecision(ok=False, raw_text="", reject_code="spy")

    class FakeRecorder:
        def buffer(self, obs):
            pass

        def capture(self, event, prior_outputs):
            return "snapshot"

    probe = ActiveProbe(decel_s=0.1, accel_s=0.1, settle_window_ms=0.0)  # total_s = 0.2 s
    planner = VlaPlanner(
        cfg=load_config("recovery/fsm_isaac.yaml"),
        policy=SpyPolicy(),
        goal_xy=np.array([5.0, 0.0]),
        dt=0.02,
        recorder=FakeRecorder(),
        compiler=None,
        probe=probe,
    )

    def _obs(t):
        return Obs(
            t=t,
            pos=np.array([1.0, 0.0]),
            heading=0.0,
            vel_body=np.zeros(2),
            yaw_rate=0.0,
            cmd_prev=np.zeros(3),
            slip_ratio=0.0,
            base_height=0.32,
            tilt=0.0,
            fallen=False,
        )

    ev = MonitorEvent(
        t=1.0, pos=np.array([1.0, 0.0]), channel="slip_ratio", value=0.5, threshold=0.4, summary="x"
    )
    assert planner.on_event(ev)
    planner.step(_obs(1.02))  # decel phase
    assert decided == [], "must NOT reflect during the probe's decel phase"
    planner.step(_obs(1.12))  # accel phase
    assert decided == [], "must NOT reflect during the probe's accel phase"
    planner.step(_obs(1.24))  # probe done (elapsed 0.24 > 0.2) ⇒ reflect on the refilled window
    assert len(decided) == 1, "reflect exactly once, after the probe refills the window"
