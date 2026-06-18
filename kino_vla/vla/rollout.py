"""Closed-loop rollout sampler at failure nodes (spec §11 Stage 2 / §12).

Embodied DPO needs *physical outcomes*: place the planner at a failure node, let it attribute and
recover, and see whether it escapes to the goal or falls / dead-loops (spec §11 "成功脱困到达 vs
跌倒/死循环"). This module runs one closed-loop episode per (scenario, sampled decision) and reports
the outcome — the raw material for the DPO preference pairs and for the M7 exit-3 success metric.

The same wiring runs on the CPU surrogate (CI + the surrogate ablation, where O3 collapse / O8
invisible-collider / O1↔O3 give genuinely strategy-dependent outcomes — continue⇒fall/stuck vs
detour⇒reach) and, with ``backend='isaac'``, on the real Go2 (the deliverable; spec §1/§8.1). The
planner's ``VlaPolicy`` is injected, so the sampler is agnostic to stub vs trained model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from kino_vla.data.snapshot import SnapshotRecorder
from kino_vla.loop import run_episode
from kino_vla.map import SemanticRegion, TraversabilityMap
from kino_vla.monitor.rule_monitor import RuleMonitor
from kino_vla.shield.cbf_shield import CbfShield
from kino_vla.shield.primitive_compiler import PrimitiveCompiler
from kino_vla.sim.operators import FailureOperator, OperatorStack
from kino_vla.utils.config import load_config
from kino_vla.vla.planner import RolloutRecord, VlaPlanner, VlaPolicy


@dataclass(frozen=True)
class Scenario:
    """One failure node: an operator placed on the path + its visual region (spec §8.2)."""

    name: str
    operator: FailureOperator
    scene_region: SemanticRegion
    operator_name: str
    appearance_class: str
    goal_xy: tuple[float, float] = (5.0, 0.0)
    start_xy: tuple[float, float] = (0.0, 0.0)
    start_heading: float = 0.0
    max_time_s: float = 16.0


@dataclass(frozen=True)
class RolloutResult:
    """Outcome of one closed-loop rollout (the DPO signal + the success metric)."""

    scenario: str
    reached: bool
    fell: bool
    stuck: bool  # neither reached nor fell within the budget (dead-loop / blocked)
    final_dist_m: float
    n_rounds: int
    first_attribution: str | None
    first_primitive: str | None
    decisions: list[RolloutRecord]
    success: bool  # reached and not fallen — the headline outcome
    first_snapshot: object = None  # the failure-node Snapshot (the DPO prompt); not serialized

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "reached": self.reached,
            "fell": self.fell,
            "stuck": self.stuck,
            "final_dist_m": round(self.final_dist_m, 3),
            "n_rounds": self.n_rounds,
            "first_attribution": self.first_attribution,
            "first_primitive": self.first_primitive,
            "success": self.success,
            "decisions": [
                {
                    "t": round(d.t, 2),
                    "attribution": d.attribution,
                    "primitive": d.primitive,
                    "code": d.code,
                }
                for d in self.decisions
            ],
        }


def run_vla_rollout(
    scenario: Scenario,
    policy: VlaPolicy,
    *,
    seed: int,
    backend: str = "surrogate",
    use_compiler: bool = True,
    use_map: bool = False,
) -> RolloutResult:
    """Run one closed-loop episode with the VLA planner; return the physical outcome."""
    start_pos = np.asarray(scenario.start_xy, dtype=np.float64)

    if backend == "surrogate":
        from kino_vla.sim.surrogate import SurrogateBackend

        backend_obj = SurrogateBackend(
            load_config("sim/surrogate.yaml"), start_pos, scenario.start_heading
        )
        monitor_cfg, fsm_cfg = "monitor/rule_v0.yaml", "recovery/fsm_v0.yaml"
    elif backend == "isaac":
        from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend

        backend_obj = IsaacPolicyBackend(
            load_config("sim/go2_skeleton.yaml"), start_pos, scenario.start_heading
        )
        monitor_cfg, fsm_cfg = "monitor/rule_v0_isaac.yaml", "recovery/fsm_isaac.yaml"
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return run_closed_loop(
        backend_obj,
        scenario,
        policy,
        monitor_cfg=monitor_cfg,
        fsm_cfg=fsm_cfg,
        seed=seed,
        use_compiler=use_compiler,
        use_map=use_map,
    )


def run_closed_loop(
    backend_obj: object,
    scenario: Scenario,
    policy: VlaPolicy,
    *,
    monitor_cfg: str,
    fsm_cfg: str,
    seed: int,
    use_compiler: bool = True,
    use_map: bool = False,
) -> RolloutResult:
    """Run the planner closed loop on an EXISTING backend (so Isaac reuses one app across lanes).

    Isaac is one-app-per-process, so the Isaac driver builds the backend once and calls this per
    lateral lane (after setting ``backend._start_pos``); the surrogate path builds a fresh backend.
    """
    goal_xy = np.asarray(scenario.goal_xy, dtype=np.float64)
    monitor = RuleMonitor(load_config(monitor_cfg), dt=backend_obj.dt)
    shield = CbfShield(load_config("shield/cbf_v0.yaml"))
    compiler = PrimitiveCompiler(shield) if use_compiler else None
    recorder = SnapshotRecorder(
        load_config("data/hindsight.yaml"),
        scene=[scenario.scene_region],
        operator_name=scenario.operator_name,
        appearance_class=scenario.appearance_class,
        privileged_fn=lambda: {},  # runtime: the planner never sees privileged θ (no leak)
        gate_rect=None,
    )
    planner = VlaPlanner(
        cfg=load_config(fsm_cfg),
        policy=policy,
        goal_xy=goal_xy,
        dt=backend_obj.dt,
        recorder=recorder,
        compiler=compiler,
    )
    nav_map = None
    if use_map:
        nav_map = TraversabilityMap(
            load_config("map/traversability_v0.yaml"), scene=[scenario.scene_region]
        )

    result = run_episode(
        backend_obj,
        OperatorStack([scenario.operator]),
        monitor,
        planner,
        shield,
        seed=seed,
        goal_xy=goal_xy,
        goal_tol_m=0.6,
        max_time_s=scenario.max_time_s,
        nav_map=nav_map,
    )
    reached = result.goal_reached
    fell = result.fell
    stuck = not reached and not fell
    first = planner.decisions[0] if planner.decisions else None
    return RolloutResult(
        scenario=scenario.name,
        reached=reached,
        fell=fell,
        stuck=stuck,
        final_dist_m=result.final_dist_m,
        n_rounds=len(planner.decisions),
        first_attribution=first.attribution if first else None,
        first_primitive=first.primitive if first else None,
        decisions=list(planner.decisions),
        success=reached and not fell,
        first_snapshot=planner.first_snapshot,
    )


def sample_rollouts(
    scenario: Scenario,
    policy_factory: Callable[[float], VlaPolicy],
    *,
    temperatures: list[float],
    seed: int,
    backend: str = "surrogate",
    use_compiler: bool = True,
) -> list[RolloutResult]:
    """Sample one rollout per temperature at the same node (spec §11 multi-temperature sampling).

    ``policy_factory(temperature)`` builds the decision policy at that sampling temperature; the
    differing decisions produce differing outcomes — the (Chosen, Rejected) raw material.
    """
    out: list[RolloutResult] = []
    for i, temp in enumerate(temperatures):
        policy = policy_factory(temp)
        out.append(
            run_vla_rollout(
                scenario, policy, seed=seed + i, backend=backend, use_compiler=use_compiler
            )
        )
    return out
