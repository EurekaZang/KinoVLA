"""Walking-skeleton assembly: configs -> wired components -> one demo episode.

Single construction point for the M1 deliverable so the demo script, the tests,
and later milestones (which each swap exactly one stub) build the identical
pipeline:

    SurrogateBackend | IsaacPolicyBackend  (Go2 + one O1 ice patch)
      -> RuleMonitor (text anomaly summary stub)
      -> FsmRecovery (scripted Backstep + Replan stub)
      -> PassThroughShield (stub)
      -> Sport-Client-style velocity interface
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kino_vla.loop import EpisodeResult, run_episode
from kino_vla.monitor.rule_monitor import RuleMonitor
from kino_vla.shield.cbf_shield import CbfShield
from kino_vla.sim.backend import LocomotionBackend
from kino_vla.sim.operators import MuField, OperatorStack
from kino_vla.sim.surrogate import SurrogateBackend
from kino_vla.sim.terrain import TerrainSpec, generate_flat_with_patch
from kino_vla.utils.config import Config, load_config
from kino_vla.vla.fsm_recovery import FsmRecovery


@dataclass
class WalkingSkeleton:
    """The wired pipeline plus the configs/terrain it was built from."""

    backend: LocomotionBackend
    operators: OperatorStack
    monitor: RuleMonitor
    policy: FsmRecovery
    shield: CbfShield
    demo_cfg: Config
    terrain: TerrainSpec


def build_walking_skeleton(
    seed: int,
    backend: str = "surrogate",
    demo_overrides: dict[str, Any] | None = None,
    record_cam: bool = False,
) -> WalkingSkeleton:
    """Assemble the full skeleton for one episode; ``backend`` is surrogate|isaac.

    ``record_cam`` (isaac only) adds a passive RGB camera for video recording.
    """
    demo_cfg = load_config("demo/walking_skeleton.yaml", demo_overrides)
    terrain = generate_flat_with_patch(demo_cfg.terrain, seed)
    ice = MuField(
        region=terrain.hazard_patch,
        mu_s=float(demo_cfg.ice.mu_s),
        mu_d=float(demo_cfg.ice.mu_d),
        restitution=float(demo_cfg.ice.restitution),
    )
    start_pos = np.asarray(demo_cfg.start.pos, dtype=np.float64)
    start_heading = float(demo_cfg.start.heading)

    backend_obj: LocomotionBackend
    if backend == "surrogate":
        sim_cfg = load_config("sim/surrogate.yaml")
        backend_obj = SurrogateBackend(sim_cfg, start_pos, start_heading)
    elif backend == "isaac":
        # Imported lazily: requires a running Isaac app (see scripts/run_demo.py).
        # M2: the trained RSL-RL policy walks the Go2 (replaces the M1 kinematic stub).
        from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend

        backend_obj = IsaacPolicyBackend(
            load_config("sim/go2_skeleton.yaml"), start_pos, start_heading, record_cam=record_cam
        )
    else:
        raise ValueError(f"unknown backend {backend!r}; expected 'surrogate' or 'isaac'")

    # Per-robot calibration: the real Go2 (Isaac) has a push-off slip transient and
    # intermittent ice slip the surrogate point-robot does not, so it uses its own
    # monitor/recovery constants (same pipeline, different thresholds).
    monitor_cfg = "monitor/rule_v0_isaac.yaml" if backend == "isaac" else "monitor/rule_v0.yaml"
    fsm_cfg = "recovery/fsm_isaac.yaml" if backend == "isaac" else "recovery/fsm_v0.yaml"
    monitor = RuleMonitor(load_config(monitor_cfg), dt=backend_obj.dt)
    policy = FsmRecovery(
        load_config(fsm_cfg),
        goal_xy=np.asarray(demo_cfg.goal.pos, dtype=np.float64),
        dt=backend_obj.dt,
    )
    # M3: the CBF-QP shield replaces the M1 pass-through stub (spec §6). It is
    # transparent at the FSM cruise speed (~0.8 m/s ≪ the trot capture bound) so the
    # demo's qualitative behaviour is unchanged; it only bites hostile commands.
    shield = CbfShield(load_config("shield/cbf_v0.yaml"))
    return WalkingSkeleton(
        backend=backend_obj,
        operators=OperatorStack([ice]),
        monitor=monitor,
        policy=policy,
        shield=shield,
        demo_cfg=demo_cfg,
        terrain=terrain,
    )


def run_walking_skeleton(
    seed: int,
    backend: str = "surrogate",
    demo_overrides: dict[str, Any] | None = None,
) -> tuple[EpisodeResult, WalkingSkeleton]:
    """Build and run one seeded walking-skeleton episode."""
    skeleton = build_walking_skeleton(seed, backend, demo_overrides)
    result = run_episode(
        skeleton.backend,
        skeleton.operators,
        skeleton.monitor,
        skeleton.policy,
        skeleton.shield,
        seed=seed,
        goal_xy=np.asarray(skeleton.demo_cfg.goal.pos, dtype=np.float64),
        goal_tol_m=float(skeleton.demo_cfg.goal.tol_m),
        max_time_s=float(skeleton.demo_cfg.max_time_s),
    )
    return result, skeleton
