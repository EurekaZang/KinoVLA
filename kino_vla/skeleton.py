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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from kino_vla.loop import EpisodeResult, run_episode
from kino_vla.map import SemanticRegion, TraversabilityMap
from kino_vla.monitor.learned_monitor import LearnedMonitor, load_deployed_monitor
from kino_vla.shield.cbf_shield import CbfShield
from kino_vla.sim.backend import LocomotionBackend
from kino_vla.sim.operators import FailureOperator, MuField, OperatorStack
from kino_vla.sim.surrogate import SurrogateBackend
from kino_vla.sim.terrain import TerrainSpec, generate_flat_with_patch
from kino_vla.utils.config import Config, load_config
from kino_vla.utils.geometry import Rect
from kino_vla.vla.fsm_recovery import FsmRecovery


@dataclass
class WalkingSkeleton:
    """The wired pipeline plus the configs/terrain it was built from."""

    backend: LocomotionBackend
    operators: OperatorStack
    monitor: LearnedMonitor
    policy: FsmRecovery
    shield: CbfShield
    demo_cfg: Config
    terrain: TerrainSpec
    nav_map: TraversabilityMap | None = None


def build_walking_skeleton(
    seed: int,
    backend: str = "surrogate",
    demo_overrides: dict[str, Any] | None = None,
    record_cam: bool = False,
    use_map: bool = True,
    operator_factory: Callable[[Rect], tuple[FailureOperator, SemanticRegion | None]] | None = None,
    map_overrides: dict[str, Any] | None = None,
    live_perception: bool = False,
    extra_operators: list[tuple[FailureOperator, SemanticRegion]] | None = None,
    monitor: object | None = None,
) -> WalkingSkeleton:
    """Assemble the full skeleton for one episode; ``backend`` is surrogate|isaac.

    ``record_cam`` (isaac only) adds a passive RGB camera for video recording.
    ``use_map`` (M5, default on) builds the semantic traversability map (spec §7): the
    ice patch is grounded as a homogeneous "ice_sheet" region, the on-ground slip
    overwrites the costmap, the mark propagates over the sheet, and the avoid discs feed
    the recovery planner. The map is backend-agnostic (numpy), so it runs identically on
    the surrogate and on the physically-simulated Isaac Go2.
    ``operator_factory`` (default None ⇒ the pinned O1 ice patch) swaps the path hazard for
    an arbitrary Kino-Fail operator placed at ``terrain.hazard_patch`` — it returns
    ``(operator, scene_region|None)`` so each operator can run the SAME closed loop
    (monitor → FSM → shield → map). Used by scripts/record_operators.py for the per-operator
    closed-loop videos; the default keeps the demo/CI path byte-identical.
    """
    demo_cfg = load_config("demo/walking_skeleton.yaml", demo_overrides)
    terrain = generate_flat_with_patch(demo_cfg.terrain, seed)
    if operator_factory is None:
        hazard_op: FailureOperator = MuField(
            region=terrain.hazard_patch,
            mu_s=float(demo_cfg.ice.mu_s),
            mu_d=float(demo_cfg.ice.mu_d),
            restitution=float(demo_cfg.ice.restitution),
        )
        scene_region: SemanticRegion | None = SemanticRegion(
            Rect(
                terrain.hazard_patch.cx,
                terrain.hazard_patch.cy,
                terrain.hazard_patch.hx,
                terrain.hazard_patch.hy,
            ),
            "ice_sheet",
        )
    else:
        hazard_op, scene_region = operator_factory(terrain.hazard_patch)
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
            load_config("sim/go2_skeleton.yaml"),
            start_pos,
            start_heading,
            record_cam=record_cam,
            perception_cam=live_perception,
        )
    else:
        raise ValueError(f"unknown backend {backend!r}; expected 'surrogate' or 'isaac'")

    # The learning-based Kino-Monitor (trained + calibrated on the real Go2, see
    # outputs/monitor_learned/RESULTS.md) is the single deployed monitor — it learned the per-robot
    # proprioception distribution from data, so no per-robot threshold config remains.
    fsm_cfg = "recovery/fsm_isaac.yaml" if backend == "isaac" else "recovery/fsm_v0.yaml"
    monitor = monitor if monitor is not None else load_deployed_monitor(backend_obj.dt)
    policy = FsmRecovery(
        load_config(fsm_cfg),
        goal_xy=np.asarray(demo_cfg.goal.pos, dtype=np.float64),
        dt=backend_obj.dt,
    )
    # M3: the CBF-QP shield replaces the M1 pass-through stub (spec §6). It is
    # transparent at the FSM cruise speed (~0.8 m/s ≪ the trot capture bound) so the
    # demo's qualitative behaviour is unchanged; it only bites hostile commands.
    shield = CbfShield(load_config("shield/cbf_v0.yaml"))
    # M5: the semantic traversability map grounds the path hazard as a homogeneous visual
    # region so the physical slip can overwrite + propagate over the whole sheet (spec §7).
    # Multi-patch readiness (#47): the primary hazard plus any ``extra_operators`` (pre-built
    # (operator, region) pairs at other locations) share the SAME closed loop. A long/multi-patch
    # course registers several patches; the default single-patch path stays byte-identical.
    operators_all: list[FailureOperator] = [hazard_op]
    scenes_all: list[SemanticRegion] = [scene_region] if scene_region is not None else []
    for op, reg in extra_operators or []:
        operators_all.append(op)
        if reg is not None:
            scenes_all.append(reg)
    nav_map = None
    if use_map and scenes_all:
        if live_perception and backend == "isaac":
            # REAL camera-grounded §7 map: texture + semantically tag EVERY hazard on the terrain so
            # the RTX perception camera sees it, and ground the costmap from those real pixels
            # (LiveRtxSegmenter: semantic mask + real-CLIP label + ray∩ground footprint).
            from kino_vla.map.clip_segmentation import APPEARANCE_TO_MATERIAL
            from kino_vla.map.live_rtx_segmenter import LiveRtxSegmenter

            for reg in scenes_all:
                material = APPEARANCE_TO_MATERIAL.get(reg.appearance_class, "concrete")
                backend_obj.add_textured_patch(reg.rect, material, material)
            nav_map = TraversabilityMap(
                load_config("map/traversability_v0.yaml", map_overrides),
                scene=scenes_all,
                segmenter=LiveRtxSegmenter(backend_obj),
            )
        else:
            # ``map_overrides`` lets a caller pick the §7 perception front-end (e.g.
            # {"segmenter": "clip"} for the synthetic-CLIP demo) or enable the rolling costmap;
            # default None keeps the shared config's surrogate front-end (CI/demo path unchanged).
            nav_map = TraversabilityMap(
                load_config("map/traversability_v0.yaml", map_overrides), scene=scenes_all
            )
    return WalkingSkeleton(
        backend=backend_obj,
        operators=OperatorStack(operators_all),
        monitor=monitor,
        policy=policy,
        shield=shield,
        demo_cfg=demo_cfg,
        terrain=terrain,
        nav_map=nav_map,
    )


def run_walking_skeleton(
    seed: int,
    backend: str = "surrogate",
    demo_overrides: dict[str, Any] | None = None,
    monitor: object | None = None,
) -> tuple[EpisodeResult, WalkingSkeleton]:
    """Build and run one seeded walking-skeleton episode."""
    skeleton = build_walking_skeleton(seed, backend, demo_overrides, monitor=monitor)
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
        nav_map=skeleton.nav_map,
    )
    return result, skeleton
