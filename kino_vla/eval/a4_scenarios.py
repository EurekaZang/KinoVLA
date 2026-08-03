"""A4 shared scenario construction — build a registry scenario's operator + rollout.Scenario.

Both the matrix runner (``scripts/a4_matrix.py``) and the closed-loop validation
(``scripts/a4_4_closedloop.py``) construct scenarios identically from the pre-registered registry
(``configs/eval/a0_registry.yaml``), so the operator↔θ mapping lives ONCE here (single source of
truth, no builder/θ drift). The two-phase O4 (A4.1) is selected by ``theta.p0_m > 0``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from kino_vla.map.types import SemanticRegion
from kino_vla.utils.geometry import Rect

if TYPE_CHECKING:
    from kino_vla.eval.registry import ScenarioSpec
    from kino_vla.sim.operators.base import FailureOperator
    from kino_vla.vla.rollout import Scenario


def build_operator(spec: ScenarioSpec, rect: Rect) -> FailureOperator:
    """Build the registry scenario's operator from its θ (registry = source of truth)."""
    from kino_vla.sim.operators import (
        ComplianceField,
        EffortDecay,
        InvisibleCollider,
        MuField,
        Payload,
        Push,
        Tether,
        VisualPhysicsRemap,
    )

    th = dict(spec.theta)
    op_name = spec.operator
    if op_name == "O2_compliance":
        return ComplianceField(rect, k_c=th["k_c"], c_c=th["c_c"], d_sink=th.get("d_sink", 0.03))
    if op_name == "O4_tether":
        if th.get("p0_m", 0.0) > 0.0:  # A4.1 two-phase (consequence-grade)
            return Tether(
                rect,
                k=th.get("k_c", 14.0),
                d=th.get("c_c", 6.0),
                l0=0.0,
                f_break=th["f_break"],
                force_offset_n=th["k_c"],
                peel_factor=1.0,
                p0_m=th["p0_m"],
                k2_n_per_m=th["k2"],
            )
        pre = spec.o4_preset or {}
        return Tether(
            rect,
            k=th["k"],
            d=th["c"],
            l0=0.0,
            f_break=th["f_break"],
            force_cap_n=pre.get("force_cap_n", float("inf")),
            force_offset_n=pre.get("force_offset_n", 0.0),
            peel_factor=pre.get("peel_factor", 0.3),
        )
    if op_name == "O1_mu_field":
        return MuField(rect, mu_s=th["mu_s"], mu_d=th["mu_d"])
    if op_name == "O8_invisible_collider":
        return InvisibleCollider(rect)
    if op_name == "O5_payload":
        return Payload(mass_kg=th["mass_kg"], com_offset_m=(0.0, 0.0))
    if op_name == "O10_effort_decay":
        return EffortDecay(
            decay_rate_per_s=th.get("decay_rate_per_s", 0.6),
            floor=th["floor"],
            t_start_s=th.get("t_start_s", 2.0),
        )
    if op_name == "O7_visual_remap":
        return VisualPhysicsRemap(
            rect,
            mu_s=th["mu_s"],
            mu_d=th["mu_d"],
            depth_bias_m=th.get("depth_bias_m", 0.6),
            appearance_class=spec.appearance_class,
        )
    if op_name == "O6_push":
        return Push(
            np.array([0.0, th["impulse_ns"]], dtype=np.float64), t_push_s=th.get("t_push_s", 2.0)
        )
    raise ValueError(f"unknown operator {op_name!r} for scenario {spec.name!r}")


def build_rollout_scenario(
    spec: ScenarioSpec, lane_y: float, *, max_time_s: float = 40.0
) -> Scenario:
    """Build a rollout.Scenario for a registry spec at the fixed A4 lane geometry."""
    from kino_vla.vla.rollout import Scenario

    rect = Rect(cx=3.0, cy=lane_y, hx=1.0, hy=1.0)
    op = build_operator(spec, rect)
    region = SemanticRegion(rect=rect, appearance_class=spec.appearance_class)
    return Scenario(
        name=spec.name,
        operator=op,
        scene_region=region,
        operator_name=spec.operator,
        appearance_class=spec.appearance_class,
        goal_xy=(6.0, lane_y),
        start_xy=(0.0, lane_y),
        max_time_s=max_time_s,
        success_mode="reach",
    )
