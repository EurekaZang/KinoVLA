"""Friction-limited velocity-tracking model shared by the surrogate and Isaac backends.

A walking quadruped demands tangential ground force both to *change* velocity
(tracking acceleration) and to *maintain* it (cyclic stance accelerations of the
gait, which scale with speed). Total demand beyond the friction cone budget
``mu * g`` produces slip: control authority degrades by the saturation ratio and
the excess shows up as the slip-ratio signal the Kino-Monitor thresholds (spec §3).

This is deliberately a reduced model: at M1 it stands in for true contact physics
(the M2 locomotion policy + PhysX contacts replace its role on the GPU machine);
its value here is that O1's privileged μ has a measurable, testable effect on the
closed loop (θ-application gate, QA 5.2a).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TractionResult:
    """Outcome of one traction-limited tracking step.

    Two independent shortfall signals (M2): ``slip_ratio`` is the fraction of demand
    the *friction* budget cannot supply (O1 μ-field, O3 collapse); ``effort_ratio`` is
    the fraction the *actuator-effort* budget cannot supply (O5 payload, O10
    effort-decay). They are the Kino-Monitor's two traction-side channels and the
    constructive O1↔O3 vs O5↔O10 ambiguity (both read as "the robot can't accelerate"
    but for opposite reasons). ``authority`` is gated by whichever budget binds first.
    """

    accel_body: np.ndarray  # (2,) achieved body-frame acceleration [m/s^2]
    slip_ratio: float  # 0 within the friction budget, in (0, 1) when friction-limited
    authority: float  # fraction of demanded acceleration delivered, in (0, 1]
    effort_ratio: float = 0.0  # 0 within the effort budget, in (0, 1) when effort-limited


def traction_step(
    vel_body: np.ndarray,
    vel_des_body: np.ndarray,
    mu: float,
    *,
    tau_track_s: float,
    gait_demand_per_speed: float,
    gravity: float,
    effort_budget: float = float("inf"),
    extra_demand: float = 0.0,
) -> TractionResult:
    """Compute achieved acceleration, slip, and effort-saturation for one tracking step.

    Demand = |a_des| + gait_demand_per_speed * |v| + extra_demand, where extra_demand
    folds in a payload's center-of-mass disturbance (O5). Two budgets cap it: the
    friction budget ``mu * gravity`` and the actuator ``effort_budget`` (both in
    m/s^2-equivalent). Achieved acceleration is scaled by ``authority =
    min(friction, effort) / demand`` when either is exceeded; each ``*_ratio`` reports
    that specific budget's shortfall, so a friction-limited and an effort-limited step
    are distinguishable even though both lose authority.

    With the defaults (``effort_budget=inf``, ``extra_demand=0``) this reduces exactly
    to the M1 friction-only model: ``slip_ratio = 1 - authority``, ``effort_ratio = 0``.
    """
    vel_body = np.asarray(vel_body, dtype=np.float64)
    accel_des = (np.asarray(vel_des_body, dtype=np.float64) - vel_body) / tau_track_s
    demand = (
        float(np.linalg.norm(accel_des))
        + gait_demand_per_speed * float(np.linalg.norm(vel_body))
        + float(extra_demand)
    )
    friction_budget = mu * gravity
    available = min(friction_budget, effort_budget)
    if demand <= available or demand == 0.0:
        return TractionResult(accel_body=accel_des, slip_ratio=0.0, authority=1.0)
    authority = available / demand
    slip_ratio = max(0.0, 1.0 - friction_budget / demand)
    effort_ratio = max(0.0, 1.0 - effort_budget / demand) if effort_budget != float("inf") else 0.0
    return TractionResult(
        accel_body=accel_des * authority,
        slip_ratio=slip_ratio,
        authority=authority,
        effort_ratio=effort_ratio,
    )
