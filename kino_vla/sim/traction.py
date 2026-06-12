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
    """Outcome of one traction-limited tracking step."""

    accel_body: np.ndarray  # (2,) achieved body-frame acceleration [m/s^2]
    slip_ratio: float  # 0 when within the friction budget, in (0, 1) when saturated
    authority: float  # fraction of demanded acceleration delivered, in (0, 1]


def traction_step(
    vel_body: np.ndarray,
    vel_des_body: np.ndarray,
    mu: float,
    *,
    tau_track_s: float,
    gait_demand_per_speed: float,
    gravity: float,
) -> TractionResult:
    """Compute achieved acceleration and slip for one step of velocity tracking.

    Demand = |a_des| + gait_demand_per_speed * |v|; available = mu * gravity.
    When demand exceeds the budget, acceleration scales down by authority =
    available / demand and slip_ratio = 1 - authority.
    """
    vel_body = np.asarray(vel_body, dtype=np.float64)
    accel_des = (np.asarray(vel_des_body, dtype=np.float64) - vel_body) / tau_track_s
    demand = float(np.linalg.norm(accel_des)) + gait_demand_per_speed * float(
        np.linalg.norm(vel_body)
    )
    available = mu * gravity
    if demand <= available or demand == 0.0:
        return TractionResult(accel_body=accel_des, slip_ratio=0.0, authority=1.0)
    authority = available / demand
    return TractionResult(
        accel_body=accel_des * authority,
        slip_ratio=1.0 - authority,
        authority=authority,
    )
