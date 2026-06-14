"""Primitive Compiler gates (spec §5) — the full library compiles or rejects cleanly.

Every primitive either compiles to an executable command or returns a structured
rejection code; mode-switch primitives are gated by the §6.7 admission rule. No
malformed command may reach the executor (groundwork for the M7 schema-validation
exit criterion).
"""

from __future__ import annotations

import numpy as np

from kino_vla.shield.cbf_shield import CbfShield
from kino_vla.shield.primitive_compiler import (
    AdjustPosture,
    Backstep,
    HoldAndRequest,
    PrimitiveCompiler,
    ReplanWaypoint,
    SetConstraint,
    SwitchGait,
    UpdateTopology,
)
from kino_vla.sim.types import Obs
from kino_vla.utils.config import load_config

CFG = "shield/cbf_v0.yaml"


def _obs(vx: float = 0.0, vy: float = 0.0) -> Obs:
    return Obs(
        t=0.0,
        pos=np.zeros(2),
        heading=0.0,
        vel_body=np.array([vx, vy]),
        yaw_rate=0.0,
        cmd_prev=np.zeros(3),
        slip_ratio=0.0,
        base_height=0.31,
        tilt=0.0,
        fallen=False,
    )


def _compiler() -> PrimitiveCompiler:
    return PrimitiveCompiler(CbfShield(load_config(CFG)))


def test_backstep_compiles_to_reverse_velocity():
    c = _compiler().compile(Backstep(distance_m=0.5), _obs())
    assert c.accepted and c.v_cmd is not None
    assert c.v_cmd[0] < 0.0 and c.backstep_m == 0.5


def test_backstep_rejects_nonpositive_distance():
    c = _compiler().compile(Backstep(distance_m=0.0), _obs())
    assert not c.accepted and c.code.startswith("REJECT")


def test_replan_waypoint_carries_point():
    c = _compiler().compile(ReplanWaypoint(point_xy=(2.0, -1.0)), _obs())
    assert c.accepted
    np.testing.assert_allclose(c.waypoint_xy, [2.0, -1.0])


def test_switch_gait_admitted_when_safe():
    c = _compiler().compile(SwitchGait(mode="crawl"), _obs(0.2, 0.0))
    assert c.accepted and c.target_mode == "crawl"


def test_switch_gait_rejected_when_unsafe():
    # Too fast for high_step ⇒ admission rejects the switch (spec §6.7).
    c = _compiler().compile(SwitchGait(mode="high_step"), _obs(1.6, 0.0))
    assert not c.accepted and c.code.startswith("REJECT")


def test_switch_gait_unknown_mode_rejected():
    c = _compiler().compile(SwitchGait(mode="gallop"), _obs())
    assert not c.accepted and "unknown gait" in c.code


def test_adjust_posture_maps_to_nearest_mode():
    # A low requested height maps to the lowest configured posture (brace, z_c=0.20).
    c = _compiler().compile(AdjustPosture(body_height_m=0.20), _obs(0.1, 0.0))
    assert c.accepted and c.target_mode == "brace"


def test_adjust_posture_rejects_bad_height():
    c = _compiler().compile(AdjustPosture(body_height_m=-0.1), _obs())
    assert not c.accepted


def test_set_constraint_carries_caps():
    c = _compiler().compile(SetConstraint(max_speed=0.4, stiffness=0.5), _obs())
    assert c.accepted and c.max_speed == 0.4 and c.stiffness == 0.5


def test_set_constraint_rejects_bad_values():
    c = _compiler().compile(SetConstraint(max_speed=-1.0, stiffness=0.5), _obs())
    assert not c.accepted


def test_update_topology_carries_region():
    c = _compiler().compile(
        UpdateTopology(region_xy=(3.0, 0.0), radius_m=1.0, status="untraversable"), _obs()
    )
    assert c.accepted and c.topology_op == (3.0, 0.0, 1.0, "untraversable")


def test_hold_and_request_halts():
    c = _compiler().compile(HoldAndRequest(reason="torque saturation deadlock"), _obs())
    assert c.accepted and c.halt
    np.testing.assert_allclose(c.v_cmd, [0.0, 0.0, 0.0])
