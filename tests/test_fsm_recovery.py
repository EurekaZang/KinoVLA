"""Scripted FSM recovery stub: detour geometry, phase logic, command shaping."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from kino_vla.monitor.event import MonitorEvent
from kino_vla.utils.config import load_config
from kino_vla.utils.geometry import segment_hits_circle
from kino_vla.vla.fsm_recovery import AvoidCircle, FsmRecovery, Phase, plan_detour
from tests._monitor_stub import make_obs

CFG = load_config("recovery/fsm_v0.yaml")
DT = 0.02


def make_event(t=1.0, pos=(2.0, 0.0), channel="slip_ratio"):
    return MonitorEvent(
        t=t,
        pos=np.asarray(pos, dtype=np.float64),
        channel=channel,
        value=0.6,
        threshold=0.4,
        summary="[anomaly] test",
    )


def make_obs_like(t, pos, heading=0.0):
    return replace(make_obs(t), pos=np.asarray(pos, dtype=np.float64), heading=heading)


# ---------------------------------------------------------------- plan_detour


def test_detour_clear_path_goes_direct():
    waypoints = plan_detour(np.array([0.0, 0.0]), np.array([6.0, 0.0]), circles=[], clearance_m=0.4)
    assert len(waypoints) == 1
    np.testing.assert_allclose(waypoints[-1], [6.0, 0.0])


def test_detour_skirts_blocking_circle():
    circle = AvoidCircle(center=np.array([3.0, 0.0]), radius=1.8)
    start, goal = np.array([0.5, 0.0]), np.array([6.0, 0.0])
    waypoints = plan_detour(start, goal, [circle], clearance_m=0.4)
    assert len(waypoints) == 3  # two box corners + goal
    reach = circle.radius + 0.4
    for corner in waypoints[:2]:
        assert np.linalg.norm(corner - circle.center) >= reach - 1e-9
    # The leg rejoining the goal is clear of the inflated circle.
    assert not segment_hits_circle(waypoints[1], goal, circle.center, reach)
    np.testing.assert_allclose(waypoints[-1], goal)


def test_detour_side_away_from_offset_circle():
    # Circle center left (+y) of the path: detour must go right (-y).
    circle = AvoidCircle(center=np.array([3.0, 0.5]), radius=1.8)
    waypoints = plan_detour(np.array([0.0, 0.0]), np.array([6.0, 0.0]), [circle], clearance_m=0.4)
    assert waypoints[0][1] < 0.0


# ---------------------------------------------------------------- FSM phases


def make_fsm():
    return FsmRecovery(CFG, goal_xy=np.array([6.0, 0.0]), dt=DT)


def test_event_starts_backstep_and_marks_region():
    fsm = make_fsm()
    assert fsm.phase is Phase.NOMINAL
    assert fsm.on_event(make_event())
    assert fsm.phase is Phase.BACKSTEP
    assert len(fsm.avoid_circles) == 1
    assert fsm.backstep_count == 1


def test_backstep_commands_reverse_then_replans():
    fsm = make_fsm()
    fsm.on_event(make_event(t=1.0, pos=(2.0, 0.0)))
    obs = make_obs_like(1.02, (2.0, 0.0))
    for _ in range(int(float(CFG.backstep.duration_s) / DT) + 2):
        cmd = fsm.step(obs)
        if fsm.phase is Phase.BACKSTEP:
            assert cmd[0] <= 0.0  # slew-limited toward reverse, never forward
        obs = make_obs_like(obs.t + DT, (2.0 - 0.1 * obs.t, 0.0))
    assert fsm.phase is Phase.DETOUR
    assert len(fsm.waypoints) >= 2  # detour corners + goal


def test_events_ignored_while_backstepping():
    fsm = make_fsm()
    assert fsm.on_event(make_event(t=1.0))
    assert not fsm.on_event(make_event(t=1.5))
    assert fsm.backstep_count == 1


def test_repeat_event_grows_avoid_radius():
    fsm = make_fsm()
    fsm.on_event(make_event(t=1.0, pos=(2.0, 0.0)))
    r0 = fsm.avoid_circles[0].radius
    # Finish the backstep so a new event is accepted, past the grace window.
    obs = make_obs_like(1.0 + float(CFG.backstep.duration_s) + DT, (1.5, 0.0))
    fsm.step(obs)
    assert fsm.phase is Phase.DETOUR
    t_late = obs.t + float(CFG.event_grace_s) + 0.1
    assert fsm.on_event(make_event(t=t_late, pos=(2.1, 0.0)))
    assert len(fsm.avoid_circles) == 1  # merged, not duplicated
    assert fsm.avoid_circles[0].radius > r0


def test_command_slew_limited_from_rest():
    fsm = make_fsm()
    cmd = fsm.step(make_obs_like(0.0, (0.0, 0.0)))
    assert abs(cmd[0]) <= float(CFG.cmd_slew_mps2) * DT + 1e-9
    assert abs(cmd[2]) <= float(CFG.yaw_slew_radps2) * DT + 1e-9
