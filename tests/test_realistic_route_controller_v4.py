from __future__ import annotations

import numpy as np
import pytest

from kino_vla.sim.realistic_route_controller_v4 import (
    SCHEMA_VERSION,
    SettledLookaheadRouteControllerV4,
)
from kino_vla.sim.realistic_route_protocol_v2 import RouteFrameV2


def _frame() -> RouteFrameV2:
    return RouteFrameV2(
        origin_xy_m=(0.0, 0.0),
        direction_xy=(1.0, 0.0),
        left_xy=(0.0, 1.0),
        heading_rad=0.0,
        route_length_m=2.0,
        surface_width_m=1.2,
    )


def _command(
    controller: SettledLookaheadRouteControllerV4,
    *,
    position: tuple[float, float],
    heading: float = 0.0,
) -> tuple[np.ndarray, dict]:
    return controller.command(
        _frame(),
        position_xy_m=np.asarray(position),
        heading_rad=heading,
        velocity_body_xy_mps=np.asarray([0.0, 0.0]),
        forward_speed_mps=0.2,
        target_lateral_offset_m=-0.25,
    )


def test_alignment_does_not_advance_toward_operator() -> None:
    controller = SettledLookaheadRouteControllerV4(alignment_dwell_steps=3)
    command, telemetry = _command(controller, position=(0.0, 0.0))
    assert command == pytest.approx([0.0, -0.12, 0.0])
    assert telemetry["controller_phase"] == "align"
    assert telemetry["alignment_transition_step"] is None


def test_dwell_switches_once_to_heading_only_traverse() -> None:
    controller = SettledLookaheadRouteControllerV4(alignment_dwell_steps=3)
    for _ in range(2):
        command, telemetry = _command(controller, position=(0.0, -0.25))
        assert command[0] == pytest.approx(0.0)
        assert telemetry["controller_phase"] == "align"
    command, telemetry = _command(controller, position=(0.0, -0.25))
    assert command == pytest.approx([0.2, 0.0, 0.0])
    assert telemetry["controller_phase"] == "traverse"
    assert telemetry["alignment_transition_step"] == 3

    command, telemetry = _command(controller, position=(0.1, 0.20))
    assert command[0] == pytest.approx(0.2)
    assert command[1] == pytest.approx(0.0)
    assert command[2] < 0.0
    assert telemetry["controller_phase"] == "traverse"


def test_lookahead_steers_back_without_lateral_strafe() -> None:
    controller = SettledLookaheadRouteControllerV4(alignment_dwell_steps=1)
    _command(controller, position=(0.0, -0.25))
    command, telemetry = _command(controller, position=(0.1, 0.05))
    assert command[0] == pytest.approx(0.2)
    assert command[1] == pytest.approx(0.0)
    assert command[2] == pytest.approx(-0.6)
    assert telemetry["lookahead_correction_angle_rad"] < 0.0


def test_contract_records_non_rescuing_phase_policy() -> None:
    contract = SettledLookaheadRouteControllerV4().contract()
    assert contract["schema_version"] == SCHEMA_VERSION
    assert contract["phase_policy"] == "one_way_align_then_traverse_no_anomaly_reacquisition"
    assert contract["traverse_cross_track_actuation"] == "heading_only_no_lateral_strafe"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alignment_gain_per_s": 0.0},
        {"alignment_lateral_limit_mps": 0.0},
        {"alignment_tolerance_m": 0.0},
        {"alignment_dwell_steps": 0},
        {"lookahead_m": 0.0},
        {"heading_gain_per_s": 0.0},
        {"yaw_rate_limit_radps": 0.0},
    ],
)
def test_invalid_contract_fails_closed(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SettledLookaheadRouteControllerV4(**kwargs)
