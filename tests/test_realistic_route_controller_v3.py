from __future__ import annotations

import numpy as np
import pytest

from kino_vla.sim.realistic_route_controller_v3 import DampedRouteControllerV3
from kino_vla.sim.realistic_route_protocol_v2 import RouteFrameV2


def _frame(direction: tuple[float, float], left: tuple[float, float], heading: float) -> RouteFrameV2:
    return RouteFrameV2(
        origin_xy_m=(0.0, 0.0),
        direction_xy=direction,
        left_xy=left,
        heading_rad=heading,
        route_length_m=2.0,
        surface_width_m=1.2,
    )


def test_damping_opposes_measured_lateral_velocity() -> None:
    controller = DampedRouteControllerV3(lateral_velocity_damping=1.0)
    command, telemetry = controller.command(
        _frame((1.0, 0.0), (0.0, 1.0), 0.0),
        position_xy_m=np.asarray([0.5, 0.0]),
        heading_rad=0.0,
        velocity_body_xy_mps=np.asarray([0.2, 0.1]),
        forward_speed_mps=0.2,
        target_lateral_offset_m=0.0,
    )
    assert command == pytest.approx([0.2, -0.1, 0.0])
    assert telemetry["measured_lateral_velocity_mps"] == pytest.approx(0.1)


def test_damping_reduces_command_that_would_amplify_return_motion() -> None:
    controller = DampedRouteControllerV3(lateral_velocity_damping=1.0)
    command, telemetry = controller.command(
        _frame((1.0, 0.0), (0.0, 1.0), 0.0),
        position_xy_m=np.asarray([0.5, 0.1]),
        heading_rad=0.0,
        velocity_body_xy_mps=np.asarray([0.2, -0.15]),
        forward_speed_mps=0.2,
        target_lateral_offset_m=0.0,
    )
    assert telemetry["raw_lateral_command_mps"] == pytest.approx(0.05)
    assert command[1] == pytest.approx(0.05)


def test_controller_is_route_orientation_invariant() -> None:
    controller = DampedRouteControllerV3()
    command, telemetry = controller.command(
        _frame((0.0, 1.0), (-1.0, 0.0), np.pi / 2.0),
        position_xy_m=np.asarray([-0.1, 0.5]),
        heading_rad=np.pi / 2.0,
        velocity_body_xy_mps=np.asarray([0.2, 0.0]),
        forward_speed_mps=0.2,
        target_lateral_offset_m=0.0,
    )
    assert telemetry["route_lateral_offset_m"] == pytest.approx(0.1)
    assert command == pytest.approx([0.2, -0.1, 0.0])
