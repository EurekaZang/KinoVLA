from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.sim.realistic_route_controller_v3 import DampedRouteControllerV3
from kino_vla.sim.realistic_route_protocol_v2 import RouteFrameV2


def _frame() -> RouteFrameV2:
    return RouteFrameV2(
        origin_xy_m=(0.0, 0.0),
        direction_xy=(1.0, 0.0),
        left_xy=(0.0, 1.0),
        heading_rad=0.0,
        route_length_m=4.0,
        surface_width_m=1.0,
    )


def test_v37_emergency_controller_opposes_recorded_lateral_momentum() -> None:
    normal = DampedRouteControllerV3(
        cross_track_gain_per_s=1.0,
        lateral_velocity_damping=0.0,
        lateral_limit_mps=0.2,
    )
    emergency = DampedRouteControllerV3(
        cross_track_gain_per_s=2.0,
        lateral_velocity_damping=1.0,
        lateral_limit_mps=0.4,
    )
    kwargs = {
        "frame": _frame(),
        "position_xy_m": np.asarray([0.404, 0.083]),
        "heading_rad": 0.0,
        "velocity_body_xy_mps": np.asarray([0.436, 0.231]),
        "forward_speed_mps": -0.48,
        "target_lateral_offset_m": 0.0,
    }
    _, normal_control = normal.command(**kwargs)
    _, emergency_control = emergency.command(**kwargs)
    assert normal_control["clipped_lateral_command_mps"] == pytest.approx(-0.083)
    assert emergency_control["clipped_lateral_command_mps"] == pytest.approx(-0.397)
    assert abs(emergency_control["clipped_lateral_command_mps"]) > 4.0 * abs(
        normal_control["clipped_lateral_command_mps"]
    )


def test_v37_emergency_controller_respects_frozen_limit() -> None:
    emergency = DampedRouteControllerV3(
        cross_track_gain_per_s=2.0,
        lateral_velocity_damping=1.0,
        lateral_limit_mps=0.4,
    )
    _, control = emergency.command(
        _frame(),
        position_xy_m=np.asarray([0.5, 0.3]),
        heading_rad=0.0,
        velocity_body_xy_mps=np.asarray([0.0, 0.8]),
        forward_speed_mps=-0.48,
        target_lateral_offset_m=0.0,
    )
    assert control["raw_lateral_command_mps"] == pytest.approx(-1.4)
    assert control["clipped_lateral_command_mps"] == pytest.approx(-0.4)
