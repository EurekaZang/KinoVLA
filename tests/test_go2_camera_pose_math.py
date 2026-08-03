from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.isaac_embodiedgen_go2_scene_qa import (
    _quat_angular_distance_rad,
    _quat_apply_wxyz,
    _quat_multiply_wxyz,
)


def test_local_pitch_is_composed_after_base_heading() -> None:
    base = np.asarray([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    pitch = np.asarray([math.cos(0.05), 0.0, math.sin(0.05), 0.0])
    camera = _quat_multiply_wxyz(base, pitch)
    forward = _quat_apply_wxyz(camera, np.asarray([1.0, 0.0, 0.0]))
    assert forward[0] == pytest.approx(0.0, abs=1.0e-7)
    assert forward[1] == pytest.approx(math.cos(0.1), abs=1.0e-7)
    assert forward[2] == pytest.approx(-math.sin(0.1), abs=1.0e-7)


def test_quaternion_distance_is_sign_invariant() -> None:
    quat = np.asarray([0.9, 0.1, -0.2, 0.3], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    assert _quat_angular_distance_rad(quat, -quat) == pytest.approx(0.0)
