from __future__ import annotations

from types import SimpleNamespace

import pytest

from kino_vla.sim.isaac_policy_backend import (
    INFERENCE_RESET_SCHEMA_VERSION,
    _freeze_inference_joint_reset,
)


def test_inference_reset_removes_hidden_joint_pose_randomization() -> None:
    reset = SimpleNamespace(
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        }
    )
    env_cfg = SimpleNamespace(
        events=SimpleNamespace(reset_robot_joints=reset)
    )
    contract = _freeze_inference_joint_reset(env_cfg)
    assert reset.params["position_range"] == (1.0, 1.0)
    assert reset.params["velocity_range"] == (0.0, 0.0)
    assert contract["schema_version"] == INFERENCE_RESET_SCHEMA_VERSION
    assert contract["observed_pre_freeze_position_scale_range"] == [0.5, 1.5]
    assert contract["seed_may_randomize_initial_joints"] is False


def test_inference_reset_fails_closed_without_joint_event() -> None:
    with pytest.raises(RuntimeError):
        _freeze_inference_joint_reset(SimpleNamespace(events=SimpleNamespace()))
