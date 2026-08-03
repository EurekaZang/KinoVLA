from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _collector():
    spec = importlib.util.spec_from_file_location(
        "collector_o4_v35",
        ROOT / "scripts/isaac_collect_o4_action_consequence_v35.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _command(**overrides):
    values = {
        "nominal_speed_mps": 0.24,
        "emergency_speed_mps": 0.48,
        "decision_progress_m": 0.20,
        "current_progress_m": 0.22,
        "preattachment_base_height_m": 0.42,
        "current_base_height_m": 0.41,
        "current_tilt_rad": 0.04,
        "maximum_forward_penetration_m": 0.025,
        "urgency_tilt_rad": 0.08,
        "urgency_height_drop_m": 0.02,
        "already_escalated": False,
    }
    values.update(overrides)
    return _collector()._adaptive_backstep_command(**values)


def test_v35_uses_nominal_reverse_below_both_guards() -> None:
    speed, escalated, reason = _command()
    assert speed == -0.24
    assert not escalated
    assert reason is None


def test_v35_escalates_on_physical_urgency_or_forward_penetration() -> None:
    urgency = _command(current_base_height_m=0.40, current_tilt_rad=0.08)
    penetration = _command(current_progress_m=0.225)
    assert urgency == (-0.48, True, "physical_urgency")
    assert penetration == (-0.48, True, "forward_penetration_guard")


def test_v35_emergency_reverse_latches_after_measurements_recover() -> None:
    speed, escalated, reason = _command(
        current_progress_m=0.19,
        current_base_height_m=0.42,
        current_tilt_rad=0.01,
        already_escalated=True,
    )
    assert speed == -0.48
    assert escalated
    assert reason is None
