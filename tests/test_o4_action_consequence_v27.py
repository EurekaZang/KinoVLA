from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "o4_action_v27", ROOT / "scripts/isaac_collect_o4_action_consequence_v27.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_posture_urgency_preempts_maximum_dwell_only_after_minimum() -> None:
    assert (
        MODULE._decision_trigger_reason(
            elapsed_steps=4,
            urgency_dwell_steps=3,
            minimum_dwell_steps=5,
            maximum_dwell_steps=25,
            required_urgency_dwell_steps=2,
        )
        is None
    )
    assert (
        MODULE._decision_trigger_reason(
            elapsed_steps=5,
            urgency_dwell_steps=2,
            minimum_dwell_steps=5,
            maximum_dwell_steps=25,
            required_urgency_dwell_steps=2,
        )
        == "posture_urgency"
    )


def test_maximum_dwell_is_failure_closed_fallback() -> None:
    assert (
        MODULE._decision_trigger_reason(
            elapsed_steps=24,
            urgency_dwell_steps=0,
            minimum_dwell_steps=5,
            maximum_dwell_steps=25,
            required_urgency_dwell_steps=2,
        )
        is None
    )
    assert (
        MODULE._decision_trigger_reason(
            elapsed_steps=25,
            urgency_dwell_steps=0,
            minimum_dwell_steps=5,
            maximum_dwell_steps=25,
            required_urgency_dwell_steps=2,
        )
        == "maximum_dwell"
    )


def test_clearance_requires_consecutive_release_and_retreat() -> None:
    dwell = MODULE._next_clearance_dwell(
        0, retreat_m=0.21, target_retreat_m=0.20, active_feet=0
    )
    assert dwell == 1
    dwell = MODULE._next_clearance_dwell(
        dwell, retreat_m=0.22, target_retreat_m=0.20, active_feet=0
    )
    assert dwell == 2
    assert (
        MODULE._next_clearance_dwell(
            dwell, retreat_m=0.23, target_retreat_m=0.20, active_feet=1
        )
        == 0
    )
