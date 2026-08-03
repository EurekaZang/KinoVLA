from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_o4_action_consequence_v35_development_postrun_v2 as audit


def test_corrected_one_step_trigger_accepts_frozen_urgency() -> None:
    runtime = {"minimum_decision_dwell_steps": 1, "maximum_decision_dwell_steps": 1}
    manifest = {
        "decision_contract": {
            "minimum_decision_dwell_steps": 1,
            "maximum_decision_dwell_steps": 1,
        },
        "takes": {
            "continue": {
                "first_attachment_step": 81,
                "predecision_step": 82,
                "decision_trigger_reason": "posture_urgency",
            },
            "backstep": {
                "first_attachment_step": 81,
                "predecision_step": 82,
                "decision_trigger_reason": "posture_urgency",
            },
        },
    }
    assert all(audit._corrected_early_branch_checks(manifest, runtime).values())


def test_corrected_one_step_trigger_rejects_mismatched_reasons() -> None:
    runtime = {"minimum_decision_dwell_steps": 1, "maximum_decision_dwell_steps": 1}
    manifest = {
        "decision_contract": {
            "minimum_decision_dwell_steps": 1,
            "maximum_decision_dwell_steps": 1,
        },
        "takes": {
            "continue": {
                "first_attachment_step": 1,
                "predecision_step": 2,
                "decision_trigger_reason": "posture_urgency",
            },
            "backstep": {
                "first_attachment_step": 1,
                "predecision_step": 2,
                "decision_trigger_reason": "maximum_dwell",
            },
        },
    }
    checks = audit._corrected_early_branch_checks(manifest, runtime)
    assert not checks["both_takes_use_same_frozen_one_step_trigger"]
