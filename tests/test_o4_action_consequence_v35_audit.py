from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_o4_action_consequence_v35_development_preflight as audit
from scripts import audit_o4_action_consequence_v35_development_postrun as postrun


def test_v35_runtime_delta_adds_only_guard_and_collector_revision() -> None:
    source = {"forward_speed_mps": 0.32, "collector_sha256": "old"}
    candidate = {
        "forward_speed_mps": 0.32,
        "collector_sha256": "new",
        "emergency_backstep_speed_mps": 0.48,
        "maximum_postdecision_forward_penetration_m": 0.025,
    }
    assert audit._v35_runtime_delta_is_exact(source, candidate, "new")
    candidate["forward_speed_mps"] = 0.33
    assert not audit._v35_runtime_delta_is_exact(source, candidate, "new")


def test_v35_phase_contract_rejects_route_defined_onset() -> None:
    contract = {
        "route_budget_m": 0.3,
        "matched_prefix_within_route_budget": True,
        "backstep_full_horizon_within_route_budget": True,
        "continue_through_physical_failure_onset_within_route_budget": True,
        "physical_failure_onset": {
            "tilt_rad_at_least": 0.08,
            "height_drop_m_at_least": 0.02,
            "conjunction_required": True,
            "route_deviation_used_to_define_onset": False,
        },
        "continue_post_onset_deviation_reported_not_gated": True,
        "legacy_full_horizon_continue_deviation_retained": True,
    }
    assert audit._phase_aware_contract_is_exact(contract)
    contract["physical_failure_onset"]["route_deviation_used_to_define_onset"] = True
    assert not audit._phase_aware_contract_is_exact(contract)


def test_v35_adaptive_contract_accepts_optional_valid_escalation() -> None:
    runtime = {
        "backstep_speed_mps": 0.24,
        "emergency_backstep_speed_mps": 0.48,
        "maximum_postdecision_forward_penetration_m": 0.025,
        "target_recovery_distance_m": 0.2,
        "adhesion_clearance_dwell_steps": 5,
        "maximum_backstep_steps": 300,
        "horizon_steps": 300,
    }
    action = (
        "reverse 0.24 m/s, latching 0.48 m/s when physical urgency or 0.025 m "
        "forward penetration is observed, until retreat reaches 0.2 m and active adhesion "
        "remains clear for 5 steps, capped at 300 steps, then hold"
    )
    manifest = {
        "decision_contract": {
            "backstep_action": action,
            "adaptive_backstep_guard": {
                "emergency_backstep_speed_mps": 0.48,
                "maximum_postdecision_forward_penetration_m": 0.025,
                "physical_urgency_uses_frozen_decision_thresholds": True,
                "latches_after_first_trigger": True,
                "scene_identity_used": False,
                "command_conditioned_release_used": False,
            },
        },
        "takes": {"backstep": {
            "predecision_step": 81,
            "backstep_escalated": True,
            "backstep_escalation_step": 90,
            "backstep_escalation_reason": "forward_penetration_guard",
        }},
    }
    assert all(postrun._adaptive_contract_checks(manifest, runtime).values())
    manifest["takes"]["backstep"].update({
        "backstep_escalated": False,
        "backstep_escalation_step": None,
        "backstep_escalation_reason": None,
    })
    assert all(postrun._adaptive_contract_checks(manifest, runtime).values())
