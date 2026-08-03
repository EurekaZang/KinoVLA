from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime() -> dict:
    return {
        "decision_tilt_urgency_rad": 0.08,
        "decision_height_drop_urgency_m": 0.02,
        "maximum_route_deviation_m": 0.3,
    }


def _row(step: int, offset: float, height: float, tilt: float) -> dict:
    return {
        "step": step,
        "route_lateral_offset_m": offset,
        "base_height_m": height,
        "tilt_rad": tilt,
    }


def _manifest() -> dict:
    summary = {"predecision_step": 1, "preattachment_base_height_m": 0.42}
    return {"takes": {"continue": dict(summary), "backstep": dict(summary)}}


def test_phase_aware_route_allows_only_post_onset_continue_deviation() -> None:
    postrun = _module(
        "postrun_o4_v34",
        "scripts/audit_o4_action_consequence_v34_confirmation_postrun.py",
    )
    rows = {
        "continue": [
            _row(0, 0.01, 0.42, 0.01),
            _row(1, 0.02, 0.41, 0.02),
            _row(2, 0.03, 0.39, 0.09),
            _row(3, 0.40, 0.20, 0.70),
        ],
        "backstep": [
            _row(0, 0.01, 0.42, 0.01),
            _row(1, 0.02, 0.41, 0.02),
            _row(2, 0.04, 0.41, 0.02),
            _row(3, 0.03, 0.42, 0.01),
        ],
    }
    checks, metrics = postrun._phase_aware_route_checks(
        _manifest(), _runtime(), rows
    )
    assert all(checks.values())
    assert metrics["physical_failure_onset_step"] == 2
    assert metrics["continue_first_over_route_budget_step"] == 3
    assert metrics["continue_first_over_route_budget_after_failure_onset"]


def test_phase_aware_route_rejects_pre_onset_or_backstep_deviation() -> None:
    postrun = _module(
        "postrun_o4_v34_negative",
        "scripts/audit_o4_action_consequence_v34_confirmation_postrun.py",
    )
    rows = {
        "continue": [
            _row(0, 0.01, 0.42, 0.01),
            _row(1, 0.02, 0.41, 0.02),
            _row(2, 0.31, 0.41, 0.02),
            _row(3, 0.32, 0.39, 0.09),
        ],
        "backstep": [
            _row(0, 0.01, 0.42, 0.01),
            _row(1, 0.02, 0.41, 0.02),
            _row(2, 0.31, 0.42, 0.01),
        ],
    }
    checks, _ = postrun._phase_aware_route_checks(_manifest(), _runtime(), rows)
    assert not checks["continue_through_failure_onset_within_route_budget"]
    assert not checks["backstep_full_horizon_within_route_budget"]


def test_v34_preflight_contract_is_exact_and_route_independent() -> None:
    preflight = _module(
        "preflight_o4_v34",
        "scripts/audit_o4_action_consequence_v34_confirmation_preflight.py",
    )
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
    assert preflight._phase_aware_contract_is_exact(contract)
    contract["physical_failure_onset"]["route_deviation_used_to_define_onset"] = True
    assert not preflight._phase_aware_contract_is_exact(contract)
