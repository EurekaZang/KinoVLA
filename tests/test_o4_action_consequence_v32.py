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


def _contract() -> dict:
    return {
        "strict_labels": ["initial", "decision", "backstep_outcome"],
        "strict_minimum_std_luminance": 0.025,
        "strict_maximum_black_fraction": 0.95,
        "strict_minimum_quantized_color_count_5bit": 32,
        "minimum_mean_luminance": 0.02,
        "maximum_mean_luminance": 0.98,
        "fallen_continue_outcome_maximum_black_fraction": 0.98,
        "fallen_continue_outcome_minimum_quantized_color_count_5bit": 16,
    }


def _capture(label: str, *, fallen: bool = False, std: float = 0.2, colors: int = 100):
    return {
        "label": label,
        "fallen": fallen,
        "metrics": {
            "std_luminance": std,
            "black_fraction": 0.01,
            "quantized_color_count_5bit": colors,
            "mean_luminance": 0.5,
        },
    }


def test_v32_runtime_delta_changes_only_three_frozen_fields() -> None:
    preflight = _module(
        "preflight_o4_v32",
        "scripts/audit_o4_action_consequence_v32_development_preflight.py",
    )
    source = {
        "minimum_decision_dwell_steps": 5,
        "maximum_decision_dwell_steps": 25,
        "maximum_backstep_steps": 120,
        "horizon_steps": 300,
        "forward_speed_mps": 0.32,
    }
    candidate = dict(source)
    candidate.update(
        minimum_decision_dwell_steps=1,
        maximum_decision_dwell_steps=1,
        maximum_backstep_steps=300,
    )
    assert preflight._runtime_delta_is_exact(source, candidate)
    candidate["forward_speed_mps"] = 0.31
    assert not preflight._runtime_delta_is_exact(source, candidate)


def test_v32_visual_qa_relaxes_only_fallen_continue_outcome() -> None:
    postrun = _module(
        "postrun_o4_v32",
        "scripts/audit_o4_action_consequence_v32_development_postrun.py",
    )
    manifest = {
        "takes": {
            "continue": {
                "fell": True,
                "captures": [
                    _capture("initial"),
                    _capture("decision"),
                    _capture("outcome", fallen=True, std=0.0, colors=16),
                ],
            },
            "backstep": {
                "fell": False,
                "captures": [
                    _capture("initial"),
                    _capture("decision"),
                    _capture("outcome"),
                ],
            },
        }
    }
    checks = postrun._state_aware_visual_checks(manifest, _contract())
    assert all(checks.values())
    manifest["takes"]["backstep"]["captures"][2] = _capture(
        "outcome", std=0.0, colors=16
    )
    checks = postrun._state_aware_visual_checks(manifest, _contract())
    assert checks["fallen_continue_outcome_sensor_valid"]
    assert not checks["backstep_outcome_strict_visual_quality"]


def test_v32_visual_qa_rejects_invalid_fallen_sensor_frame() -> None:
    postrun = _module(
        "postrun_o4_v32_invalid",
        "scripts/audit_o4_action_consequence_v32_development_postrun.py",
    )
    fallen = _capture("outcome", fallen=True, std=0.0, colors=15)
    manifest = {
        "takes": {
            "continue": {
                "fell": True,
                "captures": [_capture("initial"), _capture("decision"), fallen],
            },
            "backstep": {
                "fell": False,
                "captures": [
                    _capture("initial"),
                    _capture("decision"),
                    _capture("outcome"),
                ],
            },
        }
    }
    checks = postrun._state_aware_visual_checks(manifest, _contract())
    assert not checks["fallen_continue_outcome_sensor_valid"]
