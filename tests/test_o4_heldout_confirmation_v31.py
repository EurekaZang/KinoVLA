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


def test_v31_runner_uses_frozen_v27_collector_and_phase_contract() -> None:
    runner = _module("run_o4_v31", "scripts/run_o4_heldout_confirmation_v31.py")
    runtime = {
        "camera_profile": "go2_front_calib_b",
        "horizon_steps": 300,
        "minimum_decision_dwell_steps": 5,
        "maximum_decision_dwell_steps": 25,
        "decision_tilt_urgency_rad": 0.08,
        "decision_height_drop_urgency_m": 0.02,
        "decision_urgency_dwell_steps": 2,
        "maximum_backstep_steps": 120,
        "target_recovery_distance_m": 0.2,
        "minimum_recovery_distance_m": 0.15,
        "adhesion_clearance_dwell_steps": 5,
        "forward_speed_mps": 0.32,
        "backstep_speed_mps": 0.24,
        "maximum_route_deviation_m": 0.3,
        "tangential_force_cap_n": 18.0,
        "normal_force_cap_n": 8.0,
        "peel_height_m": 0.025,
        "unload_steps_to_peel": 3,
        "reattach_cooldown_steps": 2,
        "max_active_feet": 2,
        "cross_track_gain_per_s": 1.0,
        "lateral_velocity_damping": 0.0,
        "heading_gain_per_s": 2.0,
        "lateral_limit_mps": 0.2,
        "yaw_rate_limit_radps": 0.6,
    }
    case = {
        "episode_usd": "episode.usda",
        "output": "out",
        "material_id": "mat",
        "runtime_seed": 7,
        "prerequisites": {
            "terrain_compiled_audit": {"path": "compiled.json"}
        },
    }
    args = runner._collector_args(case, runtime)
    assert args[0] == "scripts/isaac_collect_o4_action_consequence_v27.py"
    assert args[args.index("--minimum-decision-dwell-steps") + 1] == "5"
    assert args[args.index("--decision-dwell-steps") + 1] == "25"
    assert args[args.index("--maximum-backstep-steps") + 1] == "120"
    assert args[args.index("--target-recovery-distance-m") + 1] == "0.2"
    assert args[-1] == "--headless"


def test_v31_postrun_imports_frozen_v27_auditor() -> None:
    postrun = _module(
        "audit_o4_v31", "scripts/audit_o4_heldout_confirmation_v31_postrun.py"
    )
    assert callable(postrun._audit_v27_case)
    assert callable(postrun._locked)
