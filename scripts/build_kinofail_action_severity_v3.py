#!/usr/bin/env python3
"""Freeze the complete mild/current/hard remediation confirmation."""

import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_kinofail_action_severity_v1 as base


CURRENT = {
    "O1_mu_field": {"mu_s": 0.06, "mu_d": 0.04, "restitution": 0.0},
    "O2_compliance": {"sink_depth_m": 0.045, "shear_gain": 0.66, "stiffness_n_per_m": 120.0},
    "O3_collapse": {"damage_threshold_ns": 70.0, "drop_m": 0.030, "residual_support": 0.68},
    "O4_tether": {"attachment_enabled": 1.0, "tangential_force_cap_n": 30.0, "normal_force_cap_n": 12.0, "peel_height_m": 0.025, "unload_steps_to_peel": 3.0},
    "O5_payload": {"mass_kg": 8.0, "com_offset_x_m": 0.08, "com_offset_y_m": 0.0},
    "O6_push": {"impulse_ns": 3.0, "duration_s": 0.12, "application_point_body_m": [0.0, -0.085, 0.075]},
    "O7_visual_remap": {"mu_s": 0.06, "mu_d": 0.04, "depth_bias_m": 0.32},
    "O8_invisible_collider": {"collision_enabled": 1.0, "obstacle_height_m": 0.12, "optical_transmission": 0.92},
    "O9_high_centering": {"ridge_height_m": 0.19, "ridge_width_m": 0.24, "residual_support": 0.46},
    "O10_effort_decay": {"effort_floor": 0.72, "decay_rate_per_s": 0.48, "onset_s": 0.80},
    "O11_obs_bias": {"tilt_bias_rad": 0.32, "random_walk_rad_sqrt_s": 0.018, "latency_s": 0.18},
}


def main() -> int:
    base.OUTPUT = base.ROOT / "outputs/kinofail_action_severity_v3"
    base.COLLECTOR = "scripts/isaac_collect_kinofail_remediation_v2.py"
    base.PROTOCOL_ID = "kinofail-action-severity-v3-three-level-confirmation-20260814"
    base.HORIZON_STEPS = 1200
    base.SEVERITY_LEVELS = ("mild", "current", "hard")
    base.SEVERITY_PARAMETERS["current"] = copy.deepcopy(CURRENT)
    base.O9_PRODUCTION_004["current"] = {
        "ridge_height_m": 0.22,
        "ridge_width_m": 0.28,
        "residual_support": 0.25,
    }
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
