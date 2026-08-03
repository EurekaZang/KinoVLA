#!/usr/bin/env python3
"""Paired finite-wrench dose characterization for realistic O6 pushes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SEEDS = (11, 23, 37, 42, 59)
DEFAULT_IMPULSES_NS = (2.0, 6.0, 12.0)
PULSE_DURATION_S = 0.12
# Runtime USD audit confirms this lies on the upper-right side of the Go2 base bounds.
APPLICATION_POINT_BODY_M = np.array([0.0, -0.085, 0.075], dtype=np.float64)
WARMUP_S = 2.0
IMPACT_WINDOW_S = 0.50
RECOVERY_S = 3.5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paired_command(seed: int) -> np.ndarray:
    """Generate one deterministic locomotion condition reused by all doses."""
    rng = np.random.default_rng(seed)
    return np.array(
        [
            rng.uniform(0.30, 0.42),
            rng.uniform(-0.015, 0.015),
            rng.uniform(-0.04, 0.04),
        ],
        dtype=np.float64,
    )


def _world_velocity(backend) -> tuple[np.ndarray, np.ndarray]:
    linear = backend._robot.data.root_lin_vel_w[0].detach().cpu().numpy().astype(np.float64)
    angular = backend._robot.data.root_ang_vel_w[0].detach().cpu().numpy().astype(np.float64)
    return linear, angular


def _base_local_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Read the composed Go2 base bound from the live USD, in the base local frame."""
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/base")
    if not prim.IsValid():
        raise RuntimeError("could not resolve the live Go2 base prim")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    bounds = cache.ComputeLocalBound(prim).ComputeAlignedRange()
    return np.asarray(bounds.GetMin(), dtype=np.float64), np.asarray(
        bounds.GetMax(), dtype=np.float64
    )


def _run_protocol(
    backend,
    *,
    command: np.ndarray,
    impulse_ns: float,
) -> dict[str, object]:
    """Warm up at a fixed gait phase, apply one lateral pulse, and observe recovery."""
    obs = backend._make_obs()
    while obs.t < WARMUP_S and not obs.fallen:
        obs = backend.step(command)
    pre_linear, pre_angular = _world_velocity(backend)
    pre_pos = obs.pos.copy()
    pre_tilt = float(obs.tilt)
    if impulse_ns > 0.0:
        backend.start_push_pulse(
            np.array([0.0, impulse_ns], dtype=np.float64),
            0.0,
            PULSE_DURATION_S,
            APPLICATION_POINT_BODY_M,
        )

    impact_steps = max(1, int(round(IMPACT_WINDOW_S / backend.dt)))
    total_steps = max(impact_steps, int(round(RECOVERY_S / backend.dt)))
    peak_lateral_delta = 0.0
    peak_signed_roll_delta = 0.0
    peak_tilt = pre_tilt
    peak_lateral_excursion = 0.0
    for step_index in range(total_steps):
        obs = backend.step(command)
        linear, angular = _world_velocity(backend)
        if step_index < impact_steps:
            # +Y force at +Z gives a -X roll torque. The signs therefore audit both
            # the force and application-point semantics instead of an unsigned norm alone.
            peak_lateral_delta = max(
                peak_lateral_delta, float(linear[1] - pre_linear[1])
            )
            peak_signed_roll_delta = max(
                peak_signed_roll_delta, float(-(angular[0] - pre_angular[0]))
            )
        peak_tilt = max(peak_tilt, float(obs.tilt))
        peak_lateral_excursion = max(
            peak_lateral_excursion, abs(float(obs.pos[1] - pre_pos[1]))
        )
        if obs.fallen:
            break

    final_linear, final_angular = _world_velocity(backend)
    return {
        "command_vx_vy_yaw": command.tolist(),
        "requested_impulse_ns": float(impulse_ns),
        "pre_push_pos_xy_m": pre_pos.tolist(),
        "pre_push_world_linear_velocity_mps": pre_linear.tolist(),
        "pre_push_world_angular_velocity_radps": pre_angular.tolist(),
        "pre_push_tilt_rad": pre_tilt,
        "peak_push_direction_delta_velocity_mps": peak_lateral_delta,
        "peak_expected_roll_delta_radps": peak_signed_roll_delta,
        "peak_tilt_rad": peak_tilt,
        "peak_tilt_increase_rad": max(0.0, peak_tilt - pre_tilt),
        "peak_lateral_excursion_m": peak_lateral_excursion,
        "final_pos_xy_m": obs.pos.tolist(),
        "final_world_linear_velocity_mps": final_linear.tolist(),
        "final_world_angular_velocity_radps": final_angular.tolist(),
        "final_tilt_rad": float(obs.tilt),
        "fallen": bool(obs.fallen),
        "episode_duration_s": float(obs.t),
        "pulse_telemetry": backend.push_telemetry(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/kinofail_realistic/o6_force_pulse_dose_sweep")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument(
        "--dose-impulses",
        default=",".join(str(value) for value in DEFAULT_IMPULSES_NS),
        help="mild,moderate,severe lateral impulses in N*s",
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    import isaaclab

    from kino_vla.eval.dose_response import (
        bootstrap_mean_interval,
        paired_dose_monotonicity,
    )
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import load_config

    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("--seeds must contain one or more unique comma-separated integers")
    dose_impulses = tuple(
        float(value) for value in args.dose_impulses.split(",") if value.strip()
    )
    if len(dose_impulses) != 3 or any(value <= 0.0 for value in dose_impulses):
        raise ValueError("--dose-impulses must contain three positive values")
    if not all(
        left < right for left, right in zip(dose_impulses, dose_impulses[1:], strict=False)
    ):
        raise ValueError("--dose-impulses must be strictly increasing")
    doses = tuple(
        {"dose": name, "impulse_ns": impulse}
        for name, impulse in zip(("mild", "moderate", "severe"), dose_impulses, strict=True)
    )

    cfg = load_config("sim/go2_skeleton.yaml")
    backend = IsaacPolicyBackend(cfg, np.array([0.0, 0.0]), 0.0)
    base_bound_min, base_bound_max = _base_local_bounds()
    records: list[dict[str, object]] = []
    nominal_by_seed: dict[int, dict[str, object]] = {}
    for seed in selected_seeds:
        command = _paired_command(seed)
        backend.deep_reset(seed)
        result = _run_protocol(backend, command=command, impulse_ns=0.0)
        nominal_by_seed[seed] = result
        records.append(
            {
                "seed": seed,
                "dose": "nominal",
                "impulse_ns": 0.0,
                **result,
                "final_lateral_deviation_from_nominal_m": 0.0,
                "peak_lateral_excursion_increase_m": 0.0,
            }
        )

    for dose in doses:
        for seed in selected_seeds:
            command = _paired_command(seed)
            backend.deep_reset(seed)
            result = _run_protocol(
                backend,
                command=command,
                impulse_ns=float(dose["impulse_ns"]),
            )
            nominal = nominal_by_seed[seed]
            records.append(
                {
                    "seed": seed,
                    **dose,
                    **result,
                    "final_lateral_deviation_from_nominal_m": abs(
                        float(result["final_pos_xy_m"][1])
                        - float(nominal["final_pos_xy_m"][1])
                    ),
                    "peak_lateral_excursion_increase_m": max(
                        0.0,
                        float(result["peak_lateral_excursion_m"])
                        - float(nominal["peak_lateral_excursion_m"]),
                    ),
                }
            )

    dose_order = ["nominal", "mild", "moderate", "severe"]
    monotonicity = {
        "lateral_delta_velocity": paired_dose_monotonicity(
            records,
            dose_order=dose_order,
            metric="peak_push_direction_delta_velocity_mps",
            tolerance=0.03,
        ),
        "expected_roll_rate": paired_dose_monotonicity(
            records,
            dose_order=dose_order,
            metric="peak_expected_roll_delta_radps",
            tolerance=0.08,
        ),
        "lateral_excursion": paired_dose_monotonicity(
            records,
            dose_order=dose_order,
            metric="peak_lateral_excursion_m",
            tolerance=0.03,
        ),
    }
    by_dose: dict[str, dict[str, object]] = {}
    summary_metrics = (
        "peak_push_direction_delta_velocity_mps",
        "peak_expected_roll_delta_radps",
        "peak_tilt_increase_rad",
        "peak_lateral_excursion_m",
        "final_lateral_deviation_from_nominal_m",
    )
    for dose_index, dose_name in enumerate(dose_order):
        cells = [row for row in records if row["dose"] == dose_name]
        by_dose[dose_name] = {
            "n_paired_conditions": len(cells),
            "fall_count": sum(bool(row["fallen"]) for row in cells),
        }
        for metric_index, metric in enumerate(summary_metrics):
            by_dose[dose_name][metric] = bootstrap_mean_interval(
                [float(row[metric]) for row in cells],
                seed=700 + 10 * dose_index + metric_index,
            ).to_dict()

    anomaly_rows = [row for row in records if row["dose"] != "nominal"]
    mild_rows = [row for row in records if row["dose"] == "mild"]
    severe_rows = [row for row in records if row["dose"] == "severe"]
    telemetry_rows = [row["pulse_telemetry"] for row in anomaly_rows]
    backend.deep_reset(selected_seeds[0])
    reset_telemetry = backend.push_telemetry()
    checks = {
        "complete_paired_design": len(records) == len(selected_seeds) * len(dose_order),
        "paired_nuisance_commands_match": all(
            len(
                {
                    tuple(float(value) for value in row["command_vx_vy_yaw"])
                    for row in records
                    if int(row["seed"]) == seed
                }
            )
            == 1
            for seed in selected_seeds
        ),
        "nuisance_conditions_are_distinct": len(
            {
                tuple(float(value) for value in row["command_vx_vy_yaw"])
                for row in records
                if row["dose"] == "nominal"
            }
        )
        == len(selected_seeds),
        "finite_physx_wrench_mode": all(
            row["mode"] == "finite_physx_wrench_at_body_point" for row in telemetry_rows
        ),
        "all_pulses_complete": all(
            bool(row["complete"])
            and not bool(row["active"])
            and int(row["applied_steps"]) == int(row["target_steps"])
            for row in telemetry_rows
        ),
        "commanded_linear_impulse_matches_request": all(
            np.allclose(
                np.asarray(row["pulse_telemetry"]["commanded_linear_impulse_xy_ns"]),
                np.array([0.0, float(row["impulse_ns"])]),
                atol=1.0e-5,
            )
            for row in anomaly_rows
        ),
        "body_point_and_duration_readback_match": all(
            np.allclose(
                np.asarray(row["application_point_body_m"]),
                APPLICATION_POINT_BODY_M,
                atol=1.0e-7,
            )
            and abs(float(row["effective_duration_s"]) - PULSE_DURATION_S) <= backend.dt / 2
            for row in telemetry_rows
        ),
        "application_point_lies_within_live_base_bounds": bool(
            np.all(APPLICATION_POINT_BODY_M >= base_bound_min - 1.0e-6)
            and np.all(APPLICATION_POINT_BODY_M <= base_bound_max + 1.0e-6)
        ),
        "off_center_point_generates_roll_impulse": all(
            float(row["commanded_torque_impulse_xyz_nms"][0]) < -0.05
            and abs(float(row["commanded_torque_impulse_xyz_nms"][0]))
            >= 0.05 * float(anomaly["impulse_ns"])
            for row, anomaly in zip(telemetry_rows, anomaly_rows, strict=True)
        ),
        "lateral_velocity_orders_most_pairs": monotonicity["lateral_delta_velocity"][
            "paired_seed_order_fraction"
        ]
        >= 0.8,
        "roll_response_orders_most_pairs": monotonicity["expected_roll_rate"][
            "paired_seed_order_fraction"
        ]
        >= 0.8,
        "mild_is_not_catastrophic": all(not bool(row["fallen"]) for row in mild_rows),
        "severe_has_measured_motion_consequence": all(
            bool(row["fallen"])
            or float(row["peak_push_direction_delta_velocity_mps"]) >= 0.40
            or float(row["peak_lateral_excursion_increase_m"]) >= 0.12
            for row in severe_rows
        ),
        "deep_reset_clears_push_state": reset_telemetry["mode"] == "inactive"
        and not bool(reset_telemetry["active"])
        and int(reset_telemetry["applied_steps"]) == 0,
    }
    manifest = {
        "schema_version": "kinofail.o6-finite-wrench-dose-sweep.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "publication_status": "engineering_dose_gate_not_corpus_evidence",
        "checks": checks,
        "design": {
            "paired_by_condition": True,
            "seeds": list(selected_seeds),
            "dose_order": dose_order,
            "doses": list(doses),
            "single_factor_dose": "world-frame lateral linear impulse",
            "force_pulse": {
                "requested_duration_s": PULSE_DURATION_S,
                "application_point_body_m": APPLICATION_POINT_BODY_M.tolist(),
                "live_go2_base_local_bound_min_m": base_bound_min.tolist(),
                "live_go2_base_local_bound_max_m": base_bound_max.tolist(),
                "requested_yaw_impulse_nms": 0.0,
                "force_direction_world": "+Y",
                "expected_roll_torque_direction_world": "-X",
            },
            "protocol": {
                "warmup_s": WARMUP_S,
                "impact_window_s": IMPACT_WINDOW_S,
                "recovery_s": RECOVERY_S,
            },
            "paired_nuisance_randomization": {
                "source": "numpy.default_rng(seed)",
                "command_vx_mps_range": [0.30, 0.42],
                "command_vy_mps_range": [-0.015, 0.015],
                "command_yaw_radps_range": [-0.04, 0.04],
            },
            "independent_scene_replication": False,
            "inference_warning": (
                "paired locomotion conditions form an engineering mechanics gate; "
                "independent scene and push-event clusters remain required for corpus inference"
            ),
        },
        "software": {
            "isaac_lab_version": str(isaaclab.__version__),
            "isaac_sim_version": (
                Path(sys.executable).resolve().parents[3] / "VERSION"
            ).read_text(encoding="utf-8").strip(),
            "physics_engine": "PhysX GPU",
            "physics_dt_s": float(cfg.physics_dt),
            "control_dt_s": float(backend.dt),
        },
        "input_sha256": {
            "policy": _sha256(REPO_ROOT / str(cfg.policy_path)),
            "backend": _sha256(REPO_ROOT / "kino_vla/sim/isaac_policy_backend.py"),
            "operator": _sha256(REPO_ROOT / "kino_vla/sim/operators/o6_push.py"),
            "gate_script": _sha256(Path(__file__).resolve()),
            "config": _sha256(REPO_ROOT / "configs/sim/go2_skeleton.yaml"),
        },
        "summary_by_dose": by_dose,
        "monotonicity": monotonicity,
        "reset_readback": reset_telemetry,
        "records": records,
    }
    path = output / "dose_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(
        "PASS: O6 finite-wrench dose sweep"
        if manifest["passed"]
        else "FAIL: O6 finite-wrench dose sweep"
    )
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if manifest["passed"] else 1)


if __name__ == "__main__":
    main()
