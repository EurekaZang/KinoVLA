#!/usr/bin/env python3
"""Paired raw-IMU/odometry fault gate for realistic O11 sensor degradation."""

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
DOSES = (
    {
        "dose": "mild",
        "tilt_bias_rad": 0.08,
        "random_walk_rad_sqrt_s": 0.004,
        "latency_s": 0.04,
    },
    {
        "dose": "moderate",
        "tilt_bias_rad": 0.18,
        "random_walk_rad_sqrt_s": 0.010,
        "latency_s": 0.10,
    },
    {
        "dose": "severe",
        "tilt_bias_rad": 0.35,
        "random_walk_rad_sqrt_s": 0.020,
        "latency_s": 0.20,
    },
)
PROTOCOL_S = 4.5
WARMUP_S = 1.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trajectory_hash(positions: list[np.ndarray]) -> str:
    array = np.round(np.asarray(positions, dtype=np.float64), decimals=6)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _paired_command(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array(
        [rng.uniform(0.32, 0.40), rng.uniform(-0.015, 0.015), rng.uniform(-0.03, 0.03)],
        dtype=np.float64,
    )


def _run_protocol(backend, operator, *, command: np.ndarray) -> dict[str, object]:
    truth = backend._make_obs()
    positions = []
    tilt_errors = []
    velocity_errors = []
    odom_ages = []
    raw_tilt_errors = []
    random_walk_states = []
    dropout_steps = 0
    pipeline_active = True
    imu_sources = set()
    odom_sources = set()
    while truth.t < PROTOCOL_S and not truth.fallen:
        truth = backend.step(command)
        measured = operator.transform_obs(truth)
        telemetry = operator.sensor_telemetry()
        positions.append(truth.pos.copy())
        if truth.t >= WARMUP_S:
            tilt_errors.append(abs(float(measured.tilt) - float(truth.tilt)))
            velocity_errors.append(float(np.linalg.norm(measured.vel_body - truth.vel_body)))
            odom_ages.append(float(telemetry["effective_odom_age_s"]))
            raw_tilt_errors.append(abs(float(telemetry["total_raw_tilt_error_rad"])))
            random_walk_states.append(float(telemetry["tilt_random_walk_state_rad"]))
            dropout_steps += int(bool(telemetry["odom_dropped_this_step"]))
            pipeline_active = pipeline_active and bool(telemetry["raw_pipeline_active"])
            imu_sources.add(str(telemetry["imu_source"]))
            odom_sources.add(str(telemetry["odometry_source"]))
    return {
        "command_vx_vy_yaw": command.tolist(),
        "trajectory_sha256_rounded_1e6": _trajectory_hash(positions),
        "final_truth_pos_xy_m": truth.pos.tolist(),
        "truth_fallen": bool(truth.fallen),
        "mean_abs_estimated_tilt_error_rad": float(np.mean(tilt_errors)),
        "p90_abs_estimated_tilt_error_rad": float(np.quantile(tilt_errors, 0.90)),
        "mean_abs_raw_tilt_error_rad": float(np.mean(raw_tilt_errors)),
        "mean_velocity_latency_error_mps": float(np.mean(velocity_errors)),
        "mean_effective_odom_age_s": float(np.mean(odom_ages)),
        "max_effective_odom_age_s": float(np.max(odom_ages)),
        "random_walk_state_std_rad": float(np.std(random_walk_states)),
        "dropout_steps": int(dropout_steps),
        "measurement_steps": len(tilt_errors),
        "raw_pipeline_active_every_step": pipeline_active,
        "imu_sources": sorted(imu_sources),
        "odometry_sources": sorted(odom_sources),
        "last_sensor_telemetry": operator.sensor_telemetry(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/kinofail_realistic/o11_sensor_fault_gate")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    import isaaclab

    from kino_vla.eval.dose_response import bootstrap_mean_interval, paired_dose_monotonicity
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.operators.o11_obs_bias import ObsBias
    from kino_vla.utils.config import load_config

    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("--seeds must contain one or more unique comma-separated integers")

    cfg = load_config("sim/go2_skeleton.yaml")
    backend = IsaacPolicyBackend(cfg, np.array([0.0, 0.0]), 0.0)
    records: list[dict[str, object]] = []
    for seed in selected_seeds:
        command = _paired_command(seed)
        backend.deep_reset(seed)
        nominal = ObsBias({}, seed=seed)
        nominal.on_reset(backend)
        records.append(
            {
                "seed": seed,
                "dose": "nominal",
                "tilt_bias_rad": 0.0,
                "random_walk_rad_sqrt_s": 0.0,
                "latency_s": 0.0,
                **_run_protocol(backend, nominal, command=command),
            }
        )
    for dose in DOSES:
        for seed in selected_seeds:
            command = _paired_command(seed)
            backend.deep_reset(seed)
            operator = ObsBias(
                {"tilt": float(dose["tilt_bias_rad"])},
                random_walk_per_sqrt_s={
                    "tilt": float(dose["random_walk_rad_sqrt_s"])
                },
                latency_s=float(dose["latency_s"]),
                seed=seed,
            )
            operator.on_reset(backend)
            records.append(
                {
                    "seed": seed,
                    **dose,
                    **_run_protocol(backend, operator, command=command),
                }
            )

    dose_order = ["nominal", "mild", "moderate", "severe"]
    monotonicity = {
        "raw_tilt_error": paired_dose_monotonicity(
            records,
            dose_order=dose_order,
            metric="mean_abs_raw_tilt_error_rad",
            tolerance=0.005,
        ),
        "estimated_tilt_error": paired_dose_monotonicity(
            records,
            dose_order=dose_order,
            metric="mean_abs_estimated_tilt_error_rad",
            tolerance=0.01,
        ),
        "odom_age": paired_dose_monotonicity(
            records,
            dose_order=dose_order,
            metric="mean_effective_odom_age_s",
            tolerance=backend.dt / 2,
        ),
        "velocity_latency_error": paired_dose_monotonicity(
            records,
            dose_order=dose_order,
            metric="mean_velocity_latency_error_mps",
            tolerance=0.015,
        ),
    }
    by_dose: dict[str, dict[str, object]] = {}
    for dose_index, dose_name in enumerate(dose_order):
        cells = [row for row in records if row["dose"] == dose_name]
        by_dose[dose_name] = {
            "n_paired_conditions": len(cells),
            "truth_fall_count": sum(bool(row["truth_fallen"]) for row in cells),
        }
        for metric_index, metric in enumerate(
            (
                "mean_abs_raw_tilt_error_rad",
                "mean_abs_estimated_tilt_error_rad",
                "mean_effective_odom_age_s",
                "mean_velocity_latency_error_mps",
                "random_walk_state_std_rad",
            )
        ):
            by_dose[dose_name][metric] = bootstrap_mean_interval(
                [float(row[metric]) for row in cells],
                seed=900 + 10 * dose_index + metric_index,
            ).to_dict()

    nominal_rows = [row for row in records if row["dose"] == "nominal"]
    anomaly_rows = [row for row in records if row["dose"] != "nominal"]
    checks = {
        "complete_paired_design": len(records) == len(selected_seeds) * len(dose_order),
        "paired_commands_match": all(
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
                for row in nominal_rows
            }
        )
        == len(selected_seeds),
        "raw_isaac_sensor_pipeline_active": all(
            bool(row["raw_pipeline_active_every_step"])
            and row["imu_sources"]
            == ["IsaacLab.Imu(projected_gravity_b,ang_vel_b,lin_acc_b)"]
            and row["odometry_sources"] == ["Isaac articulation root state"]
            for row in records
        ),
        "truth_trajectory_is_unchanged_by_sensor_fault": all(
            len(
                {
                    str(row["trajectory_sha256_rounded_1e6"])
                    for row in records
                    if int(row["seed"]) == seed
                }
            )
            == 1
            for seed in selected_seeds
        ),
        "all_truth_runs_remain_upright": all(not bool(row["truth_fallen"]) for row in records),
        "nominal_pipeline_is_identity": all(
            float(row["mean_abs_raw_tilt_error_rad"]) <= 1.0e-8
            and float(row["mean_effective_odom_age_s"]) <= 1.0e-8
            and int(row["dropout_steps"]) == 0
            for row in nominal_rows
        ),
        "requested_latency_is_realized_after_warmup": all(
            abs(float(row["mean_effective_odom_age_s"]) - float(row["latency_s"]))
            <= backend.dt / 2 + 1.0e-8
            for row in anomaly_rows
        ),
        "raw_tilt_error_strictly_orders_each_pair": monotonicity["raw_tilt_error"][
            "paired_seed_strict_fraction"
        ]
        == 1.0,
        "estimated_tilt_error_orders_most_pairs": monotonicity["estimated_tilt_error"][
            "paired_seed_order_fraction"
        ]
        >= 0.8,
        "odom_age_strictly_orders_each_pair": monotonicity["odom_age"][
            "paired_seed_strict_fraction"
        ]
        == 1.0,
        "random_walk_is_nonzero_for_anomalies": all(
            float(row["random_walk_state_std_rad"]) > 0.0 for row in anomaly_rows
        ),
        "no_unrequested_dropout": all(int(row["dropout_steps"]) == 0 for row in records),
    }
    manifest = {
        "schema_version": "kinofail.o11-raw-sensor-fault-gate.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "publication_status": "engineering_sensor_pipeline_gate_not_corpus_evidence",
        "checks": checks,
        "design": {
            "paired_by_condition": True,
            "seeds": list(selected_seeds),
            "dose_order": dose_order,
            "doses": list(DOSES),
            "pipeline_order": [
                "IsaacLab base-link IMU and articulation odometry sample",
                "raw projected-gravity bias plus seeded random walk",
                "timestamped odometry latency/dropout buffer",
                "shared proprioceptive state estimator",
                "monitor/planner Obs",
            ],
            "low_level_policy_uses_fault_free_proprioception": True,
            "reason": (
                "O11 evaluates diagnosis and recovery under sensor degradation without "
                "confounding the pretrained locomotion controller"
            ),
            "independent_scene_replication": False,
            "inference_warning": (
                "one flat scene and simulated ideal IMU source form an engineering gate; "
                "real stationary/motion logs and hardware clock/dropout calibration remain required"
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
            "operator": _sha256(REPO_ROOT / "kino_vla/sim/operators/o11_obs_bias.py"),
            "sensor_pipeline": _sha256(REPO_ROOT / "kino_vla/sim/proprio_pipeline.py"),
            "gate_script": _sha256(Path(__file__).resolve()),
            "config": _sha256(REPO_ROOT / "configs/sim/go2_skeleton.yaml"),
        },
        "summary_by_dose": by_dose,
        "monotonicity": monotonicity,
        "records": records,
    }
    path = output / "gate_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(
        "PASS: O11 raw sensor fault gate"
        if manifest["passed"]
        else "FAIL: O11 raw sensor fault gate"
    )
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if manifest["passed"] else 1)


if __name__ == "__main__":
    main()
