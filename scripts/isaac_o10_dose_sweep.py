#!/usr/bin/env python3
"""Paired actuator-cap dose characterization for realistic O10 derating."""

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
    {"dose": "mild", "effort_scale": 0.75},
    {"dose": "moderate", "effort_scale": 0.50},
    {"dose": "severe", "effort_scale": 0.25},
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paired_command(seed: int) -> np.ndarray:
    """Deterministic nuisance condition reused by every dose for one paired seed."""
    rng = np.random.default_rng(seed)
    return np.array(
        [
            rng.uniform(0.48, 0.62),
            rng.uniform(-0.04, 0.04),
            rng.uniform(-0.08, 0.08),
        ],
        dtype=np.float64,
    )


def _drive(backend, command: np.ndarray, *, steps: int = 240):
    speeds: list[float] = []
    tilts: list[float] = []
    for index in range(steps):
        obs = backend.step(command)
        if index >= 30:
            speeds.append(float(np.linalg.norm(obs.vel_body)))
            tilts.append(float(obs.tilt))
        if obs.fallen:
            break
    return obs, {
        "mean_speed_mps": float(np.mean(speeds)) if speeds else 0.0,
        "mean_tilt_rad": float(np.mean(tilts)) if tilts else float(obs.tilt),
    }


def _cap_readback(telemetry: dict[str, object]) -> tuple[float, float]:
    rows = telemetry["actuator_limit_readback"]
    effort = [float(row["effort_limit"]["max"]) for row in rows.values()]
    saturation = [float(row["saturation_effort"]["max"]) for row in rows.values()]
    return max(effort), max(saturation)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/kinofail_realistic/o10_dose_sweep")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
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
    cfg = load_config("sim/go2_skeleton.yaml")
    backend = IsaacPolicyBackend(cfg, np.array([0.0, 0.0]), 0.0)
    records: list[dict[str, object]] = []
    nominal_by_seed: dict[int, dict[str, float]] = {}

    for seed in selected_seeds:
        command = _paired_command(seed)
        backend.deep_reset(seed)
        obs, response = _drive(backend, command)
        telemetry = backend.actuator_telemetry()
        effort_cap, saturation_cap = _cap_readback(telemetry)
        nominal_by_seed[seed] = {
            "final_x_m": float(obs.pos[0]),
            "mean_speed_mps": float(response["mean_speed_mps"]),
        }
        records.append(
            {
                "seed": seed,
                "dose": "nominal",
                "effort_scale": 1.0,
                "derating_fraction": 0.0,
                "command_vx_vy_yaw": command.tolist(),
                "effort_limit_readback_nm": effort_cap,
                "saturation_effort_readback_nm": saturation_cap,
                "mean_utilization_p90": float(telemetry["mean_utilization_p90"]),
                "mean_binding_joint_fraction": float(
                    telemetry["mean_binding_joint_fraction"]
                ),
                "peak_joint_utilization": max(
                    float(value) for value in telemetry["peak_torque_utilization_by_joint"]
                ),
                "mean_total_abs_mechanical_power_w": float(
                    telemetry["mean_total_abs_mechanical_power_w"]
                ),
                "total_abs_mechanical_energy_j": float(
                    telemetry["total_abs_mechanical_energy_j"]
                ),
                **response,
                "speed_loss_mps": 0.0,
                "progress_deficit_m": 0.0,
                "final_x_m": float(obs.pos[0]),
                "fallen": bool(obs.fallen),
                "payload_kg": float(backend.privileged_physics()["payload_kg"]),
            }
        )

    for dose in DOSES:
        for seed in selected_seeds:
            command = _paired_command(seed)
            backend.deep_reset(seed)
            backend.set_effort_scale(float(dose["effort_scale"]))
            backend.reset(seed)
            obs, response = _drive(backend, command)
            telemetry = backend.actuator_telemetry()
            effort_cap, saturation_cap = _cap_readback(telemetry)
            nominal = nominal_by_seed[seed]
            records.append(
                {
                    "seed": seed,
                    **dose,
                    "derating_fraction": 1.0 - float(dose["effort_scale"]),
                    "command_vx_vy_yaw": command.tolist(),
                    "effort_limit_readback_nm": effort_cap,
                    "saturation_effort_readback_nm": saturation_cap,
                    "mean_utilization_p90": float(telemetry["mean_utilization_p90"]),
                    "mean_binding_joint_fraction": float(
                        telemetry["mean_binding_joint_fraction"]
                    ),
                    "peak_joint_utilization": max(
                        float(value) for value in telemetry["peak_torque_utilization_by_joint"]
                    ),
                    "mean_total_abs_mechanical_power_w": float(
                        telemetry["mean_total_abs_mechanical_power_w"]
                    ),
                    "total_abs_mechanical_energy_j": float(
                        telemetry["total_abs_mechanical_energy_j"]
                    ),
                    **response,
                    "speed_loss_mps": float(nominal["mean_speed_mps"])
                    - float(response["mean_speed_mps"]),
                    "progress_deficit_m": float(nominal["final_x_m"])
                    - float(obs.pos[0]),
                    "final_x_m": float(obs.pos[0]),
                    "fallen": bool(obs.fallen),
                    "payload_kg": float(backend.privileged_physics()["payload_kg"]),
                }
            )

    dose_order = ["nominal", "mild", "moderate", "severe"]
    monotonicity = {
        "utilization_p90": paired_dose_monotonicity(
            records,
            dose_order=dose_order,
            metric="mean_utilization_p90",
            tolerance=0.02,
        ),
        "binding_fraction": paired_dose_monotonicity(
            records,
            dose_order=dose_order,
            metric="mean_binding_joint_fraction",
            tolerance=0.01,
        ),
        "speed_loss": paired_dose_monotonicity(
            records, dose_order=dose_order, metric="speed_loss_mps", tolerance=0.03
        ),
        "progress_deficit": paired_dose_monotonicity(
            records,
            dose_order=dose_order,
            metric="progress_deficit_m",
            tolerance=0.08,
        ),
    }
    by_dose: dict[str, dict[str, object]] = {}
    for dose_name in dose_order:
        cells = [row for row in records if row["dose"] == dose_name]
        by_dose[dose_name] = {
            "n_seed_repeats": len(cells),
            "fall_count": sum(bool(row["fallen"]) for row in cells),
            "mean_utilization_p90": bootstrap_mean_interval(
                [float(row["mean_utilization_p90"]) for row in cells], seed=501
            ).to_dict(),
            "mean_binding_joint_fraction": bootstrap_mean_interval(
                [float(row["mean_binding_joint_fraction"]) for row in cells], seed=502
            ).to_dict(),
            "mean_total_abs_mechanical_power_w": bootstrap_mean_interval(
                [float(row["mean_total_abs_mechanical_power_w"]) for row in cells], seed=503
            ).to_dict(),
            "speed_loss_mps": bootstrap_mean_interval(
                [float(row["speed_loss_mps"]) for row in cells], seed=504
            ).to_dict(),
            "progress_deficit_m": bootstrap_mean_interval(
                [float(row["progress_deficit_m"]) for row in cells], seed=505
            ).to_dict(),
        }

    anomaly_rows = [row for row in records if row["dose"] != "nominal"]
    mild_rows = [row for row in records if row["dose"] == "mild"]
    severe_rows = [row for row in records if row["dose"] == "severe"]
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
        "actuator_cap_readback_matches_dose": all(
            abs(
                float(row["effort_limit_readback_nm"])
                - float(cfg.effort_limit_nm) * float(row["effort_scale"])
            )
            <= 1.0e-4
            and abs(
                float(row["saturation_effort_readback_nm"])
                - float(cfg.effort_limit_nm) * float(row["effort_scale"])
            )
            <= 1.0e-4
            for row in records
        ),
        "no_payload_confound": all(float(row["payload_kg"]) == 0.0 for row in anomaly_rows),
        "utilization_order_holds_most_seeds": monotonicity["utilization_p90"][
            "paired_seed_order_fraction"
        ]
        >= 0.8,
        "binding_order_holds_most_seeds": monotonicity["binding_fraction"][
            "paired_seed_order_fraction"
        ]
        >= 0.8,
        "mild_is_not_catastrophic": all(not bool(row["fallen"]) for row in mild_rows),
        "severe_has_actuator_or_motion_consequence": all(
            float(row["mean_binding_joint_fraction"]) >= 0.02
            or float(row["progress_deficit_m"]) >= 0.35
            or bool(row["fallen"])
            for row in severe_rows
        ),
    }
    manifest = {
        "schema_version": "kinofail.o10-dose-sweep.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "publication_status": "engineering_dose_gate_not_corpus_evidence",
        "checks": checks,
        "design": {
            "paired_by_seed": True,
            "seeds": list(selected_seeds),
            "paired_nuisance_randomization": {
                "source": "numpy.default_rng(seed)",
                "command_vx_mps_range": [0.48, 0.62],
                "command_vy_mps_range": [-0.04, 0.04],
                "command_yaw_radps_range": [-0.08, 0.08],
                "note": (
                    "Each seed defines one distinct command-demand condition; the exact same "
                    "condition is reused across all doses for paired single-factor contrasts."
                ),
            },
            "dose_order": dose_order,
            "doses": list(DOSES),
            "single_factor_dose": "global_actuator_effort_and_saturation_cap",
            "existing_motor_velocity_limit_radps": 30.0,
            "thermal_state_modeled": False,
            "inference_warning": (
                "endpoint cap states test actuator observability; a thermal ramp gate and "
                "independent scene-cluster inference remain required"
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
            "operator": _sha256(REPO_ROOT / "kino_vla/sim/operators/o10_effort_decay.py"),
            "actuator_model": _sha256(REPO_ROOT / "kino_vla/sim/actuator.py"),
            "gate_script": _sha256(Path(__file__).resolve()),
            "config": _sha256(REPO_ROOT / "configs/sim/go2_skeleton.yaml"),
        },
        "summary_by_dose": by_dose,
        "monotonicity": monotonicity,
        "records": records,
    }
    path = output / "dose_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("PASS: O10 dose sweep" if manifest["passed"] else "FAIL: O10 dose sweep")
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if manifest["passed"] else 1)


if __name__ == "__main__":
    main()
