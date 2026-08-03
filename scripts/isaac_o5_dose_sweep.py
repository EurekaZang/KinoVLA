#!/usr/bin/env python3
"""Paired mass-dose PhysX characterization for realistic O5 rigid payloads."""

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
PAYLOAD_COM_M = np.array([0.0, 0.0, 0.10], dtype=np.float64)
PAYLOAD_SIZE_M = np.array([0.30, 0.20, 0.16], dtype=np.float64)
DOSES = (
    {"dose": "mild", "payload_mass_kg": 2.0},
    {"dose": "moderate", "payload_mass_kg": 4.0},
    {"dose": "severe", "payload_mass_kg": 6.0},
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _measure(backend, *, stance_warmup: int = 25, stance_samples: int = 45, drive_steps: int = 190):
    foot_rows: list[np.ndarray] = []
    for index in range(stance_warmup + stance_samples):
        obs = backend.step(np.zeros(3))
        if index >= stance_warmup:
            forces = backend._contact.data.net_forces_w[0, backend._contact_foot_ids]
            foot_rows.append(np.maximum(forces[:, 2].detach().cpu().numpy(), 0.0))
    command = np.array([0.55, 0.0, 0.0], dtype=np.float64)
    efforts: list[float] = []
    speeds: list[float] = []
    tilts: list[float] = []
    for index in range(drive_steps):
        obs = backend.step(command)
        if index >= 25:
            efforts.append(float(obs.effort_ratio))
            speeds.append(float(np.linalg.norm(obs.vel_body)))
            tilts.append(float(obs.tilt))
        if obs.fallen:
            break
    return obs, {
        "mean_foot_normal_force_n": np.mean(foot_rows, axis=0).tolist(),
        "total_foot_normal_force_n": float(np.mean(foot_rows, axis=0).sum()),
        "mean_effort_ratio": float(np.mean(efforts)) if efforts else 0.0,
        "mean_speed_mps": float(np.mean(speeds)) if speeds else 0.0,
        "mean_tilt_rad": float(np.mean(tilts)) if tilts else float(obs.tilt),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/kinofail_realistic/o5_dose_sweep")
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
    from kino_vla.sim.operators import Payload
    from kino_vla.utils.config import load_config

    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("--seeds must contain one or more unique comma-separated integers")
    cfg = load_config("sim/go2_skeleton.yaml")
    backend = IsaacPolicyBackend(cfg, np.array([0.0, 0.0]), 0.0)
    nominal_by_seed: dict[int, dict[str, float | bool]] = {}
    records: list[dict[str, object]] = []

    for seed in selected_seeds:
        backend.deep_reset(seed)
        obs, response = _measure(backend)
        physics = backend.payload_telemetry()
        actuator = backend.actuator_telemetry()
        base_inertia = np.asarray(physics["base_inertia_kg_m2"], dtype=np.float64)
        nominal_by_seed[seed] = {
            "final_x_m": float(obs.pos[0]),
            "mean_effort_ratio": float(response["mean_effort_ratio"]),
            "mean_speed_mps": float(response["mean_speed_mps"]),
            "mean_tilt_rad": float(response["mean_tilt_rad"]),
            "base_mass_kg": float(physics["base_mass_kg"]),
            "com_x_m": float(physics["base_com_pose"][0]),
            "com_y_m": float(physics["base_com_pose"][1]),
            "com_z_m": float(physics["base_com_pose"][2]),
            "inertia_trace_kg_m2": float(np.trace(base_inertia)),
            "actuator_utilization_p90": float(actuator["mean_utilization_p90"]),
            "actuator_binding_fraction": float(actuator["mean_binding_joint_fraction"]),
            "actuator_power_w": float(actuator["mean_total_abs_mechanical_power_w"]),
        }
        records.append(
            {
                "seed": seed,
                "dose": "nominal",
                "payload_mass_kg": 0.0,
                "base_mass_delta_kg": 0.0,
                "com_shift_m": 0.0,
                "inertia_trace_delta_kg_m2": 0.0,
                **response,
                "actuator_utilization_p90": float(actuator["mean_utilization_p90"]),
                "actuator_binding_fraction": float(
                    actuator["mean_binding_joint_fraction"]
                ),
                "actuator_power_w": float(actuator["mean_total_abs_mechanical_power_w"]),
                "actuator_energy_j": float(actuator["total_abs_mechanical_energy_j"]),
                "actuator_utilization_increase": 0.0,
                "actuator_binding_increase": 0.0,
                "actuator_power_increase_w": 0.0,
                "effort_increase": 0.0,
                "speed_loss_mps": 0.0,
                "tilt_increase_rad": 0.0,
                "progress_deficit_m": 0.0,
                "final_x_m": float(obs.pos[0]),
                "fallen": bool(obs.fallen),
                "visible_payload_count": 0,
            }
        )

    for dose in DOSES:
        for seed in selected_seeds:
            backend.deep_reset(seed)
            Payload(
                mass_kg=float(dose["payload_mass_kg"]),
                com_offset_m=PAYLOAD_COM_M,
                size_m=PAYLOAD_SIZE_M,
            ).on_reset(backend)
            backend.reset(seed)
            obs, response = _measure(backend)
            physics = backend.payload_telemetry()
            actuator = backend.actuator_telemetry()
            nominal = nominal_by_seed[seed]
            com = np.asarray(physics["base_com_pose"][:3], dtype=np.float64)
            nominal_com = np.array(
                [nominal["com_x_m"], nominal["com_y_m"], nominal["com_z_m"]],
                dtype=np.float64,
            )
            inertia = np.asarray(physics["base_inertia_kg_m2"], dtype=np.float64)
            records.append(
                {
                    "seed": seed,
                    **dose,
                    "base_mass_delta_kg": float(physics["base_mass_kg"])
                    - float(nominal["base_mass_kg"]),
                    "com_shift_m": float(np.linalg.norm(com - nominal_com)),
                    "inertia_trace_delta_kg_m2": float(np.trace(inertia))
                    - float(nominal["inertia_trace_kg_m2"]),
                    **response,
                    "actuator_utilization_p90": float(actuator["mean_utilization_p90"]),
                    "actuator_binding_fraction": float(
                        actuator["mean_binding_joint_fraction"]
                    ),
                    "actuator_power_w": float(
                        actuator["mean_total_abs_mechanical_power_w"]
                    ),
                    "actuator_energy_j": float(actuator["total_abs_mechanical_energy_j"]),
                    "actuator_utilization_increase": float(
                        actuator["mean_utilization_p90"]
                    )
                    - float(nominal["actuator_utilization_p90"]),
                    "actuator_binding_increase": float(
                        actuator["mean_binding_joint_fraction"]
                    )
                    - float(nominal["actuator_binding_fraction"]),
                    "actuator_power_increase_w": float(
                        actuator["mean_total_abs_mechanical_power_w"]
                    )
                    - float(nominal["actuator_power_w"]),
                    "effort_increase": float(response["mean_effort_ratio"])
                    - float(nominal["mean_effort_ratio"]),
                    "speed_loss_mps": float(nominal["mean_speed_mps"])
                    - float(response["mean_speed_mps"]),
                    "tilt_increase_rad": float(response["mean_tilt_rad"])
                    - float(nominal["mean_tilt_rad"]),
                    "progress_deficit_m": float(nominal["final_x_m"])
                    - float(obs.pos[0]),
                    "final_x_m": float(obs.pos[0]),
                    "fallen": bool(obs.fallen),
                    "visible_payload_count": len(physics["visuals"]),
                }
            )

    dose_order = ["nominal", "mild", "moderate", "severe"]
    metrics = {
        "base_mass_delta": ("base_mass_delta_kg", 1.0e-4),
        "com_shift": ("com_shift_m", 1.0e-5),
        "inertia_trace_delta": ("inertia_trace_delta_kg_m2", 1.0e-5),
        "foot_load": ("total_foot_normal_force_n", 5.0),
        "actuator_utilization_increase": ("actuator_utilization_increase", 0.02),
        "actuator_binding_increase": ("actuator_binding_increase", 0.01),
        "actuator_power_increase": ("actuator_power_increase_w", 2.0),
        "effort_increase": ("effort_increase", 0.02),
        "speed_loss": ("speed_loss_mps", 0.03),
        "progress_deficit": ("progress_deficit_m", 0.08),
    }
    monotonicity = {
        name: paired_dose_monotonicity(
            records, dose_order=dose_order, metric=metric, tolerance=tolerance
        )
        for name, (metric, tolerance) in metrics.items()
    }
    by_dose: dict[str, dict[str, object]] = {}
    for dose_name in dose_order:
        cells = [row for row in records if row["dose"] == dose_name]
        by_dose[dose_name] = {
            "n_seed_repeats": len(cells),
            "fall_count": sum(bool(row["fallen"]) for row in cells),
            "base_mass_delta_kg": bootstrap_mean_interval(
                [float(row["base_mass_delta_kg"]) for row in cells], seed=401
            ).to_dict(),
            "com_shift_m": bootstrap_mean_interval(
                [float(row["com_shift_m"]) for row in cells], seed=402
            ).to_dict(),
            "inertia_trace_delta_kg_m2": bootstrap_mean_interval(
                [float(row["inertia_trace_delta_kg_m2"]) for row in cells], seed=403
            ).to_dict(),
            "total_foot_normal_force_n": bootstrap_mean_interval(
                [float(row["total_foot_normal_force_n"]) for row in cells], seed=404
            ).to_dict(),
            "actuator_utilization_p90": bootstrap_mean_interval(
                [float(row["actuator_utilization_p90"]) for row in cells], seed=405
            ).to_dict(),
            "actuator_binding_fraction": bootstrap_mean_interval(
                [float(row["actuator_binding_fraction"]) for row in cells], seed=406
            ).to_dict(),
            "actuator_power_w": bootstrap_mean_interval(
                [float(row["actuator_power_w"]) for row in cells], seed=407
            ).to_dict(),
            "actuator_utilization_increase": bootstrap_mean_interval(
                [float(row["actuator_utilization_increase"]) for row in cells], seed=408
            ).to_dict(),
            "effort_increase": bootstrap_mean_interval(
                [float(row["effort_increase"]) for row in cells], seed=409
            ).to_dict(),
            "progress_deficit_m": bootstrap_mean_interval(
                [float(row["progress_deficit_m"]) for row in cells], seed=410
            ).to_dict(),
        }

    anomaly_rows = [row for row in records if row["dose"] != "nominal"]
    traversable_rows = [row for row in records if row["dose"] in {"mild", "moderate"}]
    checks = {
        "complete_paired_design": len(records) == len(selected_seeds) * len(dose_order),
        "all_anomalies_have_one_visible_payload": all(
            int(row["visible_payload_count"]) == 1 for row in anomaly_rows
        ),
        "mass_readback_matches_dose": all(
            abs(float(row["base_mass_delta_kg"]) - float(row["payload_mass_kg"]))
            <= 1.0e-4
            for row in anomaly_rows
        ),
        "mass_shift_strictly_ordered_each_seed": monotonicity["base_mass_delta"][
            "paired_seed_strict_fraction"
        ]
        == 1.0,
        "com_shift_strictly_ordered_each_seed": monotonicity["com_shift"][
            "paired_seed_strict_fraction"
        ]
        == 1.0,
        "inertia_shift_strictly_ordered_each_seed": monotonicity["inertia_trace_delta"][
            "paired_seed_strict_fraction"
        ]
        == 1.0,
        "foot_load_order_holds_most_seeds": monotonicity["foot_load"][
            "paired_seed_order_fraction"
        ]
        >= 0.8,
        "actuator_utilization_strictly_ordered_each_seed": monotonicity[
            "actuator_utilization_increase"
        ]["paired_seed_strict_fraction"]
        == 1.0,
        "severe_payload_binds_actuators_each_seed": all(
            float(row["actuator_binding_fraction"]) >= 0.02
            for row in records
            if row["dose"] == "severe"
        ),
        "mild_and_moderate_not_catastrophic": all(
            not bool(row["fallen"]) for row in traversable_rows
        ),
    }
    manifest = {
        "schema_version": "kinofail.o5-dose-sweep.v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "publication_status": "engineering_dose_gate_not_corpus_evidence",
        "checks": checks,
        "design": {
            "paired_by_seed": True,
            "seeds": list(selected_seeds),
            "dose_order": dose_order,
            "doses": list(DOSES),
            "fixed_payload_com_m": PAYLOAD_COM_M.tolist(),
            "fixed_payload_size_m": PAYLOAD_SIZE_M.tolist(),
            "single_factor_dose": "payload_mass",
            "actuator_telemetry": (
                "joint torque utilization, binding fraction, mechanical power and energy"
            ),
            "legacy_effort_ratio_semantics": (
                "retained for compatibility; it thresholds mean torque above 70% nominal cap"
            ),
            "independent_scene_replication": False,
            "inference_warning": (
                "seed repeats test physics reproducibility; scene-cluster inference waits for pilot"
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
            "operator": _sha256(REPO_ROOT / "kino_vla/sim/operators/o5_payload.py"),
            "mass_model": _sha256(REPO_ROOT / "kino_vla/sim/payload.py"),
            "actuator_model": _sha256(REPO_ROOT / "kino_vla/sim/actuator.py"),
            "gate_script": _sha256(Path(__file__).resolve()),
            "config": _sha256(REPO_ROOT / "configs/sim/go2_skeleton.yaml"),
        },
        "summary_by_dose": by_dose,
        "monotonicity": monotonicity,
        "task_consequence_note": (
            "joint-level actuator utilization/power, legacy effort, speed, tilt and progress "
            "are reported but do not define this physics gate; "
            "a recovery-consequence gate is required separately"
        ),
        "records": records,
    }
    path = output / "dose_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("PASS: O5 dose sweep" if manifest["passed"] else "FAIL: O5 dose sweep")
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if manifest["passed"] else 1)


if __name__ == "__main__":
    main()
