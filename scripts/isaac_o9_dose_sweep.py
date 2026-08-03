#!/usr/bin/env python3
"""Paired geometry-dose PhysX characterization for realistic O9 high-centering."""

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
RIDGE_WIDTH_M = 0.35
DOSES = (
    {"dose": "mild", "ridge_height_m": 0.350},
    {"dose": "moderate", "ridge_height_m": 0.355},
    {"dose": "severe", "ridge_height_m": 0.360},
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _drive(backend, monitor_region, *, steps: int = 280):
    command = np.array([0.55, 0.0, 0.0], dtype=np.float64)
    for _ in range(steps):
        obs = backend.step(command)
        if obs.pos[0] > monitor_region.cx + monitor_region.hx + 0.5 or obs.fallen:
            break
    return obs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/kinofail_realistic/o9_dose_sweep")
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
    from kino_vla.sim.operators import HighCentering
    from kino_vla.utils.config import load_config
    from kino_vla.utils.geometry import Rect

    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("--seeds must contain one or more unique comma-separated integers")
    cfg = load_config("sim/go2_skeleton.yaml")
    region = Rect(1.8, 0.0, 0.75, 0.9)
    backend = IsaacPolicyBackend(cfg, np.array([1.8, 0.0]), 0.0)
    backend._spawn_z = 0.43
    nominal_by_seed: dict[int, float] = {}
    records: list[dict[str, object]] = []

    for seed in selected_seeds:
        backend.deep_reset(seed)
        backend.reset(seed, preserve_settle_telemetry=True)
        for _ in range(12):
            backend.step(np.zeros(3))
        obs = _drive(backend, region)
        nominal_by_seed[seed] = float(obs.pos[0])
        records.append(
            {
                "seed": seed,
                "dose": "nominal",
                "ridge_height_m": 0.0,
                "ridge_width_m": 0.0,
                "min_measured_support": 1.0,
                "max_belly_contact_force_n": 0.0,
                "mean_belly_contact_force_n": 0.0,
                "max_consecutive_belly_contact_steps": 0,
                "belly_contact_duty_cycle": 0.0,
                "progress_deficit_m": 0.0,
                "final_x_m": float(obs.pos[0]),
                "fallen": bool(obs.fallen),
                "measurement_steps": 0,
            }
        )

    for dose in DOSES:
        for seed in selected_seeds:
            backend.deep_reset(seed)
            operator = HighCentering(
                region,
                residual_support=0.22,
                ridge_height_m=float(dose["ridge_height_m"]),
                ridge_width_m=RIDGE_WIDTH_M,
                geometry_kind="central_pallet_runner",
            )
            operator.on_reset(backend)
            backend.reset(seed, preserve_settle_telemetry=True)
            for _ in range(12):
                backend.step(np.zeros(3))
            obs = _drive(backend, region)
            telemetry = backend.high_centering_telemetry()["regions"][0]
            records.append(
                {
                    "seed": seed,
                    "dose": dose["dose"],
                    "ridge_height_m": float(dose["ridge_height_m"]),
                    "ridge_width_m": RIDGE_WIDTH_M,
                    "min_measured_support": float(telemetry["min_measured_support"]),
                    "max_belly_contact_force_n": float(
                        telemetry["max_belly_contact_force_n"]
                    ),
                    "mean_belly_contact_force_n": float(
                        telemetry["mean_belly_contact_force_n"]
                    ),
                    "max_consecutive_belly_contact_steps": int(
                        telemetry["max_consecutive_belly_contact_steps"]
                    ),
                    "belly_contact_duty_cycle": float(
                        telemetry["belly_contact_duty_cycle"]
                    ),
                    "progress_deficit_m": nominal_by_seed[seed] - float(obs.pos[0]),
                    "final_x_m": float(obs.pos[0]),
                    "fallen": bool(obs.fallen),
                    "measurement_steps": int(telemetry["measurement_steps"]),
                }
            )

    dose_order = ["nominal", "mild", "moderate", "severe"]
    contact_duration_monotonicity = paired_dose_monotonicity(
        records,
        dose_order=dose_order,
        metric="max_consecutive_belly_contact_steps",
        tolerance=2.0,
    )
    duty_monotonicity = paired_dose_monotonicity(
        records,
        dose_order=dose_order,
        metric="belly_contact_duty_cycle",
        tolerance=0.02,
    )
    progress_monotonicity = paired_dose_monotonicity(
        records,
        dose_order=dose_order,
        metric="progress_deficit_m",
        tolerance=0.08,
    )
    by_dose: dict[str, dict[str, object]] = {}
    for dose_name in dose_order:
        cells = [row for row in records if row["dose"] == dose_name]
        by_dose[dose_name] = {
            "n_seed_repeats": len(cells),
            "fall_count": sum(bool(row["fallen"]) for row in cells),
            "max_consecutive_belly_contact_steps": bootstrap_mean_interval(
                [float(row["max_consecutive_belly_contact_steps"]) for row in cells], seed=301
            ).to_dict(),
            "belly_contact_duty_cycle": bootstrap_mean_interval(
                [float(row["belly_contact_duty_cycle"]) for row in cells], seed=302
            ).to_dict(),
            "progress_deficit_m": bootstrap_mean_interval(
                [float(row["progress_deficit_m"]) for row in cells], seed=303
            ).to_dict(),
        }

    supercritical_rows = [row for row in records if row["dose"] in {"moderate", "severe"}]
    mild_rows = [row for row in records if row["dose"] == "mild"]
    severe_rows = [row for row in records if row["dose"] == "severe"]
    checks = {
        "complete_paired_design": len(records) == len(selected_seeds) * len(dose_order),
        "supercritical_doses_measure_real_belly_contact": all(
            float(row["max_belly_contact_force_n"]) > 2.0
            and int(row["measurement_steps"]) >= 3
            for row in supercritical_rows
        ),
        "contact_duration_order_holds_most_seeds": contact_duration_monotonicity[
            "paired_seed_order_fraction"
        ]
        >= 0.8,
        "contact_duty_order_holds_most_seeds": duty_monotonicity[
            "paired_seed_order_fraction"
        ]
        >= 0.8,
        "progress_deficit_order_holds_most_seeds": progress_monotonicity[
            "paired_seed_order_fraction"
        ]
        >= 0.8,
        "mild_is_subcritical_and_traversable": all(
            not bool(row["fallen"])
            and float(row["progress_deficit_m"]) <= 0.15
            for row in mild_rows
        ),
        "severe_is_sustained_beaching": all(
            int(row["max_consecutive_belly_contact_steps"]) >= 10
            and float(row["belly_contact_duty_cycle"]) >= 0.10
            and float(row["progress_deficit_m"]) >= 0.35
            for row in severe_rows
        ),
    }
    manifest = {
        "schema_version": "kinofail.o9-dose-sweep.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "publication_status": "engineering_dose_gate_not_corpus_evidence",
        "checks": checks,
        "design": {
            "paired_by_seed": True,
            "seeds": list(selected_seeds),
            "dose_order": dose_order,
            "doses": list(DOSES),
            "fixed_ridge_width_m": RIDGE_WIDTH_M,
            "single_factor_dose": "ridge_height_around_measured_chassis_clearance",
            "matched_initial_root_z_m": 0.43,
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
            "operator": _sha256(REPO_ROOT / "kino_vla/sim/operators/o9_high_centering.py"),
            "gate_script": _sha256(Path(__file__).resolve()),
            "config": _sha256(REPO_ROOT / "configs/sim/go2_skeleton.yaml"),
        },
        "summary_by_dose": by_dose,
        "monotonicity": {
            "contact_duration": contact_duration_monotonicity,
            "contact_duty_cycle": duty_monotonicity,
            "progress_deficit": progress_monotonicity,
        },
        "records": records,
    }
    path = output / "dose_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("PASS: O9 dose sweep" if manifest["passed"] else "FAIL: O9 dose sweep")
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if manifest["passed"] else 1)


if __name__ == "__main__":
    main()
