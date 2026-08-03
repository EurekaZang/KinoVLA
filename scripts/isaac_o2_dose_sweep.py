#!/usr/bin/env python3
"""Paired multi-dose, multi-seed PhysX characterization for realistic O2."""

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
        "sink_depth_m": 0.035,
        "shear_retention": 0.72,
        "vertical_stiffness_n_per_m": 720.0,
    },
    {
        "dose": "moderate",
        "sink_depth_m": 0.050,
        "shear_retention": 0.60,
        "vertical_stiffness_n_per_m": 560.0,
    },
    {
        "dose": "severe",
        "sink_depth_m": 0.110,
        "shear_retention": 0.32,
        "vertical_stiffness_n_per_m": 240.0,
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _drive(backend, region, *, steps: int = 320):
    command = np.array([0.55, 0.0, 0.0], dtype=np.float64)
    rows = []
    for _ in range(steps):
        obs = backend.step(command)
        if region.contains(obs.pos):
            rows.append(
                {
                    "t_s": float(obs.t),
                    "base_height_m": float(obs.base_height),
                    "speed_mps": float(np.linalg.norm(obs.vel_body)),
                }
            )
        if obs.pos[0] > region.cx + region.hx + 0.4 or obs.fallen:
            break
    return obs, rows


def _median(rows: list[dict[str, float]], key: str) -> float:
    if not rows:
        raise RuntimeError(f"empty O2 measurement window for {key}")
    return float(np.median([row[key] for row in rows]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/kinofail_realistic/o2_dose_sweep")
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
    from kino_vla.sim.operators import ComplianceField
    from kino_vla.utils.config import load_config
    from kino_vla.utils.geometry import Rect

    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("--seeds must contain one or more unique comma-separated integers")
    cfg = load_config("sim/go2_skeleton.yaml")
    region = Rect(1.9, 0.0, 0.9, 0.8)
    backend = IsaacPolicyBackend(cfg, np.array([0.0, 0.0]), 0.0)
    nominal_by_seed: dict[int, dict[str, float | bool]] = {}
    records: list[dict[str, object]] = []

    for seed in selected_seeds:
        backend.deep_reset(seed)
        nominal_obs, nominal_rows = _drive(backend, region)
        nominal_by_seed[seed] = {
            "median_base_height_m": _median(nominal_rows, "base_height_m"),
            "median_speed_mps": _median(nominal_rows, "speed_mps"),
            "final_x_m": float(nominal_obs.pos[0]),
            "fallen": bool(nominal_obs.fallen),
        }
        records.append(
            {
                "seed": seed,
                "dose": "nominal",
                "sink_depth_m": 0.0,
                "shear_retention": 1.0,
                "vertical_stiffness_n_per_m": None,
                "max_foot_sinkage_m": 0.0,
                "total_shear_work_j": 0.0,
                "base_height_loss_m": 0.0,
                "speed_loss_mps": 0.0,
                "progress_deficit_m": 0.0,
                "fallen": bool(nominal_obs.fallen),
                "loaded_feet": 4,
                "default_ground_replaced": False,
                "legacy_trunk_drag_count": 0,
            }
        )

    for dose in DOSES:
        for seed in selected_seeds:
            backend.deep_reset(seed)
            operator = ComplianceField(
                region=region,
                k_c=float(dose["vertical_stiffness_n_per_m"]),
                c_c=float(dose["shear_retention"]),
                d_sink=float(dose["sink_depth_m"]),
                realistic_foot_model=True,
            )
            operator.on_reset(backend)
            backend.reset(seed)
            obs, rows = _drive(backend, region)
            telemetry = backend.foot_compliance_telemetry()
            nominal = nominal_by_seed[seed]
            max_sinkage = max(float(foot["max_sinkage_m"]) for foot in telemetry["feet"])
            total_shear_work = sum(float(foot["shear_work_j"]) for foot in telemetry["feet"])
            loaded_feet = sum(int(foot["contact_steps"]) > 0 for foot in telemetry["feet"])
            records.append(
                {
                    "seed": seed,
                    **dose,
                    "max_foot_sinkage_m": max_sinkage,
                    "total_shear_work_j": total_shear_work,
                    "max_applied_foot_force_n": max(
                        float(foot["max_applied_force_n"]) for foot in telemetry["feet"]
                    ),
                    "base_height_loss_m": float(nominal["median_base_height_m"])
                    - _median(rows, "base_height_m"),
                    "speed_loss_mps": float(nominal["median_speed_mps"])
                    - _median(rows, "speed_mps"),
                    "progress_deficit_m": float(nominal["final_x_m"])
                    - float(obs.pos[0]),
                    "final_x_m": float(obs.pos[0]),
                    "fallen": bool(obs.fallen),
                    "loaded_feet": loaded_feet,
                    "default_ground_replaced": bool(
                        backend._disabled_default_ground_colliders
                    ),
                    "legacy_trunk_drag_count": len(backend._resistance),
                }
            )

    dose_order = ["nominal", "mild", "moderate", "severe"]
    sink_monotonicity = paired_dose_monotonicity(
        records,
        dose_order=dose_order,
        metric="max_foot_sinkage_m",
        tolerance=0.002,
    )
    height_monotonicity = paired_dose_monotonicity(
        records,
        dose_order=dose_order,
        metric="base_height_loss_m",
        tolerance=0.006,
    )
    speed_monotonicity = paired_dose_monotonicity(
        records,
        dose_order=dose_order,
        metric="speed_loss_mps",
        tolerance=0.025,
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
            "max_foot_sinkage_m": bootstrap_mean_interval(
                [float(row["max_foot_sinkage_m"]) for row in cells], seed=101
            ).to_dict(),
            "base_height_loss_m": bootstrap_mean_interval(
                [float(row["base_height_loss_m"]) for row in cells], seed=102
            ).to_dict(),
            "speed_loss_mps": bootstrap_mean_interval(
                [float(row["speed_loss_mps"]) for row in cells], seed=103
            ).to_dict(),
            "progress_deficit_m": bootstrap_mean_interval(
                [float(row["progress_deficit_m"]) for row in cells], seed=104
            ).to_dict(),
        }

    anomaly_rows = [row for row in records if row["dose"] != "nominal"]
    severe_rows = [row for row in records if row["dose"] == "severe"]
    traversable_rows = [row for row in records if row["dose"] in {"mild", "moderate"}]
    checks = {
        "complete_paired_design": len(records) == len(selected_seeds) * len(dose_order),
        "all_anomalies_use_local_foot_model": all(
            bool(row["default_ground_replaced"])
            and int(row["legacy_trunk_drag_count"]) == 0
            and int(row["loaded_feet"]) >= 2
            for row in anomaly_rows
        ),
        "sinkage_median_strictly_increases": bool(
            sink_monotonicity["median_strictly_ordered"]
        ),
        "sinkage_order_holds_each_seed": sink_monotonicity[
            "paired_seed_order_fraction"
        ]
        == 1.0,
        "height_loss_order_holds_each_seed": height_monotonicity[
            "paired_seed_order_fraction"
        ]
        == 1.0,
        "progress_deficit_order_holds_most_seeds": progress_monotonicity[
            "paired_seed_order_fraction"
        ]
        >= 0.8,
        "mild_and_moderate_remain_traversable": all(
            not bool(row["fallen"]) and float(row["final_x_m"]) >= region.cx + region.hx
            for row in traversable_rows
        ),
        "severe_has_large_sinkage": min(
            float(row["max_foot_sinkage_m"]) for row in severe_rows
        )
        >= 0.07,
    }
    manifest = {
        "schema_version": "kinofail.o2-dose-sweep.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "publication_status": "engineering_dose_gate_not_corpus_evidence",
        "checks": checks,
        "design": {
            "paired_by_seed": True,
            "seeds": list(selected_seeds),
            "dose_order": dose_order,
            "doses": list(DOSES),
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
            "operator": _sha256(REPO_ROOT / "kino_vla/sim/operators/o2_compliance.py"),
            "terramechanics": _sha256(REPO_ROOT / "kino_vla/sim/terramechanics.py"),
            "gate_script": _sha256(Path(__file__).resolve()),
            "config": _sha256(REPO_ROOT / "configs/sim/go2_skeleton.yaml"),
        },
        "summary_by_dose": by_dose,
        "monotonicity": {
            "sinkage": sink_monotonicity,
            "base_height_loss": height_monotonicity,
            "speed_loss": speed_monotonicity,
            "progress_deficit": progress_monotonicity,
        },
        "metric_censoring_note": (
            "speed loss is descriptive only because a fall censors the within-patch speed window; "
            "ordered consequence uses completion/progress deficit instead"
        ),
        "records": records,
    }
    manifest_path = output / "dose_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))
    print("PASS: O2 dose sweep" if manifest["passed"] else "FAIL: O2 dose sweep")
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if manifest["passed"] else 1)


if __name__ == "__main__":
    main()
