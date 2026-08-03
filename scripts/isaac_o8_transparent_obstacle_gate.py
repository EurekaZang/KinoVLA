#!/usr/bin/env python3
"""Paired visual-contract and physics gate for realistic transparent O8 obstacles."""

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
    {"dose": "mild", "height_m": 0.38, "optical_transmission": 0.88},
    {"dose": "moderate", "height_m": 0.38, "optical_transmission": 0.94},
    {"dose": "severe", "height_m": 0.38, "optical_transmission": 0.98},
)
WALL_CENTER_X_M = 1.35
WALL_HALF_THICKNESS_M = 0.018
WALL_HALF_WIDTH_M = 1.0
PROTOCOL_S = 7.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paired_command(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array(
        [rng.uniform(0.32, 0.38), rng.uniform(-0.01, 0.01), rng.uniform(-0.025, 0.025)],
        dtype=np.float64,
    )


def _contact_force_peaks(backend) -> tuple[float, float]:
    forces = backend._contact.data.net_forces_w[0, backend._contact_all_ids]
    norms = backend._torch.linalg.norm(forces, dim=-1).detach().cpu().numpy()
    foot = [
        float(norms[index])
        for index, name in enumerate(backend._contact_all_names)
        if "foot" in name
    ]
    nonfoot = [
        float(norms[index])
        for index, name in enumerate(backend._contact_all_names)
        if "foot" not in name
    ]
    return max(foot, default=0.0), max(nonfoot, default=0.0)


def _run_protocol(backend, *, command: np.ndarray) -> dict[str, object]:
    obs = backend._make_obs()
    initial_x = float(obs.pos[0])
    left_edge = WALL_CENTER_X_M - WALL_HALF_THICKNESS_M
    peak_tracking_error = 0.0
    peak_foot_force = 0.0
    peak_nonfoot_force = 0.0
    peak_tilt = float(obs.tilt)
    stall_s = 0.0
    barrier_window_steps = 0
    while obs.t < PROTOCOL_S and not obs.fallen:
        obs = backend.step(command)
        foot_force, nonfoot_force = _contact_force_peaks(backend)
        if float(obs.pos[0]) >= left_edge - 0.35:
            barrier_window_steps += 1
            peak_tracking_error = max(
                peak_tracking_error,
                max(0.0, float(command[0]) - float(obs.vel_body[0])),
            )
            peak_foot_force = max(peak_foot_force, foot_force)
            peak_nonfoot_force = max(peak_nonfoot_force, nonfoot_force)
            if abs(float(obs.vel_body[0])) < 0.12:
                stall_s += backend.dt
        peak_tilt = max(peak_tilt, float(obs.tilt))
    return {
        "command_vx_vy_yaw": command.tolist(),
        "initial_x_m": initial_x,
        "final_x_m": float(obs.pos[0]),
        "max_forward_progress_m": float(obs.pos[0]) - initial_x,
        "peak_tracking_error_mps": peak_tracking_error,
        "peak_foot_contact_force_n": peak_foot_force,
        "peak_nonfoot_contact_force_n": peak_nonfoot_force,
        "peak_tilt_rad": peak_tilt,
        "stall_duration_s": stall_s,
        "barrier_window_steps": barrier_window_steps,
        "fallen": bool(obs.fallen),
        "episode_duration_s": float(obs.t),
        "obstacle_readback": backend.blocking_telemetry(),
    }


def _obstacle_row(record: dict[str, object]) -> dict[str, object]:
    return record["obstacle_readback"]["obstacles"][0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/kinofail_realistic/o8_transparent_gate")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    import isaaclab

    from kino_vla.eval.dose_response import bootstrap_mean_interval, paired_dose_monotonicity
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.operators.o8_invisible_collider import InvisibleCollider
    from kino_vla.utils.config import load_config
    from kino_vla.utils.geometry import Rect

    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("--seeds must contain one or more unique comma-separated integers")

    cfg = load_config("sim/go2_skeleton.yaml")
    backend = IsaacPolicyBackend(cfg, np.array([0.0, 0.0]), 0.0)
    region = Rect(
        cx=WALL_CENTER_X_M,
        cy=0.0,
        hx=WALL_HALF_THICKNESS_M,
        hy=WALL_HALF_WIDTH_M,
    )
    records: list[dict[str, object]] = []
    effect_records: list[dict[str, object]] = []
    for dose in DOSES:
        for seed in selected_seeds:
            command = _paired_command(seed)
            pair: dict[str, dict[str, object]] = {}
            for condition, collision_enabled in (("nominal", False), ("anomaly", True)):
                backend.deep_reset(seed)
                operator = InvisibleCollider(
                    region,
                    height_m=float(dose["height_m"]),
                    collision_enabled=collision_enabled,
                    geometry_kind="transparent_acrylic",
                    optical_transmission=float(dose["optical_transmission"]),
                )
                operator.on_reset(backend)
                backend.reset(seed)
                result = _run_protocol(backend, command=command)
                row = {
                    "seed": seed,
                    **dose,
                    "condition": condition,
                    "collision_enabled": collision_enabled,
                    **result,
                }
                pair[condition] = row
                records.append(row)
            nominal = pair["nominal"]
            anomaly = pair["anomaly"]
            effect_records.append(
                {
                    "seed": seed,
                    "dose": dose["dose"],
                    "height_m": dose["height_m"],
                    "frosting_roughness": float(
                        _obstacle_row(anomaly)["visual_material_parameters"][0][
                            "frosting_roughness"
                        ]
                    ),
                    "progress_deficit_m": float(nominal["final_x_m"])
                    - float(anomaly["final_x_m"]),
                    "tracking_error_increase_mps": float(anomaly["peak_tracking_error_mps"])
                    - float(nominal["peak_tracking_error_mps"]),
                    "stall_increase_s": float(anomaly["stall_duration_s"])
                    - float(nominal["stall_duration_s"]),
                    "tilt_increase_rad": float(anomaly["peak_tilt_rad"])
                    - float(nominal["peak_tilt_rad"]),
                    "foot_force_increase_n": float(anomaly["peak_foot_contact_force_n"])
                    - float(nominal["peak_foot_contact_force_n"]),
                    "nonfoot_force_increase_n": float(anomaly["peak_nonfoot_contact_force_n"])
                    - float(nominal["peak_nonfoot_contact_force_n"]),
                    "anomaly_fallen": bool(anomaly["fallen"]),
                    "nominal_fallen": bool(nominal["fallen"]),
                }
            )

    dose_order = [str(row["dose"]) for row in DOSES]
    monotonicity = {
        "optical_ambiguity": paired_dose_monotonicity(
            effect_records,
            dose_order=dose_order,
            metric="frosting_roughness",
            direction="decreasing",
        ),
        "progress_deficit": paired_dose_monotonicity(
            effect_records,
            dose_order=dose_order,
            metric="progress_deficit_m",
            tolerance=0.12,
        ),
        "stall_increase": paired_dose_monotonicity(
            effect_records,
            dose_order=dose_order,
            metric="stall_increase_s",
            tolerance=0.20,
        ),
    }
    by_dose: dict[str, dict[str, object]] = {}
    for dose_index, dose_name in enumerate(dose_order):
        cells = [row for row in effect_records if row["dose"] == dose_name]
        by_dose[dose_name] = {
            "n_paired_conditions": len(cells),
            "anomaly_fall_count": sum(bool(row["anomaly_fallen"]) for row in cells),
        }
        for metric_index, metric in enumerate(
            (
                "progress_deficit_m",
                "tracking_error_increase_mps",
                "stall_increase_s",
                "tilt_increase_rad",
                "foot_force_increase_n",
                "nonfoot_force_increase_n",
            )
        ):
            by_dose[dose_name][metric] = bootstrap_mean_interval(
                [float(row[metric]) for row in cells],
                seed=800 + 10 * dose_index + metric_index,
            ).to_dict()

    nominal_rows = [row for row in records if row["condition"] == "nominal"]
    anomaly_rows = [row for row in records if row["condition"] == "anomaly"]
    mild_effects = [row for row in effect_records if row["dose"] == "mild"]
    pair_keys = {(int(row["seed"]), str(row["dose"])) for row in records}
    visual_contract = []
    for seed, dose_name in sorted(pair_keys):
        pair = [
            row
            for row in records
            if int(row["seed"]) == seed and str(row["dose"]) == dose_name
        ]
        nominal = next(row for row in pair if row["condition"] == "nominal")
        anomaly = next(row for row in pair if row["condition"] == "anomaly")
        nominal_obstacle = _obstacle_row(nominal)
        anomaly_obstacle = _obstacle_row(anomaly)
        visual_contract.append(
            {
                "seed": seed,
                "dose": dose_name,
                "same_prim_path": nominal_obstacle["prim_path"]
                == anomaly_obstacle["prim_path"],
                "same_geometry": nominal_obstacle["size_xyz_m"]
                == anomaly_obstacle["size_xyz_m"],
                "same_visual_material": nominal_obstacle["visual_material_paths"]
                == anomaly_obstacle["visual_material_paths"],
                "same_visual_parameters": nominal_obstacle["visual_material_parameters"]
                == anomaly_obstacle["visual_material_parameters"],
                "same_optical_metadata": nominal_obstacle["optical_transmission_metadata"]
                == anomaly_obstacle["optical_transmission_metadata"],
            }
        )

    checks = {
        "complete_paired_design": len(records) == len(selected_seeds) * len(DOSES) * 2,
        "paired_commands_match": all(
            len(
                {
                    tuple(float(value) for value in row["command_vx_vy_yaw"])
                    for row in records
                    if int(row["seed"]) == seed and row["dose"] == dose
                }
            )
            == 1
            for seed, dose in pair_keys
        ),
        "nuisance_conditions_are_distinct": len(
            {
                tuple(float(value) for value in row["command_vx_vy_yaw"])
                for row in records
                if row["condition"] == "nominal" and row["dose"] == "mild"
            }
        )
        == len(selected_seeds),
        "all_obstacle_prims_and_glass_materials_exist": all(
            bool(_obstacle_row(row)["prim_valid"])
            and _obstacle_row(row)["geometry_kind"] == "transparent_acrylic"
            and any(
                "material" in str(path).lower()
                for path in _obstacle_row(row)["visual_material_paths"]
            )
            for row in records
        ),
        "counterfactual_visual_contract_matches": all(
            row["same_prim_path"]
            and row["same_geometry"]
            and row["same_visual_material"]
            and row["same_visual_parameters"]
            and row["same_optical_metadata"]
            for row in visual_contract
        ),
        "nominal_colliders_are_disabled": all(
            _obstacle_row(row)["colliders"]
            and all(
                not bool(collider["enabled"])
                for collider in _obstacle_row(row)["colliders"]
            )
            for row in nominal_rows
        ),
        "anomaly_colliders_are_enabled": all(
            _obstacle_row(row)["colliders"]
            and all(
                bool(collider["enabled"])
                for collider in _obstacle_row(row)["colliders"]
            )
            for row in anomaly_rows
        ),
        "all_nominal_counterfactuals_traverse": all(
            not bool(row["fallen"])
            and float(row["final_x_m"])
            >= WALL_CENTER_X_M + WALL_HALF_THICKNESS_M + 0.5
            for row in nominal_rows
        ),
        "optical_ambiguity_strictly_ordered_each_pair": monotonicity[
            "optical_ambiguity"
        ][
            "paired_seed_strict_fraction"
        ]
        == 1.0,
        "glass_shader_parameters_read_back": all(
            len(_obstacle_row(row)["visual_material_parameters"]) == 1
            and bool(_obstacle_row(row)["visual_material_parameters"][0]["shader_valid"])
            and abs(
                float(_obstacle_row(row)["visual_material_parameters"][0]["glass_ior"])
                - 1.49
            )
            <= 1.0e-5
            and not bool(_obstacle_row(row)["visual_material_parameters"][0]["thin_walled"])
            for row in records
        ),
        "mild_is_not_catastrophic": all(not bool(row["anomaly_fallen"]) for row in mild_effects),
        "every_active_barrier_has_measured_consequence": all(
            bool(row["anomaly_fallen"])
            or float(row["progress_deficit_m"]) >= 0.08
            or float(row["tilt_increase_rad"]) >= 0.04
            or float(row["foot_force_increase_n"]) >= 10.0
            or float(row["nonfoot_force_increase_n"]) >= 10.0
            for row in effect_records
        ),
        "all_glass_panels_block_or_destabilize": all(
            bool(row["anomaly_fallen"])
            or float(row["progress_deficit_m"]) >= 0.35
            or float(row["stall_increase_s"]) >= 0.30
            for row in effect_records
        ),
        "physics_response_is_invariant_to_visual_dose": all(
            max(
                float(row["progress_deficit_m"])
                for row in effect_records
                if int(row["seed"]) == seed
            )
            - min(
                float(row["progress_deficit_m"])
                for row in effect_records
                if int(row["seed"]) == seed
            )
            <= 0.02
            for seed in selected_seeds
        ),
    }
    manifest = {
        "schema_version": "kinofail.o8-transparent-obstacle-gate.v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "publication_status": "engineering_visual_physics_gate_not_corpus_evidence",
        "checks": checks,
        "design": {
            "paired_by_seed_and_geometry": True,
            "seeds": list(selected_seeds),
            "doses": list(DOSES),
            "geometry_kind": "transparent_acrylic",
            "panel_rect": {
                "center_x_m": WALL_CENTER_X_M,
                "half_thickness_m": WALL_HALF_THICKNESS_M,
                "half_width_m": WALL_HALF_WIDTH_M,
            },
            "counterfactual_contract": (
                "nominal and anomaly retain identical visible USD geometry/material; only the "
                "collisionEnabled attribute changes"
            ),
            "mechanics_note": (
                "panel collision geometry is fixed at 0.38 m; visual ambiguity is the dose. "
                "Mechanical response should therefore be invariant across visual doses"
            ),
            "independent_scene_replication": False,
            "inference_warning": (
                "one flat scene and one panel realization form an engineering gate; rendered "
                "multi-scene RGB pairs and scene-cluster inference remain required"
            ),
            "command_vx_mps_range": [0.32, 0.38],
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
            "operator": _sha256(REPO_ROOT / "kino_vla/sim/operators/o8_invisible_collider.py"),
            "types": _sha256(REPO_ROOT / "kino_vla/sim/types.py"),
            "gate_script": _sha256(Path(__file__).resolve()),
            "config": _sha256(REPO_ROOT / "configs/sim/go2_skeleton.yaml"),
        },
        "summary_by_dose": by_dose,
        "monotonicity": monotonicity,
        "visual_counterfactual_contract": visual_contract,
        "effect_records": effect_records,
        "records": records,
    }
    path = output / "gate_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(
        "PASS: O8 transparent obstacle gate"
        if manifest["passed"]
        else "FAIL: O8 transparent obstacle gate"
    )
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if manifest["passed"] else 1)


if __name__ == "__main__":
    main()
