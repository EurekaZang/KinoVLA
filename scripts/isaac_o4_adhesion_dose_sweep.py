#!/usr/bin/env python3
"""Paired foot-local adhesion dose and recovery gate in the realistic indoor scene."""

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
DEFAULT_CAPS_N = (8.0, 16.0, 32.0)
FORWARD_S = 4.7
RECOVERY_S = 2.2
STOP_S = 0.4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paired_commands(seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    return float(rng.uniform(0.53, 0.58)), float(rng.uniform(-0.35, -0.29))


def _run_protocol(backend, *, forward_mps: float, reverse_mps: float) -> dict[str, object]:
    obs = backend._make_obs()
    initial_x = float(obs.pos[0])
    reversal_x: float | None = None
    peak_force_n = 0.0
    peak_per_foot_force_n = 0.0
    peak_raw_force_n = 0.0
    force_impulse_ns = 0.0
    active_foot_seconds = 0.0
    attachment_events = 0
    peel_events = 0
    break_events = 0
    max_active_feet = 0
    while obs.t < FORWARD_S + RECOVERY_S + STOP_S and not obs.fallen:
        if obs.t < FORWARD_S:
            command = forward_mps
        elif obs.t < FORWARD_S + RECOVERY_S:
            command = reverse_mps
            if reversal_x is None:
                reversal_x = float(obs.pos[0])
        else:
            command = 0.0
        obs = backend.step(np.array([command, 0.0, 0.0], dtype=np.float64))
        telemetry = backend.adhesion_telemetry()
        applied_force = float(telemetry["total_applied_force_n"])
        active_feet = int(telemetry["active_feet"])
        peak_force_n = max(peak_force_n, applied_force)
        peak_per_foot_force_n = max(
            [peak_per_foot_force_n]
            + [float(foot["applied_force_n"]) for foot in telemetry["feet"]]
        )
        peak_raw_force_n = max(
            [peak_raw_force_n]
            + [float(foot["raw_force_n"]) for foot in telemetry["feet"]]
        )
        force_impulse_ns += applied_force * backend.dt
        active_foot_seconds += active_feet * backend.dt
        max_active_feet = max(max_active_feet, active_feet)
        attachment_events += sum(
            foot["event"] == "attached" for foot in telemetry["feet"]
        )
        peel_events += sum(foot["event"] == "peeled" for foot in telemetry["feet"])
        break_events += sum(foot["event"] == "broken" for foot in telemetry["feet"])
    if reversal_x is None:
        reversal_x = float(obs.pos[0])
    return {
        "initial_x_m": initial_x,
        "reversal_x_m": reversal_x,
        "final_x_m": float(obs.pos[0]),
        "forward_progress_m": reversal_x - initial_x,
        "backtrack_m": max(0.0, reversal_x - float(obs.pos[0])),
        "peak_applied_adhesion_force_n": peak_force_n,
        "peak_applied_per_foot_force_n": peak_per_foot_force_n,
        "peak_raw_adhesion_force_n": peak_raw_force_n,
        "adhesion_force_impulse_ns": force_impulse_ns,
        "active_foot_seconds": active_foot_seconds,
        "max_active_feet": max_active_feet,
        "attachment_events": int(attachment_events),
        "peel_events": int(peel_events),
        "break_events": int(break_events),
        "fallen": bool(obs.fallen),
        "episode_duration_s": float(obs.t),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/kinofail_realistic/o4_adhesion_dose_sweep")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument(
        "--dose-caps",
        default=",".join(str(value) for value in DEFAULT_CAPS_N),
        help="mild,moderate,severe per-foot force caps in newtons",
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
    from kino_vla.sim.adhesion import FootAdhesionConfig
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.realistic_scene import (
        build_indoor_adhesion_scene_spec,
        compile_indoor_scene_layers,
    )
    from kino_vla.utils.config import load_config

    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("--seeds must contain one or more unique comma-separated integers")
    dose_caps = tuple(float(value) for value in args.dose_caps.split(",") if value.strip())
    if len(dose_caps) != 3 or any(value <= 0.0 for value in dose_caps):
        raise ValueError("--dose-caps must contain three positive values")
    if not all(
        left < right for left, right in zip(dose_caps, dose_caps[1:], strict=False)
    ):
        raise ValueError("--dose-caps must be strictly increasing")
    doses = tuple(
        {"dose": name, "force_cap_n": cap}
        for name, cap in zip(("mild", "moderate", "severe"), dose_caps, strict=True)
    )

    demo_cfg = load_config("demo/indoor_adhesion_icra.yaml")
    sim_cfg = load_config("sim/go2_skeleton.yaml")
    scene_spec = build_indoor_adhesion_scene_spec(int(demo_cfg.scene.seed))
    compiled = compile_indoor_scene_layers(scene_spec, output / "scene_usd")
    if not compiled["audit"]["passed"]:
        raise RuntimeError(f"indoor scene audit failed: {compiled['audit']}")
    backend = IsaacPolicyBackend(
        sim_cfg,
        np.asarray(demo_cfg.start.pos, dtype=np.float64),
        float(demo_cfg.start.heading),
    )
    backend.load_realistic_scene(str(compiled["episode_usd"]))

    fixed = demo_cfg.adhesion.to_dict()
    records: list[dict[str, object]] = []
    nominal_by_seed: dict[int, dict[str, object]] = {}
    for seed in selected_seeds:
        forward_mps, reverse_mps = _paired_commands(seed)
        backend.deep_reset(seed)
        result = _run_protocol(
            backend, forward_mps=forward_mps, reverse_mps=reverse_mps
        )
        nominal_by_seed[seed] = result
        records.append(
            {
                "seed": seed,
                "dose": "nominal",
                "force_cap_n": 0.0,
                "forward_command_mps": forward_mps,
                "reverse_command_mps": reverse_mps,
                **result,
                "recovery_deficit_m": 0.0,
                "legacy_trunk_resistance_count": len(backend._resistance),
            }
        )

    if dose_caps[-1] >= float(fixed["break_force_n"]):
        raise ValueError("the severe force cap must stay below the fixed break force")

    for dose in doses:
        for seed in selected_seeds:
            forward_mps, reverse_mps = _paired_commands(seed)
            backend.deep_reset(seed)
            backend.add_foot_adhesion(
                FootAdhesionConfig(
                    region=scene_spec.adhesion_region,
                    surface_z_m=float(fixed["surface_z_m"]),
                    attach_contact_force_n=float(fixed["attach_contact_force_n"]),
                    attach_height_tolerance_m=float(fixed["attach_height_tolerance_m"]),
                    stiffness_xy_n_per_m=float(fixed["stiffness_xy_n_per_m"]),
                    damping_xy_ns_per_m=float(fixed["damping_xy_ns_per_m"]),
                    stiffness_z_n_per_m=float(fixed["stiffness_z_n_per_m"]),
                    damping_z_ns_per_m=float(fixed["damping_z_ns_per_m"]),
                    force_cap_n=float(dose["force_cap_n"]),
                    break_force_n=float(fixed["break_force_n"]),
                    peel_release_force_n=float(fixed["peel_release_force_n"]),
                    peel_velocity_threshold_mps=float(fixed["peel_velocity_threshold_mps"]),
                    progress_axis_xy=tuple(float(v) for v in fixed["progress_axis_xy"]),
                    max_active_feet=int(fixed["max_active_feet"]),
                )
            )
            backend.reset(seed)
            result = _run_protocol(
                backend, forward_mps=forward_mps, reverse_mps=reverse_mps
            )
            records.append(
                {
                    "seed": seed,
                    **dose,
                    "forward_command_mps": forward_mps,
                    "reverse_command_mps": reverse_mps,
                    **result,
                    "recovery_deficit_m": float(nominal_by_seed[seed]["backtrack_m"])
                    - float(result["backtrack_m"]),
                    "legacy_trunk_resistance_count": len(backend._resistance),
                }
            )

    dose_order = ["nominal", "mild", "moderate", "severe"]
    monotonicity = {
        "peak_applied_force": paired_dose_monotonicity(
            records,
            dose_order=dose_order,
            metric="peak_applied_per_foot_force_n",
            tolerance=1.0,
        ),
        "force_impulse": paired_dose_monotonicity(
            records,
            dose_order=dose_order,
            metric="adhesion_force_impulse_ns",
            tolerance=2.0,
        ),
        "recovery_deficit": paired_dose_monotonicity(
            records,
            dose_order=dose_order,
            metric="recovery_deficit_m",
            tolerance=0.08,
        ),
    }
    by_dose: dict[str, dict[str, object]] = {}
    for dose_name in dose_order:
        cells = [row for row in records if row["dose"] == dose_name]
        by_dose[dose_name] = {
            "n_paired_conditions": len(cells),
            "fall_count": sum(bool(row["fallen"]) for row in cells),
            "peak_applied_adhesion_force_n": bootstrap_mean_interval(
                [float(row["peak_applied_adhesion_force_n"]) for row in cells], seed=601
            ).to_dict(),
            "peak_applied_per_foot_force_n": bootstrap_mean_interval(
                [float(row["peak_applied_per_foot_force_n"]) for row in cells], seed=606
            ).to_dict(),
            "adhesion_force_impulse_ns": bootstrap_mean_interval(
                [float(row["adhesion_force_impulse_ns"]) for row in cells], seed=602
            ).to_dict(),
            "active_foot_seconds": bootstrap_mean_interval(
                [float(row["active_foot_seconds"]) for row in cells], seed=603
            ).to_dict(),
            "backtrack_m": bootstrap_mean_interval(
                [float(row["backtrack_m"]) for row in cells], seed=604
            ).to_dict(),
            "recovery_deficit_m": bootstrap_mean_interval(
                [float(row["recovery_deficit_m"]) for row in cells], seed=605
            ).to_dict(),
        }

    anomaly_rows = [row for row in records if row["dose"] != "nominal"]
    mild_rows = [row for row in records if row["dose"] == "mild"]
    severe_rows = [row for row in records if row["dose"] == "severe"]
    backend.deep_reset(selected_seeds[0])
    reset_telemetry = backend.adhesion_telemetry()
    checks = {
        "complete_paired_design": len(records) == len(selected_seeds) * len(dose_order),
        "paired_commands_match": all(
            len(
                {
                    (float(row["forward_command_mps"]), float(row["reverse_command_mps"]))
                    for row in records
                    if int(row["seed"]) == seed
                }
            )
            == 1
            for seed in selected_seeds
        ),
        "paired_conditions_are_distinct": len(
            {
                (float(row["forward_command_mps"]), float(row["reverse_command_mps"]))
                for row in records
                if row["dose"] == "nominal"
            }
        )
        == len(selected_seeds),
        "nominal_has_no_attachment": all(
            int(row["attachment_events"]) == 0
            for row in records
            if row["dose"] == "nominal"
        ),
        "all_anomalies_attach_at_named_feet": all(
            int(row["attachment_events"]) >= 1 and int(row["max_active_feet"]) >= 1
            for row in anomaly_rows
        ),
        "all_anomalies_release_or_break": all(
            int(row["peel_events"]) + int(row["break_events"]) >= 1
            for row in anomaly_rows
        ),
        "no_legacy_trunk_wrench": all(
            int(row["legacy_trunk_resistance_count"]) == 0 for row in records
        ),
        "per_foot_force_respects_cap": all(
            float(row["peak_applied_per_foot_force_n"])
            <= float(row["force_cap_n"]) + 1.0e-5
            for row in anomaly_rows
        ),
        "applied_force_strictly_orders_each_pair": monotonicity[
            "peak_applied_force"
        ]["paired_seed_strict_fraction"]
        == 1.0,
        "force_impulse_orders_most_pairs": monotonicity["force_impulse"][
            "paired_seed_order_fraction"
        ]
        >= 0.8,
        "mild_is_not_catastrophic": all(not bool(row["fallen"]) for row in mild_rows),
        "severe_has_physical_or_recovery_consequence": all(
            bool(row["fallen"])
            or float(row["recovery_deficit_m"]) >= 0.20
            or float(row["adhesion_force_impulse_ns"]) >= 20.0
            for row in severe_rows
        ),
        "deep_reset_clears_adhesion": not bool(reset_telemetry["enabled"])
        and int(reset_telemetry["active_feet"]) == 0,
    }
    manifest = {
        "schema_version": "kinofail.o4-adhesion-dose-sweep.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "publication_status": "engineering_dose_gate_not_corpus_evidence",
        "checks": checks,
        "design": {
            "paired_by_condition": True,
            "seeds": list(selected_seeds),
            "dose_order": dose_order,
            "doses": list(doses),
            "single_factor_dose": "per-foot applied adhesion force cap",
            "fixed_contact_law": fixed,
            "protocol": {
                "forward_s": FORWARD_S,
                "reverse_peel_s": RECOVERY_S,
                "stop_s": STOP_S,
            },
            "scene_id": compiled["scene_id"],
            "independent_scene_replication": False,
            "inference_warning": (
                "one indoor scene and paired command-demand conditions form an engineering gate; "
                "scene-cluster corpus inference remains required"
            ),
        },
        "software": {
            "isaac_lab_version": str(isaaclab.__version__),
            "isaac_sim_version": (
                Path(sys.executable).resolve().parents[3] / "VERSION"
            ).read_text(encoding="utf-8").strip(),
            "physics_engine": "PhysX GPU",
            "physics_dt_s": float(sim_cfg.physics_dt),
            "control_dt_s": float(backend.dt),
        },
        "scene_layers_sha256": compiled["sha256"],
        "input_sha256": {
            "policy": _sha256(REPO_ROOT / str(sim_cfg.policy_path)),
            "backend": _sha256(REPO_ROOT / "kino_vla/sim/isaac_policy_backend.py"),
            "operator_model": _sha256(REPO_ROOT / "kino_vla/sim/adhesion.py"),
            "scene_compiler": _sha256(REPO_ROOT / "kino_vla/sim/realistic_scene.py"),
            "gate_script": _sha256(Path(__file__).resolve()),
            "sim_config": _sha256(REPO_ROOT / "configs/sim/go2_skeleton.yaml"),
            "demo_config": _sha256(REPO_ROOT / "configs/demo/indoor_adhesion_icra.yaml"),
        },
        "summary_by_dose": by_dose,
        "monotonicity": monotonicity,
        "reset_readback": reset_telemetry,
        "records": records,
    }
    path = output / "dose_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("PASS: O4 adhesion dose sweep" if manifest["passed"] else "FAIL: O4 adhesion dose sweep")
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if manifest["passed"] else 1)


if __name__ == "__main__":
    main()
