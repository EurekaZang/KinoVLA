#!/usr/bin/env python3
"""Paired multi-dose PhysX characterization for realistic O3 support collapse."""

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
DAMAGE_THRESHOLD_NS = 25.0
DOSES = (
    {
        "dose": "mild",
        "drop_m": 0.015,
        "residual_support": 0.9375,
        "mu_collapsed": 0.55,
    },
    {
        "dose": "moderate",
        "drop_m": 0.040,
        "residual_support": 0.75,
        "mu_collapsed": 0.30,
    },
    {
        "dose": "severe",
        "drop_m": 0.120,
        "residual_support": 0.18,
        "mu_collapsed": 0.07,
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _drive(backend, region, *, steps: int = 320) -> tuple[object, dict[str, object]]:
    """Drive through the patch and retain trigger-local, uncensored response telemetry."""
    command = np.array([0.55, 0.0, 0.0], dtype=np.float64)
    heights_before_trigger: list[float] = []
    heights_after_trigger: list[float] = []
    trigger_time_s: float | None = None
    trigger_x_m: float | None = None
    for _ in range(steps):
        collapsed_before = bool(backend.collapse_telemetry()["regions"][0]["collapsed"])
        obs = backend.step(command)
        telemetry = backend.collapse_telemetry()["regions"][0]
        collapsed_after = bool(telemetry["collapsed"])
        if not collapsed_before:
            heights_before_trigger.append(float(obs.base_height))
        if collapsed_after:
            heights_after_trigger.append(float(obs.base_height))
        if collapsed_after and trigger_time_s is None:
            trigger_time_s = float(obs.t)
            trigger_x_m = float(obs.pos[0])
        if obs.pos[0] > region.cx + region.hx + 0.4 or obs.fallen:
            break
    baseline_height = (
        float(np.median(heights_before_trigger[-20:]))
        if heights_before_trigger
        else float(obs.base_height)
    )
    minimum_post_height = (
        min(heights_after_trigger) if heights_after_trigger else baseline_height
    )
    return obs, {
        "trigger_time_s": trigger_time_s,
        "trigger_x_m": trigger_x_m,
        "baseline_base_height_m": baseline_height,
        "minimum_post_trigger_base_height_m": minimum_post_height,
        "vertical_drop_m": max(0.0, baseline_height - minimum_post_height),
        "post_trigger_observations": len(heights_after_trigger),
    }


def _drive_nominal(backend, region, *, steps: int = 320):
    command = np.array([0.55, 0.0, 0.0], dtype=np.float64)
    for _ in range(steps):
        obs = backend.step(command)
        if obs.pos[0] > region.cx + region.hx + 0.4 or obs.fallen:
            break
    return obs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/kinofail_realistic/o3_dose_sweep")
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
    from kino_vla.sim.operators import Collapse
    from kino_vla.utils.config import load_config
    from kino_vla.utils.geometry import Rect

    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("--seeds must contain one or more unique comma-separated integers")
    cfg = load_config("sim/go2_skeleton.yaml")
    region = Rect(1.8, 0.0, 0.9, 0.8)
    backend = IsaacPolicyBackend(cfg, np.array([0.0, 0.0]), 0.0)
    nominal_by_seed: dict[int, dict[str, float | bool]] = {}
    records: list[dict[str, object]] = []

    for seed in selected_seeds:
        backend.deep_reset(seed)
        obs = _drive_nominal(backend, region)
        nominal_by_seed[seed] = {
            "final_x_m": float(obs.pos[0]),
            "fallen": bool(obs.fallen),
        }
        records.append(
            {
                "seed": seed,
                "dose": "nominal",
                "drop_m": 0.0,
                "residual_support": 1.0,
                "failed_support_fraction": 0.0,
                "mu_collapsed": None,
                "normal_impulse_at_trigger_ns": 0.0,
                "trigger_time_s": None,
                "trigger_x_m": None,
                "vertical_drop_m": 0.0,
                "progress_deficit_m": 0.0,
                "final_x_m": float(obs.pos[0]),
                "fallen": bool(obs.fallen),
                "catastrophic_outcome": float(bool(obs.fallen)),
                "topology_changed": False,
                "default_ground_replaced": False,
            }
        )

    for dose in DOSES:
        for seed in selected_seeds:
            backend.deep_reset(seed)
            operator = Collapse(
                region=region,
                mu_intact=0.8,
                damage_threshold_ns=DAMAGE_THRESHOLD_NS,
                **{key: value for key, value in dose.items() if key != "dose"},
            )
            operator.on_reset(backend)
            backend.reset(seed)
            obs, response = _drive(backend, region)
            telemetry = backend.collapse_telemetry()["regions"][0]
            damage = telemetry["last_damage_update"] or {}
            failed_count = int(telemetry["failed_support_cells"])
            total_cells = len(backend._collapse[0]["cell_paths"])
            records.append(
                {
                    "seed": seed,
                    **dose,
                    "failed_support_fraction": failed_count / total_cells,
                    "normal_impulse_at_trigger_ns": float(
                        damage.get("normal_impulse_ns", 0.0)
                    ),
                    **response,
                    "progress_deficit_m": float(nominal_by_seed[seed]["final_x_m"])
                    - float(obs.pos[0]),
                    "final_x_m": float(obs.pos[0]),
                    "fallen": bool(obs.fallen),
                    "catastrophic_outcome": float(bool(obs.fallen)),
                    "topology_changed": bool(telemetry["collapsed"])
                    and int(telemetry["disabled_cell_colliders"]) == failed_count,
                    "default_ground_replaced": int(
                        telemetry["default_ground_colliders_disabled"]
                    )
                    > 0,
                }
            )

    dose_order = ["nominal", "mild", "moderate", "severe"]
    failed_monotonicity = paired_dose_monotonicity(
        records,
        dose_order=dose_order,
        metric="failed_support_fraction",
    )
    drop_monotonicity = paired_dose_monotonicity(
        records,
        dose_order=dose_order,
        metric="vertical_drop_m",
        tolerance=0.006,
    )
    progress_monotonicity = paired_dose_monotonicity(
        records,
        dose_order=dose_order,
        metric="progress_deficit_m",
        tolerance=0.08,
    )
    catastrophe_monotonicity = paired_dose_monotonicity(
        records,
        dose_order=dose_order,
        metric="catastrophic_outcome",
    )
    by_dose: dict[str, dict[str, object]] = {}
    for dose_name in dose_order:
        cells = [row for row in records if row["dose"] == dose_name]
        by_dose[dose_name] = {
            "n_seed_repeats": len(cells),
            "fall_count": sum(bool(row["fallen"]) for row in cells),
            "failed_support_fraction": bootstrap_mean_interval(
                [float(row["failed_support_fraction"]) for row in cells], seed=201
            ).to_dict(),
            "vertical_drop_m": bootstrap_mean_interval(
                [float(row["vertical_drop_m"]) for row in cells], seed=202
            ).to_dict(),
            "progress_deficit_m": bootstrap_mean_interval(
                [float(row["progress_deficit_m"]) for row in cells], seed=203
            ).to_dict(),
        }

    anomaly_rows = [row for row in records if row["dose"] != "nominal"]
    mild_rows = [row for row in records if row["dose"] == "mild"]
    severe_rows = [row for row in records if row["dose"] == "severe"]
    checks = {
        "complete_paired_design": len(records) == len(selected_seeds) * len(dose_order),
        "fixed_measured_load_trigger": all(
            bool(row["topology_changed"])
            and float(row["normal_impulse_at_trigger_ns"]) >= DAMAGE_THRESHOLD_NS
            for row in anomaly_rows
        ),
        "all_anomalies_change_real_support_topology": all(
            bool(row["default_ground_replaced"]) and bool(row["topology_changed"])
            for row in anomaly_rows
        ),
        "failed_support_fraction_strictly_increases": bool(
            failed_monotonicity["median_strictly_ordered"]
        )
        and failed_monotonicity["paired_seed_strict_fraction"] == 1.0,
        "catastrophic_outcome_never_decreases": catastrophe_monotonicity[
            "paired_seed_order_fraction"
        ]
        == 1.0,
        "progress_deficit_order_holds_most_seeds": progress_monotonicity[
            "paired_seed_order_fraction"
        ]
        >= 0.8,
        "mild_is_not_catastrophic": all(not bool(row["fallen"]) for row in mild_rows),
        "severe_has_clear_consequence": all(
            bool(row["fallen"])
            or float(row["vertical_drop_m"]) >= 0.06
            or float(row["progress_deficit_m"]) >= 0.6
            for row in severe_rows
        ),
    }
    manifest = {
        "schema_version": "kinofail.o3-dose-sweep.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "publication_status": "engineering_dose_gate_not_corpus_evidence",
        "checks": checks,
        "design": {
            "paired_by_seed": True,
            "seeds": list(selected_seeds),
            "dose_order": dose_order,
            "doses": list(DOSES),
            "damage_threshold_ns": DAMAGE_THRESHOLD_NS,
            "trigger_is_fixed_across_anomaly_doses": True,
            "independent_scene_replication": False,
            "inference_warning": (
                "seed repeats test physics reproducibility; scene-cluster inference waits for pilot"
            ),
            "dose_interpretation": (
                "post-failure severity jointly increases geometric drop, failed support area, "
                "and loss of catch-bed friction while holding the measured-load trigger fixed"
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
            "operator": _sha256(REPO_ROOT / "kino_vla/sim/operators/o3_collapse.py"),
            "collapse_model": _sha256(REPO_ROOT / "kino_vla/sim/collapse.py"),
            "gate_script": _sha256(Path(__file__).resolve()),
            "config": _sha256(REPO_ROOT / "configs/sim/go2_skeleton.yaml"),
        },
        "summary_by_dose": by_dose,
        "monotonicity": {
            "failed_support_fraction": failed_monotonicity,
            "vertical_drop": drop_monotonicity,
            "progress_deficit": progress_monotonicity,
            "catastrophic_outcome": catastrophe_monotonicity,
        },
        "metric_censoring_note": (
            "vertical drop is descriptive after a fall because termination censors the tail; "
            "ordered consequence is evaluated with catastrophic outcome and progress deficit"
        ),
        "records": records,
    }
    path = output / "dose_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("PASS: O3 dose sweep" if manifest["passed"] else "FAIL: O3 dose sweep")
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if manifest["passed"] else 1)


if __name__ == "__main__":
    main()
