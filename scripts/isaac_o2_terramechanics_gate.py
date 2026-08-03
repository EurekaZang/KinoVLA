#!/usr/bin/env python3
"""GPU gate for O2 per-foot sinkage and load-gated soil shear."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _footprint_readback(rows: list[dict[str, object]], env_origin: np.ndarray) -> dict[str, object]:
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    position_errors = []
    visible = []
    collision_api_counts = []
    failure_states = []
    for row in rows:
        prim = stage.GetPrimAtPath(str(row["prim_path"]))
        visible.append(
            prim.IsValid()
            and UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
        )
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
        translation = np.asarray(matrix.ExtractTranslation(), dtype=np.float64) - env_origin
        contact = np.asarray(row["contact_position_xyz_m"], dtype=np.float64)
        position_errors.append(float(np.linalg.norm(translation[:2] - contact[:2])))
        collision_api_counts.append(
            sum(candidate.HasAPI(UsdPhysics.CollisionAPI) for candidate in Usd.PrimRange(prim))
        )
        attribute = prim.GetAttribute("kino:failureState")
        failure_states.append(attribute.Get() if attribute.IsValid() else None)
    return {
        "all_visible": bool(rows) and all(visible),
        "max_contact_to_visual_xy_error_m": max(position_errors, default=float("inf")),
        "collision_api_counts": collision_api_counts,
        "failure_states": failure_states,
    }


def _drive(backend, region, initial_obs, *, steps: int = 400):
    obs = initial_obs
    rows = []
    for _ in range(steps):
        command = np.array(
            [
                0.32,
                float(np.clip(-float(obs.pos[1]), -0.16, 0.16)),
                float(np.clip(-2.0 * float(obs.heading), -0.60, 0.60)),
            ],
            dtype=np.float64,
        )
        obs = backend.step(command)
        if region.contains(obs.pos):
            rows.append(
                {
                    "t_s": float(obs.t),
                    "x_m": float(obs.pos[0]),
                    "base_height_m": float(obs.base_height),
                    "speed_mps": float(np.linalg.norm(obs.vel_body)),
                }
            )
        if obs.pos[0] > region.cx + region.hx + 0.4 or obs.fallen:
            break
    return obs, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/kinofail_realistic/o2_terramechanics_gate")
    parser.add_argument("--episode-usd", type=Path)
    parser.add_argument("--compiled-audit", type=Path)
    parser.add_argument(
        "--material-lock",
        type=Path,
        default=REPO_ROOT / "outputs/assets/terrain_pbr_v1/terrain_assets.lock.json",
    )
    parser.add_argument("--material-id", default="train_ground037")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.operators import ComplianceField
    from kino_vla.sim.terrain_materials import (
        appearance_binding_from_record,
        load_terrain_asset_lock,
    )
    from kino_vla.utils.config import load_config
    from kino_vla.utils.geometry import Rect

    output = Path(args.out).resolve()
    cfg = load_config("sim/go2_skeleton.yaml")
    region = Rect(1.5, 0.0, 0.65, 0.60)
    seed = 42
    backend = IsaacPolicyBackend(cfg, np.array([0.0, 0.0]), 0.0)
    compiled = None
    if args.episode_usd is not None:
        if args.compiled_audit is None:
            raise ValueError("--compiled-audit is required with --episode-usd")
        episode = args.episode_usd.resolve()
        compiled_path = args.compiled_audit.resolve()
        compiled = json.loads(compiled_path.read_text(encoding="utf-8"))
        if compiled.get("passed") is not True:
            raise RuntimeError("compiled scene did not pass")
        if compiled["files"].get(episode.name) != _sha256(episode):
            raise RuntimeError("episode USD differs from compiled audit")
        backend.load_realistic_scene(str(episode))

    material_lock_path = args.material_lock.resolve()
    material_lock = load_terrain_asset_lock(material_lock_path)
    binding = appearance_binding_from_record(
        {
            "material_family": args.material_id,
            "appearance_id": f"o2-capability-{args.material_id}",
            "uv_scale": 1.0,
            "uv_rotation_deg": 0.0,
            "surface_state": "damp",
        },
        lock=material_lock,
        asset_root=REPO_ROOT / material_lock["asset_root"],
    )

    initial_nominal = backend.reset(seed)
    nominal_obs, nominal_rows = _drive(backend, region, initial_nominal)
    backend.deep_reset(seed)
    backend.set_terrain_appearance(binding)
    operator = ComplianceField(
        region=region,
        k_c=240.0,
        c_c=0.32,
        d_sink=0.11,
        realistic_foot_model=True,
    )
    operator.on_reset(backend)
    initial_anomaly = backend.reset(seed)
    anomaly_obs, anomaly_rows = _drive(backend, region, initial_anomaly)
    telemetry = backend.foot_compliance_telemetry()
    footprints = telemetry["visual_footprints"]
    footprint_readback = _footprint_readback(
        footprints,
        backend._env.scene.env_origins[0].detach().cpu().numpy(),
    )

    max_sinkage = max(float(foot["max_sinkage_m"]) for foot in telemetry["feet"])
    total_shear_work = sum(float(foot["shear_work_j"]) for foot in telemetry["feet"])
    loaded_feet = sum(int(foot["contact_steps"]) > 0 for foot in telemetry["feet"])
    max_applied_force = max(float(foot["max_applied_force_n"]) for foot in telemetry["feet"])
    nominal_height = float(np.median([row["base_height_m"] for row in nominal_rows]))
    anomaly_height = float(np.median([row["base_height_m"] for row in anomaly_rows]))
    nominal_speed = float(np.median([row["speed_mps"] for row in nominal_rows]))
    anomaly_speed = float(np.median([row["speed_mps"] for row in anomaly_rows]))
    checks = {
        "actual_foot_sinkage": max_sinkage >= 0.07,
        "multiple_feet_load_bearing": loaded_feet >= 2,
        "per_foot_shear_dissipates_energy": total_shear_work > 0.02,
        "per_foot_force_is_nontrivial": max_applied_force > 2.0,
        "base_height_responds": nominal_height - anomaly_height >= 0.035,
        "tracking_speed_responds": anomaly_speed < nominal_speed - 0.03
        or bool(anomaly_obs.fallen),
        "no_legacy_trunk_drag": len(backend._resistance) == 0,
        "isaac_default_ground_replaced": bool(
            telemetry["isaac_default_ground_colliders_disabled"]
        ),
        "realistic_scene_floor_replaced": compiled is None
        or bool(telemetry["realistic_scene_floor_colliders_disabled"]),
        "operator_route_visualization_valid": compiled is None
        or bool(telemetry["nominal_route_appearance_hidden"])
        or bool(telemetry.get("continuous_visual_surface", {}).get("applied")),
        "load_triggered_footprints_exist": len(footprints) >= 4,
        "footprints_cover_multiple_named_feet": len(
            {str(row["foot_name"]) for row in footprints}
        )
        >= 2,
        "footprint_provenance_is_physical": all(
            float(row["contact_force_n"]) >= 2.0
            and float(row["sinkage_m"]) > 0.003
            and region.contains(np.asarray(row["contact_position_xyz_m"])[:2])
            and row["trigger"] == "measured_load_bearing_contact"
            for row in footprints
        ),
        "footprint_visuals_match_contact_xy": footprint_readback[
            "max_contact_to_visual_xy_error_m"
        ]
        < 1.0e-4,
        "footprints_are_visual_only": not any(footprint_readback["collision_api_counts"]),
        "footprints_are_visible_and_state_tagged": footprint_readback["all_visible"]
        and set(footprint_readback["failure_states"]) == {"load_bearing_soil_imprint"},
    }
    manifest = {
        "schema_version": "kinofail.o2-terramechanics-gate.v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "publication_status": "physics_vertical_slice_not_corpus_evidence",
        "checks": checks,
        "operator": {
            "id": "O2_compliance",
            "mode": "per_foot_lowered_collision_bed_plus_shear_yield",
            "sink_depth_m": 0.11,
            "shear_retention": 0.32,
            "vertical_stiffness_n_per_m": 240.0,
            "seed": seed,
        },
        "scene": {
            "compiled_audit": str(args.compiled_audit.resolve())
            if args.compiled_audit
            else None,
            "compiled_audit_sha256": _sha256(args.compiled_audit.resolve())
            if args.compiled_audit
            else None,
            "episode_usd": str(args.episode_usd.resolve()) if args.episode_usd else None,
            "episode_usd_sha256": _sha256(args.episode_usd.resolve())
            if args.episode_usd
            else None,
            "scene_id": compiled.get("scene_id") if compiled else None,
            "terrain_material_id": args.material_id,
            "terrain_material_lock": str(material_lock_path),
            "terrain_material_lock_sha256": _sha256(material_lock_path),
        },
        "measurements": {
            "max_foot_sinkage_m": max_sinkage,
            "loaded_feet": loaded_feet,
            "total_shear_work_j": total_shear_work,
            "max_applied_foot_force_n": max_applied_force,
            "nominal_median_base_height_m": nominal_height,
            "anomaly_median_base_height_m": anomaly_height,
            "base_height_delta_m": nominal_height - anomaly_height,
            "nominal_median_speed_mps": nominal_speed,
            "anomaly_median_speed_mps": anomaly_speed,
            "nominal_final_x_m": float(nominal_obs.pos[0]),
            "anomaly_final_x_m": float(anomaly_obs.pos[0]),
            "anomaly_fallen": bool(anomaly_obs.fallen),
            "visual_footprint_count": len(footprints),
            "visual_footprint_named_feet": sorted(
                {str(row["foot_name"]) for row in footprints}
            ),
        },
        "footprint_visual_readback": footprint_readback,
        "software": {
            "isaac_lab_version": importlib.metadata.version("isaaclab"),
            "isaac_sim_version": importlib.metadata.version("isaacsim"),
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
            "compiled_scene": _sha256(args.compiled_audit.resolve())
            if args.compiled_audit
            else None,
        },
        "telemetry": telemetry,
    }
    _write_json(output / "gate_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    print("PASS: O2 terramechanics gate" if manifest["passed"] else "FAIL: O2 terramechanics gate")
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if manifest["passed"] else 1)


if __name__ == "__main__":
    main()
