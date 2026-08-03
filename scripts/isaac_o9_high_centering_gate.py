#!/usr/bin/env python3
"""GPU gate for O9 real ridge geometry and measured foot/belly contact."""

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _drive(backend, monitor_region, steps: int = 280):
    rows = []
    command = np.array([0.55, 0.0, 0.0], dtype=np.float64)
    for _ in range(steps):
        obs = backend.step(command)
        if monitor_region.contains(obs.pos):
            rows.append(
                {
                    "t_s": float(obs.t),
                    "x_m": float(obs.pos[0]),
                    "base_height_m": float(obs.base_height),
                    "support_ratio": float(obs.support_ratio),
                    "speed_mps": float(np.linalg.norm(obs.vel_body)),
                }
            )
        if obs.pos[0] > monitor_region.cx + monitor_region.hx + 0.5 or obs.fallen:
            break
    return obs, rows


def _geometry_readback(prim_path: str) -> dict[str, object]:
    import omni.usd
    from pxr import Usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() == "Cylinder":
            return {
                "type": "Cylinder",
                "radius_m": float(prim.GetAttribute("radius").Get()),
                "length_m": float(prim.GetAttribute("height").Get()),
                "axis": str(prim.GetAttribute("axis").Get()),
            }
        if prim.GetTypeName() == "Cube":
            return {"type": "Cube"}
    return {"type": "missing"}


def _base_collision_readback() -> dict[str, object]:
    """Audit the active Go2 USD, rather than assuming its source URDF collider survived export."""
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    base = stage.GetPrimAtPath("/World/envs/env_0/Robot/base")
    rows: list[dict[str, object]] = []
    if not base.IsValid():
        return {"root_valid": False, "root_is_instance": False, "prim_count": 0, "colliders": []}
    prims = []
    frontier = [base]
    while frontier:
        prim = frontier.pop()
        prims.append(prim)
        frontier.extend(prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))
    for prim in prims:
        collision_api = UsdPhysics.CollisionAPI(prim)
        if not collision_api:
            continue
        enabled_attr = collision_api.GetCollisionEnabledAttr()
        row: dict[str, object] = {
            "path": str(prim.GetPath()),
            "type": prim.GetTypeName(),
            "collision_enabled": bool(enabled_attr.Get()) if enabled_attr else True,
        }
        if prim.IsA(UsdGeom.Cube):
            row["size"] = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
            scale = Gf.Transform(
                UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
            ).GetScale()
            row["world_scale_xyz"] = [float(scale[0]), float(scale[1]), float(scale[2])]
        rows.append(row)
    return {
        "root_valid": True,
        "root_is_instance": bool(base.IsInstance()),
        "prim_count": len(prims),
        "colliders": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/kinofail_realistic/o9_high_centering_gate")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    import isaaclab

    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.operators import HighCentering
    from kino_vla.utils.config import load_config
    from kino_vla.utils.geometry import Rect

    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config("sim/go2_skeleton.yaml")
    seed = 42
    # O9 evaluates attribution/recovery from an already-beached failure state.  Both conditions
    # use the same initial base pose; only the central support geometry differs.
    backend = IsaacPolicyBackend(cfg, np.array([1.8, 0.0]), 0.0)
    # The Go2 collision model uses a 0.114 m-tall base box.  Starting at 0.43 m leaves
    # 13 mm clearance above the 0.36 m runner.  The nominal controller settles
    # toward ~0.408 m, so gravity establishes the chassis contact continuously; starting with
    # interpenetrating meshes would create a one-substep depenetration impulse, not beaching.
    backend._spawn_z = 0.43
    monitor_region = Rect(1.8, 0.0, 0.75, 0.9)
    backend.reset(seed, preserve_settle_telemetry=True)
    for _ in range(12):
        backend.step(np.zeros(3))
    nominal_obs, nominal_rows = _drive(backend, monitor_region)
    backend.deep_reset(seed)
    operator = HighCentering(
        monitor_region,
        residual_support=0.22,
        ridge_height_m=0.36,
        ridge_width_m=0.35,
        geometry_kind="central_pallet_runner",
    )
    operator.on_reset(backend)
    backend.reset(seed, preserve_settle_telemetry=True)
    for _ in range(12):
        backend.step(np.zeros(3))
    anomaly_obs, anomaly_rows = _drive(backend, monitor_region)
    telemetry = backend.high_centering_telemetry()["regions"][0]
    geometry = _geometry_readback(str(telemetry["prim_path"]))
    base_colliders = _base_collision_readback()
    nominal_height = float(np.median([row["base_height_m"] for row in nominal_rows]))
    nominal_support = min(row["support_ratio"] for row in nominal_rows)
    measured_support = float(telemetry["min_measured_support"])
    belly_force = float(telemetry["max_belly_contact_force_n"])
    checks = {
        "pallet_runner_geometry_readback": geometry.get("type") == "Cube",
        "go2_base_collider_present": any(
            row["collision_enabled"] for row in base_colliders["colliders"]
        ),
        "foot_support_measured_loss": measured_support <= 0.75,
        "belly_contact_measured": belly_force > 2.0,
        "belly_contact_sustained": int(telemetry["max_consecutive_belly_contact_steps"])
        >= 10,
        "belly_contact_not_single_impulse": float(telemetry["belly_contact_duty_cycle"])
        >= 0.10,
        "support_not_copied_from_target": abs(measured_support - 0.22) > 1.0e-3,
        "chassis_height_responds": float(telemetry["max_base_height_m"]) > nominal_height + 0.02,
        "motion_consequence": float(anomaly_obs.pos[0]) < float(nominal_obs.pos[0]) - 0.35
        or bool(anomaly_obs.fallen),
        "measurement_window_nonempty": int(telemetry["measurement_steps"]) >= 3,
    }
    manifest = {
        "schema_version": "kinofail.o9-high-centering-gate.v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "operator": {
            "id": "O9_high_centering",
            "geometry_kind": "central_pallet_runner",
            "ridge_height_m": 0.36,
            "ridge_width_m": 0.35,
            "target_residual_support": 0.22,
            "seed": seed,
            "initialization": "matched_root_z_0.43_with_13mm_clearance_then_gravity_settle",
            "termination_contract": (
                "training_base_contact_auto_reset_disabled; "
                "Kino-Fail height/tilt fall checks active"
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
        "geometry_readback": geometry,
        "go2_base_collision_readback": base_colliders,
        "measurements": {
            "nominal_min_support": nominal_support,
            "measured_min_support": measured_support,
            "max_belly_contact_force_n": belly_force,
            "mean_belly_contact_force_n": telemetry["mean_belly_contact_force_n"],
            "belly_contact_steps": telemetry["belly_contact_steps"],
            "max_consecutive_belly_contact_steps": telemetry[
                "max_consecutive_belly_contact_steps"
            ],
            "belly_contact_duty_cycle": telemetry["belly_contact_duty_cycle"],
            "nominal_median_base_height_m": nominal_height,
            "anomaly_max_base_height_m": telemetry["max_base_height_m"],
            "nominal_final_x_m": float(nominal_obs.pos[0]),
            "anomaly_final_x_m": float(anomaly_obs.pos[0]),
            "anomaly_fallen": bool(anomaly_obs.fallen),
        },
        "telemetry": telemetry,
    }
    manifest_path = output / "gate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))
    print("PASS: O9 high-centering gate" if manifest["passed"] else "FAIL: O9 high-centering gate")
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if manifest["passed"] else 1)


if __name__ == "__main__":
    main()
