#!/usr/bin/env python3
"""GPU gate for O5 visible rigid payload mass, CoM, inertia and load transfer."""

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


def _stance_loads(backend, *, warmup: int = 35, samples: int = 60) -> np.ndarray:
    rows: list[np.ndarray] = []
    for index in range(warmup + samples):
        backend.step(np.zeros(3))
        if index >= warmup:
            forces = backend._contact.data.net_forces_w[0, backend._contact_foot_ids]
            rows.append(np.maximum(forces[:, 2].detach().cpu().numpy(), 0.0))
    return np.mean(rows, axis=0)


def _visual_readback(path: str) -> dict[str, object]:
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(path)
    if not root.IsValid():
        return {"valid": False}
    matrix = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    cube_prim = next((prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Cube)), None)
    if cube_prim is None:
        return {"valid": False, "reason": "no Cube below payload visual root"}
    cube = UsdGeom.Cube(cube_prim)
    local_scale = cube_prim.GetAttribute("xformOp:scale").Get()
    size = float(cube.GetSizeAttr().Get())
    dimensions = [size * abs(float(local_scale[index])) for index in range(3)]
    collision_paths = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    return {
        "valid": True,
        "type": cube_prim.GetTypeName(),
        "dimensions_m": dimensions,
        "world_translation_m": [float(value) for value in matrix.ExtractTranslation()],
        "collision_paths": collision_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/kinofail_realistic/o5_payload_gate")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    import isaaclab
    from isaaclab.utils.math import quat_apply

    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.operators import Payload
    from kino_vla.sim.payload import combine_with_cuboid_payload
    from kino_vla.utils.config import load_config

    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config("sim/go2_skeleton.yaml")
    seed = 42
    payload_mass = 6.0
    payload_com = np.array([0.10, 0.045, 0.14], dtype=np.float64)
    payload_size = np.array([0.30, 0.20, 0.16], dtype=np.float64)
    backend = IsaacPolicyBackend(cfg, np.array([0.0, 0.0]), 0.0)

    backend.deep_reset(seed)
    nominal = backend.payload_telemetry()
    nominal_loads = _stance_loads(backend)
    nominal_total_load = float(nominal_loads.sum())

    backend.deep_reset(seed)
    Payload(
        mass_kg=payload_mass,
        com_offset_m=payload_com,
        size_m=payload_size,
    ).on_reset(backend)
    backend.reset(seed)
    payload_loads = _stance_loads(backend)
    loaded = backend.payload_telemetry()
    visual = _visual_readback(str(loaded["visuals"][0]["prim_path"]))

    nominal_mass = float(nominal["base_mass_kg"])
    nominal_com = np.asarray(nominal["base_com_pose"][:3], dtype=np.float64)
    nominal_inertia = np.asarray(nominal["base_inertia_kg_m2"], dtype=np.float64)
    expected = combine_with_cuboid_payload(
        base_mass_kg=nominal_mass,
        base_com_m=nominal_com,
        base_inertia_kg_m2=nominal_inertia,
        payload_mass_kg=payload_mass,
        payload_com_m=payload_com,
        payload_size_m=payload_size,
    )
    measured_com = np.asarray(loaded["base_com_pose"][:3], dtype=np.float64)
    measured_inertia = np.asarray(loaded["base_inertia_kg_m2"], dtype=np.float64)
    base_pos = backend._robot.data.root_pos_w[0].detach().cpu()
    base_quat = backend._robot.data.root_quat_w[0].detach().cpu()
    local_com = backend._torch.as_tensor(payload_com, dtype=base_quat.dtype).reshape(1, 3)
    expected_visual_center = (
        base_pos + quat_apply(base_quat.reshape(1, 4), local_com)[0]
    ).numpy()
    visual_center = np.asarray(visual["world_translation_m"], dtype=np.float64)
    visual_follow_error = float(np.linalg.norm(visual_center - expected_visual_center))
    payload_total_load = float(payload_loads.sum())
    nominal_shares = nominal_loads / max(nominal_total_load, 1.0e-9)
    payload_shares = payload_loads / max(payload_total_load, 1.0e-9)
    redistribution_l1 = float(np.abs(payload_shares - nominal_shares).sum())

    backend.deep_reset(seed)
    restored = backend.payload_telemetry()
    checks = {
        "mass_readback_matches": abs(float(loaded["base_mass_kg"]) - expected.mass_kg)
        <= 1.0e-4,
        "com_readback_matches": float(np.linalg.norm(measured_com - expected.com_m)) <= 1.0e-5,
        "inertia_readback_matches": float(
            np.linalg.norm(measured_inertia - expected.inertia_kg_m2)
        )
        <= 1.0e-4,
        "visible_payload_dimensions_match": bool(visual.get("valid"))
        and np.allclose(visual["dimensions_m"], payload_size, atol=1.0e-6),
        "visible_payload_has_no_duplicate_collider": visual.get("collision_paths") == [],
        "visible_payload_follows_base": visual_follow_error <= 1.0e-4,
        "feet_bear_added_weight": payload_total_load
        >= nominal_total_load + 0.55 * payload_mass * 9.81,
        "offset_changes_load_distribution": redistribution_l1 >= 0.015,
        "deep_reset_restores_mass": abs(float(restored["base_mass_kg"]) - nominal_mass)
        <= 1.0e-5,
        "deep_reset_restores_com": np.allclose(
            restored["base_com_pose"], nominal["base_com_pose"], atol=1.0e-6
        ),
        "deep_reset_restores_inertia": np.allclose(
            restored["base_inertia_kg_m2"], nominal["base_inertia_kg_m2"], atol=1.0e-6
        ),
        "deep_reset_removes_visual": not restored["visuals"],
    }
    manifest = {
        "schema_version": "kinofail.o5-payload-gate.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "publication_status": "physics_vertical_slice_not_corpus_evidence",
        "checks": checks,
        "operator": {
            "id": "O5_payload",
            "mass_kg": payload_mass,
            "com_offset_m": payload_com.tolist(),
            "size_m": payload_size.tolist(),
            "physics_contract": "equivalent_rigid_attachment_composite_mass_properties",
            "visual_contract": "collision_free_visible_cuboid_follows_base_pose",
            "seed": seed,
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
            "gate_script": _sha256(Path(__file__).resolve()),
            "config": _sha256(REPO_ROOT / "configs/sim/go2_skeleton.yaml"),
        },
        "readback": {
            "nominal": nominal,
            "loaded": loaded,
            "expected_composite": {
                "mass_kg": expected.mass_kg,
                "com_m": expected.com_m.tolist(),
                "inertia_kg_m2": expected.inertia_kg_m2.tolist(),
            },
            "visual": visual,
            "visual_follow_error_m": visual_follow_error,
            "nominal_mean_foot_normal_force_n": nominal_loads.tolist(),
            "payload_mean_foot_normal_force_n": payload_loads.tolist(),
            "nominal_total_foot_normal_force_n": nominal_total_load,
            "payload_total_foot_normal_force_n": payload_total_load,
            "load_share_redistribution_l1": redistribution_l1,
            "restored": restored,
        },
    }
    path = output / "gate_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("PASS: O5 payload gate" if manifest["passed"] else "FAIL: O5 payload gate")
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if manifest["passed"] else 1)


if __name__ == "__main__":
    main()
