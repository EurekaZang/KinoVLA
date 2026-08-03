#!/usr/bin/env python3
"""GPU gate for O3 measured load damage and real support-topology change."""

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


def _collision_readback(prim_path: str) -> list[bool]:
    import omni.usd
    from pxr import Usd, UsdPhysics

    root = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
    values: list[bool] = []
    if not root.IsValid():
        return values
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            attr = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
            values.append(bool(attr.Get()) if attr.IsValid() else True)
    return values


def _fracture_visual_readback(paths: list[str]) -> dict[str, object]:
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    visible = []
    collision_counts = []
    failure_states = []
    source_cells = []
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        visible.append(
            prim.IsValid()
            and UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
        )
        collision_counts.append(
            sum(candidate.HasAPI(UsdPhysics.CollisionAPI) for candidate in Usd.PrimRange(prim))
        )
        failure_states.append(prim.GetAttribute("kino:failureState").Get())
        source_cells.append(int(prim.GetAttribute("kino:sourceCellIndex").Get()))
    return {
        "all_visible": bool(paths) and all(visible),
        "collision_api_counts": collision_counts,
        "failure_states": failure_states,
        "source_cells": source_cells,
    }


def _visible(prim_path: str) -> bool:
    import omni.usd
    from pxr import UsdGeom

    prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
    return bool(
        prim.IsValid()
        and UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/kinofail_realistic/o3_topology_gate")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    import isaaclab

    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.operators import Collapse
    from kino_vla.utils.config import load_config
    from kino_vla.utils.geometry import Rect

    output = Path(args.out).resolve()
    cfg = load_config("sim/go2_skeleton.yaml")
    backend = IsaacPolicyBackend(cfg, np.array([0.0, 0.0]), 0.0)
    threshold_ns = 25.0
    drop_m = 0.12
    residual_support = 0.18
    operator = Collapse(
        region=Rect(1.8, 0.0, 0.9, 0.8),
        mu_intact=0.8,
        mu_collapsed=0.07,
        damage_threshold_ns=threshold_ns,
        drop_m=drop_m,
        residual_support=residual_support,
    )
    operator.on_reset(backend)
    obs = backend.reset(42)
    command = np.array([0.55, 0.0, 0.0], dtype=np.float64)
    pre_trigger_heights: list[float] = []
    post_trigger_heights: list[float] = []
    trigger_time_s: float | None = None
    for _ in range(320):
        telemetry = backend.collapse_telemetry()["regions"][0]
        collapsed_before = bool(telemetry["collapsed"])
        obs = backend.step(command if not collapsed_before else np.zeros(3))
        if not collapsed_before:
            pre_trigger_heights.append(float(obs.base_height))
        else:
            post_trigger_heights.append(float(obs.base_height))
        telemetry = backend.collapse_telemetry()["regions"][0]
        if bool(telemetry["collapsed"]) and trigger_time_s is None:
            trigger_time_s = float(obs.t)
        if trigger_time_s is not None and len(post_trigger_heights) >= 35:
            break

    state = backend._collapse[0]
    telemetry = backend.collapse_telemetry()["regions"][0]
    failed_paths = [state["cell_paths"][index] for index in state["failed_cell_indices"]]
    failed_collision_values = [
        value for path in failed_paths for value in _collision_readback(path)
    ]
    catch_collision_values = _collision_readback(state["catch_path"])
    fracture_paths = list(state["fracture_visual_paths"])
    fracture_readback = _fracture_visual_readback(fracture_paths)
    baseline_height = float(np.median(pre_trigger_heights[-20:])) if pre_trigger_heights else 0.0
    minimum_post_height = min(post_trigger_heights) if post_trigger_heights else baseline_height
    vertical_drop = baseline_height - minimum_post_height
    last_damage = telemetry["last_damage_update"] or {}
    checks = {
        "measured_normal_impulse_triggered": bool(telemetry["collapsed"])
        and float(last_damage.get("normal_impulse_ns", 0.0)) >= threshold_ns,
        "support_cells_selected": int(telemetry["failed_support_cells"]) == 13,
        "failed_cell_colliders_disabled": bool(failed_collision_values)
        and not any(failed_collision_values),
        "lower_catch_surface_enabled": bool(catch_collision_values)
        and all(catch_collision_values),
        "default_monolithic_ground_disabled": int(
            telemetry["default_ground_colliders_disabled"]
        )
        > 0,
        "vertical_response_observed": vertical_drop >= 0.025 or bool(obs.fallen),
        "failed_cells_are_visually_removed": bool(failed_paths)
        and not any(_visible(path) for path in failed_paths),
        "fracture_visuals_match_failed_topology": len(fracture_paths)
        == 2 * len(failed_paths)
        and set(fracture_readback["source_cells"]) == set(state["failed_cell_indices"]),
        "fracture_visuals_are_visible_state_tagged": fracture_readback["all_visible"]
        and set(fracture_readback["failure_states"])
        == {"fractured_after_impulse_trigger"},
        "fracture_visuals_do_not_add_hidden_physics": not any(
            fracture_readback["collision_api_counts"]
        ),
        "visual_and_physics_share_trigger_step": telemetry["visual_sync_trigger_step"]
        == last_damage.get("trigger_step"),
    }
    manifest = {
        "schema_version": "kinofail.o3-topology-gate.v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "publication_status": "physics_vertical_slice_not_corpus_evidence",
        "checks": checks,
        "operator": {
            "id": "O3_collapse",
            "mode": "impulse_triggered_support_topology",
            "damage_threshold_ns": threshold_ns,
            "drop_m": drop_m,
            "residual_support": residual_support,
            "seed": 42,
        },
        "readback": telemetry,
        "trigger_time_s": trigger_time_s,
        "baseline_base_height_m": baseline_height,
        "minimum_post_trigger_base_height_m": minimum_post_height,
        "vertical_drop_m": vertical_drop,
        "fallen": bool(obs.fallen),
        "failed_collider_readback": failed_collision_values,
        "catch_collider_readback": catch_collision_values,
        "fracture_visual_readback": fracture_readback,
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
    }
    _write_json(output / "gate_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    print("PASS: O3 topology gate" if manifest["passed"] else "FAIL: O3 topology gate")
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if manifest["passed"] else 1)


if __name__ == "__main__":
    main()
