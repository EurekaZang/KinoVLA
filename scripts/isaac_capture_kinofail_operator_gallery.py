#!/usr/bin/env python3
"""Capture traceable RTX views for Kino-Fail operators that lack saved imagery.

The publication gallery is assembled from existing gate/demo frames wherever possible.  This
script only fills the six historical evidence gaps (O5/O6/O8/O9/O10/O11).  Each invocation
starts one Isaac process and freezes one representative operator state before moving a passive
RTX review camera around it.  It does not modify any gate output or claim new benchmark trials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def _advance(backend, command: np.ndarray, steps: int):
    obs = backend._make_obs()
    for _ in range(steps):
        obs = backend.step(command)
        if obs.fallen:
            break
    return obs


def _prepare_state(operator_id: str, backend, seed: int):
    """Instantiate the exact severe gate configuration and return its frozen review state."""
    from kino_vla.sim.operators import HighCentering, Payload
    from kino_vla.sim.operators.o11_obs_bias import ObsBias
    from kino_vla.sim.operators.o8_invisible_collider import InvisibleCollider
    from kino_vla.utils.geometry import Rect

    zero = np.zeros(3, dtype=np.float64)
    metadata: dict[str, Any] = {"seed": seed, "dose": "severe"}

    if operator_id == "O5":
        backend.deep_reset(seed)
        operator = Payload(
            mass_kg=6.0,
            com_offset_m=np.array([0.10, 0.045, 0.14], dtype=np.float64),
            size_m=np.array([0.30, 0.20, 0.16], dtype=np.float64),
        )
        operator.on_reset(backend)
        backend.reset(seed)
        obs = _advance(backend, np.array([0.30, 0.0, 0.0]), 70)
        metadata.update(
            {
                "operator": "O5_payload",
                "mass_kg": 6.0,
                "com_offset_m": [0.10, 0.045, 0.14],
                "payload_telemetry": backend.payload_telemetry(),
            }
        )
        target = [float(obs.pos[0]), float(obs.pos[1]), 0.30]
    elif operator_id == "O6":
        backend.deep_reset(seed)
        command = np.array([0.36, 0.0, 0.0], dtype=np.float64)
        obs = _advance(backend, command, max(1, int(round(2.0 / backend.dt))))
        pre_push = {
            "t_s": float(obs.t),
            "pos_xy_m": obs.pos.tolist(),
            "tilt_rad": float(obs.tilt),
        }
        backend.start_push_pulse(
            np.array([0.0, 12.0], dtype=np.float64),
            0.0,
            0.12,
            np.array([0.0, -0.085, 0.075], dtype=np.float64),
        )
        obs = _advance(backend, command, max(1, int(round(0.22 / backend.dt))))
        metadata.update(
            {
                "operator": "O6_push",
                "impulse_ns": 12.0,
                "duration_s": 0.12,
                "application_point_body_m": [0.0, -0.085, 0.075],
                "pre_push": pre_push,
                "pulse_telemetry": backend.push_telemetry(),
            }
        )
        target = [float(obs.pos[0]), float(obs.pos[1]), 0.28]
    elif operator_id == "O8":
        backend.deep_reset(seed)
        region = Rect(cx=1.35, cy=0.0, hx=0.018, hy=1.0)
        operator = InvisibleCollider(
            region,
            height_m=0.38,
            collision_enabled=True,
            geometry_kind="transparent_acrylic",
            optical_transmission=0.98,
        )
        operator.on_reset(backend)
        backend.reset(seed)
        obs = _advance(backend, np.array([0.36, 0.0, 0.0]), max(1, int(5.2 / backend.dt)))
        metadata.update(
            {
                "operator": "O8_invisible_collider",
                "geometry_kind": "transparent_acrylic",
                "optical_transmission": 0.98,
                "height_m": 0.38,
                "blocking_telemetry": backend.blocking_telemetry(),
            }
        )
        target = [1.25, 0.0, 0.27]
    elif operator_id == "O9":
        backend.deep_reset(seed)
        region = Rect(1.8, 0.0, 0.75, 0.9)
        operator = HighCentering(
            region,
            residual_support=0.22,
            ridge_height_m=0.36,
            ridge_width_m=0.35,
            geometry_kind="central_pallet_runner",
        )
        operator.on_reset(backend)
        backend.reset(seed, preserve_settle_telemetry=True)
        _advance(backend, zero, 12)
        obs = _advance(backend, np.array([0.55, 0.0, 0.0]), 150)
        metadata.update(
            {
                "operator": "O9_high_centering",
                "geometry_kind": "central_pallet_runner",
                "ridge_height_m": 0.36,
                "ridge_width_m": 0.35,
                "high_centering_telemetry": backend.high_centering_telemetry(),
            }
        )
        target = [1.8, 0.0, 0.30]
    elif operator_id == "O10":
        backend.deep_reset(seed)
        backend.set_effort_scale(0.25)
        backend.reset(seed)
        command = np.array([0.58, 0.0, 0.0], dtype=np.float64)
        speeds: list[float] = []
        tilts: list[float] = []
        obs = backend._make_obs()
        for _ in range(180):
            obs = backend.step(command)
            speeds.append(float(np.linalg.norm(obs.vel_body)))
            tilts.append(float(obs.tilt))
            if obs.fallen:
                break
        metadata.update(
            {
                "operator": "O10_actuator_derating",
                "effort_scale": 0.25,
                "command_vx_vy_yaw": command.tolist(),
                "mean_speed_mps": float(np.mean(speeds)),
                "peak_tilt_rad": float(max(tilts)),
                "actuator_telemetry": backend.actuator_telemetry(),
            }
        )
        target = [float(obs.pos[0]), float(obs.pos[1]), 0.28]
    elif operator_id == "O11":
        backend.deep_reset(seed)
        operator = ObsBias(
            {"tilt": 0.35},
            random_walk_per_sqrt_s={"tilt": 0.020},
            latency_s=0.20,
            seed=seed,
        )
        operator.on_reset(backend)
        command = np.array([0.36, 0.0, 0.0], dtype=np.float64)
        trace: list[dict[str, float]] = []
        truth = backend._make_obs()
        while truth.t < 4.5 and not truth.fallen:
            truth = backend.step(command)
            measured = operator.transform_obs(truth)
            telemetry = operator.sensor_telemetry()
            trace.append(
                {
                    "t_s": float(truth.t),
                    "truth_tilt_rad": float(truth.tilt),
                    "measured_tilt_rad": float(measured.tilt),
                    "effective_odom_age_s": float(telemetry["effective_odom_age_s"]),
                }
            )
        obs = truth
        metadata.update(
            {
                "operator": "O11_obs_bias",
                "tilt_bias_rad": 0.35,
                "random_walk_rad_sqrt_s": 0.020,
                "latency_s": 0.20,
                "sensor_telemetry": operator.sensor_telemetry(),
                "trace": trace,
                "truth_trajectory_affected": False,
            }
        )
        target = [float(obs.pos[0]), float(obs.pos[1]), 0.28]
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(operator_id)

    metadata["frozen_state"] = {
        "t_s": float(obs.t),
        "pos_xy_m": obs.pos.tolist(),
        "tilt_rad": float(obs.tilt),
        "base_height_m": float(obs.base_height),
        "fallen": bool(obs.fallen),
    }
    return obs, np.asarray(target, dtype=np.float64), metadata


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--operator", required=True, choices=("O5", "O6", "O8", "O9", "O10", "O11"))
    pre.add_argument("--out", default="outputs/kinofail_realistic/operator_gallery/source")
    pre_args, _ = pre.parse_known_args()

    parser = argparse.ArgumentParser(description="Capture Kino-Fail operator gallery evidence")
    parser.add_argument("--operator", required=True, choices=("O5", "O6", "O8", "O9", "O10", "O11"))
    parser.add_argument("--out", default=pre_args.out)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app

    import isaaclab
    import omni.timeline
    from PIL import Image

    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import load_config

    operator_id = str(args.operator)
    output = (REPO_ROOT / str(args.out) / operator_id).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config("sim/go2_skeleton.yaml", {"cam_width": 1280, "cam_height": 720})
    start = np.array([1.8, 0.0], dtype=np.float64) if operator_id == "O9" else np.zeros(2)
    backend = IsaacPolicyBackend(cfg, start, 0.0, record_cam=True)
    if operator_id == "O9":
        backend._spawn_z = 0.43

    seed = 42
    obs, target, metadata = _prepare_state(operator_id, backend, seed)
    offsets = (
        ("01_overview_left", np.array([-2.55, -3.25, 1.95], dtype=np.float64)),
        ("02_side_low", np.array([0.0, -2.25, 0.72], dtype=np.float64)),
        ("03_front_oblique", np.array([2.45, -1.75, 1.30], dtype=np.float64)),
    )
    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()
    frames: list[dict[str, Any]] = []
    for name, offset in offsets:
        eye = target + offset
        backend.aim_record_camera(eye, target)
        for _ in range(4):
            app.update()
        frame = backend.capture_rgb(force_recompute=True)
        if frame is None or float(frame.std()) < 8.0:
            raise RuntimeError(f"invalid RTX frame for {operator_id}/{name}")
        path = output / f"{name}.png"
        Image.fromarray(frame).save(path)
        frames.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256(path),
                "eye_xyz_m": eye.tolist(),
                "target_xyz_m": target.tolist(),
            }
        )

    manifest = {
        "schema_version": "kinofail.operator-gallery-source.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "purpose": "publication_gallery_visualization_not_additional_benchmark_trials",
        "operator_id": operator_id,
        "state_frozen_across_views": True,
        "camera_resolution": [1280, 720],
        "physics_engine": "PhysX GPU",
        "isaac_lab_version": str(isaaclab.__version__),
        "metadata": metadata,
        "frames": frames,
        "input_sha256": {
            "capture_script": _sha256(Path(__file__).resolve()),
            "backend": _sha256(REPO_ROOT / "kino_vla/sim/isaac_policy_backend.py"),
            "config": _sha256(REPO_ROOT / "configs/sim/go2_skeleton.yaml"),
        },
    }
    _write_json(output / "capture_manifest.json", manifest)
    print(json.dumps({"operator": operator_id, "output": str(output), "frames": len(frames)}))
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0)


if __name__ == "__main__":
    main()
