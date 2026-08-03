#!/usr/bin/env python3
"""Collect a same-seed nominal/O4 pair in an admitted EmbodiedGen Office scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(root: Path) -> dict[str, object]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": True}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_frozen_protocol(
    path: Path,
    *,
    args: argparse.Namespace,
) -> dict[str, object]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "kinofail.embodiedgen-o4-formal-protocol.v3":
        raise ValueError("unsupported EmbodiedGen O4 formal protocol schema")
    if protocol.get("status") != "frozen":
        raise ValueError("EmbodiedGen O4 formal protocol is not frozen")

    files = protocol["frozen_files"]
    live_files = {
        "collector": Path(__file__).resolve(),
        "backend": REPO_ROOT / "kino_vla/sim/isaac_policy_backend.py",
        "adhesion_model": REPO_ROOT / "kino_vla/sim/adhesion.py",
        "episode_usd": args.episode_usd.resolve(),
        "compiled_audit": args.compiled_audit.resolve(),
        "rtx_audit": args.episode_usd.resolve().parent / "rtx_qa/rtx_scene_audit.json",
        "go2_audit": args.episode_usd.resolve().parent / "go2_qa/go2_scene_audit.json",
    }
    for name, live_path in live_files.items():
        expected = files[name]
        if Path(expected["path"]).resolve() != live_path:
            raise ValueError(f"formal protocol {name} path differs from the live path")
        if expected["sha256"] != _sha256(live_path):
            raise ValueError(f"formal protocol {name} changed after freeze")

    expected_run = protocol["run_contract"]
    live_run = {
        "forward_s": float(args.forward_s),
        "peel_pulse_s": float(args.peel_pulse_s),
        "recovery_s": float(args.recovery_s),
        "reverse_s": float(args.reverse_s),
        "stop_s": float(args.stop_s),
        "rgb_fps": float(args.rgb_fps),
    }
    if live_run != expected_run:
        raise ValueError(f"formal run contract differs: live={live_run}, frozen={expected_run}")
    if int(args.seed) not in [int(seed) for seed in protocol["formal_episode_seeds"]]:
        raise ValueError(f"episode seed {args.seed} is not authorized by the frozen protocol")
    return protocol


def _frame_metrics(rgb: np.ndarray) -> dict[str, float | int]:
    values = rgb[..., :3].astype(np.float32) / 255.0
    luminance = 0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]
    return {
        "mean_luminance": float(luminance.mean()),
        "std_luminance": float(luminance.std()),
        "black_fraction": float((luminance < 0.02).mean()),
        "quantized_color_count_5bit": int(
            len(np.unique((values * 31).astype(np.uint8).reshape(-1, 3), axis=0))
        ),
    }


def _command_phase_at(
    t: float,
    *,
    forward_s: float,
    peel_pulse_s: float,
    recovery_s: float,
    reverse_s: float,
) -> tuple[np.ndarray, str]:
    """Return the pre-registered O4 release/recovery command at episode time ``t``.

    Peeling and retreat are deliberately separated.  A short reverse pulse authorizes bond
    release, a zero-command interval lets the locomotion policy recover posture, and only then
    does the robot retreat.  This avoids conflating O4's causal consequence with an abrupt
    full-speed direction change while the base is already tilted.
    """
    if t < forward_s:
        return np.array([0.50, 0.0, 0.0], dtype=np.float64), "forward"
    if t < forward_s + peel_pulse_s:
        return np.array([-0.30, 0.0, 0.0], dtype=np.float64), "peel_release"
    if t < forward_s + peel_pulse_s + recovery_s:
        return np.zeros(3, dtype=np.float64), "post_peel_recovery"
    if t < forward_s + peel_pulse_s + recovery_s + reverse_s:
        return np.array([-0.30, 0.0, 0.0], dtype=np.float64), "reverse_retreat"
    return np.zeros(3, dtype=np.float64), "stop"


def _obs_record(obs: object) -> dict[str, object]:
    return {
        "time_s": float(obs.t),
        "pos_xy_m": np.asarray(obs.pos).tolist(),
        "heading_rad": float(obs.heading),
        "vel_body_mps": np.asarray(obs.vel_body).tolist(),
        "yaw_rate_radps": float(obs.yaw_rate),
        "base_height_m": float(obs.base_height),
        "tilt_rad": float(obs.tilt),
        "slip_ratio": float(obs.slip_ratio),
        "effort_ratio": float(obs.effort_ratio),
        "support_ratio": float(obs.support_ratio),
        "fallen": bool(obs.fallen),
    }


def _paired_consequence_metrics(
    nominal_rows: list[dict[str, object]],
    anomaly_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Measure paired O4 consequences after the first attachment.

    The locomotion channel uses route-progress gain, rather than an endpoint,
    so the metric is invariant to scene heading and excludes divergence before
    the intervention becomes active.  The posture channel is evaluated over
    the same forward-command window.
    """

    nominal_by_step = {int(row["step"]): row for row in nominal_rows}
    anomaly_by_step = {int(row["step"]): row for row in anomaly_rows}
    common_steps = sorted(nominal_by_step.keys() & anomaly_by_step.keys())
    if not common_steps:
        raise ValueError("paired telemetry has no common steps")

    first_attachment_step: int | None = None
    for step in common_steps:
        feet = anomaly_by_step[step]["adhesion"]["feet"]
        if any(foot["event"] == "attached" for foot in feet):
            first_attachment_step = step
            break
    if first_attachment_step is None:
        return {
            "available": False,
            "failure_reason": "no_attachment_event",
            "first_attachment_step": None,
            "paired_samples": 0,
        }

    effect_steps = [
        step
        for step in common_steps
        if step >= first_attachment_step
        and nominal_by_step[step]["phase"] == "forward"
        and anomaly_by_step[step]["phase"] == "forward"
    ]
    if len(effect_steps) < 2:
        return {
            "available": False,
            "failure_reason": "fewer_than_two_paired_forward_samples_after_attachment",
            "first_attachment_step": first_attachment_step,
            "paired_samples": len(effect_steps),
        }

    first_step, last_step = effect_steps[0], effect_steps[-1]
    n_first, n_last = nominal_by_step[first_step], nominal_by_step[last_step]
    a_first, a_last = anomaly_by_step[first_step], anomaly_by_step[last_step]
    duration_s = float(n_last["time_s"]) - float(n_first["time_s"])
    if duration_s <= 0.0:
        raise ValueError("paired effect window has non-positive duration")

    nominal_gain = float(n_last["progress_m"]) - float(n_first["progress_m"])
    anomaly_gain = float(a_last["progress_m"]) - float(a_first["progress_m"])
    progress_gain_lag = nominal_gain - anomaly_gain
    nominal_speed = nominal_gain / duration_s
    anomaly_speed = anomaly_gain / duration_s
    speed_suppression = nominal_speed - anomaly_speed
    relative_speed_suppression = speed_suppression / max(abs(nominal_speed), 1.0e-9)

    nominal_window = [nominal_by_step[step] for step in effect_steps]
    anomaly_window = [anomaly_by_step[step] for step in effect_steps]
    nominal_max_tilt = max(float(row["tilt_rad"]) for row in nominal_window)
    anomaly_max_tilt = max(float(row["tilt_rad"]) for row in anomaly_window)
    nominal_min_height = min(float(row["base_height_m"]) for row in nominal_window)
    anomaly_min_height = min(float(row["base_height_m"]) for row in anomaly_window)

    return {
        "available": True,
        "first_attachment_step": first_attachment_step,
        "first_attachment_time_s": float(anomaly_by_step[first_attachment_step]["time_s"]),
        "window_last_step": last_step,
        "window_end_time_s": float(anomaly_by_step[last_step]["time_s"]),
        "window_duration_s": duration_s,
        "paired_samples": len(effect_steps),
        "nominal_progress_gain_m": nominal_gain,
        "o4_progress_gain_m": anomaly_gain,
        "progress_gain_lag_m": progress_gain_lag,
        "nominal_forward_progress_speed_mps": nominal_speed,
        "o4_forward_progress_speed_mps": anomaly_speed,
        "forward_speed_suppression_mps": speed_suppression,
        "relative_forward_speed_suppression": relative_speed_suppression,
        "nominal_max_tilt_rad": nominal_max_tilt,
        "o4_max_tilt_rad": anomaly_max_tilt,
        "max_tilt_increase_rad": anomaly_max_tilt - nominal_max_tilt,
        "nominal_min_base_height_m": nominal_min_height,
        "o4_min_base_height_m": anomaly_min_height,
        "min_base_height_drop_m": nominal_min_height - anomaly_min_height,
    }


def _registered_terminal_outcome(
    nominal: dict[str, object],
    anomaly: dict[str, object],
    consequence: dict[str, object],
    *,
    total_steps: int,
) -> dict[str, object]:
    """Adjudicate fixed-horizon recovery or a causally ordered O4 terminal fall.

    Kino-Fail is a *failure* attribution benchmark, so a severe anomaly-induced fall is an
    outcome, not missing data.  It is admitted only when the nominal take completes, the paired
    forward-effect window is complete, a physical peel is observed, and the fall occurs later.
    Everything else remains unregistered censoring.  This function deliberately depends only on
    recorded summaries/telemetry-derived metrics so the decision is independently auditable.
    """

    nominal_complete = int(nominal["steps"]) == int(total_steps)
    anomaly_complete = int(anomaly["steps"]) == int(total_steps)
    recoverable_complete = (
        nominal_complete
        and anomaly_complete
        and not bool(nominal["fell"])
        and not bool(anomaly["fell"])
    )

    peel_steps = [
        int(event["step"])
        for event in anomaly.get("event_trace", [])
        if event.get("event") == "peeled"
        and event.get("phase") in ("peel_release", "reverse_retreat")
    ]
    first_peel_step = min(peel_steps) if peel_steps else None
    fall_step_value = anomaly.get("first_fall_step")
    fall_step = None if fall_step_value is None else int(fall_step_value)
    window_last_value = consequence.get("window_last_step")
    window_last_step = None if window_last_value is None else int(window_last_value)
    fall_phase = anomaly.get("first_fall_phase")
    ordered_terminal_fall = (
        nominal_complete
        and not bool(nominal["fell"])
        and bool(anomaly["fell"])
        and bool(consequence.get("available"))
        and fall_step is not None
        and window_last_step is not None
        and fall_step > window_last_step
        and first_peel_step is not None
        and window_last_step < first_peel_step < fall_step
        and fall_phase in ("peel_release", "post_peel_recovery", "reverse_retreat", "stop")
        and int(anomaly["steps"]) == fall_step + 1
        and bool(anomaly.get("terminal_frame_at_fall"))
    )
    outcome_class = (
        "recoverable_fixed_horizon"
        if recoverable_complete
        else "o4_induced_terminal_fall"
        if ordered_terminal_fall
        else "unregistered_censoring"
    )
    return {
        "registered": recoverable_complete or ordered_terminal_fall,
        "outcome_class": outcome_class,
        "nominal_completed_fixed_horizon": nominal_complete,
        "anomaly_completed_fixed_horizon": anomaly_complete,
        "recoverable_fixed_horizon": recoverable_complete,
        "ordered_o4_terminal_fall": ordered_terminal_fall,
        "first_peel_step": first_peel_step,
        "first_fall_step": fall_step,
        "first_fall_phase": fall_phase,
        "effect_window_last_step": window_last_step,
        "terminal_frame_at_fall": bool(anomaly.get("terminal_frame_at_fall")),
    }


def main() -> int:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--episode-usd", type=Path, required=True)
    preliminary.add_argument("--compiled-audit", type=Path, required=True)
    preliminary.add_argument("--out", type=Path, required=True)
    preliminary.add_argument("--seed", type=int, default=20260721)
    preliminary.add_argument("--forward-s", type=float, default=9.0)
    preliminary.add_argument("--peel-pulse-s", type=float, default=0.10)
    preliminary.add_argument("--recovery-s", type=float, default=1.00)
    preliminary.add_argument("--reverse-s", type=float, default=3.0)
    preliminary.add_argument("--stop-s", type=float, default=1.0)
    preliminary.add_argument("--rgb-fps", type=float, default=5.0)
    preliminary.add_argument(
        "--protocol-role", choices=("calibration", "formal"), default="formal"
    )
    preliminary.add_argument("--protocol", type=Path)
    pre, _ = preliminary.parse_known_args()

    parser = argparse.ArgumentParser(description="EmbodiedGen O4 counterfactual pair collector")
    parser.add_argument("--episode-usd", type=Path, default=pre.episode_usd)
    parser.add_argument("--compiled-audit", type=Path, default=pre.compiled_audit)
    parser.add_argument("--out", type=Path, default=pre.out)
    parser.add_argument("--seed", type=int, default=pre.seed)
    parser.add_argument("--forward-s", type=float, default=pre.forward_s)
    parser.add_argument("--peel-pulse-s", type=float, default=pre.peel_pulse_s)
    parser.add_argument("--recovery-s", type=float, default=pre.recovery_s)
    parser.add_argument("--reverse-s", type=float, default=pre.reverse_s)
    parser.add_argument("--stop-s", type=float, default=pre.stop_s)
    parser.add_argument("--rgb-fps", type=float, default=pre.rgb_fps)
    parser.add_argument(
        "--protocol-role",
        choices=("calibration", "formal"),
        default=pre.protocol_role,
        help="Calibration outputs are never counted as formal benchmark evidence.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=pre.protocol,
        help="Frozen protocol JSON; mandatory for formal collection.",
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    duration_fields = {
        "forward_s": args.forward_s,
        "peel_pulse_s": args.peel_pulse_s,
        "recovery_s": args.recovery_s,
        "reverse_s": args.reverse_s,
        "stop_s": args.stop_s,
    }
    if any(float(value) < 0.0 for value in duration_fields.values()):
        parser.error(f"protocol durations must be non-negative: {duration_fields}")
    if args.forward_s <= 0.0 or args.peel_pulse_s <= 0.0 or args.recovery_s <= 0.0:
        parser.error("forward, peel-pulse, and recovery durations must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"refusing to overwrite non-empty output directory: {args.out}")
    frozen_protocol = None
    if args.protocol_role == "formal":
        if args.protocol is None:
            parser.error("--protocol is mandatory when --protocol-role=formal")
        frozen_protocol = _load_frozen_protocol(args.protocol.resolve(), args=args)
    args.enable_cameras = True
    simulation_app = AppLauncher(args).app

    passed = False
    try:
        from PIL import Image

        from kino_vla.sim.adhesion import FootAdhesionConfig, IrregularRegion
        from kino_vla.sim.isaac_policy_backend import (
            IsaacPolicyBackend,
            body_command_to_world_xy,
        )
        from kino_vla.utils.config import load_config

        episode = args.episode_usd.resolve()
        compiled_path = args.compiled_audit.resolve()
        compiled = json.loads(compiled_path.read_text())
        if not compiled.get("passed"):
            raise RuntimeError("compiled scene audit failed")
        if compiled["files"].get(episode.name) != _sha256(episode):
            raise RuntimeError("episode hash differs from compiled audit")
        rtx_audit = episode.parent / "rtx_qa/rtx_scene_audit.json"
        go2_audit = episode.parent / "go2_qa/go2_scene_audit.json"
        for required in (rtx_audit, go2_audit):
            evidence = json.loads(required.read_text())
            if not evidence.get("passed") or evidence.get("episode_usd_sha256") != _sha256(episode):
                raise RuntimeError(f"prerequisite scene QA is stale or failed: {required}")
        source_manifest_path = Path(compiled["source_manifest"]).resolve()
        source_manifest_value = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        room_type = str(source_manifest_value["generation"]["room_type"])
        source_scene_seed = int(source_manifest_value["generation"]["seed"])

        route = np.asarray(compiled["route"]["waypoints_xy_m"], dtype=np.float64)
        start = route[len(route) // 2]
        target = route[min(len(route) - 1, len(route) // 2 + 1)]
        direction = (target - start) / np.linalg.norm(target - start)
        heading = math.atan2(float(direction[1]), float(direction[0]))
        region = IrregularRegion(
            tuple(tuple(float(value) for value in point) for point in compiled["operator"]["vertices_xy_m"])
        )
        adhesion_cfg = FootAdhesionConfig(
            region=region,
            surface_z_m=float(compiled["operator"]["surface_z_m"]),
            attach_contact_force_n=5.0,
            attach_height_tolerance_m=0.08,
            stiffness_xy_n_per_m=150.0,
            damping_xy_ns_per_m=3.0,
            stiffness_z_n_per_m=120.0,
            damping_z_ns_per_m=2.0,
            force_cap_n=35.0,
            break_force_n=105.0,
            peel_release_force_n=10.0,
            peel_velocity_threshold_mps=0.03,
            progress_axis_xy=(float(direction[0]), float(direction[1])),
            max_active_feet=1,
        )
        sim_cfg = load_config(
            "sim/go2_skeleton.yaml",
            {
                "cam_width": 640,
                "cam_height": 360,
                "front_cam_width": 640,
                "front_cam_height": 360,
                "front_cam_focal_mm": 19.0,
                "front_cam_aperture_mm": 24.0,
                "front_cam_x_m": 0.335,
                "front_cam_y_m": 0.0,
                "front_cam_z_m": 0.065,
            },
        )
        backend = IsaacPolicyBackend(sim_cfg, start, heading, front_cam=True)
        scene_prim = backend.load_realistic_scene(str(episode))
        frame_stride = max(1, round(1.0 / (float(args.rgb_fps) * backend.dt)))
        total_s = float(
            args.forward_s
            + args.peel_pulse_s
            + args.recovery_s
            + args.reverse_s
            + args.stop_s
        )
        total_steps = round(total_s / backend.dt)
        args.out.mkdir(parents=True, exist_ok=True)

        summaries: dict[str, dict[str, object]] = {}
        telemetry_rows: dict[str, list[dict[str, object]]] = {}
        initial_records: dict[str, dict[str, object]] = {}
        initial_rgbs: dict[str, np.ndarray] = {}
        for take in ("nominal", "o4_adhesion"):
            take_dir = args.out / take
            rgb_dir = take_dir / "rgb"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            obs = backend.deep_reset(int(args.seed))
            if take == "o4_adhesion":
                backend.add_foot_adhesion(adhesion_cfg)
            initial_records[take] = _obs_record(obs)
            rows: list[dict[str, object]] = []
            frames: list[dict[str, object]] = []

            sensor = backend.capture_front_camera()
            if sensor is None:
                raise RuntimeError("front camera returned no initial frame")
            initial_rgb = np.asarray(sensor["rgb"])[..., :3].astype(np.uint8)
            initial_rgbs[take] = initial_rgb.copy()
            initial_path = rgb_dir / "frame_000000.png"
            Image.fromarray(initial_rgb).save(initial_path, compress_level=3)
            frames.append(
                {
                    "frame_id": 0,
                    "step": -1,
                    "time_s": 0.0,
                    "path": str(initial_path.relative_to(args.out)),
                    "sha256": _sha256(initial_path),
                    "metrics": _frame_metrics(initial_rgb),
                }
            )

            for step in range(total_steps):
                t = step * backend.dt
                command, phase = _command_phase_at(
                    t,
                    forward_s=float(args.forward_s),
                    peel_pulse_s=float(args.peel_pulse_s),
                    recovery_s=float(args.recovery_s),
                    reverse_s=float(args.reverse_s),
                )
                obs = backend.step(command)
                adhesion = backend.adhesion_telemetry()
                world_command = body_command_to_world_xy(command[:2], obs.heading)
                record = {
                    **_obs_record(obs),
                    "step": step,
                    "phase": phase,
                    "command_body": command.tolist(),
                    "command_world_xy": world_command.tolist(),
                    "progress_m": float(np.dot(obs.pos - start, direction)),
                    "adhesion": adhesion,
                }
                rows.append(record)
                # Always capture the terminal state, even when it falls between the regular RGB
                # sampling instants.  This makes a registered terminal outcome visual evidence,
                # rather than a privileged-telemetry-only assertion.
                if step % frame_stride == 0 or obs.fallen:
                    sensor = backend.capture_front_camera()
                    if sensor is None:
                        raise RuntimeError("front camera returned no episode frame")
                    rgb = np.asarray(sensor["rgb"])[..., :3].astype(np.uint8)
                    frame_id = len(frames)
                    path = rgb_dir / f"frame_{frame_id:06d}.png"
                    Image.fromarray(rgb).save(path, compress_level=3)
                    frames.append(
                        {
                            "frame_id": frame_id,
                            "step": step,
                            "time_s": float(obs.t),
                            "path": str(path.relative_to(args.out)),
                            "sha256": _sha256(path),
                            "metrics": _frame_metrics(rgb),
                        }
                    )
                if obs.fallen:
                    break

            telemetry_path = take_dir / "telemetry.jsonl"
            with telemetry_path.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            frames_path = take_dir / "frames.jsonl"
            with frames_path.open("w", encoding="utf-8") as stream:
                for frame in frames:
                    stream.write(json.dumps(frame, ensure_ascii=False) + "\n")
            event_trace = [
                {
                    "step": int(row["step"]),
                    "time_s": float(row["time_s"]),
                    "phase": str(row["phase"]),
                    "foot": str(foot["name"]),
                    "event": str(foot["event"]),
                    "raw_force_n": float(foot["raw_force_n"]),
                }
                for row in rows
                for foot in row["adhesion"]["feet"]
                if foot["event"] != "none"
            ]
            events = [event["event"] for event in event_trace]
            first_fall = next((row for row in rows if bool(row["fallen"])), None)
            valid_frames = [
                frame
                for frame in frames
                if frame["metrics"]["std_luminance"] >= 0.025
                and frame["metrics"]["black_fraction"] < 0.95
                and frame["metrics"]["quantized_color_count_5bit"] >= 32
            ]
            summary = {
                "take": take,
                "seed": int(args.seed),
                "steps": len(rows),
                "duration_s": float(rows[-1]["time_s"] if rows else 0.0),
                "rgb_frames": len(frames),
                "initial_state": initial_records[take],
                "max_progress_m": max(float(row["progress_m"]) for row in rows),
                "final_progress_m": float(rows[-1]["progress_m"]),
                "min_base_height_m": min(float(row["base_height_m"]) for row in rows),
                "max_tilt_rad": max(float(row["tilt_rad"]) for row in rows),
                "fell": any(bool(row["fallen"]) for row in rows),
                "first_fall_step": (
                    None if first_fall is None else int(first_fall["step"])
                ),
                "first_fall_time_s": (
                    None if first_fall is None else float(first_fall["time_s"])
                ),
                "first_fall_phase": (
                    None if first_fall is None else str(first_fall["phase"])
                ),
                "peak_adhesion_force_n": max(
                    float(row["adhesion"]["total_applied_force_n"]) for row in rows
                ),
                "max_active_feet": max(int(row["adhesion"]["active_feet"]) for row in rows),
                "attachment_events": events.count("attached"),
                "peel_events": events.count("peeled"),
                "break_events": events.count("broken"),
                "forward_break_events": sum(
                    event["event"] == "broken" and event["phase"] == "forward"
                    for event in event_trace
                ),
                "reverse_peel_events": sum(
                    event["event"] == "peeled"
                    and event["phase"] in ("peel_release", "reverse_retreat")
                    for event in event_trace
                ),
                "reverse_break_events": sum(
                    event["event"] == "broken"
                    and event["phase"] in ("peel_release", "reverse_retreat")
                    for event in event_trace
                ),
                "event_trace": event_trace,
                "telemetry": str(telemetry_path.relative_to(args.out)),
                "telemetry_sha256": _sha256(telemetry_path),
                "frames_manifest": str(frames_path.relative_to(args.out)),
                "frames_manifest_sha256": _sha256(frames_path),
                "no_black_frame": all(
                    frame["metrics"]["black_fraction"] < 0.95 for frame in frames
                ),
                "valid_frame_fraction": len(valid_frames) / len(frames),
                "terminal_frame_at_fall": bool(
                    first_fall is not None and int(frames[-1]["step"]) == int(first_fall["step"])
                ),
            }
            _write_json(take_dir / "summary.json", summary)
            summaries[take] = summary
            telemetry_rows[take] = rows
            print(
                f"[pair] {take}: max_progress={summary['max_progress_m']:.3f}m "
                f"force={summary['peak_adhesion_force_n']:.1f}N "
                f"attach={summary['attachment_events']} peel={summary['peel_events']} "
                f"frames={summary['rgb_frames']}",
                flush=True,
            )

        nominal = summaries["nominal"]
        anomaly = summaries["o4_adhesion"]
        initial_state_delta = {
            key: float(
                np.max(
                    np.abs(
                        np.asarray(initial_records["nominal"][key], dtype=np.float64)
                        - np.asarray(initial_records["o4_adhesion"][key], dtype=np.float64)
                    )
                )
            )
            for key in ("pos_xy_m", "heading_rad", "vel_body_mps", "base_height_m", "tilt_rad")
        }
        initial_rgb_l1 = float(
            np.abs(
                initial_rgbs["nominal"].astype(np.float32)
                - initial_rgbs["o4_adhesion"].astype(np.float32)
            ).mean()
            / 255.0
        )
        consequence = _paired_consequence_metrics(
            telemetry_rows["nominal"], telemetry_rows["o4_adhesion"]
        )
        consequence_thresholds = {
            "minimum_effect_window_s": 1.0,
            "minimum_progress_gain_lag_m": 0.10,
            "minimum_relative_speed_suppression": 0.08,
            "minimum_tilt_increase_rad": 0.12,
            "minimum_base_height_drop_m": 0.04,
        }
        consequence_available = bool(consequence.get("available"))
        locomotion_consequence = consequence_available and (
            float(consequence["window_duration_s"])
            >= consequence_thresholds["minimum_effect_window_s"]
            and float(consequence["progress_gain_lag_m"])
            >= consequence_thresholds["minimum_progress_gain_lag_m"]
            and float(consequence["relative_forward_speed_suppression"])
            >= consequence_thresholds["minimum_relative_speed_suppression"]
        )
        posture_consequence = consequence_available and (
            float(consequence["max_tilt_increase_rad"])
            >= consequence_thresholds["minimum_tilt_increase_rad"]
            or float(consequence["min_base_height_drop_m"])
            >= consequence_thresholds["minimum_base_height_drop_m"]
        )
        terminal_outcome = _registered_terminal_outcome(
            nominal,
            anomaly,
            consequence,
            total_steps=total_steps,
        )
        checks = {
            "same_seed": nominal["seed"] == anomaly["seed"] == int(args.seed),
            "same_episode": True,
            "same_initial_physical_state": max(initial_state_delta.values()) <= 1.0e-5,
            "same_initial_visual_state": initial_rgb_l1 <= 0.03,
            "nominal_completed_fixed_horizon_without_fall": bool(
                terminal_outcome["nominal_completed_fixed_horizon"]
            )
            and not nominal["fell"],
            "registered_recovery_or_terminal_outcome": bool(terminal_outcome["registered"]),
            "both_sequences_visually_valid": bool(nominal["no_black_frame"])
            and bool(anomaly["no_black_frame"])
            and nominal["valid_frame_fraction"] >= 0.90
            and anomaly["valid_frame_fraction"] >= 0.90,
            "nominal_operator_inactive": nominal["peak_adhesion_force_n"] == 0.0,
            "o4_attached": anomaly["attachment_events"] >= 1
            and anomaly["max_active_feet"] >= 1,
            "o4_force_observable": anomaly["peak_adhesion_force_n"] >= 10.0,
            "o4_holds_through_forward": anomaly["forward_break_events"] == 0,
            "o4_reverse_peel_observable": anomaly["reverse_peel_events"] >= 1,
            "counterfactual_locomotion_consequence": locomotion_consequence,
            "counterfactual_posture_consequence": posture_consequence,
            "counterfactual_multichannel_consequence": (
                locomotion_consequence and posture_consequence
            ),
        }
        manifest = {
            "schema_version": "kinofail.embodiedgen-o4-paired-sequence.v4",
            "created_utc": datetime.now(UTC).isoformat(),
            "scene_id": compiled["scene_id"],
            "scene_family": f"EmbodiedGen-v2-{room_type}",
            "source_scene_seed": source_scene_seed,
            "episode_usd": str(episode),
            "episode_usd_sha256": _sha256(episode),
            "source_manifest": compiled["source_manifest"],
            "source_manifest_sha256": compiled["source_manifest_sha256"],
            "compiled_audit": str(compiled_path),
            "compiled_audit_sha256": _sha256(compiled_path),
            "rtx_qa": str(rtx_audit),
            "rtx_qa_sha256": _sha256(rtx_audit),
            "go2_qa": str(go2_audit),
            "go2_qa_sha256": _sha256(go2_audit),
            "seed": int(args.seed),
            "protocol_role": args.protocol_role,
            "formal_protocol": (
                None
                if frozen_protocol is None
                else {
                    "path": str(args.protocol.resolve()),
                    "sha256": _sha256(args.protocol.resolve()),
                    "protocol_id": frozen_protocol["protocol_id"],
                    "scene_seed": frozen_protocol["scene_seed"],
                    "episode_seed_semantics": frozen_protocol["episode_seed_semantics"],
                }
            ),
            "pairing": {
                "intervention_only_difference": "O4 foot-local adhesion enabled",
                "start_xy_m": start.tolist(),
                "heading_rad": heading,
                "route_direction_world_xy": direction.tolist(),
                "initial_state_max_abs_delta": initial_state_delta,
                "initial_rgb_mean_absolute_difference": initial_rgb_l1,
            },
            "operator": {
                "id": "O4",
                "name": "foot_adhesion",
                "fidelity": "per-foot world-anchor force constraint",
                "region_xy_m": [list(point) for point in region.vertices_xy],
                "parameters": {
                    key: value
                    for key, value in adhesion_cfg.__dict__.items()
                    if key != "region"
                },
            },
            "protocol": {
                "forward_s": float(args.forward_s),
                "peel_pulse_s": float(args.peel_pulse_s),
                "post_peel_recovery_s": float(args.recovery_s),
                "reverse_retreat_s": float(args.reverse_s),
                "stop_s": float(args.stop_s),
                "control_dt_s": backend.dt,
                "rgb_fps": float(args.rgb_fps),
                "camera": "actual body-fixed Go2 front RTX RGB",
                "lighting": compiled["lighting_contract"],
            },
            "takes": summaries,
            "paired_consequence": {
                "definition": (
                    "paired route-progress and posture effects from first O4 attachment "
                    "through the end of the matched forward-command phase"
                ),
                "metrics": consequence,
                "thresholds": consequence_thresholds,
                "locomotion_channel_passed": locomotion_consequence,
                "posture_channel_passed": posture_consequence,
                "endpoint_max_progress_separation_m_report_only": (
                    nominal["max_progress_m"] - anomaly["max_progress_m"]
                ),
            },
            "terminal_outcome": {
                **terminal_outcome,
                "contract": {
                    "recoverable_path": (
                        "both takes complete the frozen horizon without a fall"
                    ),
                    "terminal_failure_path": (
                        "nominal completes without fall; O4 has a complete paired forward-effect "
                        "window, then an observed peel, then a captured terminal fall"
                    ),
                    "invalid": (
                        "nominal fall, missing attachment/effect/peel, fall before the completed "
                        "effect window, or an uncaptured/otherwise early termination"
                    ),
                },
                "sequence_diagnostics_report_only": {
                    "equal_sequence_length": nominal["steps"] == anomaly["steps"] == total_steps,
                    "equal_rgb_frame_count": nominal["rgb_frames"] == anomaly["rgb_frames"],
                },
            },
            "checks": checks,
            "passed": all(checks.values()),
            "dataset_status": (
                "calibration_pair_passed_not_formal_benchmark_evidence"
                if all(checks.values()) and args.protocol_role == "calibration"
                else "admitted_cross_scene_o4_pair_not_full_benchmark_coverage"
                if all(checks.values()) and args.protocol_role == "formal"
                else "pair_failed"
            ),
            "provenance": {
                "git": _git_state(REPO_ROOT),
                "command": " ".join(sys.argv),
                "python": sys.executable,
                "collector_sha256": _sha256(Path(__file__).resolve()),
                "backend_sha256": _sha256(REPO_ROOT / "kino_vla/sim/isaac_policy_backend.py"),
                "adhesion_model_sha256": _sha256(REPO_ROOT / "kino_vla/sim/adhesion.py"),
            },
            "isaac_scene_prim": scene_prim,
        }
        manifest_path = args.out / "pair_manifest.json"
        _write_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "passed": manifest["passed"],
                    "checks": checks,
                    "manifest": str(manifest_path),
                },
                indent=2,
            ),
            flush=True,
        )
        passed = bool(manifest["passed"])
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        closer = threading.Thread(target=simulation_app.close, daemon=True)
        closer.start()
        closer.join(timeout=8.0)
    return 0 if passed else 2


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
