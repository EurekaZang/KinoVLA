#!/usr/bin/env python3
"""Collect realistic A4 Continue/recovery consequences with actual Go2 actions.

This is intentionally a one-pass scale collector. Action branches are frozen, route-
relative, and begin only after an operator-specific physical-onset dwell. It rejects
missing onset and unequal paired prefixes, but never rejects an unfavorable outcome.
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _prefix_hash(rows: list[dict[str, Any]], decision_step: int) -> str:
    digest = hashlib.sha256()
    for row in rows:
        if int(row["step"]) > decision_step:
            break
        compact = {
            key: row[key]
            for key in (
                "step", "time_s", "progress_m", "lateral_m", "position_xy_m",
                "heading_rad", "velocity_body_mps", "base_height_m", "tilt_rad",
                "slip_ratio", "effort_ratio", "support_ratio", "fallen",
                "command_body", "operator_telemetry",
            )
        }
        digest.update(json.dumps(compact, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _operator_engaged(operator: str, telemetry: dict[str, Any]) -> bool:
    if operator == "O2_compliance":
        return any(int(row.get("contact_steps", 0)) > 0 for row in telemetry.get("feet", []))
    if operator == "O4_tether":
        return int(telemetry.get("total_attachment_cycles", 0)) > 0
    if operator == "O5_payload":
        return float(telemetry.get("payload_kg", 0.0)) > 0.0
    if operator == "O8_invisible_collider":
        return any(bool(row.get("collision_requested")) for row in telemetry.get("obstacles", []))
    if operator == "O9_high_centering":
        return any(int(row.get("measurement_steps", 0)) > 0 for row in telemetry.get("regions", []))
    raise ValueError(operator)


def _decision_ready(operator: str, telemetry: dict[str, Any], engaged_dwell: int, time_s: float) -> bool:
    if operator == "O5_payload":
        return time_s >= 2.2 and _operator_engaged(operator, telemetry)
    required = {
        "O2_compliance": 10,
        "O4_tether": 10,
        "O8_invisible_collider": 3,
        "O9_high_centering": 5,
    }[operator]
    return engaged_dwell >= required


def _terminal_success(
    operator: str,
    *,
    fell: bool,
    final_speed_mps: float,
    final_progress_m: float,
    region_far_m: float,
    action: str,
    decision_progress_m: float,
    minimum_postdecision_progress_m: float,
    o4_active_feet: int,
) -> bool:
    if fell:
        return False
    if operator == "O4_tether":
        return (
            o4_active_feet == 0
            and decision_progress_m - minimum_postdecision_progress_m >= 0.15
        )
    if operator == "O5_payload":
        return action == "hold_and_request" and final_speed_mps < 0.08
    return final_progress_m >= region_far_m + 0.20


def main() -> int:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--schedule", required=True)
    preliminary.add_argument("--protocol", required=True)
    preliminary.add_argument("--scene-registry", required=True)
    preliminary.add_argument("--scene", required=True)
    preliminary.add_argument("--out", required=True)
    pre, _ = preliminary.parse_known_args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", default=pre.schedule)
    parser.add_argument("--protocol", default=pre.protocol)
    parser.add_argument("--scene-registry", default=pre.scene_registry)
    parser.add_argument("--scene", default=pre.scene)
    parser.add_argument("--out", default=pre.out)
    parser.add_argument("--horizon-steps", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = False
    app = AppLauncher(args).app
    passed = False
    try:
        import yaml

        from kino_vla.sim.isaac_o4_v3_backend import IsaacPolicyBackendO4V3
        from kino_vla.sim.realistic_route_controller_v3 import DampedRouteControllerV3
        from kino_vla.sim.realistic_route_protocol_v2 import scene_route_binding
        from kino_vla.utils.config import load_config
        from scripts.isaac_collect_kinofail_realistic_pair_v1 import _install_operator, _telemetry

        schedule_path = (ROOT / args.schedule).resolve()
        protocol_path = (ROOT / args.protocol).resolve()
        registry_path = (ROOT / args.scene_registry).resolve()
        out_root = (ROOT / args.out).resolve()
        protocol = _json(protocol_path)
        for key, actual in (
            ("schedule_sha256", _sha(schedule_path)),
            ("collector_sha256", _sha(Path(__file__).resolve())),
        ):
            if protocol.get(key) != actual:
                raise RuntimeError(f"protocol {key} mismatch: {protocol.get(key)} != {actual}")
        cases = [row for row in _jsonl(schedule_path) if row["scene_cluster"] == args.scene]
        if len(cases) != 10:
            raise RuntimeError(f"expected 10 cases for scene, got {len(cases)}")
        registry = _json(registry_path)
        registry_rows = [row for row in registry["scenes"] if row["scene_id"] == args.scene]
        if len(registry_rows) != 1:
            raise RuntimeError("scene registry binding is not unique")
        scene_row = registry_rows[0]
        episode_path = (ROOT / scene_row["episode_usd"]).resolve()
        compiled_path = (ROOT / scene_row["compiled_audit"]).resolve()
        if _sha(episode_path) != scene_row["episode_sha256"] or _sha(compiled_path) != scene_row["compiled_audit_sha256"]:
            raise RuntimeError("scene registry hash mismatch")
        compiled = _json(compiled_path)
        frame = scene_route_binding(compiled).frame
        config = load_config("sim/go2_skeleton.yaml")
        backend = IsaacPolicyBackendO4V3(
            config, frame.point(0.0, 0.0), frame.heading_rad
        )
        backend.load_realistic_scene(str(episode_path))
        controller = DampedRouteControllerV3(
            cross_track_gain_per_s=2.0,
            lateral_velocity_damping=1.0,
            heading_gain_per_s=2.0,
            lateral_limit_mps=0.45,
            yaw_rate_limit_radps=0.60,
        )
        result_dir = out_root / args.scene
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / "results.jsonl"
        prior: dict[tuple[str, str], dict[str, Any]] = {}
        if args.resume and result_path.exists():
            for row in _jsonl(result_path):
                prior[(row["case_id"], row["action"])] = row
        elif result_path.exists():
            raise FileExistsError(f"refusing to overwrite {result_path}")

        written: list[dict[str, Any]] = list(prior.values())
        for case in cases:
            record = dict(case["source_record"])
            operator_name = str(case["operator"])
            pair_rows: dict[str, dict[str, Any]] = {}
            for action in case["actions"]:
                key = (case["case_id"], action)
                if key in prior:
                    pair_rows[action] = prior[key]
                    continue
                seed = int(case["reset_seed"])
                obs = backend.deep_reset(seed)
                operator, region = _install_operator(backend, record, frame, seed)
                engaged_dwell = 0
                decision_step: int | None = None
                decision_progress = float("nan")
                action_phase = "shared_prefix"
                detour_phase = "back"
                detour_sign = 1.0 if int(case["replicate"]) == 0 else -1.0
                min_post_progress = float("inf")
                rows: list[dict[str, Any]] = []
                final_telemetry: dict[str, Any] = {}
                for step in range(args.horizon_steps):
                    if operator is not None:
                        operator.on_step(backend, float(obs.t))
                    telemetry = _telemetry(backend, operator_name, operator)
                    engaged = _operator_engaged(operator_name, telemetry)
                    engaged_dwell = engaged_dwell + 1 if engaged else 0
                    progress, lateral = frame.project(obs.pos)
                    if decision_step is None and _decision_ready(
                        operator_name, telemetry, engaged_dwell, float(obs.t)
                    ):
                        decision_step = step
                        decision_progress = float(progress)
                    post = decision_step is not None and step > decision_step
                    speed = 0.32
                    target_lateral = 0.0
                    posture: float | None = None
                    if post and action != "continue":
                        if operator_name == "O2_compliance":
                            speed, posture, action_phase = 0.16, 0.30, "slow_high_step"
                        elif operator_name == "O4_tether":
                            retreat = decision_progress - float(progress)
                            speed = -0.32 if retreat < 0.28 else 0.0
                            action_phase = "backstep" if speed < 0.0 else "released_hold"
                        elif operator_name == "O5_payload":
                            speed, action_phase = 0.0, "hold_and_request"
                        elif operator_name == "O8_invisible_collider":
                            retreat = decision_progress - float(progress)
                            if detour_phase == "back" and retreat >= 0.25:
                                detour_phase = "lateral"
                            if detour_phase == "back":
                                speed = -0.24
                            elif detour_phase == "lateral":
                                speed, target_lateral = 0.10, detour_sign * 0.72
                                if abs(float(lateral)) >= 0.62:
                                    detour_phase = "cross"
                            elif detour_phase == "cross":
                                speed, target_lateral = 0.24, detour_sign * 0.72
                                if float(progress) >= 2.05:
                                    detour_phase = "return"
                            else:
                                speed, target_lateral = 0.24, 0.0
                            action_phase = f"detour_{detour_phase}"
                        elif operator_name == "O9_high_centering":
                            speed, posture, action_phase = 0.16, 0.33, "raise_body_slow_cross"
                    elif post:
                        action_phase = "continue"
                    backend.set_posture(posture, 1.5 if posture is not None else 1.0)
                    command, control = controller.command(
                        frame,
                        position_xy_m=np.asarray(obs.pos),
                        heading_rad=float(obs.heading),
                        velocity_body_xy_mps=np.asarray(obs.vel_body[:2]),
                        forward_speed_mps=speed,
                        target_lateral_offset_m=target_lateral,
                    )
                    row = {
                        "step": step,
                        "time_s": float(obs.t),
                        "action_phase": action_phase,
                        "progress_m": float(progress),
                        "lateral_m": float(lateral),
                        "position_xy_m": np.asarray(obs.pos).tolist(),
                        "heading_rad": float(obs.heading),
                        "velocity_body_mps": np.asarray(obs.vel_body).tolist(),
                        "base_height_m": float(obs.base_height),
                        "tilt_rad": float(obs.tilt),
                        "slip_ratio": float(obs.slip_ratio),
                        "effort_ratio": float(obs.effort_ratio),
                        "support_ratio": float(obs.support_ratio),
                        "fallen": bool(obs.fallen),
                        "command_body": command.tolist(),
                        "route_controller": control,
                        "operator_telemetry": telemetry,
                    }
                    rows.append(row)
                    if post:
                        min_post_progress = min(min_post_progress, float(progress))
                    final_telemetry = telemetry
                    obs = backend.step(command)
                    new_progress, _ = frame.project(obs.pos)
                    if obs.fallen:
                        break
                    if operator_name in {"O2_compliance", "O8_invisible_collider", "O9_high_centering"} and new_progress >= frame.route_length_m - 0.20:
                        break
                    if post and operator_name in {"O4_tether", "O5_payload"} and step - decision_step >= 160:
                        break
                if decision_step is None:
                    raise RuntimeError(f"operator never reached decision onset: {case['case_id']}/{action}")
                progress, lateral = frame.project(obs.pos)
                final_speed = float(np.linalg.norm(np.asarray(obs.vel_body[:2])))
                if not np.isfinite(min_post_progress):
                    min_post_progress = float(progress)
                o4_active = int(final_telemetry.get("active_feet", 0)) if operator_name == "O4_tether" else 0
                success = _terminal_success(
                    operator_name,
                    fell=bool(obs.fallen),
                    final_speed_mps=final_speed,
                    final_progress_m=float(progress),
                    region_far_m=1.80,
                    action=action,
                    decision_progress_m=decision_progress,
                    minimum_postdecision_progress_m=min_post_progress,
                    o4_active_feet=o4_active,
                )
                prefix = _prefix_hash(rows, decision_step)
                terminal_cost = (
                    100.0 * float(obs.fallen)
                    + 30.0 * float(not success)
                    + 10.0 * max(0.0, abs(float(obs.tilt)) - 0.35)
                    + max(0.0, frame.route_length_m - float(progress))
                )
                result = {
                    "schema_version": "kinofail.realistic-a4-actual-action-result.v1",
                    "created_utc": datetime.now(UTC).isoformat(),
                    "protocol_id": protocol["protocol_id"],
                    "case_id": case["case_id"],
                    "scene_cluster": case["scene_cluster"],
                    "domain": case["domain"],
                    "operator": operator_name,
                    "replicate": case["replicate"],
                    "reset_seed": seed,
                    "action": action,
                    "decision_step": decision_step,
                    "decision_time_s": rows[decision_step]["time_s"],
                    "decision_progress_m": decision_progress,
                    "predecision_rows_sha256": prefix,
                    "operator_engaged": _operator_engaged(operator_name, final_telemetry),
                    "steps": len(rows),
                    "fell": bool(obs.fallen),
                    "success": success,
                    "terminal_cost": float(terminal_cost),
                    "final_progress_m": float(progress),
                    "final_lateral_m": float(lateral),
                    "final_speed_mps": final_speed,
                    "final_tilt_rad": float(obs.tilt),
                    "minimum_postdecision_progress_m": min_post_progress,
                    "o4_final_active_feet": o4_active,
                    "telemetry_rows": rows,
                }
                pair_rows[action] = result
                written.append(result)
                with result_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, sort_keys=True) + "\n")
                print(
                    json.dumps(
                        {key: result[key] for key in (
                            "case_id", "action", "success", "fell", "terminal_cost",
                            "decision_step", "steps", "final_progress_m",
                        )}, sort_keys=True
                    ), flush=True
                )
            if len(pair_rows) == 2:
                values = list(pair_rows.values())
                if values[0]["decision_step"] != values[1]["decision_step"]:
                    raise RuntimeError(f"paired decision step mismatch: {case['case_id']}")
                if values[0]["predecision_rows_sha256"] != values[1]["predecision_rows_sha256"]:
                    raise RuntimeError(f"paired predecision hash mismatch: {case['case_id']}")
        expected = {(row["case_id"], action) for row in cases for action in row["actions"]}
        complete = expected <= {(row["case_id"], row["action"]) for row in written}
        summary = {
            "schema_version": "kinofail.realistic-a4-scene-summary.v1",
            "scene_cluster": args.scene,
            "protocol_id": protocol["protocol_id"],
            "passed": complete,
            "expected_episodes": len(expected),
            "completed_episodes": len(expected & {(row["case_id"], row["action"]) for row in written}),
            "unfavorable_outcomes_are_retained": True,
        }
        (result_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        passed = complete
    finally:
        sys.stdout.flush()
        closer = threading.Thread(target=app.close, daemon=True)
        closer.start()
        closer.join(timeout=15.0)
    os._exit(0 if passed else 2)


if __name__ == "__main__":
    main()
