from __future__ import annotations

import json
from pathlib import Path

from kino_vla.eval.visual_shell_go2_matrix import audit_go2_profile_matrix
from kino_vla.eval.visual_shell_preflight import sha256_file


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value), encoding="utf-8")
    return sha256_file(path)


def test_profile_matrix_keeps_development_result_out_of_registry(tmp_path: Path) -> None:
    episode = tmp_path / "episode.usda"
    episode_hash = _write(episode, "#usda 1.0\n")
    compiled = tmp_path / "compiled.json"
    compiled_hash = _write(
        compiled,
        {"passed": True, "scene_id": "forest", "files": {"episode.usda": episode_hash}},
    )
    script = tmp_path / "qa.py"
    script_hash = _write(script, "# qa\n")
    runs = []
    telemetry_hash = None
    for profile in ("a", "b", "c"):
        directory = tmp_path / profile
        telemetry = directory / "telemetry.jsonl"
        telemetry_hash = _write(telemetry, "same\n")
        captures = []
        for index in range(3):
            image = directory / f"{index}.png"
            image_hash = _write(image, f"image-{profile}-{index}")
            captures.append({"path": image.name, "sha256": image_hash})
        audit = {
            "telemetry": telemetry.name,
            "telemetry_sha256": telemetry_hash,
            "camera": {
                "profile": profile,
                "calibration_status": "engineering_pending_physical_fixture_calibration",
                "captures": captures,
                "max_rigid_mount_position_error_m": 0.0,
                "max_rigid_mount_orientation_error_rad": 0.0,
            },
            "compiled_audit_sha256": compiled_hash,
            "episode_usd_sha256": episode_hash,
            "passed": True,
            "checks": {"ok": True},
            "command_body_mps": [0.32, 0.0, 0.0],
            "route_tracking_controller": {
                "enabled": True,
                "cross_track_gain": 1.0,
                "heading_gain": 2.0,
                "lateral_command_limit_mps": 0.16,
                "yaw_rate_command_limit_radps": 0.6,
            },
            "route_progress_m": 0.5,
            "max_route_deviation_m": 0.01,
            "max_tilt_rad": 0.1,
            "min_base_height_m": 0.35,
        }
        audit_path = directory / "audit.json"
        runs.append(
            {"profile": profile, "audit": str(audit_path.relative_to(tmp_path)), "sha256": _write(audit_path, audit)}
        )
    history = {}
    for name, progress, tilt in (
        ("open_loop_failure", 0.5, 0.7),
        ("open_loop_repeat_failure", 0.5, 0.7),
        ("lower_speed_failure", 0.4, 0.8),
    ):
        path = tmp_path / f"{name}.json"
        history[name] = {
            "audit": path.name,
            "sha256": _write(
                path,
                {
                    "passed": False,
                    "route_progress_m": progress,
                    "max_route_deviation_m": 0.2,
                    "max_tilt_rad": tilt,
                },
            ),
        }
    history["interpretation"] = "preserved"
    config = {
        "schema_version": "kinofail.visual-shell-go2-profile-matrix.v1-development",
        "scene_id": "forest",
        "compiled_scene_audit": {"path": compiled.name, "sha256": compiled_hash},
        "episode_usd": {"path": episode.name, "sha256": episode_hash},
        "qa_script": {"path": script.name, "sha256": script_hash},
        "required_profiles": ["a", "b", "c"],
        "qa_runs": runs,
        "development_history": history,
        "required_controller": {
            "enabled": True,
            "command_speed_mps": 0.32,
            "cross_track_gain": 1.0,
            "heading_gain": 2.0,
            "lateral_command_limit_mps": 0.16,
            "yaw_rate_command_limit_radps": 0.6,
        },
    }
    result = audit_go2_profile_matrix(config, root=tmp_path)
    assert result["passed"] is True
    assert result["may_proceed_to_operator_development"] is True
    assert result["physical_fixture_camera_calibrated"] is False
    assert result["scene_registry_eligible"] is False
    assert result["counts_as_a0_a7_evidence"] is False
