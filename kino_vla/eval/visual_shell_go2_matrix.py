"""Aggregate the three engineering Go2-front profiles without pre-admitting a scene."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from kino_vla.eval.visual_shell_preflight import sha256_file


SCHEMA_VERSION = "kinofail.visual-shell-go2-profile-matrix-audit.v1"


def _resolve(root: Path, path_value: str, expected_hash: str, label: str) -> Path:
    path = (root / path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"stale {label}: {path}")
    return path


def audit_go2_profile_matrix(config: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    if config.get("schema_version") != "kinofail.visual-shell-go2-profile-matrix.v1-development":
        raise ValueError("unsupported Go2 profile-matrix contract")
    compiled_path = _resolve(
        root,
        config["compiled_scene_audit"]["path"],
        config["compiled_scene_audit"]["sha256"],
        "compiled scene audit",
    )
    episode_path = _resolve(
        root,
        config["episode_usd"]["path"],
        config["episode_usd"]["sha256"],
        "episode USD",
    )
    _resolve(
        root,
        config["qa_script"]["path"],
        config["qa_script"]["sha256"],
        "Go2 QA script",
    )
    compiled = json.loads(compiled_path.read_text(encoding="utf-8"))
    if compiled.get("passed") is not True or compiled.get("scene_id") != config["scene_id"]:
        raise ValueError("compiled scene is not the passed contracted scene")
    if compiled["files"].get(episode_path.name) != sha256_file(episode_path):
        raise ValueError("compiled audit does not bind the contracted episode")

    required = set(config["required_profiles"])
    controller = config["required_controller"]
    rows = []
    telemetry_hashes = set()
    for spec in config["qa_runs"]:
        path = _resolve(root, spec["audit"], spec["sha256"], f"{spec['profile']} audit")
        audit = json.loads(path.read_text(encoding="utf-8"))
        telemetry_path = path.parent / audit["telemetry"]
        if not telemetry_path.is_file() or sha256_file(telemetry_path) != audit["telemetry_sha256"]:
            raise ValueError(f"stale telemetry for {spec['profile']}")
        telemetry_hashes.add(audit["telemetry_sha256"])
        captures_valid = True
        for capture in audit["camera"]["captures"]:
            image_path = path.parent / capture["path"]
            captures_valid &= image_path.is_file() and sha256_file(image_path) == capture["sha256"]
        observed_controller = audit["route_tracking_controller"]
        controller_matches = (
            observed_controller["enabled"] is controller["enabled"]
            and audit["command_body_mps"][0] == controller["command_speed_mps"]
            and observed_controller["cross_track_gain"] == controller["cross_track_gain"]
            and observed_controller["heading_gain"] == controller["heading_gain"]
            and observed_controller["lateral_command_limit_mps"]
            == controller["lateral_command_limit_mps"]
            and observed_controller["yaw_rate_command_limit_radps"]
            == controller["yaw_rate_command_limit_radps"]
        )
        checks = {
            "profile_matches": audit["camera"]["profile"] == spec["profile"],
            "engineering_calibration_status_explicit": audit["camera"].get(
                "calibration_status"
            )
            == "engineering_pending_physical_fixture_calibration",
            "compiled_scene_matches": audit["compiled_audit_sha256"]
            == config["compiled_scene_audit"]["sha256"],
            "episode_matches": audit["episode_usd_sha256"] == config["episode_usd"]["sha256"],
            "qa_passed": audit["passed"] is True,
            "all_runtime_checks_passed": all(audit["checks"].values()),
            "controller_matches": controller_matches,
            "three_hash_verified_captures": len(audit["camera"]["captures"]) == 3
            and captures_valid,
        }
        rows.append(
            {
                "profile": spec["profile"],
                "audit": str(path),
                "audit_sha256": sha256_file(path),
                "telemetry_sha256": audit["telemetry_sha256"],
                "route_progress_m": audit["route_progress_m"],
                "max_route_deviation_m": audit["max_route_deviation_m"],
                "max_tilt_rad": audit["max_tilt_rad"],
                "min_base_height_m": audit["min_base_height_m"],
                "max_camera_position_error_m": audit["camera"][
                    "max_rigid_mount_position_error_m"
                ],
                "max_camera_orientation_error_rad": audit["camera"][
                    "max_rigid_mount_orientation_error_rad"
                ],
                "checks": checks,
                "passed": all(checks.values()),
            }
        )

    history_rows = {}
    for name in ("open_loop_failure", "open_loop_repeat_failure", "lower_speed_failure"):
        spec = config["development_history"][name]
        path = _resolve(root, spec["audit"], spec["sha256"], name)
        audit = json.loads(path.read_text(encoding="utf-8"))
        history_rows[name] = {
            "audit": str(path),
            "audit_sha256": sha256_file(path),
            "passed": audit["passed"],
            "route_progress_m": audit["route_progress_m"],
            "max_route_deviation_m": audit["max_route_deviation_m"],
            "max_tilt_rad": audit["max_tilt_rad"],
        }
    observed_profiles = {row["profile"] for row in rows}
    checks = {
        "required_profile_set_exact": observed_profiles == required and len(rows) == len(required),
        "all_profiles_passed": all(row["passed"] for row in rows),
        "physics_telemetry_identical_across_profiles": len(telemetry_hashes) == 1,
        "open_loop_failure_preserved": history_rows["open_loop_failure"]["passed"] is False,
        "open_loop_failure_reproduced": history_rows["open_loop_repeat_failure"]["passed"]
        is False
        and history_rows["open_loop_failure"]["route_progress_m"]
        == history_rows["open_loop_repeat_failure"]["route_progress_m"]
        and history_rows["open_loop_failure"]["max_tilt_rad"]
        == history_rows["open_loop_repeat_failure"]["max_tilt_rad"],
        "lower_speed_failure_preserved": history_rows["lower_speed_failure"]["passed"] is False,
    }
    passed = all(checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "scene_id": config["scene_id"],
        "development_only": True,
        "profiles": rows,
        "development_history": {
            **history_rows,
            "interpretation": config["development_history"]["interpretation"],
        },
        "checks": checks,
        "passed": passed,
        "may_proceed_to_operator_development": passed,
        "physical_fixture_camera_calibrated": False,
        "independent_human_review_completed": False,
        "scene_registry_eligible": False,
        "counts_as_a0_a7_evidence": False,
        "remaining_gates": [
            "physical_go2_fixture_camera_calibration",
            "independent_human_scene_review",
            "operator_capability_audit",
            "scene_registry_binding",
            "paired_realistic_a0_a7_collection",
        ],
        "interpretation": (
            "All three frozen engineering camera profiles and route-tracked Go2 runs passed, "
            "with byte-identical physics telemetry across camera profiles. This authorizes "
            "operator development only; it is not physical camera calibration, scene admission, "
            "or realistic A0-A7 evidence."
        ),
    }
