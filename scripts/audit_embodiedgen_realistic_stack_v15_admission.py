#!/usr/bin/env python3
"""Fail-closed visual-plus-physics admission for one realistic EmbodiedGen scene."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "kinofail.embodiedgen-realistic-stack-v15-admission.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _evidence(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-preflight", type=Path, required=True)
    parser.add_argument("--route-compiled-audit", type=Path, required=True)
    parser.add_argument("--rtx-audit", type=Path, required=True)
    parser.add_argument("--terrain-compiled-audit", type=Path, required=True)
    parser.add_argument("--terrain-offline-audit", type=Path, required=True)
    parser.add_argument("--go2-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "source_preflight": args.source_preflight.resolve(),
        "route_compiled_audit": args.route_compiled_audit.resolve(),
        "rtx_audit": args.rtx_audit.resolve(),
        "terrain_compiled_audit": args.terrain_compiled_audit.resolve(),
        "terrain_offline_audit": args.terrain_offline_audit.resolve(),
        "go2_audit": args.go2_audit.resolve(),
    }
    records = {name: _json(path) for name, path in paths.items()}
    source = records["source_preflight"]
    route = records["route_compiled_audit"]
    rtx = records["rtx_audit"]
    terrain = records["terrain_compiled_audit"]
    offline = records["terrain_offline_audit"]
    go2 = records["go2_audit"]

    route_path = paths["route_compiled_audit"]
    terrain_path = paths["terrain_compiled_audit"]
    route_episode = route_path.parent / "episode_v4.usda"
    terrain_episode = terrain_path.parent / "episode_terrain_v2.usda"
    material = terrain.get("physics_contract", {}).get("nominal_floor_material", {})
    go2_checks = go2.get("checks", {})
    checks: dict[str, bool] = {
        "source_preflight_passed": source.get("passed") is True,
        "route_surface_compile_passed": route.get("passed") is True,
        "rtx_qa_passed": rtx.get("passed") is True,
        "terrain_compile_passed": terrain.get("passed") is True,
        "terrain_offline_audit_passed": offline.get("passed") is True,
        "go2_qa_passed": go2.get("passed") is True,
        "scene_id_consistent": len(
            {
                str(value)
                for value in (
                    source.get("scene_id"),
                    route.get("scene_id"),
                    terrain.get("scene_id"),
                    offline.get("scene_id"),
                    go2.get("scene_id"),
                )
                if value is not None
            }
        )
        == 1,
        "rtx_binds_route_audit": rtx.get("compiled_audit_sha256") == _sha256(route_path),
        "rtx_binds_route_episode": route_episode.is_file()
        and route.get("files", {}).get(route_episode.name) == _sha256(route_episode)
        and rtx.get("episode_usd_sha256") == _sha256(route_episode),
        "terrain_binds_route_audit": terrain.get("base_compiled_audit_sha256")
        == _sha256(route_path),
        "offline_binds_terrain_audit": offline.get("compiled_audit", {}).get("sha256")
        == _sha256(terrain_path),
        "terrain_episode_hash_verified": terrain_episode.is_file()
        and terrain.get("files", {}).get(terrain_episode.name) == _sha256(terrain_episode),
        "go2_binds_terrain_audit": go2.get("compiled_audit_sha256") == _sha256(terrain_path),
        "go2_binds_terrain_episode": terrain_episode.is_file()
        and go2.get("episode_usd_sha256") == _sha256(terrain_episode),
        "kino_owned_floor_material": material.get("owned_by") == "Kino-Fail"
        and material.get("api_schema") == "PhysicsMaterialAPI",
        "nominal_friction_exact": abs(float(material.get("static_friction", -1.0)) - 0.8)
        <= 1.0e-9
        and abs(float(material.get("dynamic_friction", -1.0)) - 0.6) <= 1.0e-9,
        "single_ground_contact_authority": go2_checks.get("single_ground_contact_authority")
        is True,
        "actual_front_camera_valid": go2_checks.get("three_front_frames") is True
        and go2_checks.get("front_frames_non_degenerate") is True
        and go2_checks.get("front_camera_position_rigid_mount") is True
        and go2_checks.get("front_camera_orientation_rigid_mount") is True,
        "route_progress_and_stability_valid": go2_checks.get(
            "target_progress_reached_within_step_budget"
        )
        is True
        and go2_checks.get("robot_did_not_fall") is True
        and go2_checks.get("route_tracking_within_planned_clearance") is True,
        "development_boundary_retained": terrain.get("development_only") is True
        and terrain.get("counts_as_a0_a7_evidence") is False,
    }
    passed = all(checks.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "scene_id": terrain.get("scene_id"),
        "passed": passed,
        "admission_state": (
            "realistic_visual_physics_stack_admitted_for_nominal_policy_confirmation"
            if passed
            else "realistic_visual_physics_stack_rejected"
        ),
        "checks": checks,
        "evidence": {name: _evidence(path) for name, path in paths.items()},
        "measurements": {
            "route_progress_m": go2.get("route_progress_m"),
            "max_route_deviation_m": go2.get("max_route_deviation_m"),
            "min_base_height_m": go2.get("min_base_height_m"),
            "max_tilt_rad": go2.get("max_tilt_rad"),
            "front_camera_profile": go2.get("camera", {}).get("profile"),
            "front_camera_capture_count": len(go2.get("camera", {}).get("captures", [])),
        },
        "counts_as_realistic_corpus_evidence": False,
        "counts_as_a0_a7_evidence": False,
        "remaining_gates": [
            "frozen_nominal_policy_confirmation",
            "all_11_operator_recalibration",
            "registered_counterfactual_corpus_collection",
            "new_a0_training_and_inference",
            "new_a1_a7_replication",
        ],
    }
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite admission audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "scene_id": result["scene_id"], "passed": passed}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
