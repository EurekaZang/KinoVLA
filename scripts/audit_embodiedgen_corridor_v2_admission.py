#!/usr/bin/env python3
"""Independently admit a robust-corridor v2 scene after RTX and articulated-Go2 QA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled-audit", type=Path, required=True)
    parser.add_argument("--rtx-audit", type=Path, required=True)
    parser.add_argument("--go2-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    compiled_path = args.compiled_audit.resolve()
    rtx_path = args.rtx_audit.resolve()
    go2_path = args.go2_audit.resolve()
    compiled = _json(compiled_path)
    rtx = _json(rtx_path)
    go2 = _json(go2_path)
    route = compiled.get("route", {})
    episode_name = next(
        (name for name in compiled.get("files", {}) if name.startswith("episode_v2")),
        "",
    )
    episode_path = compiled_path.parent / episode_name
    base_path = Path(compiled.get("base_compiled_audit", "")).resolve()
    robot_radius = float(route.get("robot_radius_m", math.nan))
    physical_margin = float(route.get("safety_margin_m", math.nan))
    tracking_budget = float(
        route.get("maximum_admissible_tracking_error_m", math.nan)
    )
    robust_clearance = float(route.get("robust_centerline_clearance_m", math.nan))
    measured_clearance = float(route.get("measured_min_clearance_m", math.nan))
    checks = {
        "compiled_schema_v2": compiled.get("schema_version")
        == "kinofail.embodiedgen-compiled-scene.v2",
        "compiled_passed": compiled.get("passed") is True,
        "base_audit_present": base_path.is_file(),
        "base_audit_hash_verified": base_path.is_file()
        and compiled.get("base_compiled_audit_sha256") == _sha256(base_path),
        "episode_present": bool(episode_name) and episode_path.is_file(),
        "episode_hash_verified": bool(episode_name)
        and episode_path.is_file()
        and compiled["files"].get(episode_name) == _sha256(episode_path),
        "robust_clearance_contract": math.isclose(
            robust_clearance,
            robot_radius + physical_margin + tracking_budget,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ),
        "measured_clearance_meets_robust_contract": measured_clearance + 1.0e-6
        >= robust_clearance,
        "tracking_budget_positive": tracking_budget > 0.0,
        "rtx_passed": rtx.get("passed") is True,
        "rtx_binds_compiled": rtx.get("compiled_audit_sha256")
        == _sha256(compiled_path),
        "rtx_binds_episode": episode_path.is_file()
        and rtx.get("episode_usd_sha256") == _sha256(episode_path),
        "go2_passed": go2.get("passed") is True,
        "go2_binds_compiled": go2.get("compiled_audit_sha256")
        == _sha256(compiled_path),
        "go2_binds_episode": episode_path.is_file()
        and go2.get("episode_usd_sha256") == _sha256(episode_path),
        "go2_tracking_within_frozen_budget": float(
            go2.get("max_route_deviation_m", math.inf)
        )
        <= tracking_budget,
        "go2_stable_tilt": float(go2.get("max_tilt_rad", math.inf)) <= 0.55,
        "go2_stable_height": float(go2.get("min_base_height_m", -math.inf))
        >= 0.20,
        "go2_no_fall": go2.get("checks", {}).get("robot_did_not_fall") is True,
        "actual_front_camera": go2.get("camera", {}).get("role")
        == "actual body-fixed Go2 front RTX RGB",
    }
    payload = {
        "schema_version": "kinofail.embodiedgen-corridor-v2-admission-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "scene_id": compiled.get("scene_id"),
        "compiled_audit": {"path": str(compiled_path), "sha256": _sha256(compiled_path)},
        "rtx_audit": {"path": str(rtx_path), "sha256": _sha256(rtx_path)},
        "go2_audit": {"path": str(go2_path), "sha256": _sha256(go2_path)},
        "episode": {
            "path": str(episode_path),
            "sha256": _sha256(episode_path) if episode_path.is_file() else None,
        },
        "clearance_contract_m": {
            "robot_radius": robot_radius,
            "physical_safety_margin": physical_margin,
            "maximum_tracking_error": tracking_budget,
            "required_centerline_clearance": robust_clearance,
            "measured_centerline_clearance": measured_clearance,
            "observed_go2_tracking_error": go2.get("max_route_deviation_m"),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "admission_state": (
            "corridor_v2_admitted_for_operator_protocol_freeze"
            if all(checks.values())
            else "corridor_v2_rejected"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

