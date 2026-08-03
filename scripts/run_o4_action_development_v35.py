#!/usr/bin/env python3
"""Execute every frozen v35 development case exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ISAAC_SETUP = Path("/home/eureka/nvidia/isaacsim/setup_conda_env.sh")
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
COLLECTOR = ROOT / "scripts/isaac_collect_o4_action_consequence_v35.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _collector_args(case: dict[str, Any], runtime: dict[str, Any]) -> list[str]:
    compiled = _resolve(case["prerequisites"]["terrain_compiled_audit"]["path"])
    return [
        str(COLLECTOR.relative_to(ROOT)),
        "--episode-usd", str(_resolve(case["episode_usd"])),
        "--compiled-audit", str(compiled),
        "--out", str(_resolve(case["output"])),
        "--material-id", case["material_id"],
        "--camera-profile", runtime["camera_profile"],
        "--seed", str(case["runtime_seed"]),
        "--horizon-steps", str(runtime["horizon_steps"]),
        "--minimum-decision-dwell-steps", str(runtime["minimum_decision_dwell_steps"]),
        "--decision-dwell-steps", str(runtime["maximum_decision_dwell_steps"]),
        "--decision-tilt-urgency-rad", str(runtime["decision_tilt_urgency_rad"]),
        "--decision-height-drop-urgency-m", str(runtime["decision_height_drop_urgency_m"]),
        "--decision-urgency-dwell-steps", str(runtime["decision_urgency_dwell_steps"]),
        "--maximum-backstep-steps", str(runtime["maximum_backstep_steps"]),
        "--target-recovery-distance-m", str(runtime["target_recovery_distance_m"]),
        "--minimum-recovery-distance-m", str(runtime["minimum_recovery_distance_m"]),
        "--adhesion-clearance-dwell-steps", str(runtime["adhesion_clearance_dwell_steps"]),
        "--forward-speed-mps", str(runtime["forward_speed_mps"]),
        "--backstep-speed-mps", str(runtime["backstep_speed_mps"]),
        "--emergency-backstep-speed-mps", str(runtime["emergency_backstep_speed_mps"]),
        "--maximum-postdecision-forward-penetration-m",
        str(runtime["maximum_postdecision_forward_penetration_m"]),
        "--max-route-deviation-m", str(runtime["maximum_route_deviation_m"]),
        "--tangential-force-cap-n", str(runtime["tangential_force_cap_n"]),
        "--normal-force-cap-n", str(runtime["normal_force_cap_n"]),
        "--peel-height-m", str(runtime["peel_height_m"]),
        "--unload-steps-to-peel", str(runtime["unload_steps_to_peel"]),
        "--reattach-cooldown-steps", str(runtime["reattach_cooldown_steps"]),
        "--max-active-feet", str(runtime["max_active_feet"]),
        "--cross-track-gain-per-s", str(runtime["cross_track_gain_per_s"]),
        "--lateral-velocity-damping", str(runtime["lateral_velocity_damping"]),
        "--heading-gain-per-s", str(runtime["heading_gain_per_s"]),
        "--lateral-limit-mps", str(runtime["lateral_limit_mps"]),
        "--yaw-rate-limit-radps", str(runtime["yaw_rate_limit_radps"]),
        "--headless",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    preflight_path = args.preflight.resolve()
    config = _json(config_path)
    preflight = _json(preflight_path)
    if (
        preflight.get("passed") is not True
        or preflight.get("execution_authorized") is not True
        or preflight.get("config_sha256") != _sha256(config_path)
    ):
        raise RuntimeError("v35 preflight did not authorize this exact config")

    output_root = _resolve(config["operator_output_root"])
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse operator output root: {output_root}")
    audit_root = output_root / "launcher_audits"
    audit_root.mkdir(parents=True)
    rows = []
    for index, case in enumerate(config["cases"]):
        output = _resolve(case["output"])
        if output.exists():
            raise FileExistsError(f"case output appeared before execution: {output}")
        native_args = _collector_args(case, config["runtime"])
        command = [
            "/bin/bash", "-lc",
            'source "$1" >/dev/null 2>&1 && exec "$2" "${@:3}"',
            "v35-o4-development-launcher", str(ISAAC_SETUP), str(PYTHON), *native_args,
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        manifest_path = output / "pair_manifest.json"
        manifest = _json(manifest_path) if manifest_path.is_file() else {}
        row = {
            "schema_version": "kinofail.o4-action-consequence-v35-launcher-audit.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "case_index": index,
            "scene_id": case["scene_id"],
            "returncode": completed.returncode,
            "collector_terminal_returncode": completed.returncode in {0, 2},
            "manifest": str(manifest_path),
            "manifest_exists": manifest_path.is_file(),
            "manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
            "manifest_passed": manifest.get("passed"),
            "native_arguments": native_args,
        }
        audit_path = audit_root / f"{index:02d}_{case['scene_id']}.json"
        audit_path.write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        rows.append(row)

    result = {
        "schema_version": "kinofail.o4-action-consequence-v35-runner-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "preflight": str(preflight_path),
        "preflight_sha256": _sha256(preflight_path),
        "attempted_scene_ids": [row["scene_id"] for row in rows],
        "all_cases_attempted": len(rows) == len(config["cases"]),
        "all_cases_terminal": all(row["collector_terminal_returncode"] for row in rows),
        "all_manifests_present": all(row["manifest_exists"] for row in rows),
        "cases": rows,
    }
    runner_path = output_root / "runner_audit.json"
    runner_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "runner_audit": str(runner_path),
        "all_cases_attempted": result["all_cases_attempted"],
        "all_cases_terminal": result["all_cases_terminal"],
        "all_manifests_present": result["all_manifests_present"],
    }, indent=2))
    return 0 if result["all_cases_terminal"] and result["all_manifests_present"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
