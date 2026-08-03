#!/usr/bin/env python3
"""Audit the v9 reset hypothesis without relabeling the failed nominal lane."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_embodiedgen_fixed_joint_reset_v9_calibration_v1.json",
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_fixed_joint_reset_v9_calibration_v1/preflight_audit.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_fixed_joint_reset_v9_calibration_v1/kitchen31/o2_nominal_attempt1/lane_manifest.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_fixed_joint_reset_v9_calibration_v1/postrun_audit.json",
    )
    args = parser.parse_args()
    config = _json(args.config.resolve())
    preflight = _json(args.preflight.resolve())
    manifest = _json(args.manifest.resolve())
    baseline_path = _resolve(config["calibration_basis"]["v5_o2_nominal_manifest"]["path"])
    baseline = _json(baseline_path)
    reset = manifest.get("inference_reset_contract", {})
    checks = {
        "preflight_passed_before_run": preflight.get("passed") is True,
        "manifest_is_retained_failure": manifest.get("passed") is False,
        "only_tracking_gate_failed": {
            key for key, value in manifest.get("checks", {}).items() if value is not True
        }
        == {"full_lane_tracking_within_admitted_budget"},
        "nominal_completed_without_fall": manifest.get("checks", {}).get(
            "nominal_route_completed"
        )
        is True
        and manifest.get("checks", {}).get("nominal_robot_not_fallen") is True,
        "runtime_go2_reset_was_already_fixed": reset.get(
            "removed_hidden_position_scale_range"
        )
        == [1.0, 1.0]
        and reset.get("removed_hidden_velocity_scale_range") == [0.0, 0.0],
        "trajectory_matches_v5_tracking_failure": abs(
            float(
                manifest.get("measurements", {}).get(
                    "maximum_absolute_route_lateral_offset_m", -1.0
                )
            )
            - float(
                baseline.get("measurements", {}).get(
                    "maximum_absolute_route_lateral_offset_m", 1.0
                )
            )
        )
        <= 0.001,
        "source_hashes_match_frozen_run": manifest.get("provenance", {}).get(
            "collector_engine_sha256"
        )
        == config["source_freeze"]["collector_engine"]["sha256"]
        and manifest.get("provenance", {}).get("backend_sha256")
        == config["source_freeze"]["backend"]["sha256"]
        and manifest.get("provenance", {}).get("script_sha256")
        == config["source_freeze"]["collector_wrapper"]["sha256"],
    }
    hypothesis_supported = False
    result = {
        "schema_version": "kinofail.embodiedgen-fixed-joint-reset-v9-postrun.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "hypothesis": "hidden_seed_dependent_joint_reset_caused_o2_nominal_instability",
        "hypothesis_supported": hypothesis_supported,
        "calibration_outcome": "falsified_effective_go2_reset_was_already_fixed",
        "observed_pre_freeze_position_scale_range": reset.get(
            "removed_hidden_position_scale_range"
        ),
        "v5_maximum_absolute_route_lateral_offset_m": baseline.get(
            "measurements", {}
        ).get("maximum_absolute_route_lateral_offset_m"),
        "v9_maximum_absolute_route_lateral_offset_m": manifest.get(
            "measurements", {}
        ).get("maximum_absolute_route_lateral_offset_m"),
        "next_scientific_action": (
            "Increase the common route controller's lateral authority under a new frozen "
            "version because the command saturated for 106 pre-peak steps; do not run O2 anomaly."
        ),
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
        "inputs": {
            "config": {"path": str(args.config.resolve()), "sha256": _sha256(args.config.resolve())},
            "preflight": {"path": str(args.preflight.resolve()), "sha256": _sha256(args.preflight.resolve())},
            "manifest": {"path": str(args.manifest.resolve()), "sha256": _sha256(args.manifest.resolve())},
            "v5_baseline": {"path": str(baseline_path), "sha256": _sha256(baseline_path)},
        },
    }
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite postrun audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "passed": result["passed"], "hypothesis_supported": False}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
