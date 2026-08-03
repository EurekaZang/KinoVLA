#!/usr/bin/env python3
"""Audit the preregistered v10 lateral-authority calibration."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    base = ROOT / "outputs/kinofail_realistic/operator_confirmation"
    parser.add_argument(
        "--preflight",
        type=Path,
        default=base / "embodiedgen_lateral_authority_v10_calibration_v1/preflight_audit.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=base
        / "embodiedgen_lateral_authority_v10_calibration_v1/kitchen31/o2_nominal_attempt1/lane_manifest.json",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=base
        / "embodiedgen_fixed_joint_reset_v9_calibration_v1/kitchen31/o2_nominal_attempt1/lane_manifest.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=base / "embodiedgen_lateral_authority_v10_calibration_v1/postrun_audit.json",
    )
    args = parser.parse_args()
    preflight = _json(args.preflight.resolve())
    manifest = _json(args.manifest.resolve())
    baseline = _json(args.baseline.resolve())
    old_peak = float(
        baseline["measurements"]["maximum_absolute_route_lateral_offset_m"]
    )
    new_peak = float(manifest["measurements"]["maximum_absolute_route_lateral_offset_m"])
    required = old_peak - 0.30
    improvement = old_peak - new_peak
    checks = {
        "preflight_passed": preflight.get("passed") is True,
        "retained_as_failed_lane": manifest.get("passed") is False,
        "only_tracking_gate_failed": {
            key for key, value in manifest.get("checks", {}).items() if value is not True
        }
        == {"full_lane_tracking_within_admitted_budget"},
        "controller_limit_is_0_20_mps": abs(
            float(manifest["route_controller"]["controller_contract"]["lateral_limit_mps"])
            - 0.20
        )
        <= 1.0e-9,
        "nominal_completed_without_fall": manifest["checks"]["nominal_route_completed"]
        is True
        and manifest["checks"]["nominal_robot_not_fallen"] is True,
        "peak_reduced_by_at_least_0_04_m": improvement >= 0.04,
        "strict_0_30_m_gate_not_relabelled": new_peak > 0.30,
    }
    result = {
        "schema_version": "kinofail.embodiedgen-lateral-authority-v10-postrun.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "baseline_peak_m": old_peak,
        "v10_peak_m": new_peak,
        "improvement_m": improvement,
        "required_improvement_m": required,
        "fraction_of_required_improvement": improvement / required,
        "remaining_excess_m": new_peak - 0.30,
        "calibration_outcome": "mechanism_supported_but_strict_gate_failed",
        "next_preregistered_change": "forward_speed_mps_from_0.20_to_0.18",
        "stop_rule_after_next_failure": "retrain_locomotion_path_tracking_policy",
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
        "inputs": {
            "preflight": {"path": str(args.preflight.resolve()), "sha256": _sha256(args.preflight.resolve())},
            "manifest": {"path": str(args.manifest.resolve()), "sha256": _sha256(args.manifest.resolve())},
            "baseline": {"path": str(args.baseline.resolve()), "sha256": _sha256(args.baseline.resolve())},
        },
    }
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite postrun audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "passed": result["passed"], "v10_peak_m": new_peak}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
