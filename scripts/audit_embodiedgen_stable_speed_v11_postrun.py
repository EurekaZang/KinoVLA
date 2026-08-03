#!/usr/bin/env python3
"""Audit the v11 Kitchen31 stable-speed calibration pass."""

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
        default=base / "embodiedgen_stable_speed_v11_calibration_v1/preflight_audit.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=base
        / "embodiedgen_stable_speed_v11_calibration_v1/kitchen31/o2_nominal_attempt1/lane_manifest.json",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=base
        / "embodiedgen_lateral_authority_v10_calibration_v1/kitchen31/o2_nominal_attempt1/lane_manifest.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=base / "embodiedgen_stable_speed_v11_calibration_v1/postrun_audit.json",
    )
    args = parser.parse_args()
    preflight = _json(args.preflight.resolve())
    manifest = _json(args.manifest.resolve())
    baseline = _json(args.baseline.resolve())
    old_peak = float(baseline["measurements"]["maximum_absolute_route_lateral_offset_m"])
    new_peak = float(manifest["measurements"]["maximum_absolute_route_lateral_offset_m"])
    reset = manifest.get("inference_reset_contract", {})
    checks = {
        "preflight_passed": preflight.get("passed") is True,
        "lane_passed_all_frozen_checks": manifest.get("passed") is True
        and all(value is True for value in manifest.get("checks", {}).values()),
        "strict_tracking_gate_passed": new_peak <= 0.30,
        "forward_speed_is_0_18_mps": abs(
            float(manifest["route_controller"]["forward_speed_mps"]) - 0.18
        )
        <= 1.0e-9,
        "lateral_authority_remains_0_20_mps": abs(
            float(manifest["route_controller"]["controller_contract"]["lateral_limit_mps"])
            - 0.20
        )
        <= 1.0e-9,
        "nominal_completed_without_fall": manifest["checks"]["nominal_route_completed"]
        is True
        and manifest["checks"]["nominal_robot_not_fallen"] is True,
        "tracking_improved_over_v10": new_peak < old_peak,
        "effective_go2_reset_was_already_fixed": reset.get(
            "observed_pre_freeze_position_scale_range"
        )
        == [1.0, 1.0],
    }
    result = {
        "schema_version": "kinofail.embodiedgen-stable-speed-v11-postrun.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "v10_peak_m": old_peak,
        "v11_peak_m": new_peak,
        "tracking_margin_m": 0.30 - new_peak,
        "calibration_outcome": "kitchen31_nominal_pass_with_narrow_margin",
        "next_required_action": (
            "Freeze the same controller and confirm O2 nominal on untouched Office27 and "
            "Bedroom28 before any anomaly lane."
        ),
        "counts_as_a0_a7_evidence": False,
        "scene_registry_eligible": False,
        "realistic_a0_a7_readiness": "0/8",
        "inputs": {
            "preflight": {"path": str(args.preflight.resolve()), "sha256": _sha256(args.preflight.resolve())},
            "manifest": {"path": str(args.manifest.resolve()), "sha256": _sha256(args.manifest.resolve())},
            "v10_baseline": {"path": str(args.baseline.resolve()), "sha256": _sha256(args.baseline.resolve())},
        },
    }
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite postrun audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "passed": result["passed"], "tracking_margin_m": result["tracking_margin_m"]}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
