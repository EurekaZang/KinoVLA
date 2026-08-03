#!/usr/bin/env python3
"""Audit the Kitchen31-only calibration of the long-route controller contract."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--failed-preflight-postrun",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_native_v3_preflight_v1/postrun_audit.json",
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_route_controller_calibration_v1/kitchen31/target_m025_speed020_attempt1/lane_manifest.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/operator_confirmation/embodiedgen_route_controller_calibration_v1/calibration_audit.json",
    )
    args = parser.parse_args()

    failed_path = args.failed_preflight_postrun.resolve()
    manifest_path = args.calibration_manifest.resolve()
    failed = _json(failed_path)
    manifest = _json(manifest_path)
    telemetry_path = manifest_path.parent / str(manifest.get("telemetry", ""))
    rows = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line
    ] if telemetry_path.is_file() else []
    maximum_absolute_lateral_offset = max(
        (abs(float(row["route_lateral_offset_m"])) for row in rows),
        default=float("inf"),
    )
    controller = manifest.get("route_controller", {})
    measurements = manifest.get("measurements", {})
    failed_checks = {
        key for key, value in failed.get("checks", {}).items() if value is not True
    }
    checks = {
        "failed_v3_preflight_retained": failed_path.is_file() and failed.get("passed") is False,
        "v3_failure_was_long_route_tracking": {
            "indoor_diningroom_26_full_lane_tracking_within_admitted_budget",
            "indoor_livingroom_30_full_lane_tracking_within_admitted_budget",
        }.issubset(failed_checks),
        "calibration_is_kitchen31_only": manifest.get("scene_id") == "indoor_kitchen_31",
        "calibration_manifest_passed_local_checks": manifest.get("passed") is True,
        "calibration_not_a0_a7_evidence": manifest.get("counts_as_a0_a7_evidence") is False,
        "calibration_uses_native_static_collector": manifest.get("collector_lifecycle")
        == "native_v3_protocol_candidate"
        and "collector_adapter_provenance" not in manifest,
        "telemetry_hash_verified": telemetry_path.is_file()
        and _sha256(telemetry_path) == manifest.get("telemetry_sha256"),
        "frozen_forward_speed_0p20": abs(
            float(controller.get("forward_speed_mps", -1.0)) - 0.20
        ) <= 1.0e-9,
        "frozen_target_lateral_offset_minus_0p25": abs(
            float(controller.get("target_lateral_offset_m", 1.0)) + 0.25
        ) <= 1.0e-9,
        "frozen_operator_lateral_center_minus_0p15": abs(
            float(controller.get("operator_region_lateral_offset_m", 1.0)) + 0.15
        ) <= 1.0e-9,
        "frozen_operator_half_width_0p40": abs(
            float(controller.get("operator_region_half_width_m", -1.0)) - 0.40
        ) <= 1.0e-9,
        "full_lane_within_0p30_m_budget": maximum_absolute_lateral_offset <= 0.30,
        "operator_region_has_at_least_40_samples": int(
            measurements.get("region_sample_count", 0)
        ) >= 40,
        "route_completed_without_fall": manifest.get("checks", {}).get(
            "nominal_route_completed"
        ) is True
        and measurements.get("fallen") is False,
    }
    result = {
        "schema_version": "kinofail.embodiedgen-route-controller-calibration.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "failed_v3_preflight": {"path": str(failed_path), "sha256": _sha256(failed_path)},
        "calibration_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "calibration_measurements": {
            "maximum_absolute_route_lateral_offset_m": maximum_absolute_lateral_offset,
            "region_sample_count": measurements.get("region_sample_count"),
            "final_route_progress_m": measurements.get("final_route_progress_m"),
            "fallen": measurements.get("fallen"),
        },
        "candidate_controller_contract": {
            "forward_speed_mps": 0.20,
            "target_lateral_offset_m": -0.25,
            "operator_region_lateral_offset_m": -0.15,
            "operator_region_half_width_m": 0.40,
            "maximum_absolute_route_lateral_offset_m": 0.30,
            "minimum_operator_region_samples": 40,
        },
        "evidence_role": "calibration_only",
        "counts_as_a0_a7_evidence": False,
        "promotion_requirement": (
            "Freeze this contract before any Office27/Bedroom28 lane, then confirm without "
            "retuning or scene replacement."
        ),
    }
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibration audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "passed": result["passed"]}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
