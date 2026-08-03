#!/usr/bin/env python3
"""Audit the stopped v11 O2 unseen-scene nominal confirmation batch.

The audit distinguishes protocol compliance from scientific success: a retained
negative result can pass this audit while the controller confirmation fails.
"""

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


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(path)
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _at_step(rows: list[dict[str, Any]], step: int) -> dict[str, Any]:
    return next(row for row in rows if int(row["step"]) == step)


def _trajectory_delta(
    calibration: list[dict[str, Any]], confirmation: list[dict[str, Any]], step: int
) -> dict[str, float]:
    old = _at_step(calibration, step)
    new = _at_step(confirmation, step)
    return {
        "route_progress_m": abs(
            float(new["route_progress_m"]) - float(old["route_progress_m"])
        ),
        "route_lateral_offset_m": abs(
            float(new["route_lateral_offset_m"])
            - float(old["route_lateral_offset_m"])
        ),
        "heading_rad": abs(float(new["heading_rad"]) - float(old["heading_rad"])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    base = ROOT / "outputs/kinofail_realistic/operator_confirmation"
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_embodiedgen_o2_v11_unseen_nominal_confirmation_v1.json",
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=base
        / "embodiedgen_o2_v11_unseen_nominal_confirmation_v1/preflight_audit.json",
    )
    parser.add_argument(
        "--office-manifest",
        type=Path,
        default=base
        / "embodiedgen_o2_v11_unseen_nominal_confirmation_v1/office27/o2_nominal/lane_manifest.json",
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=base
        / "embodiedgen_stable_speed_v11_calibration_v1/kitchen31/o2_nominal_attempt1/lane_manifest.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=base
        / "embodiedgen_o2_v11_unseen_nominal_confirmation_v1/postrun_audit.json",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    preflight_path = args.preflight.resolve()
    office_manifest_path = args.office_manifest.resolve()
    calibration_manifest_path = args.calibration_manifest.resolve()
    config = _json(config_path)
    preflight = _json(preflight_path)
    office = _json(office_manifest_path)
    calibration = _json(calibration_manifest_path)
    office_telemetry_path = office_manifest_path.parent / office["telemetry"]
    calibration_telemetry_path = calibration_manifest_path.parent / calibration["telemetry"]
    office_rows = _jsonl(office_telemetry_path)
    calibration_rows = _jsonl(calibration_telemetry_path)

    office_scene = config["scenes"][0]
    bedroom_scene = config["scenes"][1]
    bedroom_output = (ROOT / bedroom_scene["planned_output"]).resolve()
    batch_root = office_manifest_path.parents[2]
    anomaly_manifests = sorted(batch_root.glob("**/o2_anomaly*/lane_manifest.json"))
    checks = {
        "preflight_passed_before_execution": preflight.get("passed") is True,
        "config_hash_matches_preflight": preflight.get("config", {}).get("sha256")
        == _sha256(config_path),
        "office_scene_is_first_preregistered_scene": office_scene.get("scene_id")
        == "indoor_office_27",
        "office_manifest_bound_to_preregistered_scene": office.get("scene_id")
        == office_scene.get("scene_id"),
        "office_manifest_is_nominal_o2": office.get("lane") == "nominal"
        and office.get("operator_family") == "o2",
        "office_confirmation_failed": office.get("passed") is False,
        "failure_is_strict_tracking_only": office.get("checks", {}).get(
            "full_lane_tracking_within_admitted_budget"
        )
        is False
        and all(
            value is True
            for name, value in office.get("checks", {}).items()
            if name != "full_lane_tracking_within_admitted_budget"
        ),
        "frozen_tracking_budget_retained": abs(
            float(office["route_controller"]["admitted_maximum_route_deviation_m"])
            - 0.30
        )
        <= 1.0e-9,
        "failed_result_retained_not_relabelled": float(
            office["measurements"]["maximum_absolute_route_lateral_offset_m"]
        )
        > 0.30,
        "bedroom_stopped_after_first_failure": not bedroom_output.exists(),
        "no_anomaly_lane_executed": not anomaly_manifests,
        "still_not_registry_evidence": office.get("scene_registry_eligible") is False,
        "still_not_a0_a7_evidence": office.get("counts_as_a0_a7_evidence") is False,
    }

    maximum_deviation = float(
        office["measurements"]["maximum_absolute_route_lateral_offset_m"]
    )
    first_violation = next(
        row for row in office_rows if abs(float(row["route_lateral_offset_m"])) > 0.30
    )
    result = {
        "schema_version": "kinofail.embodiedgen-o2-v11-unseen-nominal-postrun.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "audit_passed": all(checks.values()),
        "controller_confirmation_passed": False,
        "batch_stopped_as_preregistered": checks["bedroom_stopped_after_first_failure"]
        and checks["no_anomaly_lane_executed"],
        "checks": checks,
        "measurements": {
            "office_maximum_absolute_route_lateral_offset_m": maximum_deviation,
            "tracking_budget_m": 0.30,
            "tracking_budget_excess_m": maximum_deviation - 0.30,
            "first_tracking_violation_step": int(first_violation["step"]),
            "first_tracking_violation_progress_m": float(
                first_violation["route_progress_m"]
            ),
            "office_completed_route": office["checks"]["nominal_route_completed"],
            "office_fallen": office["measurements"]["fallen"],
            "office_region_sample_count": office["measurements"]["region_sample_count"],
            "trajectory_delta_vs_kitchen_v11": {
                "step_125": _trajectory_delta(calibration_rows, office_rows, 125),
                "step_150": _trajectory_delta(calibration_rows, office_rows, 150),
                "step_200": _trajectory_delta(calibration_rows, office_rows, 200),
            },
        },
        "scientific_outcome": (
            "v11 Kitchen calibration did not generalize to the first untouched scene; "
            "the frozen controller is rejected for formal realistic O2 collection."
        ),
        "diagnosis": {
            "observed": (
                "Relative trajectories remain close through step 150, then diverge; the "
                "Office run completes without falling but exceeds the lateral budget late."
            ),
            "ruled_out_by_current_evidence": [
                "route-coordinate binding failure at reset",
                "camera mount failure",
                "operator physics accidentally active in the nominal lane",
            ],
            "working_hypothesis_not_yet_proven": (
                "The saturated proportional lateral command and learned locomotion dynamics "
                "form a weakly damped closed loop whose small cross-scene numerical or contact "
                "differences amplify into a late lateral excursion."
            ),
        },
        "evidence_boundary": {
            "counts_as_a0_a7_evidence": False,
            "scene_registry_eligible": False,
            "realistic_a0_a7_readiness": "0/8",
            "old_a0_a7_results_are_reference_only": True,
            "completion_requires_new_realistic_corpus_training_inference_and_a0_a7": True,
        },
        "next_required_action": (
            "Retire Office27 from untouched confirmation status; develop a more robust "
            "controller on declared development scenes, then use Bedroom28 plus a newly "
            "generated untouched scene family for a fresh frozen confirmation."
        ),
        "inputs": {
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "preflight": {
                "path": str(preflight_path),
                "sha256": _sha256(preflight_path),
            },
            "office_manifest": {
                "path": str(office_manifest_path),
                "sha256": _sha256(office_manifest_path),
            },
            "office_telemetry": {
                "path": str(office_telemetry_path),
                "sha256": _sha256(office_telemetry_path),
            },
            "kitchen_calibration_manifest": {
                "path": str(calibration_manifest_path),
                "sha256": _sha256(calibration_manifest_path),
            },
            "kitchen_calibration_telemetry": {
                "path": str(calibration_telemetry_path),
                "sha256": _sha256(calibration_telemetry_path),
            },
        },
    }
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite postrun audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(output),
                "audit_passed": result["audit_passed"],
                "controller_confirmation_passed": False,
                "tracking_budget_excess_m": result["measurements"][
                    "tracking_budget_excess_m"
                ],
            },
            indent=2,
        )
    )
    return 0 if result["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
