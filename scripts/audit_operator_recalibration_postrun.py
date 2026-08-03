#!/usr/bin/env python3
"""Adjudicate frozen realistic-operator recalibration runs and paired effects."""

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


def _resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _locked(record: dict[str, Any]) -> dict[str, Any]:
    path = _resolve(record["path"])
    exists = path.is_file()
    actual = _sha256(path) if exists else None
    json_ok = True
    if exists and "json_passed" in record:
        json_ok = _json(path).get("passed") is record["json_passed"]
    return {
        "path": str(path),
        "exists": exists,
        "expected_sha256": record["sha256"],
        "actual_sha256": actual,
        "hash_matches": actual == record["sha256"],
        "json_passed_matches": json_ok,
        "passed": exists and actual == record["sha256"] and json_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    preflight_path = args.preflight.resolve()
    config = _json(config_path)
    preflight = _json(preflight_path)
    acceptance = config["acceptance"]
    runtime = config["runtime"]
    current_config_hash = _sha256(config_path)
    locked_files = [_locked(item) for item in config["locked_files"]]

    rows: list[dict[str, Any]] = []
    for case in config["cases"]:
        anomaly_path = _resolve(case["output"]) / "lane_manifest.json"
        anomaly = _json(anomaly_path)
        nominal_record = _locked(case["nominal_manifest"])
        nominal = _json(Path(nominal_record["path"]))
        required_checks = {
            name: anomaly.get("checks", {}).get(name) is True
            for name in acceptance["required_manifest_checks"]
        }
        anomaly_measurements = anomaly.get("measurements", {})
        nominal_measurements = nominal.get("measurements", {})
        controller = anomaly.get("route_controller", {})
        controller_contract = controller.get("controller_contract", {})
        identity_checks = {
            "manifest_exists": anomaly_path.is_file(),
            "manifest_passed": anomaly.get("passed") is True,
            "scene_id_matches": anomaly.get("scene_id") == case["scene_id"],
            "seed_matches": anomaly.get("seed") == case["runtime_seed"],
            "operator_matches": anomaly.get("operator_family") == runtime["operator"],
            "lane_matches": anomaly.get("lane") == runtime["lane"],
            "material_matches": anomaly.get("appearance", {}).get("material_id")
            == case["material_id"],
            "development_only": anomaly.get("development_only") is True,
            "not_a0_a7_evidence": anomaly.get("counts_as_a0_a7_evidence") is False,
            "forward_speed_matches": controller.get("forward_speed_mps")
            == runtime["forward_speed_mps"],
            "region_progress_matches": controller.get("operator_region_progress_m")
            == runtime["region_progress_m"],
            "region_half_length_matches": controller.get("operator_region_half_length_m")
            == runtime["region_half_length_m"],
            "region_half_width_matches": controller.get("operator_region_half_width_m")
            == runtime["region_half_width_m"],
            "controller_gains_match": controller_contract.get("cross_track_gain_per_s")
            == runtime["cross_track_gain_per_s"]
            and controller_contract.get("lateral_velocity_damping")
            == runtime["lateral_velocity_damping"]
            and controller_contract.get("heading_gain_per_s")
            == runtime["heading_gain_per_s"],
            "controller_limits_match": controller_contract.get("lateral_limit_mps")
            == runtime["lateral_limit_mps"]
            and controller_contract.get("yaw_rate_limit_radps")
            == runtime["yaw_rate_limit_radps"],
        }
        quantitative_checks = {
            "minimum_sinkage": float(anomaly_measurements.get("max_foot_sinkage_m", -1.0))
            >= float(acceptance["minimum_max_foot_sinkage_m"]),
            "minimum_loaded_feet": int(anomaly_measurements.get("loaded_feet", -1))
            >= int(acceptance["minimum_loaded_feet"]),
            "minimum_shear_work": float(anomaly_measurements.get("total_shear_work_j", -1.0))
            >= float(acceptance["minimum_total_shear_work_j"]),
            "route_deviation_within_budget": float(
                anomaly_measurements.get("maximum_absolute_route_lateral_offset_m", 1.0e9)
            )
            <= float(acceptance["maximum_route_deviation_m"]),
            "nominal_zero_sinkage": float(nominal_measurements.get("max_foot_sinkage_m", -1.0))
            == 0.0,
            "nominal_zero_shear_work": float(nominal_measurements.get("total_shear_work_j", -1.0))
            == 0.0,
            "nominal_not_fallen": nominal_measurements.get("fallen") is False,
            "anomaly_physical_effect_exceeds_nominal": float(
                anomaly_measurements.get("max_foot_sinkage_m", -1.0)
            )
            > float(nominal_measurements.get("max_foot_sinkage_m", 1.0e9))
            and float(anomaly_measurements.get("total_shear_work_j", -1.0))
            > float(nominal_measurements.get("total_shear_work_j", 1.0e9)),
        }
        passed = (
            nominal_record["passed"]
            and all(required_checks.values())
            and all(identity_checks.values())
            and all(quantitative_checks.values())
        )
        rows.append(
            {
                "scene_id": case["scene_id"],
                "room_family": case["room_family"],
                "material_id": case["material_id"],
                "anomaly_manifest": {
                    "path": str(anomaly_path),
                    "sha256": _sha256(anomaly_path) if anomaly_path.is_file() else None,
                },
                "nominal_manifest": nominal_record,
                "required_checks": required_checks,
                "identity_checks": identity_checks,
                "quantitative_checks": quantitative_checks,
                "paired_measurements": {
                    "nominal": nominal_measurements,
                    "anomaly": anomaly_measurements,
                },
                "passed": passed,
            }
        )

    batch_checks = {
        "preflight_passed": preflight.get("passed") is True,
        "preflight_config_hash_matches": preflight.get("config_sha256")
        == current_config_hash,
        "config_remained_frozen": config.get("freeze_status")
        == "frozen_before_first_anomaly_execution",
        "locked_files_unchanged": all(item["passed"] for item in locked_files),
        "exactly_three_cases": len(rows) == 3,
        "three_distinct_room_families": len({row["room_family"] for row in rows}) == 3,
        "three_distinct_materials": len({row["material_id"] for row in rows}) == 3,
        "all_cases_passed": all(row["passed"] for row in rows),
        "remains_development_only": config.get("evidence_policy", {}).get(
            "counts_as_a0_a7_evidence"
        )
        is False,
    }
    passed = all(batch_checks.values())
    audit = {
        "schema_version": "kinofail.operator-recalibration-postrun.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": str(config_path),
        "config_sha256": current_config_hash,
        "preflight": str(preflight_path),
        "preflight_sha256": _sha256(preflight_path) if preflight_path.is_file() else None,
        "locked_files": locked_files,
        "cases": rows,
        "batch_checks": batch_checks,
        "passed": passed,
        "sealed": passed,
        "counts_as_a0_a7_evidence": False,
        "realistic_a0_a7_readiness": "0/8",
        "next_gate": config["next_gate_if_passed"],
    }
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "passed": passed,
                "sealed": passed,
                "case_pass_count": sum(row["passed"] for row in rows),
                "case_count": len(rows),
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
