#!/usr/bin/env python3
"""Seal paired nominal/anomaly recalibration for realistic Kino-Fail operators."""

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


def _resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _same_number(value: Any, expected: Any) -> bool:
    try:
        return math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=1.0e-9)
    except (TypeError, ValueError):
        return False


def _locked(record: dict[str, Any]) -> dict[str, Any]:
    path = _resolve(record["path"])
    exists = path.is_file()
    actual = _sha256(path) if exists else None
    json_ok = True
    if exists and "json_passed" in record:
        json_ok = _json(path).get("passed") is record["json_passed"]
    passed = exists and actual == record["sha256"] and json_ok
    return {
        "path": str(path),
        "exists": exists,
        "expected_sha256": record["sha256"],
        "actual_sha256": actual,
        "hash_matches": actual == record["sha256"],
        "json_passed_matches": json_ok,
        "passed": passed,
    }


def _manifest_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": _sha256(path) if path.is_file() else None,
    }


def _identity_checks(
    manifest: dict[str, Any],
    path: Path,
    case: dict[str, Any],
    lane: str,
    runtime: dict[str, Any],
) -> dict[str, bool]:
    controller = manifest.get("route_controller", {})
    contract = controller.get("controller_contract", {})
    return {
        "manifest_exists": path.is_file(),
        "manifest_passed": manifest.get("passed") is True,
        "scene_id_matches": manifest.get("scene_id") == case["scene_id"],
        "operator_matches": manifest.get("operator_family") == case["operator"],
        "lane_matches": manifest.get("lane") == lane,
        "seed_matches": manifest.get("seed") == case["runtime_seed"],
        "material_matches": manifest.get("appearance", {}).get("material_id")
        == case["material_id"],
        "camera_profile_matches": manifest.get("camera", {}).get("profile")
        == runtime["camera_profile"],
        "development_only": manifest.get("development_only") is True,
        "not_a0_a7_evidence": manifest.get("counts_as_a0_a7_evidence") is False,
        "forward_speed_matches": _same_number(
            controller.get("forward_speed_mps"), runtime["forward_speed_mps"]
        ),
        "route_budget_matches": _same_number(
            controller.get("admitted_maximum_route_deviation_m"),
            runtime["maximum_route_deviation_m"],
        ),
        "region_matches": all(
            (
                _same_number(controller.get("operator_region_progress_m"), runtime["region_progress_m"]),
                _same_number(
                    controller.get("operator_region_lateral_offset_m"),
                    runtime["region_lateral_offset_m"],
                ),
                _same_number(
                    controller.get("operator_region_half_length_m"),
                    runtime["region_half_length_m"],
                ),
                _same_number(
                    controller.get("operator_region_half_width_m"),
                    runtime["region_half_width_m"],
                ),
            )
        ),
        "start_and_target_match": all(
            (
                _same_number(
                    controller.get("start_lateral_offset_m"),
                    runtime["start_lateral_offset_m"],
                ),
                _same_number(
                    controller.get("start_heading_offset_rad"),
                    runtime["start_heading_offset_rad"],
                ),
                _same_number(
                    controller.get("target_lateral_offset_m"),
                    runtime["target_lateral_offset_m"],
                ),
            )
        ),
        "controller_gains_match": all(
            (
                _same_number(
                    contract.get("cross_track_gain_per_s"), runtime["cross_track_gain_per_s"]
                ),
                _same_number(
                    contract.get("lateral_velocity_damping"),
                    runtime["lateral_velocity_damping"],
                ),
                _same_number(
                    contract.get("heading_gain_per_s"), runtime["heading_gain_per_s"]
                ),
            )
        ),
        "controller_limits_match": all(
            (
                _same_number(contract.get("lateral_limit_mps"), runtime["lateral_limit_mps"]),
                _same_number(
                    contract.get("yaw_rate_limit_radps"), runtime["yaw_rate_limit_radps"]
                ),
            )
        ),
    }


def _o1_effect_checks(
    nominal: dict[str, Any], anomaly: dict[str, Any], config: dict[str, Any]
) -> dict[str, bool]:
    params = config["operator_parameters"]["o1"]
    nominal_t = nominal.get("operator", {}).get("telemetry", {})
    anomaly_t = anomaly.get("operator", {}).get("telemetry", {})
    nominal_m = nominal.get("measurements", {})
    anomaly_m = anomaly.get("measurements", {})
    return {
        "matched_visual_material": nominal.get("appearance") == anomaly.get("appearance"),
        "matched_region_geometry": nominal_t.get("region") == anomaly_t.get("region"),
        "nominal_uses_scene_friction": _same_number(
            nominal_t.get("readback_static_friction"),
            nominal_t.get("nominal_readback_static_friction"),
        )
        and _same_number(
            nominal_t.get("readback_dynamic_friction"),
            nominal_t.get("nominal_readback_dynamic_friction"),
        ),
        "anomaly_friction_matches_freeze": _same_number(
            anomaly_t.get("requested_static_friction"), params["anomaly_static_friction"]
        )
        and _same_number(
            anomaly_t.get("requested_dynamic_friction"), params["anomaly_dynamic_friction"]
        )
        and _same_number(
            anomaly_t.get("readback_static_friction"), params["anomaly_static_friction"]
        )
        and _same_number(
            anomaly_t.get("readback_dynamic_friction"), params["anomaly_dynamic_friction"]
        ),
        "physics_intervention_is_separated": float(
            anomaly_t.get("readback_static_friction", 1.0e9)
        )
        < float(nominal_t.get("readback_static_friction", -1.0)),
        "anomaly_has_measured_slip": float(anomaly_m.get("max_region_slip_ratio", -1.0))
        >= 0.05,
        "anomaly_mean_slip_exceeds_nominal": float(
            anomaly_m.get("mean_region_slip_ratio", -1.0)
        )
        > float(nominal_m.get("mean_region_slip_ratio", 1.0e9)),
        "nominal_not_fallen": nominal_m.get("fallen") is False,
        "anomaly_fallen": anomaly_m.get("fallen") is True,
    }


def _o3_effect_checks(
    nominal: dict[str, Any], anomaly: dict[str, Any], config: dict[str, Any]
) -> dict[str, bool]:
    params = config["operator_parameters"]["o3"]
    nominal_regions = nominal.get("operator", {}).get("telemetry", {}).get("regions", [])
    anomaly_regions = anomaly.get("operator", {}).get("telemetry", {}).get("regions", [])
    nominal_r = nominal_regions[0] if len(nominal_regions) == 1 else {}
    anomaly_r = anomaly_regions[0] if len(anomaly_regions) == 1 else {}
    nominal_m = nominal.get("measurements", {})
    anomaly_m = anomaly.get("measurements", {})
    anomaly_visual = anomaly_r.get("continuous_visual_surface", {})
    return {
        "matched_visual_material": nominal.get("appearance") == anomaly.get("appearance"),
        "matched_region_geometry": nominal.get("operator", {}).get("parameters", {}).get("region")
        == anomaly.get("operator", {}).get("parameters", {}).get("region"),
        "parameters_match_freeze": _same_number(
            anomaly_r.get("damage_threshold_ns"), params["damage_threshold_ns"]
        )
        and _same_number(anomaly_r.get("drop_m"), params["drop_m"])
        and _same_number(anomaly_r.get("residual_support"), params["residual_support"]),
        "nominal_never_collapsed": nominal_m.get("collapse_triggered") is False
        and nominal_r.get("collapsed") is False
        and nominal_r.get("disabled_cell_colliders") == 0,
        "anomaly_collapsed": anomaly_m.get("collapse_triggered") is True
        and anomaly_r.get("collapsed") is True
        and int(anomaly_r.get("disabled_cell_colliders", 0)) > 0,
        "visual_and_physics_trigger_steps_match": anomaly_m.get("collapse_trigger_step")
        == anomaly_r.get("visual_sync_trigger_step")
        == anomaly_visual.get("visual_trigger_step"),
        "trigger_reached_after_measured_impulse": float(
            anomaly_r.get("last_damage_update", {}).get("normal_impulse_ns", -1.0)
        )
        >= float(params["damage_threshold_ns"]),
        "nominal_not_fallen": nominal_m.get("fallen") is False,
        "anomaly_fallen": anomaly_m.get("fallen") is True,
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
    runtime = config["runtime"]
    acceptance = config["acceptance"]
    current_config_hash = _sha256(config_path)
    locked_files = [_locked(item) for item in config["locked_files"]]

    rows: list[dict[str, Any]] = []
    for case in config["cases"]:
        operator = case["operator"]
        lane_data: dict[str, dict[str, Any]] = {}
        lane_rows: dict[str, dict[str, Any]] = {}
        for lane in ("nominal", "anomaly"):
            path = _resolve(case["outputs"][lane]) / "lane_manifest.json"
            manifest = _json(path)
            required = {
                name: manifest.get("checks", {}).get(name) is True
                for name in acceptance[operator][f"{lane}_required_checks"]
            }
            identity = _identity_checks(manifest, path, case, lane, runtime)
            measurements = manifest.get("measurements", {})
            quantitative = {
                "route_deviation_within_budget": float(
                    measurements.get("maximum_absolute_route_lateral_offset_m", 1.0e9)
                )
                <= float(acceptance["maximum_route_deviation_m"]),
                "operator_region_sampled": int(measurements.get("region_sample_count", -1)) > 0,
                "route_surface_visible": measurements.get("route_surface_visible") is True,
                "scene_floor_disabled": measurements.get("custom_scene_floor_enabled") is False,
            }
            lane_data[lane] = manifest
            lane_rows[lane] = {
                "manifest": _manifest_record(path),
                "required_checks": required,
                "identity_checks": identity,
                "quantitative_checks": quantitative,
                "measurements": measurements,
                "passed": all(required.values())
                and all(identity.values())
                and all(quantitative.values()),
            }

        effect = (
            _o1_effect_checks(lane_data["nominal"], lane_data["anomaly"], config)
            if operator == "o1"
            else _o3_effect_checks(lane_data["nominal"], lane_data["anomaly"], config)
        )
        prerequisites = {
            name: _locked(record) for name, record in case.get("prerequisites", {}).items()
        }
        passed = (
            all(row["passed"] for row in lane_rows.values())
            and all(effect.values())
            and all(record["passed"] for record in prerequisites.values())
        )
        rows.append(
            {
                "scene_id": case["scene_id"],
                "room_family": case["room_family"],
                "operator": operator,
                "material_id": case["material_id"],
                "prerequisites": prerequisites,
                "lanes": lane_rows,
                "paired_effect_checks": effect,
                "passed": passed,
            }
        )

    scene_operator_pairs = {(row["scene_id"], row["operator"]) for row in rows}
    expected_pairs = {
        (case["scene_id"], case["operator"]) for case in config.get("cases", [])
    }
    batch_checks = {
        "preflight_passed": preflight.get("passed") is True,
        "preflight_config_hash_matches": preflight.get("config_sha256")
        == current_config_hash,
        "config_frozen_before_execution": config.get("freeze_status")
        == "frozen_before_first_anomaly_execution",
        "locked_files_unchanged": all(record["passed"] for record in locked_files),
        "exactly_six_scene_operator_pairs": len(rows) == 6
        and len(scene_operator_pairs) == 6,
        "matches_frozen_case_matrix": scene_operator_pairs == expected_pairs,
        "covers_o1_and_o3": {row["operator"] for row in rows} == {"o1", "o3"},
        "three_distinct_room_families": len({row["room_family"] for row in rows}) == 3,
        "three_distinct_scenes": len({row["scene_id"] for row in rows}) == 3,
        "three_distinct_materials": len({row["material_id"] for row in rows}) == 3,
        "all_six_pairs_passed": all(row["passed"] for row in rows),
        "one_attempt_policy_declared": config.get("evidence_policy", {}).get(
            "one_attempt_per_lane"
        )
        is True,
        "no_replacement_policy_declared": config.get("evidence_policy", {}).get(
            "no_replacement_scenes"
        )
        is True,
        "remains_development_only": config.get("evidence_policy", {}).get(
            "counts_as_a0_a7_evidence"
        )
        is False,
        "a8_excluded": config.get("evidence_policy", {}).get("a8_in_scope") is False,
    }
    passed = all(batch_checks.values())
    audit = {
        "schema_version": "kinofail.operator-pair-recalibration-postrun.v1",
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
        "interpretation": (
            "Operator calibration only. These paired effects validate realistic-scene "
            "interventions but do not reproduce any A0-A7 conclusion."
        ),
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
                "pair_pass_count": sum(row["passed"] for row in rows),
                "pair_count": len(rows),
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
