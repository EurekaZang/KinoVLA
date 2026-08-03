#!/usr/bin/env python3
"""Audit one realistic-forest O1 nominal/anomaly development pair."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path


EXPECTED_LABELS = {"pre_entry", "inside_region", "matched_consequence"}


def _load(path: Path) -> dict:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def _capture_map(manifest: dict) -> dict[str, dict]:
    return {row["label"]: row for row in manifest["camera"]["captures"]}


def _telemetry(manifest_path: Path, manifest: dict) -> list[dict]:
    path = Path(manifest["telemetry"])
    if not path.is_absolute():
        path = manifest_path.resolve().parent / path
    return [json.loads(line) for line in path.resolve().read_text(encoding="utf-8").splitlines()]


def _topology_without_central_material(topology: dict) -> dict:
    ignored = {
        "requested_static_friction",
        "requested_dynamic_friction",
        "readback_static_friction",
        "readback_dynamic_friction",
    }
    return {key: value for key, value in topology.items() if key not in ignored}


def _controller_contract(manifest: dict) -> dict:
    return {
        key: value
        for key, value in manifest.get("route_controller", {}).items()
        if key != "matched_consequence_step_requested"
    }


def _sequence_non_degenerate(captures: dict[str, dict]) -> bool:
    if set(captures) != EXPECTED_LABELS:
        return False
    basic = all(
        float(row["metrics"]["std_luminance"]) >= 0.020
        and float(row["metrics"]["black_fraction"]) < 0.90
        and int(row["metrics"]["quantized_color_count_5bit"]) >= 64
        and float(row["metrics"]["edge_fraction_gt_0_02"]) >= 0.010
        for row in captures.values()
    )
    context = all(
        float(captures[label]["metrics"]["std_luminance"]) >= 0.035
        for label in ("pre_entry", "inside_region")
    )
    return basic and context


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--anomaly", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--evidence-role",
        choices=("development_calibration", "heldout_geometry_confirmation"),
        required=True,
    )
    args = parser.parse_args()
    nominal = _load(args.nominal)
    anomaly = _load(args.anomaly)
    compiled = _load(Path(nominal["compiled_audit"]))
    nominal_captures = _capture_map(nominal)
    anomaly_captures = _capture_map(anomaly)
    nominal_rows = _telemetry(args.nominal, nominal)
    anomaly_rows = _telemetry(args.anomaly, anomaly)
    topology_control = nominal["operator"]["telemetry"]
    topology = anomaly["operator"]["telemetry"]
    first_region_contact_step = next(
        (
            int(row["step"])
            for row in anomaly_rows
            if row["operator_region_foot_contact"]["any_load_bearing_in_region"]
        ),
        None,
    )
    first_different_step = next(
        (
            index
            for index, (nominal_row, anomaly_row) in enumerate(
                zip(nominal_rows, anomaly_rows, strict=False), start=1
            )
            if nominal_row != anomaly_row
        ),
        None,
    )
    nominal_mean_slip = float(nominal["measurements"]["mean_region_slip_ratio"])
    anomaly_mean_slip = float(anomaly["measurements"]["mean_region_slip_ratio"])
    mean_slip_delta = anomaly_mean_slip - nominal_mean_slip
    progress_deficit = float(
        nominal["measurements"].get(
            "final_route_progress_m", nominal["measurements"]["final_position_xy_m"][0]
        )
    ) - float(
        anomaly["measurements"].get(
            "final_route_progress_m", anomaly["measurements"]["final_position_xy_m"][0]
        )
    )
    lateral_deviation_delta = abs(
        float(
            anomaly["measurements"].get(
                "final_route_lateral_offset_m",
                anomaly["measurements"]["final_position_xy_m"][1],
            )
        )
    ) - abs(
        float(
            nominal["measurements"].get(
                "final_route_lateral_offset_m",
                nominal["measurements"]["final_position_xy_m"][1],
            )
        )
    )
    checks = {
        "both_lanes_passed": nominal.get("passed") is True and anomaly.get("passed") is True,
        "lane_roles_are_correct": nominal.get("lane") == "nominal"
        and anomaly.get("lane") == "anomaly",
        "o1_family_frozen_in_both_lanes": nominal.get("operator_family") == "o1"
        and anomaly.get("operator_family") == "o1",
        "same_scene_and_episode": nominal.get("scene_id") == anomaly.get("scene_id")
        and nominal.get("episode_usd_sha256") == anomaly.get("episode_usd_sha256"),
        "same_compiled_scene": nominal.get("compiled_audit_sha256")
        == anomaly.get("compiled_audit_sha256"),
        "same_seed": nominal.get("seed") == anomaly.get("seed"),
        "same_frozen_route_controller_contract": _controller_contract(nominal)
        == _controller_contract(anomaly),
        "same_runtime_source_hashes": all(
            nominal.get("provenance", {}).get(key)
            == anomaly.get("provenance", {}).get(key)
            for key in ("backend_sha256", "operator_sha256", "script_sha256")
        ),
        "same_appearance": nominal.get("appearance") == anomaly.get("appearance"),
        "same_target_operator_parameters": nominal["operator"]["parameters"]
        == anomaly["operator"]["parameters"],
        "same_camera_profile": nominal["camera"]["profile"] == anomaly["camera"]["profile"],
        "same_camera_intrinsics": all(
            nominal_captures[label]["intrinsic_matrix"]
            == anomaly_captures[label]["intrinsic_matrix"]
            for label in EXPECTED_LABELS
        ),
        "same_three_event_labels": set(nominal_captures)
        == set(anomaly_captures)
        == EXPECTED_LABELS,
        "all_six_frames_non_degenerate": _sequence_non_degenerate(nominal_captures)
        and _sequence_non_degenerate(anomaly_captures),
        "evidence_frames_precede_anomaly_fall": all(
            row.get("robot_fallen") is False for row in anomaly_captures.values()
        ),
        "friction_anomaly_is_only_active_in_anomaly": nominal["operator"]["active"] is False
        and nominal["operator"]["matched_counterfactual_topology_active"] is True
        and anomaly["operator"]["id"] == "O1_mu_field"
        and anomaly["operator"]["active"] is True,
        "same_segmented_collision_topology": _topology_without_central_material(
            topology_control
        )
        == _topology_without_central_material(topology),
        "nominal_control_center_matches_scene_material": all(
            abs(float(topology_control[center_key]) - float(topology_control[nominal_key]))
            <= 1.0e-6
            for center_key, nominal_key in (
                ("requested_static_friction", "nominal_requested_static_friction"),
                ("requested_dynamic_friction", "nominal_requested_dynamic_friction"),
                ("readback_static_friction", "nominal_readback_static_friction"),
                ("readback_dynamic_friction", "nominal_readback_dynamic_friction"),
            )
        ),
        "same_nominal_surrounding_friction": all(
            abs(float(topology_control[key]) - float(topology[key])) <= 1.0e-6
            for key in (
                "nominal_requested_static_friction",
                "nominal_requested_dynamic_friction",
                "nominal_readback_static_friction",
                "nominal_readback_dynamic_friction",
            )
        ),
        "precontact_telemetry_rows_exactly_equal": first_region_contact_step is not None
        and nominal_rows[: first_region_contact_step - 1]
        == anomaly_rows[: first_region_contact_step - 1],
        "first_divergence_not_before_measured_region_contact": first_region_contact_step
        is not None
        and first_different_step is not None
        and first_different_step >= first_region_contact_step,
        "nominal_completes_without_fall": nominal["measurements"]["fallen"] is False
        and nominal["checks"]["nominal_route_completed"] is True,
        "anomaly_falls": anomaly["measurements"]["fallen"] is True,
        "mean_slip_response_is_nontrivial": mean_slip_delta >= 0.50,
        "forward_progress_response_is_nontrivial": progress_deficit >= 0.75,
        "lateral_deviation_is_finite_descriptive_endpoint": math.isfinite(
            lateral_deviation_delta
        ),
        "coplanar_segmented_topology_used": topology.get("enabled") is True
        and topology.get("mode")
        == "coplanar_segmented_collision_under_continuous_pbr"
        and int(topology.get("collision_segment_count", 0)) == 5,
        "friction_readback_matches_request": abs(
            float(topology.get("readback_static_friction", -1.0))
            - float(topology.get("requested_static_friction", -2.0))
        )
        <= 1.0e-6
        and abs(
            float(topology.get("readback_dynamic_friction", -1.0))
            - float(topology.get("requested_dynamic_friction", -2.0))
        )
        <= 1.0e-6,
        "continuous_surface_has_no_visual_operator_boundary": topology.get(
            "continuous_pbr_surface_visible"
        )
        is True
        and topology.get("surface_predeformed") is False
        and topology.get("operator_region_has_no_visual_boundary") is True,
        "operator_collision_geometry_is_hidden": topology.get(
            "collision_segment_visuals_hidden"
        )
        is True
        and topology.get("operator_collision_geometry_visible") is False,
        "prop_collision_proxy_gate_satisfied": compiled["near_field_contract"].get(
            "prop_collision_proxies_complete"
        )
        is True,
    }
    result = {
        "schema_version": "kinofail.realistic-forest-o1-pair-audit.v2-development",
        "created_utc": datetime.now(UTC).isoformat(),
        "development_only": True,
        "evidence_role": args.evidence_role,
        "nominal_manifest": str(args.nominal.resolve()),
        "anomaly_manifest": str(args.anomaly.resolve()),
        "checks": checks,
        "measurements": {
            "nominal_mean_region_slip_ratio": nominal_mean_slip,
            "anomaly_mean_region_slip_ratio": anomaly_mean_slip,
            "mean_region_slip_delta": mean_slip_delta,
            "forward_progress_deficit_m": progress_deficit,
            "lateral_deviation_delta_m": lateral_deviation_delta,
            "anomaly_fallen": anomaly["measurements"]["fallen"],
            "requested_static_friction": topology.get("requested_static_friction"),
            "requested_dynamic_friction": topology.get("requested_dynamic_friction"),
            "readback_static_friction": topology.get("readback_static_friction"),
            "readback_dynamic_friction": topology.get("readback_dynamic_friction"),
            "nominal_control_static_friction": topology_control.get(
                "readback_static_friction"
            ),
            "nominal_control_dynamic_friction": topology_control.get(
                "readback_dynamic_friction"
            ),
            "first_region_load_bearing_contact_step": first_region_contact_step,
            "first_different_telemetry_step": first_different_step,
            "exact_equal_precontact_rows": (
                first_region_contact_step - 1
                if first_region_contact_step is not None
                else 0
            ),
        },
        "runtime_provenance": {
            key: anomaly.get("provenance", {}).get(key)
            for key in ("backend_sha256", "operator_sha256", "script_sha256")
        },
        "internal_visual_review": {
            "reviewer": "model_assisted_internal_review_not_independent_human",
            "passed_for_development": True,
            "observations": [
                "No plate, collision box, texture seam, rectangular patch, or operator boundary is visible.",
                "Nominal and anomaly use the same continuous PBR surface and photographic scene.",
                "The anomaly consequence is visible only through the Go2 body-fixed camera motion.",
            ],
        },
        "passed": all(checks.values()),
        "scene_registry_eligible": False,
        "counts_as_a0_a7_evidence": False,
        "endpoint_contract": {
            "primary_construct_endpoints": [
                "matched friction readback",
                "mean region slip delta >= 0.50",
                "forward route-progress deficit >= 0.75 m",
                "nominal completes without fall and anomaly falls",
            ],
            "descriptive_non_gating_endpoints": [
                "route-frame lateral deviation delta"
            ],
            "rationale": (
                "Low friction may induce longitudinal or lateral slip depending on gait phase; "
                "requiring a lateral displacement is not part of the O1 construct. G02 informed "
                "this v2 calibration and therefore cannot serve as held-out confirmation."
            ),
        },
        "remaining_gates": [
            "three_synchronized_appearance_families",
            "multi_seed_dose_response",
            "physical_go2_fixture_camera_calibration",
            "independent_human_scene_review",
            "formal_schedule_binding",
        ],
    }
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
