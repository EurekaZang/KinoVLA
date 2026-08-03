#!/usr/bin/env python3
"""Audit one realistic forest O2 nominal/anomaly development pair."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


EXPECTED_LABELS = {"pre_entry", "inside_region", "matched_consequence"}


def _controller_contract(manifest: dict) -> dict:
    return {
        key: value
        for key, value in manifest.get("route_controller", {}).items()
        if key != "matched_consequence_step_requested"
    }


def _capture_map(manifest: dict) -> dict[str, dict]:
    return {row["label"]: row for row in manifest["camera"]["captures"]}


def _telemetry(manifest_path: Path, manifest: dict) -> list[dict]:
    path = Path(manifest["telemetry"])
    if not path.is_absolute():
        path = manifest_path.resolve().parent / path
    return [json.loads(line) for line in path.resolve().read_text(encoding="utf-8").splitlines()]


def _sequence_non_degenerate(captures: dict[str, dict]) -> bool:
    if set(captures) != EXPECTED_LABELS:
        return False
    basic_texture_gate = all(
        float(row["metrics"]["std_luminance"]) >= 0.020
        and int(row["metrics"]["quantized_color_count_5bit"]) >= 64
        and float(row["metrics"]["edge_fraction_gt_0_02"]) >= 0.010
        for row in captures.values()
    )
    context_gate = all(
        float(captures[label]["metrics"]["std_luminance"]) >= 0.035
        for label in ("pre_entry", "inside_region")
    )
    return basic_texture_gate and context_gate


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
    nominal = json.loads(args.nominal.read_text(encoding="utf-8"))
    anomaly = json.loads(args.anomaly.read_text(encoding="utf-8"))
    compiled = json.loads(Path(nominal["compiled_audit"]).read_text(encoding="utf-8"))
    nominal_captures = _capture_map(nominal)
    anomaly_captures = _capture_map(anomaly)
    nominal_rows = _telemetry(args.nominal, nominal)
    anomaly_rows = _telemetry(args.anomaly, anomaly)
    first_nominal_region_contact_step = next(
        (
            int(row["step"])
            for row in nominal_rows
            if row["operator_region_foot_contact"]["any_load_bearing_in_region"]
        ),
        None,
    )
    first_anomaly_region_contact_step = next(
        (
            int(row["step"])
            for row in anomaly_rows
            if row["operator_region_foot_contact"]["any_load_bearing_in_region"]
        ),
        None,
    )
    contact_candidates = [
        step
        for step in (
            first_nominal_region_contact_step,
            first_anomaly_region_contact_step,
        )
        if step is not None
    ]
    first_surface_interaction_step = min(contact_candidates) if contact_candidates else None
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
    height_delta = float(nominal["measurements"]["median_region_base_height_m"]) - float(
        anomaly["measurements"]["median_region_base_height_m"]
    )
    progress_deficit = float(
        nominal["measurements"].get(
            "final_route_progress_m", nominal["measurements"]["final_position_xy_m"][0]
        )
    ) - float(
        anomaly["measurements"].get(
            "final_route_progress_m", anomaly["measurements"]["final_position_xy_m"][0]
        )
    )
    control_telemetry = nominal["operator"]["telemetry"]
    anomaly_telemetry = anomaly["operator"]["telemetry"]
    control_topology = control_telemetry.get("collision_topology", {})
    anomaly_topology = anomaly_telemetry.get("collision_topology", {})
    visual_surface = anomaly_telemetry.get("continuous_visual_surface", {})
    checks = {
        "both_lanes_passed": nominal.get("passed") is True and anomaly.get("passed") is True,
        "lane_roles_are_correct": nominal.get("lane") == "nominal"
        and anomaly.get("lane") == "anomaly",
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
        "soil_treatment_is_only_active_in_anomaly": nominal["operator"]["active"] is False
        and nominal["operator"]["matched_counterfactual_topology_active"] is True
        and anomaly["operator"]["id"] == "O2_compliance"
        and anomaly["operator"]["active"] is True,
        "same_target_operator_parameters": nominal["operator"]["parameters"]
        == anomaly["operator"]["parameters"],
        "same_segment_partition": all(
            control_topology.get(key) == anomaly_topology.get(key)
            for key in (
                "mode",
                "collision_segment_count",
                "surrounding_segment_count",
                "central_segment_count",
                "central_xy_layout",
                "collision_geometry_visible",
                "operator_region_has_no_precontact_visual_boundary",
            )
        ),
        "same_nominal_surrounding_friction": all(
            abs(
                float(control_telemetry["nominal_ground_friction"][key])
                - float(anomaly_telemetry["nominal_ground_friction"][key])
            )
            <= 1.0e-6
            for key in ("static", "dynamic", "readback_static", "readback_dynamic")
        )
        and abs(float(anomaly_telemetry["nominal_ground_friction"]["static"]) - 0.8)
        <= 1.0e-6
        and abs(float(anomaly_telemetry["nominal_ground_friction"]["dynamic"]) - 0.6)
        <= 1.0e-6,
        "control_center_is_flat_nominal_material": control_topology.get("matched_control")
        is True
        and control_topology.get("central_surface_state") == "flat_nominal_material"
        and all(abs(float(value)) <= 1.0e-9 for value in control_topology["central_top_height_m"])
        and all(
            abs(float(material[key]) - float(control_topology["nominal_material"][key]))
            <= 1.0e-6
            for material in control_topology["central_material_readback"]
            for key in ("static", "dynamic")
        ),
        "anomaly_center_is_ramped_soft_bed": anomaly_topology.get("matched_control")
        is False
        and anomaly_topology.get("central_surface_state") == "ramped_lowered_soft_bed"
        and min(float(value) for value in anomaly_topology["central_top_height_m"]) <= -0.10,
        "preinteraction_telemetry_rows_exactly_equal": first_surface_interaction_step
        is not None
        and nominal_rows[: first_surface_interaction_step - 1]
        == anomaly_rows[: first_surface_interaction_step - 1],
        "first_divergence_not_before_either_lane_surface_interaction": first_surface_interaction_step
        is not None
        and first_different_step is not None
        and first_different_step >= first_surface_interaction_step,
        "nominal_completes_without_fall": nominal["measurements"]["fallen"] is False
        and nominal["checks"]["nominal_route_completed"] is True,
        "anomaly_produces_failure": anomaly["measurements"]["fallen"] is True,
        "forward_progress_response_is_nontrivial": progress_deficit >= 0.75,
        "base_height_response_is_nontrivial": height_delta >= 0.07,
        "measured_foot_sinkage_is_nontrivial": float(
            anomaly["measurements"]["max_foot_sinkage_m"]
        )
        >= 0.07,
        "measured_shear_dissipation_is_nonzero": float(
            anomaly["measurements"]["total_shear_work_j"]
        )
        > 0.02,
        "contact_local_visual_deformation_used": visual_surface.get("applied") is True
        and visual_surface.get("mode") == "contact_local_load_triggered_imprints"
        and int(visual_surface.get("contact_local_imprint_count", 0)) >= 4
        and float(visual_surface.get("minimum_visual_offset_m", 0.0)) <= -0.07,
        "surface_is_not_predeformed": visual_surface.get("surface_predeformed") is False,
        "operator_region_has_no_visual_boundary": visual_surface.get(
            "operator_region_has_no_visual_boundary"
        )
        is True,
        "operator_collision_geometry_is_hidden": visual_surface.get(
            "operator_collision_geometry_visible"
        )
        is False,
        "prop_collision_proxy_gate_satisfied": compiled["near_field_contract"].get(
            "prop_collision_proxies_complete"
        )
        is True,
    }
    result = {
        "schema_version": "kinofail.realistic-forest-o2-pair-audit.v2-development",
        "created_utc": datetime.now(UTC).isoformat(),
        "development_only": True,
        "evidence_role": args.evidence_role,
        "nominal_manifest": str(args.nominal.resolve()),
        "anomaly_manifest": str(args.anomaly.resolve()),
        "checks": checks,
        "measurements": {
            "median_region_base_height_delta_m": height_delta,
            "anomaly_max_foot_sinkage_m": anomaly["measurements"]["max_foot_sinkage_m"],
            "anomaly_total_shear_work_j": anomaly["measurements"]["total_shear_work_j"],
            "anomaly_loaded_feet": anomaly["measurements"]["loaded_feet"],
            "anomaly_fallen": anomaly["measurements"]["fallen"],
            "forward_progress_deficit_m": progress_deficit,
            "first_nominal_region_load_bearing_contact_step": first_nominal_region_contact_step,
            "first_anomaly_region_load_bearing_contact_step": first_anomaly_region_contact_step,
            "first_either_lane_surface_interaction_step": first_surface_interaction_step,
            "first_different_telemetry_step": first_different_step,
            "exact_equal_preinteraction_rows": (
                first_surface_interaction_step - 1
                if first_surface_interaction_step is not None
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
                "No rectangular route strip, pre-sunk band, transparent film, collision box, or ramp is visible.",
                "Nominal and anomaly retain the same PBR texture identity.",
                "Only measured load-bearing foot contacts create local PBR-mesh depressions.",
                "All evidence frames are captured before the anomaly fall terminal state.",
            ],
        },
        "passed": all(checks.values()),
        "scene_registry_eligible": False,
        "counts_as_a0_a7_evidence": False,
        "causal_boundary_contract": {
            "definition": (
                "The earliest load-bearing interaction with the treated partition in either "
                "lane. A lowered/removed-support anomaly can diverge when the nominal foot "
                "contacts the control surface even though the anomaly foot is still descending."
            ),
            "calibration_disclosure": (
                "G02 exposed the flaw in using anomaly-only contact, so G02 is calibration-only "
                "and cannot serve as held-out geometry confirmation."
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
