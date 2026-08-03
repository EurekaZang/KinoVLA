from __future__ import annotations

import pytest

from kino_vla.eval.realistic_operator_admission_v3 import (
    audit_o1_phase_robust_candidate,
    audit_o2_sinkage_normalized_visual_candidate,
)


def _true_checks(names: tuple[str, ...]) -> dict[str, bool]:
    return {name: True for name in names}


def test_o1_accepts_phase_shifted_lateral_response() -> None:
    # A low-friction fall with strong progress/lateral response remains causal even
    # when mean slip is below the old direction-specific 0.50 threshold.
    names = (
        "both_lanes_passed",
        "lane_roles_are_correct",
        "same_scene_and_episode",
        "same_seed",
        "same_appearance",
        "same_camera_profile",
        "same_camera_intrinsics",
        "same_runtime_source_hashes",
        "same_frozen_route_controller_contract",
        "same_segmented_collision_topology",
        "same_target_operator_parameters",
        "same_nominal_surrounding_friction",
        "precontact_telemetry_rows_exactly_equal",
        "first_divergence_not_before_measured_region_contact",
        "friction_anomaly_is_only_active_in_anomaly",
        "friction_readback_matches_request",
        "continuous_surface_has_no_visual_operator_boundary",
        "operator_collision_geometry_is_hidden",
        "nominal_completes_without_fall",
        "anomaly_falls",
    )
    result = audit_o1_phase_robust_candidate(
        {
            "checks": _true_checks(names),
            "measurements": {
                "mean_region_slip_delta": 0.20,
                "lateral_deviation_delta_m": -0.14,
                "forward_progress_deficit_m": 1.1,
            },
        }
    )
    assert result["passed"] is True


def test_o1_rejects_no_kinematic_response() -> None:
    result = audit_o1_phase_robust_candidate(
        {
            "checks": {},
            "measurements": {
                "mean_region_slip_delta": 0.10,
                "lateral_deviation_delta_m": 0.02,
                "forward_progress_deficit_m": 1.1,
            },
        }
    )
    assert result["passed"] is False


def test_o2_uses_sinkage_normalized_visual_fidelity() -> None:
    names = (
        "lane_roles_are_correct",
        "same_scene_and_episode",
        "same_seed",
        "same_appearance",
        "same_camera_profile",
        "same_camera_intrinsics",
        "same_runtime_source_hashes",
        "same_frozen_route_controller_contract",
        "same_segment_partition",
        "same_target_operator_parameters",
        "same_nominal_surrounding_friction",
        "preinteraction_telemetry_rows_exactly_equal",
        "first_divergence_not_before_either_lane_surface_interaction",
        "soil_treatment_is_only_active_in_anomaly",
        "operator_region_has_no_visual_boundary",
        "operator_collision_geometry_is_hidden",
        "surface_is_not_predeformed",
        "control_center_is_flat_nominal_material",
        "anomaly_center_is_ramped_soft_bed",
        "measured_foot_sinkage_is_nontrivial",
        "measured_shear_dissipation_is_nonzero",
        "nominal_completes_without_fall",
        "anomaly_produces_failure",
        "base_height_response_is_nontrivial",
        "forward_progress_response_is_nontrivial",
    )
    pair = {
        "checks": _true_checks(names),
        "measurements": {"anomaly_max_foot_sinkage_m": 0.09},
    }
    anomaly = {
        "operator": {
            "telemetry": {
                "continuous_visual_surface": {
                    "mode": "contact_local_load_triggered_imprints",
                    "applied": True,
                    "contact_local_imprint_count": 5,
                    "deformed_vertex_count": 12,
                    "minimum_visual_offset_m": -0.06,
                }
            }
        }
    }
    result = audit_o2_sinkage_normalized_visual_candidate(pair, anomaly)
    assert result["passed"] is True
    assert result["measurements"]["visual_depth_to_sinkage_ratio"] > 0.65


def test_confirmation_role_is_explicit() -> None:
    result = audit_o1_phase_robust_candidate(
        {"checks": {}, "measurements": {}},
        evidence_role="unseen_scene_confirmation",
    )
    assert result["evidence_role"] == "unseen_scene_confirmation"
    assert result["calibration_only"] is False


def test_unknown_evidence_role_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported evidence_role"):
        audit_o1_phase_robust_candidate({}, evidence_role="retrospective")
