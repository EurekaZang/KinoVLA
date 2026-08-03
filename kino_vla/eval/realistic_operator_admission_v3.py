"""Candidate phase-robust admission gates for realistic terrain operators.

The v2 development gates deliberately remain immutable.  This module is a new
candidate architecture motivated by crossed-factor calibration failures:

* O1 low friction can express itself as longitudinal slip or lateral escape,
  depending on foot-contact phase.  The causal readback and outcome contrast
  remain mandatory, while the kinematic construct uses a predeclared evidence
  union rather than one direction-specific scalar.
* O2 visual deformation is a rendering of measured physical sinkage.  Its
  fidelity should therefore be normalized by that sinkage instead of requiring
  every mesh/contact phase to exceed one absolute depth.

Passing this module on calibration data is not confirmatory evidence.  A new,
unseen scene/geometry batch must be frozen before these gates can be promoted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = "kinofail.realistic-terrain-operator-admission.v3-candidate"


def _required_checks(pair: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, bool]:
    available = pair.get("checks", {})
    return {name: available.get(name) is True for name in names}


def audit_o1_phase_robust_candidate(
    pair: Mapping[str, Any], *, evidence_role: str = "calibration"
) -> dict[str, Any]:
    """Audit O1 without assuming that low friction always causes lateral slip."""
    if evidence_role not in {"calibration", "unseen_scene_confirmation"}:
        raise ValueError(f"unsupported evidence_role: {evidence_role}")
    measurements = pair.get("measurements", {})
    slip_delta = float(measurements.get("mean_region_slip_delta", float("-inf")))
    lateral_delta = abs(float(measurements.get("lateral_deviation_delta_m", 0.0)))
    progress_deficit = float(measurements.get("forward_progress_deficit_m", float("-inf")))
    causal = _required_checks(
        pair,
        (
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
        ),
    )
    endpoint_checks = {
        "forward_progress_deficit_at_least_0p75_m": progress_deficit >= 0.75,
        "phase_robust_kinematic_response": slip_delta >= 0.30 or lateral_delta >= 0.10,
    }
    checks = {**causal, **endpoint_checks}
    return {
        "schema_version": SCHEMA_VERSION,
        "operator": "O1",
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": {
            "mean_region_slip_delta": slip_delta,
            "absolute_lateral_deviation_delta_m": lateral_delta,
            "forward_progress_deficit_m": progress_deficit,
        },
        "candidate_contract": {
            "slip_delta_min": 0.30,
            "absolute_lateral_deviation_delta_m_min": 0.10,
            "kinematic_combination": "OR",
            "forward_progress_deficit_m_min": 0.75,
        },
        "evidence_role": evidence_role,
        "calibration_only": evidence_role == "calibration",
        "counts_as_a0_a7_evidence": False,
    }


def audit_o2_sinkage_normalized_visual_candidate(
    pair: Mapping[str, Any],
    anomaly_manifest: Mapping[str, Any],
    *,
    evidence_role: str = "calibration",
) -> dict[str, Any]:
    """Audit O2 visual fidelity relative to the measured physical deformation."""
    if evidence_role not in {"calibration", "unseen_scene_confirmation"}:
        raise ValueError(f"unsupported evidence_role: {evidence_role}")
    measurements = pair.get("measurements", {})
    sinkage = float(measurements.get("anomaly_max_foot_sinkage_m", 0.0))
    visual = (
        anomaly_manifest.get("operator", {})
        .get("telemetry", {})
        .get("continuous_visual_surface", {})
    )
    visual_depth = abs(float(visual.get("minimum_visual_offset_m", 0.0)))
    visual_to_sinkage = visual_depth / sinkage if sinkage > 0.0 else 0.0
    causal = _required_checks(
        pair,
        (
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
        ),
    )
    visual_checks = {
        "contact_local_visual_mode": visual.get("mode")
        == "contact_local_load_triggered_imprints",
        "visual_deformation_applied": visual.get("applied") is True,
        "at_least_four_contact_local_imprints": int(
            visual.get("contact_local_imprint_count", 0)
        )
        >= 4,
        "deformed_vertices_nonzero": int(visual.get("deformed_vertex_count", 0)) > 0,
        "absolute_visual_depth_at_least_0p05_m": visual_depth >= 0.05,
        "visual_depth_tracks_at_least_65_percent_of_sinkage": visual_to_sinkage >= 0.65,
    }
    checks = {**causal, **visual_checks}
    return {
        "schema_version": SCHEMA_VERSION,
        "operator": "O2",
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": {
            "physical_max_sinkage_m": sinkage,
            "absolute_minimum_visual_offset_m": visual_depth,
            "visual_depth_to_sinkage_ratio": visual_to_sinkage,
            "contact_local_imprint_count": int(visual.get("contact_local_imprint_count", 0)),
            "deformed_vertex_count": int(visual.get("deformed_vertex_count", 0)),
        },
        "candidate_contract": {
            "absolute_visual_depth_m_min": 0.05,
            "visual_depth_to_sinkage_ratio_min": 0.65,
            "contact_local_imprint_count_min": 4,
        },
        "evidence_role": evidence_role,
        "calibration_only": evidence_role == "calibration",
        "counts_as_a0_a7_evidence": False,
    }
