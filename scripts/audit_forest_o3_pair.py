#!/usr/bin/env python3
"""Audit one matched-topology realistic-forest O3 development pair."""

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


def _load(path: Path) -> dict:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def _telemetry(manifest_path: Path, manifest: dict) -> list[dict]:
    path = Path(manifest["telemetry"])
    if not path.is_absolute():
        path = manifest_path.resolve().parent / path
    return [json.loads(line) for line in path.resolve().read_text(encoding="utf-8").splitlines()]


def _captures(manifest: dict) -> dict[str, dict]:
    return {row["label"]: row for row in manifest["camera"]["captures"]}


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
    nominal_rows = _telemetry(args.nominal, nominal)
    anomaly_rows = _telemetry(args.anomaly, anomaly)
    nominal_captures = _captures(nominal)
    anomaly_captures = _captures(anomaly)
    topology_control = nominal["operator"]["telemetry"]["regions"][0]
    topology_anomaly = anomaly["operator"]["telemetry"]["regions"][0]
    visual = topology_anomaly["continuous_visual_surface"]
    trigger_capture_step = int(anomaly_captures["matched_consequence"]["step"])
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
    checks = {
        "both_lanes_passed": nominal.get("passed") is True and anomaly.get("passed") is True,
        "lane_roles_are_correct": nominal.get("lane") == "nominal"
        and anomaly.get("lane") == "anomaly",
        "same_scene_episode_seed": nominal.get("episode_usd_sha256")
        == anomaly.get("episode_usd_sha256")
        and nominal.get("compiled_audit_sha256") == anomaly.get("compiled_audit_sha256")
        and nominal.get("seed") == anomaly.get("seed"),
        "same_frozen_route_controller_contract": _controller_contract(nominal)
        == _controller_contract(anomaly),
        "same_runtime_source_hashes": all(
            nominal["provenance"].get(key) == anomaly["provenance"].get(key)
            for key in ("backend_sha256", "operator_sha256", "script_sha256")
        ),
        "same_appearance": nominal.get("appearance") == anomaly.get("appearance"),
        "same_target_operator_parameters": nominal["operator"]["parameters"]
        == anomaly["operator"]["parameters"],
        "anomaly_transition_only": nominal["operator"]["active"] is False
        and nominal["operator"]["matched_counterfactual_topology_active"] is True
        and anomaly["operator"]["active"] is True,
        "same_hidden_zero_seam_pretrigger_topology": topology_control["support_cell_gap_m"]
        == topology_anomaly["support_cell_gap_m"]
        == 0.0
        and topology_control["collision_geometry_visible"] is False
        and topology_anomaly["collision_geometry_visible"] is False,
        "same_nominal_friction_readback": all(
            topology_control[key] == topology_anomaly[key]
            for key in (
                "nominal_requested_static_friction",
                "nominal_requested_dynamic_friction",
                "nominal_readback_static_friction",
                "nominal_readback_dynamic_friction",
            )
        )
        and abs(float(topology_anomaly["nominal_readback_static_friction"]) - 0.8)
        <= 1.0e-6
        and abs(float(topology_anomaly["nominal_readback_dynamic_friction"]) - 0.6)
        <= 1.0e-6,
        "pretrigger_telemetry_rows_exactly_equal": nominal_rows[: trigger_capture_step - 1]
        == anomaly_rows[: trigger_capture_step - 1],
        "first_divergence_is_trigger_capture": first_different_step == trigger_capture_step,
        "matched_capture_step_identical": nominal_captures["matched_consequence"]["step"]
        == anomaly_captures["matched_consequence"]["step"]
        == trigger_capture_step,
        "same_camera_intrinsics_and_labels": set(nominal_captures)
        == set(anomaly_captures)
        == EXPECTED_LABELS
        and all(
            nominal_captures[label]["intrinsic_matrix"]
            == anomaly_captures[label]["intrinsic_matrix"]
            for label in EXPECTED_LABELS
        ),
        "all_evidence_frames_precede_fall": all(
            row["robot_fallen"] is False for row in anomaly_captures.values()
        ),
        "measured_impulse_triggered": topology_anomaly["collapsed"] is True
        and float(topology_anomaly["last_damage_update"]["normal_impulse_ns"])
        >= float(anomaly["operator"]["parameters"]["damage_threshold_ns"]),
        "physical_and_visual_trigger_synchronized": topology_anomaly[
            "visual_sync_trigger_step"
        ]
        == topology_anomaly["last_damage_update"]["trigger_step"]
        == visual["visual_trigger_step"],
        "support_topology_really_changed": topology_anomaly["disabled_cell_colliders"]
        == topology_anomaly["failed_support_cells"]
        == 13,
        "continuous_pbr_collapse_is_posttrigger_only": visual["applied"] is True
        and visual["visual_triggered"] is True
        and visual["surface_predeformed"] is False
        and visual["operator_region_has_no_pretrigger_visual_boundary"] is True
        and visual["operator_collision_geometry_visible"] is False
        and int(visual["deformed_vertex_count"]) > 50
        and float(visual["minimum_visual_offset_m"]) <= -0.10,
        "nominal_completes_anomaly_falls": nominal["measurements"]["fallen"] is False
        and nominal["checks"]["nominal_route_completed"] is True
        and anomaly["measurements"]["fallen"] is True,
        "base_height_response_nontrivial": height_delta >= 0.10,
        "progress_response_nontrivial": progress_deficit >= 1.0,
    }
    result = {
        "schema_version": "kinofail.realistic-forest-o3-pair-audit.v2-development",
        "created_utc": datetime.now(UTC).isoformat(),
        "development_only": True,
        "evidence_role": args.evidence_role,
        "nominal_manifest": str(args.nominal.resolve()),
        "anomaly_manifest": str(args.anomaly.resolve()),
        "checks": checks,
        "measurements": {
            "trigger_capture_step": trigger_capture_step,
            "first_different_telemetry_step": first_different_step,
            "exact_equal_pretrigger_rows": trigger_capture_step - 1,
            "normal_impulse_at_trigger_ns": topology_anomaly["last_damage_update"][
                "normal_impulse_ns"
            ],
            "disabled_support_cells": topology_anomaly["disabled_cell_colliders"],
            "median_region_base_height_delta_m": height_delta,
            "forward_progress_deficit_m": progress_deficit,
            "minimum_visual_offset_m": visual["minimum_visual_offset_m"],
            "deformed_vertex_count": visual["deformed_vertex_count"],
            "anomaly_fallen": anomaly["measurements"]["fallen"],
        },
        "internal_visual_review": {
            "reviewer": "model_assisted_internal_review_not_independent_human",
            "passed_for_development": True,
            "observations": [
                "No support-cell grid, pre-authored pit, collision plate, or operator rectangle is visible before trigger.",
                "The body-fixed front camera cannot directly see the under-foot basin at trigger; the visual claim is limited to synchronized mesh state, not guaranteed RGB observability.",
            ],
        },
        "passed": all(checks.values()),
        "scene_registry_eligible": False,
        "counts_as_a0_a7_evidence": False,
        "event_alignment_contract": {
            "event_source": "anomaly measured foot-normal impulse threshold crossing",
            "nominal_capture": "deterministic matched-control replay at the anomaly event step",
            "statistical_unit": "one nominal/anomaly physical pair, not two independent runs",
        },
        "remaining_gates": [
            "three_synchronized_appearance_families",
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
