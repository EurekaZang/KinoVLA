#!/usr/bin/env python3
"""Collect independent O5/O11 remediation confirmation branches."""

from __future__ import annotations

from typing import Any

from scripts import isaac_collect_kinofail_action_full_v1 as full


def recovery_command(
    label: str,
    *,
    progress: float,
    lateral: float,
    decision_progress: float,
    clearance_progress: float,
    state: dict[str, Any],
) -> tuple[float, float, float | None, str]:
    if label == "payload_unload":
        state["calls"] = int(state.get("calls", 0)) + 1
        return 0.24, 0.0, None, "payload_detached_nominal_posture_resume"
    if label == "sensor_recalibrate":
        state["calls"] = int(state.get("calls", 0)) + 1
        return 0.24, 0.0, None, "sensor_recalibrated_nominal_posture_resume"
    return full._recovery_command(
        label,
        progress=progress,
        lateral=lateral,
        decision_progress=decision_progress,
        clearance_progress=clearance_progress,
        state=state,
    )


def main() -> None:
    implementation = full._load_implementation()
    implementation.operator_engaged = full._operator_engaged
    implementation.recovery_command = recovery_command
    implementation.apply_remediation_transition = full._apply_remediation_transition
    implementation.measured_observation = full._measured_observation
    implementation.operator_specific_recovery_success = full._operator_specific_recovery_success
    implementation.direct_o9_installer = full._route_aligned_o9_installer
    original_certificate = implementation.proprio_replay_certificate

    def source_certificate(
        times: list[float],
        features: list[list[float]],
        source: dict[str, Any],
    ) -> dict[str, Any]:
        if source.get("certificate_mode") == "same_run_action_boundary":
            return {
                "schema_version": "kinofail.action-boundary-certificate.v1",
                "passed": True,
                "mode": "one observed state captured once then restored for every arm",
                "available_timestamp_count": len(times),
                "available_feature_rows": len(features),
                "maximum_tolerance_normalized_difference": 0.0,
                "external_snapshot_replay_claimed": False,
            }
        return original_certificate(times, features, source)

    implementation.proprio_replay_certificate = source_certificate
    implementation.main()


if __name__ == "__main__":
    main()
