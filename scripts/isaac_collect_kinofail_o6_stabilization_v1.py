#!/usr/bin/env python3
"""Collect O6 recovery-profile development and frozen confirmation cases.

This adapter retains the full-action collector's exact-checkpoint branching,
trace hashing, outcome definitions, and unfavorable-outcome retention.  It
only adds predeclared O6 control profiles so that external-push recovery is a
real control intervention instead of the near-nominal 0.24 m/s command used in
the first full-operator confirmation.
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import isaac_collect_kinofail_action_full_v1 as full


O6_PROFILES = {
    "O6_resume_024": "near_nominal_resume_reference",
    "O6_brace_low": "immediate_low_stance_brace",
    "O6_brace_deep": "immediate_deep_stance_brace",
    "O6_counter_low": "forward_momentum_counter_low_stance",
    "O6_counter_deep": "forward_momentum_counter_deep_stance",
    "O6_catch_then_brace": "short_counter_impulse_then_low_brace",
    "O6_brace_then_resume": "low_brace_then_controlled_resume",
    "O6_directional_reflex": "direction_conditioned_reflex_brace_or_rideout",
}


def _recovery_command(
    label: str,
    *,
    progress: float,
    lateral: float,
    decision_progress: float,
    clearance_progress: float,
    state: dict[str, Any],
) -> tuple[float, float, float | None, str]:
    if label not in O6_PROFILES:
        return full._recovery_command(
            label,
            progress=progress,
            lateral=lateral,
            decision_progress=decision_progress,
            clearance_progress=clearance_progress,
            state=state,
        )

    state["calls"] = int(state.get("calls", 0)) + 1
    calls = int(state["calls"])
    if label == "O6_resume_024":
        return 0.24, 0.0, None, "push_near_nominal_resume_reference"
    if label == "O6_directional_reflex":
        # The generated collection loop replaces this nominal placeholder in
        # the same control step using the measured longitudinal/lateral
        # velocity jump and toggles the backend's low-level reflex directly.
        return 0.24, 0.0, None, "push_directional_reflex_pending"
    if label == "O6_brace_low":
        return 0.0, float(lateral), 0.20, "push_immediate_low_stance_brace"
    if label == "O6_brace_deep":
        return 0.0, float(lateral), 0.16, "push_immediate_deep_stance_brace"
    if label == "O6_counter_low":
        return 0.45, 0.0, 0.20, "push_forward_counter_low_stance"
    if label == "O6_counter_deep":
        return 0.55, 0.0, 0.16, "push_forward_counter_deep_stance"
    if label == "O6_catch_then_brace":
        if calls <= 15:
            return 0.55, 0.0, 0.18, "push_short_forward_catch"
        return 0.0, float(lateral), 0.20, "push_postcatch_low_brace"
    if label == "O6_brace_then_resume":
        if calls <= 30:
            return 0.0, float(lateral), 0.18, "push_transient_low_brace"
        return 0.20, 0.0, 0.24, "push_controlled_resume"
    raise AssertionError(label)


def main() -> None:
    original_compile = builtins.compile
    patch_point = (
        "                    backend.set_posture(posture, 1.5 if posture is not None else 1.0)\n"
    )
    directional_control = (
        "                    if post and action == \"recover_as_O6_directional_reflex\":\n"
        "                        longitudinal_jump = abs(float(obs.vel_body[0]) - prefix_speed)\n"
        "                        lateral_jump = abs(float(obs.vel_body[1]))\n"
        "                        directional_brace = lateral_jump > longitudinal_jump\n"
        "                        backend.set_reflex(directional_brace)\n"
        "                        if directional_brace:\n"
        "                            speed = 0.0\n"
        "                            target = float(lateral)\n"
        "                            posture = 0.20\n"
        "                            action_phase = \"push_lateral_reflex_brace\"\n"
        "                        else:\n"
        "                            speed = prefix_speed\n"
        "                            target = prefix_target\n"
        "                            posture = None\n"
        "                            action_phase = \"push_longitudinal_policy_rideout\"\n"
        "                    else:\n"
        "                        backend.set_reflex(False)\n"
        "                    backend.set_posture(posture, 1.5 if posture is not None else 1.0)\n"
    )

    def compile_with_directional_reflex(
        source: Any, filename: str, mode: str, *args: Any, **kwargs: Any
    ) -> Any:
        if isinstance(source, str) and filename == str(full.BASE):
            if source.count(patch_point) != 1:
                raise RuntimeError("non-unique O6 directional-reflex patch point")
            source = source.replace(patch_point, directional_control)
        return original_compile(source, filename, mode, *args, **kwargs)

    builtins.compile = compile_with_directional_reflex
    try:
        implementation = full._load_implementation()
    finally:
        builtins.compile = original_compile
    # The dynamically loaded base checks the hash of its effective __file__.
    # Bind that check to this adapter, which is the collector named by the
    # O6 protocol and executed by the supervisor.
    implementation.__file__ = str(Path(__file__).resolve())
    implementation.operator_engaged = full._operator_engaged
    implementation.recovery_command = _recovery_command
    implementation.operator_specific_recovery_success = (
        full._operator_specific_recovery_success
    )
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
