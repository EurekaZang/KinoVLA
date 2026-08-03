from __future__ import annotations

import numpy as np
import pytest

from kino_vla.eval.c2_temporal_v5 import invariant_summary
from scripts.isaac_collect_kinofail_reconfirmation_a4_v7 import (
    TRACE_FIELDS,
    array_sha256,
    operator_engaged,
    proprio_replay_certificate,
    recovery_command,
    safety_exposure_auc,
    source_early_region,
)


class _Frame:
    route_length_m = 3.0
    surface_width_m = 1.0

    def __init__(self, direction: tuple[float, float]) -> None:
        self.direction = np.asarray(direction, dtype=np.float64)

    def point(self, progress: float, lateral: float) -> np.ndarray:
        left = np.asarray((-self.direction[1], self.direction[0]))
        return self.direction * progress + left * lateral


@pytest.mark.parametrize(
    ("direction", "expected"),
    [((1.0, 0.0), (0.35, 0.0, 0.30, 0.40)), ((0.0, 1.0), (0.0, 0.35, 0.40, 0.30))],
)
def test_source_early_region_matches_frozen_v4_geometry(direction, expected) -> None:
    region = source_early_region(_Frame(direction))
    assert (region.cx, region.cy, region.hx, region.hy) == pytest.approx(expected)


def test_proprio_replay_certificate_accepts_exact_f35_observation() -> None:
    rng = np.random.default_rng(2027014)
    raw = rng.normal(size=(21, 19)).astype(np.float32)
    timestamps = np.arange(1, 22, dtype=np.float64) * 0.02
    expected = invariant_summary(raw)
    source = {
        "sample_id": "fixture__primary",
        "proprio_timestamp_s": timestamps.tolist(),
        "invariant_proprio_80": expected.tolist(),
        "invariant_proprio_80_sha256": array_sha256(expected),
    }
    certificate = proprio_replay_certificate(
        timestamps.tolist(), raw.tolist(), source
    )
    assert certificate["passed"] is True
    assert certificate["maximum_tolerance_normalized_difference"] == 0.0


def test_proprio_replay_certificate_rejects_source_state_drift() -> None:
    raw = np.zeros((21, 19), dtype=np.float32)
    timestamps = np.arange(1, 22, dtype=np.float64) * 0.02
    expected = invariant_summary(raw)
    source = {
        "sample_id": "fixture__primary",
        "proprio_timestamp_s": timestamps.tolist(),
        "invariant_proprio_80": expected.tolist(),
        "invariant_proprio_80_sha256": array_sha256(expected),
    }
    drifted = raw.copy()
    drifted[:, 15] = 0.10
    with pytest.raises(RuntimeError, match="does not match"):
        proprio_replay_certificate(
            timestamps.tolist(), drifted.tolist(), source
        )


def test_proprio_replay_certificate_rejects_timestamp_substitution() -> None:
    raw = np.zeros((21, 19), dtype=np.float32)
    timestamps = np.arange(1, 22, dtype=np.float64) * 0.02
    expected = invariant_summary(raw)
    source = {
        "sample_id": "fixture__primary",
        "proprio_timestamp_s": timestamps.tolist(),
        "invariant_proprio_80": expected.tolist(),
        "invariant_proprio_80_sha256": array_sha256(expected),
    }
    shifted = timestamps.copy()
    shifted[-1] += 0.01
    with pytest.raises(RuntimeError, match="timestamps"):
        proprio_replay_certificate(shifted.tolist(), raw.tolist(), source)


def test_safety_exposure_auc_integrates_only_postdecision_exceedances() -> None:
    trace = np.zeros((3, len(TRACE_FIELDS)), dtype=np.float64)
    trace[:, 0] = (0.0, 0.02, 0.04)
    trace[:, 10] = 0.75
    trace[1, 7] = 0.55  # (0.55 - 0.35) / 0.20 = 1
    trace[2, 10] = 0.0  # (0.75 - 0.0) / 0.75 = 1
    assert safety_exposure_auc(trace, 0) == pytest.approx(0.04)


def test_o9_engagement_uses_belly_contact_semantics_not_payload() -> None:
    telemetry = {
        "enabled": True,
        "payload_kg": 0.0,
        "regions": [
            {
                "max_belly_contact_force_n": 30.0,
                "max_consecutive_belly_contact_steps": 12,
                "belly_contact_duty_cycle": 0.20,
                "min_measured_support": 0.50,
                "max_nonfoot_contact_by_body_n": {"base": 30.0},
            }
        ],
    }
    assert operator_engaged(
        "O9_high_centering",
        telemetry,
        progress_m=0.5,
        forward_speed_mps=0.0,
        source_region_exposed=True,
    )


def test_five_recovery_labels_start_distinct_physical_programs() -> None:
    labels = (
        "O2_compliance",
        "O4_tether",
        "O5_payload",
        "O8_invisible_collider",
        "O9_high_centering",
    )
    commands = []
    phases = []
    for label in labels:
        command = recovery_command(
            label,
            progress=0.5,
            lateral=0.0,
            decision_progress=0.5,
            clearance_progress=1.25,
            state={"phase": "back", "detour_sign": 1.0},
        )
        commands.append(command[:3])
        phases.append(command[3])
    assert len(set(commands)) == 5
    assert len(set(phases)) == 5


def test_o2_recovery_uses_low_brace_posture() -> None:
    command = recovery_command(
        "O2_compliance",
        progress=0.5,
        lateral=0.1,
        decision_progress=0.5,
        clearance_progress=1.25,
        state={"phase": "back", "detour_sign": 1.0},
    )
    assert command[:3] == pytest.approx((0.0, 0.1, 0.245))
    assert command[3] == "compliance_low_brace_and_request_assistance"


def test_environmental_recoveries_keep_distinct_crossing_controls() -> None:
    commands = {
        label: recovery_command(
            label,
            progress=0.8,
            lateral=0.7,
            decision_progress=0.5,
            clearance_progress=1.25,
            state={"phase": "cross", "detour_sign": 1.0},
        )
        for label in (
            "O4_tether",
            "O8_invisible_collider",
            "O9_high_centering",
        )
    }
    assert len({value[:3] for value in commands.values()}) == 3
    assert len({value[3] for value in commands.values()}) == 3
