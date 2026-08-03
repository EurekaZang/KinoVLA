#!/usr/bin/env python3
"""Collect the frozen A7 legacy-trunk-surrogate arm for O2/O4.

This is deliberately a thin adapter around the frozen scale-v8 collector.  It preserves the
scene, camera, controller, paired nuisance profile, RGB randomization, and capture cadence, while
replacing only the anomaly mechanism with the repository's legacy base-wrench implementation.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v8.py"
EXPECTED_V8_SHA256 = "e5b03314e055230d224f7c45e5e0ca91608734bf4c70b39c1395013d71326570"
PROFILE_INDEX = 2
ALLOWED_OPERATORS = {"O2_compliance", "O4_tether"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_v8():
    actual = _sha256(V8)
    if actual != EXPECTED_V8_SHA256:
        raise RuntimeError(f"A7 legacy-surrogate dependency hash mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("kinofail_a7_scale_v8_frozen", V8)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen collector dependency: {V8}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fixed_nuisance_contract(implementation, v8) -> None:
    profile = dict(v8.PHYSICAL_NUISANCE_PROFILES[PROFILE_INDEX])

    def physical_nuisance(record: dict[str, Any]) -> dict[str, Any]:
        if int(record.get("physical_nuisance_profile_index", -1)) != PROFILE_INDEX:
            raise RuntimeError("A7 legacy-surrogate schedule must use nuisance profile 2")
        return {
            **profile,
            "scene_seed": int(record["scene_seed"]),
            "operator_seed": int(record["operator_seed"]),
            "pair_shared": True,
            "derivation": "frozen A7 matched-design physical_nuisance_profile_index=2",
        }

    implementation._physical_nuisance = physical_nuisance


def _install_legacy_surrogate(implementation) -> None:
    original_install = implementation._install_operator
    original_telemetry = implementation._telemetry
    original_active = implementation._active_mechanism

    def install_operator(backend, record, frame, seed):
        del seed
        operator_id = str(record["target_operator"])
        if operator_id not in ALLOWED_OPERATORS:
            return original_install(backend, record, frame, int(record["operator_seed"]))
        region = implementation._region(frame)
        if record["condition"] != "anomaly":
            return None, region

        from kino_vla.sim.operators import ComplianceField, Tether

        parameters = dict(record["physics_parameters"])
        if operator_id == "O2_compliance":
            operator = ComplianceField(
                region,
                parameters["stiffness_n_per_m"],
                parameters["damping_ns_per_m"],
                parameters["sink_depth_m"],
                realistic_foot_model=False,
            )
        else:
            operator = Tether(
                region,
                parameters["stiffness_n_per_m"],
                parameters["damping_ns_per_m"],
                parameters["free_length_m"],
                parameters["snap_force_n"],
                peel_factor=parameters["peel_factor"],
                force_cap_n=parameters["force_cap_n"],
            )
        operator._a7_legacy_ever_inside = False
        operator._a7_legacy_peak_resistance_n = 0.0
        operator.on_reset(backend)
        return operator, region

    def telemetry(backend, operator_id, operator):
        if operator_id not in ALLOWED_OPERATORS:
            return original_telemetry(backend, operator_id, operator)
        states = list(getattr(backend, "_resistance", []))
        speed_mps = 0.0
        robot = getattr(backend, "_robot", None)
        if robot is not None:
            speed_mps = float(
                np.linalg.norm(robot.data.root_lin_vel_b[0].detach().cpu().numpy()[:2])
            )
        state_rows = []
        peak_now = 0.0
        for state in states:
            region = state["region"]
            inside = bool(state.get("inside", False))
            if inside:
                if region.kind == "compliance":
                    peak_now = max(
                        peak_now,
                        float(region.stiffness_n_per_m)
                        + float(region.damping_ns_per_m) * speed_mps,
                    )
                else:
                    peak_now = max(
                        peak_now,
                        float(state.get("max_grip", 0.0))
                        + float(region.damping_ns_per_m) * speed_mps,
                    )
            state_rows.append(
                {
                    "kind": str(region.kind),
                    "inside": inside,
                    "broken": bool(state.get("broken", False)),
                    "max_penetration_m": float(state.get("max_pen", 0.0)),
                    "max_grip_n": float(state.get("max_grip", 0.0)),
                    "phase": str(state.get("phase", "none")),
                }
            )
        if operator is not None:
            operator._a7_legacy_ever_inside = bool(
                getattr(operator, "_a7_legacy_ever_inside", False)
                or any(row["inside"] for row in state_rows)
            )
            operator._a7_legacy_peak_resistance_n = max(
                float(getattr(operator, "_a7_legacy_peak_resistance_n", 0.0)),
                peak_now,
            )
        return {
            "enabled": bool(states),
            "implementation": "legacy_base_wrench",
            "registered_regions": len(states),
            "ever_inside": bool(
                operator is not None
                and getattr(operator, "_a7_legacy_ever_inside", False)
            ),
            "peak_resistance_n": float(
                getattr(operator, "_a7_legacy_peak_resistance_n", 0.0)
                if operator is not None
                else 0.0
            ),
            "states": state_rows,
        }

    def active_mechanism(operator_id, telemetry_value, operator):
        if operator_id not in ALLOWED_OPERATORS:
            return original_active(operator_id, telemetry_value, operator)
        return (
            telemetry_value.get("enabled") is True
            and telemetry_value.get("ever_inside") is True
            and float(telemetry_value.get("peak_resistance_n", 0.0)) > 0.0
        )

    implementation._install_operator = install_operator
    implementation._telemetry = telemetry
    implementation._active_mechanism = active_mechanism


def main() -> None:
    v8 = _load_v8()
    v4 = v8._load_v4()
    v4._install_runtime_validation_adapter()
    implementation = v8._load_10hz_nuisance_implementation()
    _install_fixed_nuisance_contract(implementation, v8)
    _install_legacy_surrogate(implementation)
    v4._install_reachable_exposure_contract(implementation)
    v8._install_route_aligned_o9(implementation)
    implementation.__file__ = __file__
    implementation.main()


if __name__ == "__main__":
    main()
