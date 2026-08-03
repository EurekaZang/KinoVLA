#!/usr/bin/env python3
"""Collect the scale-v8 O9 repair with direct belly-contact admission.

This wrapper preserves the frozen scale-v8 scenes, counterfactual pairing,
appearance interventions, five physical nuisance profiles, controller, camera,
and 10 Hz logging.  It changes only the part invalidated by the O9 audit:

* the robot starts above the center of a route-transverse pallet crossbar;
* the crossbar reaches the Go2 underside instead of stopping at the head/legs;
* O9 is admitted only after sustained base contact and partial foot unloading.

The wrapper depends on the hash-locked scale-v8 collector and does not edit it,
so the running independent reconfirmation collection is unaffected.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v8.py"
EXPECTED_V8_SHA256 = "e5b03314e055230d224f7c45e5e0ca91608734bf4c70b39c1395013d71326570"
REPAIR_START_PROGRESS_M = 0.35
REPAIR_SPAWN_BASE_HEIGHT_M = 0.43


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_v8():
    actual = _sha256(V8)
    if actual != EXPECTED_V8_SHA256:
        raise RuntimeError(f"O9 semantic repair v1 scale-v8 dependency mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("kinofail_o9_semantic_repair_scale_v8", V8)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {V8}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_beached_start_contract(implementation) -> None:
    prior_nuisance = implementation._physical_nuisance
    prior_collect = implementation._collect_one

    def physical_nuisance(record):
        profile = dict(prior_nuisance(record))
        profile["pre_repair_start_progress_m"] = float(profile["start_progress_m"])
        profile["start_progress_m"] = REPAIR_START_PROGRESS_M
        profile["spawn_base_height_m"] = REPAIR_SPAWN_BASE_HEIGHT_M
        profile["o9_semantic_repair"] = (
            "matched pair starts at ridge center; anomaly settles onto transverse "
            "belly crossbar and nominal settles onto unobstructed floor"
        )
        return profile

    def collect_one(backend, record, *args, **kwargs):
        if record["target_operator"] != "O9_high_centering":
            raise RuntimeError("O9 semantic repair collector refuses non-O9 records")
        backend._spawn_z = REPAIR_SPAWN_BASE_HEIGHT_M
        return prior_collect(backend, record, *args, **kwargs)

    implementation._physical_nuisance = physical_nuisance
    implementation._collect_one = collect_one


def _install_route_transverse_pallet_crossbar(implementation) -> None:
    prior_install = implementation._install_operator

    def install_operator(backend, record, frame, seed):
        if record["target_operator"] != "O9_high_centering":
            raise RuntimeError("O9 semantic repair collector refuses non-O9 records")
        if record["condition"] != "anomaly":
            return prior_install(backend, record, frame, seed)
        if record["physical_realization"] != "pallet_edge":
            raise RuntimeError("O9 semantic repair requires pallet_edge")

        import kino_vla.sim.operators as operators
        from kino_vla.sim.operators.base import FailureOperator
        from kino_vla.sim.types import SupportLossRegion
        from kino_vla.utils.geometry import Rect

        original = operators.HighCentering

        class RouteTransversePalletCrossbar(FailureOperator):
            name = "O9_high_centering"
            axis = "III_geometry_visual_decoupling"

            def __init__(
                self,
                region,
                residual_support=0.15,
                *,
                ridge_height_m=None,
                ridge_width_m=None,
                geometry_kind="pallet_edge",
            ):
                if geometry_kind != "pallet_edge":
                    raise RuntimeError(
                        f"unexpected O9 semantic-repair geometry {geometry_kind}"
                    )
                if ridge_height_m is None or ridge_width_m is None:
                    raise RuntimeError("O9 semantic repair requires explicit dimensions")
                self._region = region
                self._residual_support = float(residual_support)
                self._ridge_height_m = float(ridge_height_m)
                self._ridge_width_m = float(ridge_width_m)

            @property
            def region(self):
                return self._region

            def on_reset(self, runtime_backend):
                if abs(float(frame.direction[0])) >= 1.0 - 1.0e-6:
                    physical = Rect(
                        self._region.cx,
                        self._region.cy,
                        0.5 * self._ridge_width_m,
                        self._region.hy,
                    )
                elif abs(float(frame.direction[1])) >= 1.0 - 1.0e-6:
                    physical = Rect(
                        self._region.cx,
                        self._region.cy,
                        self._region.hx,
                        0.5 * self._ridge_width_m,
                    )
                else:
                    raise ValueError(
                        "O9 semantic repair requires an axis-aligned audited route"
                    )
                runtime_backend.add_support_loss_regions(
                    [
                        SupportLossRegion(
                            rect=physical,
                            residual_support=self._residual_support,
                        )
                    ],
                    height_m=self._ridge_height_m,
                    geometry_kind="pallet_edge",
                )

            def get_privileged_state(self):
                return {
                    "region_cx": self._region.cx,
                    "region_cy": self._region.cy,
                    "region_hx": self._region.hx,
                    "region_hy": self._region.hy,
                    "residual_support": self._residual_support,
                    "ridge_height_m": self._ridge_height_m,
                    "ridge_width_m": self._ridge_width_m,
                }

        operators.HighCentering = RouteTransversePalletCrossbar
        try:
            return prior_install(backend, record, frame, seed)
        finally:
            operators.HighCentering = original

    implementation._install_operator = install_operator


def _install_direct_semantic_gate(implementation) -> None:
    prior_active = implementation._active_mechanism

    def active_mechanism(operator_id, telemetry, operator):
        if operator_id != "O9_high_centering":
            return prior_active(operator_id, telemetry, operator)
        from kino_vla.data.o9_semantics import evaluate_o9_high_centering

        exposed = int(telemetry.get("scale_region_exposure_steps", 0)) > 0
        return exposed and bool(evaluate_o9_high_centering(telemetry)["passed"])

    implementation._active_mechanism = active_mechanism


def main() -> None:
    v8 = _load_v8()
    v4 = v8._load_v4()
    v4._install_runtime_validation_adapter()
    implementation = v8._load_10hz_nuisance_implementation()
    v8._install_balanced_nuisance_contract(implementation)
    v4._install_reachable_exposure_contract(implementation)
    _install_beached_start_contract(implementation)
    _install_route_transverse_pallet_crossbar(implementation)
    _install_direct_semantic_gate(implementation)
    implementation.__file__ = __file__
    implementation.main()


if __name__ == "__main__":
    main()
