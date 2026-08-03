#!/usr/bin/env python3
"""Collect confirmatory O9 pairs with direct chassis-contact admission.

This is an outer, hash-bound adapter around the frozen confirmatory collector.
It changes only the O9 mechanism realization and its physical admission gate:
the obstacle is a route-transverse pallet crossbar, the robot approaches with
pair-shared diagnostic creep, and a sample is admitted only when the frozen
telemetry contract proves sustained belly/base loading and foot unloading.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any

from scripts import isaac_collect_kinofail_confirmatory_pair_v1 as confirmatory


ROOT = Path(__file__).resolve().parents[1]
NUISANCE_KEYS = {
    "profile_index",
    "start_progress_m",
    "start_lateral_offset_m",
    "start_heading_offset_rad",
    "forward_speed_mps",
    "controller_target_lateral_offset_m",
    "physics_seed",
    "pair_shared",
    "spawn_base_height_m",
}


def _install_o9_nuisance_contract(implementation: Any) -> None:
    def physical_nuisance(record: dict[str, Any]) -> dict[str, Any]:
        value = record.get("physical_nuisance")
        if not isinstance(value, dict) or set(value) != NUISANCE_KEYS:
            raise RuntimeError("O9 direct schedule has an invalid nuisance contract")
        nuisance = dict(value)
        if nuisance["pair_shared"] is not True:
            raise RuntimeError("O9 direct nuisance must be pair-shared")
        if int(nuisance["physics_seed"]) != int(record["operator_seed"]):
            raise RuntimeError("O9 direct operator and physics seeds differ")
        if abs(float(nuisance["start_progress_m"]) - 0.35) > 1.0e-9:
            raise RuntimeError("O9 direct start progress drift")
        if abs(float(nuisance["forward_speed_mps"]) - 0.08) > 1.0e-9:
            raise RuntimeError("O9 direct diagnostic speed drift")
        if abs(float(nuisance["spawn_base_height_m"]) - 0.43) > 1.0e-9:
            raise RuntimeError("O9 direct spawn height drift")
        if abs(float(nuisance["start_lateral_offset_m"])) > 0.02:
            raise RuntimeError("O9 direct lateral nuisance outside frozen range")
        if abs(float(nuisance["start_heading_offset_rad"])) > 0.015:
            raise RuntimeError("O9 direct heading nuisance outside frozen range")
        return nuisance

    prior_collect = implementation._collect_one

    def collect_one(backend: Any, record: dict[str, Any], *args: Any, **kwargs: Any):
        if record["target_operator"] != "O9_high_centering":
            raise RuntimeError("O9 direct collector refuses non-O9 records")
        backend._spawn_z = float(record["physical_nuisance"]["spawn_base_height_m"])
        return prior_collect(backend, record, *args, **kwargs)

    implementation._physical_nuisance = physical_nuisance
    implementation._collect_one = collect_one


def _install_transverse_belly_crossbar(implementation: Any) -> None:
    prior_install = implementation._install_operator

    def install_operator(backend: Any, record: dict[str, Any], frame: Any, seed: int):
        if record["target_operator"] != "O9_high_centering":
            raise RuntimeError("O9 direct collector refuses non-O9 records")
        if record["condition"] != "anomaly":
            return prior_install(backend, record, frame, seed)
        if record["physical_realization"] != "pallet_edge":
            raise RuntimeError("O9 direct collection requires pallet_edge")

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
                region: Any,
                residual_support: float = 0.15,
                *,
                ridge_height_m: float | None = None,
                ridge_width_m: float | None = None,
                geometry_kind: str = "pallet_edge",
            ) -> None:
                if geometry_kind != "pallet_edge":
                    raise RuntimeError(f"unexpected O9 geometry: {geometry_kind}")
                if ridge_height_m is None or ridge_width_m is None:
                    raise RuntimeError("O9 direct geometry requires explicit dimensions")
                self._region = region
                self._residual_support = float(residual_support)
                self._ridge_height_m = float(ridge_height_m)
                self._ridge_width_m = float(ridge_width_m)

            @property
            def region(self):
                return self._region

            def on_reset(self, runtime_backend: Any) -> None:
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
                    raise RuntimeError("O9 direct route must be axis-aligned")
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

            def get_privileged_state(self) -> dict[str, float]:
                return {
                    "region_cx": float(self._region.cx),
                    "region_cy": float(self._region.cy),
                    "region_hx": float(self._region.hx),
                    "region_hy": float(self._region.hy),
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


def _install_direct_semantic_gate(implementation: Any) -> None:
    prior_active = implementation._active_mechanism

    def active_mechanism(operator_id: str, telemetry: dict[str, Any], operator: Any) -> bool:
        if operator_id != "O9_high_centering":
            return bool(prior_active(operator_id, telemetry, operator))
        from kino_vla.data.o9_semantics import evaluate_o9_high_centering

        exposed = int(telemetry.get("scale_region_exposure_steps", 0)) > 0
        return exposed and bool(evaluate_o9_high_centering(telemetry)["passed"])

    implementation._active_mechanism = active_mechanism


def main() -> None:
    try:
        v4 = confirmatory._load_v4()
        v4._install_runtime_validation_adapter()
        implementation = confirmatory._load_confirmatory_implementation()
        _install_o9_nuisance_contract(implementation)
        v4._install_reachable_exposure_contract(implementation)
        _install_transverse_belly_crossbar(implementation)
        _install_direct_semantic_gate(implementation)
        implementation.__file__ = __file__
        result = implementation.main()
        status = int(result) if result is not None else 0
    except BaseException:
        traceback.print_exc()
        status = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(status)


if __name__ == "__main__":
    main()
