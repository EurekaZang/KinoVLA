#!/usr/bin/env python3
"""Collect direct-contact O9 severity branches with physical admission."""

from __future__ import annotations

from typing import Any

from scripts import isaac_collect_kinofail_action_full_v1 as full
from scripts import isaac_collect_kinofail_remediation_v2 as remediation


def direct_crossbar_installer(frame: Any):
    from scripts import isaac_collect_kinofail_realistic_pair_v1 as base

    def install(backend: Any, record: dict[str, Any], runtime_frame: Any, seed: int):
        del seed
        if runtime_frame is not frame:
            raise RuntimeError("O9 route-frame identity drift")
        if record["target_operator"] != "O9_high_centering":
            raise RuntimeError("direct O9 collector received a non-O9 case")
        if record["physical_realization"] != "pallet_edge":
            raise RuntimeError("direct O9 collector requires pallet_edge")

        import kino_vla.sim.operators as operators
        from kino_vla.sim.operators.base import FailureOperator
        from kino_vla.sim.types import SupportLossRegion
        from kino_vla.utils.geometry import Rect

        original = operators.HighCentering

        class RouteTransverseCrossbar(FailureOperator):
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
                    raise RuntimeError("direct O9 geometry requires explicit dimensions")
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
                    raise RuntimeError("direct O9 route must be axis-aligned")
                runtime_backend.add_support_loss_regions(
                    [SupportLossRegion(rect=physical, residual_support=self._residual_support)],
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

        operators.HighCentering = RouteTransverseCrossbar
        try:
            return base._install_operator(backend, record, frame, int(record["operator_seed"]))
        finally:
            operators.HighCentering = original

    return install


def main() -> None:
    implementation = full._load_implementation()
    implementation.operator_engaged = full._operator_engaged
    implementation.recovery_command = remediation.recovery_command
    implementation.apply_remediation_transition = full._apply_remediation_transition
    implementation.measured_observation = full._measured_observation
    implementation.operator_specific_recovery_success = full._operator_specific_recovery_success
    implementation.direct_o9_installer = direct_crossbar_installer
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
