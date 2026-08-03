"""O9 High-Centering — belly-out beaching over a low ridge (spec §8.2, Axis III).

θ = (region, ridge geometry). A low ridge lifts the feet partially clear so the
belly bears load: friction is nominal but the legs can no longer transmit tangential
force, so control authority collapses (the robot is "beached"). Spec instance
[B: a pose primitive must raise the chassis].  In the realistic Isaac path, support and belly
contact are measured from contact sensors; ``residual_support`` remains only a surrogate target.
"""

from __future__ import annotations

from typing import ClassVar

from kino_vla.sim.backend import LocomotionBackend
from kino_vla.sim.operators.base import FailureOperator
from kino_vla.sim.types import SupportLossRegion
from kino_vla.utils.geometry import Rect


class HighCentering(FailureOperator):
    """Author a ridge/pallet edge and measure whether the chassis grounds out."""

    name: ClassVar[str] = "O9_high_centering"
    axis: ClassVar[str] = "III_geometry_visual_decoupling"

    def __init__(
        self,
        region: Rect,
        residual_support: float = 0.15,
        *,
        ridge_height_m: float | None = None,
        ridge_width_m: float | None = None,
        geometry_kind: str = "box_ridge",
    ) -> None:
        if not 0.0 < residual_support <= 1.0:
            raise ValueError(f"residual_support must be in (0, 1], got {residual_support}")
        self._region = region
        self._residual_support = float(residual_support)
        if ridge_height_m is not None and ridge_height_m <= 0.0:
            raise ValueError("ridge_height_m must be positive")
        if ridge_width_m is not None and ridge_width_m <= 0.0:
            raise ValueError("ridge_width_m must be positive")
        if geometry_kind not in {
            "box_ridge",
            "rounded_ridge",
            "longitudinal_rounded_ridge",
            "central_pallet_runner",
            "pallet_edge",
        }:
            raise ValueError("unsupported high-centering geometry")
        self._ridge_height_m = None if ridge_height_m is None else float(ridge_height_m)
        self._ridge_width_m = None if ridge_width_m is None else float(ridge_width_m)
        self._geometry_kind = geometry_kind

    @property
    def region(self) -> Rect:
        return self._region

    def on_reset(self, backend: LocomotionBackend) -> None:
        if self._ridge_width_m is None:
            physical_region = self._region
        elif self._geometry_kind == "longitudinal_rounded_ridge":
            physical_region = Rect(
                self._region.cx,
                self._region.cy,
                self._region.hx,
                0.5 * self._ridge_width_m,
            )
        elif self._geometry_kind == "central_pallet_runner":
            physical_region = Rect(
                self._region.cx,
                self._region.cy,
                0.5 * self._ridge_width_m,
                min(self._region.hy, 0.05),
            )
        else:
            physical_region = Rect(
                self._region.cx,
                self._region.cy,
                0.5 * self._ridge_width_m,
                self._region.hy,
            )
        backend.add_support_loss_regions(
            [SupportLossRegion(rect=physical_region, residual_support=self._residual_support)],
            height_m=self._ridge_height_m,
            geometry_kind=self._geometry_kind,
        )

    def get_privileged_state(self) -> dict[str, float]:
        state = {
            "region_cx": self._region.cx,
            "region_cy": self._region.cy,
            "region_hx": self._region.hx,
            "region_hy": self._region.hy,
            "residual_support": self._residual_support,
        }
        if self._ridge_height_m is not None:
            state["ridge_height_m"] = self._ridge_height_m
        if self._ridge_width_m is not None:
            state["ridge_width_m"] = self._ridge_width_m
        return state
