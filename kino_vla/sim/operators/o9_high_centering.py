"""O9 High-Centering — belly-out beaching over a low ridge (spec §8.2, Axis III).

θ = (region, residual_support). A low ridge lifts the feet partially clear so the
belly bears load: friction is nominal but the legs can no longer transmit tangential
force, so control authority collapses (the robot is "beached"). Spec instance
[B: a pose primitive must raise the chassis]. ``residual_support`` in (0, 1] is the
fraction of foot support that remains inside the ridge footprint.
"""

from __future__ import annotations

from typing import ClassVar

from kino_vla.sim.backend import LocomotionBackend
from kino_vla.sim.operators.base import FailureOperator
from kino_vla.sim.types import SupportLossRegion
from kino_vla.utils.geometry import Rect


class HighCentering(FailureOperator):
    """Mark a region where foot support drops to ``residual_support`` (chassis grounds out)."""

    name: ClassVar[str] = "O9_high_centering"
    axis: ClassVar[str] = "III_geometry_visual_decoupling"

    def __init__(self, region: Rect, residual_support: float = 0.15) -> None:
        if not 0.0 < residual_support <= 1.0:
            raise ValueError(f"residual_support must be in (0, 1], got {residual_support}")
        self._region = region
        self._residual_support = float(residual_support)

    @property
    def region(self) -> Rect:
        return self._region

    def on_reset(self, backend: LocomotionBackend) -> None:
        backend.add_support_loss_regions(
            [SupportLossRegion(rect=self._region, residual_support=self._residual_support)]
        )

    def get_privileged_state(self) -> dict[str, float]:
        return {
            "region_cx": self._region.cx,
            "region_cy": self._region.cy,
            "region_hx": self._region.hx,
            "region_hy": self._region.hy,
            "residual_support": self._residual_support,
        }
