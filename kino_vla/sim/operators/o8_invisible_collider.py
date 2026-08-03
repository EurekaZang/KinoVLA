"""O8 visually weak obstacle — geometry/appearance decoupled from collision (Axis III).

θ = region geometry. The "glass door": a collider with nothing rendered, so RGB gives
no warning and only proprioception (a velocity command that produces no motion — a
tracking-error spike) reveals it. Spec instance [B: topology-mark + detour]; the
recovery is to mark the region untraversable and replan around it. Pure scripted
collider, negligible cost.
"""

from __future__ import annotations

from typing import ClassVar

from kino_vla.sim.backend import LocomotionBackend
from kino_vla.sim.operators.base import FailureOperator
from kino_vla.sim.types import BlockingRegion
from kino_vla.utils.geometry import Rect


class InvisibleCollider(FailureOperator):
    """Place a legacy hidden wall or a realistic transparent/low-contrast obstacle."""

    name: ClassVar[str] = "O8_invisible_collider"
    axis: ClassVar[str] = "III_geometry_visual_decoupling"

    def __init__(
        self,
        region: Rect,
        *,
        height_m: float | None = None,
        collision_enabled: bool = True,
        geometry_kind: str = "legacy_invisible_wall",
        optical_transmission: float = 1.0,
    ) -> None:
        if height_m is not None and height_m <= 0.0:
            raise ValueError("obstacle height must be positive when provided")
        if geometry_kind not in {
            "legacy_invisible_wall",
            "transparent_acrylic",
            "occluded_low_bar",
        }:
            raise ValueError(f"unsupported O8 geometry {geometry_kind!r}")
        if not 0.0 <= optical_transmission <= 1.0:
            raise ValueError("optical transmission must be in [0, 1]")
        self._region = region
        self._height_m = height_m
        self._collision_enabled = bool(collision_enabled)
        self._geometry_kind = geometry_kind
        self._optical_transmission = float(optical_transmission)

    @property
    def region(self) -> Rect:
        return self._region

    def on_reset(self, backend: LocomotionBackend) -> None:
        backend.add_blocking_regions(
            [
                BlockingRegion(
                    rect=self._region,
                    height_m=self._height_m,
                    collision_enabled=self._collision_enabled,
                    geometry_kind=self._geometry_kind,
                    optical_transmission=self._optical_transmission,
                )
            ]
        )

    def get_privileged_state(self) -> dict[str, float]:
        return {
            "region_cx": self._region.cx,
            "region_cy": self._region.cy,
            "region_hx": self._region.hx,
            "region_hy": self._region.hy,
            "height_m": 0.0 if self._height_m is None else self._height_m,
            "collision_enabled": float(self._collision_enabled),
            "rendered": float(self._geometry_kind != "legacy_invisible_wall"),
            "optical_transmission": self._optical_transmission,
        }
