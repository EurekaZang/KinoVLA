"""O8 Invisible Collider — an impassable collider with no render geometry (spec §8.2, Axis III).

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
    """Place an impassable, unrendered wall over a rectangular region."""

    name: ClassVar[str] = "O8_invisible_collider"
    axis: ClassVar[str] = "III_geometry_visual_decoupling"

    def __init__(self, region: Rect) -> None:
        self._region = region

    @property
    def region(self) -> Rect:
        return self._region

    def on_reset(self, backend: LocomotionBackend) -> None:
        backend.add_blocking_regions([BlockingRegion(rect=self._region)])

    def get_privileged_state(self) -> dict[str, float]:
        return {
            "region_cx": self._region.cx,
            "region_cy": self._region.cy,
            "region_hx": self._region.hx,
            "region_hy": self._region.hy,
            "rendered": 0.0,  # the defining property: collider present, visual absent
        }
