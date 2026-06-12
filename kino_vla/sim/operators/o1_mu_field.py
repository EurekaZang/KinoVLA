"""O1 μ-Field — per-region friction rewrite (spec §8.2, Axis I: contact material field).

θ = (μ_s, μ_d, e, region mask). Isaac implementation path: PhysX physics-material
API on a patch collider (static configuration, zero runtime cost); surrogate path:
the backend's friction-region lookup feeding the traction model. Typical instances:
uniform ice [A], one-sided oil slick [A], friction gradient strip [A].
"""

from __future__ import annotations

from typing import ClassVar

from kino_vla.sim.backend import LocomotionBackend
from kino_vla.sim.operators.base import FailureOperator
from kino_vla.sim.types import FrictionRegion
from kino_vla.utils.geometry import Rect


class MuField(FailureOperator):
    """Override ground friction inside a rectangular region."""

    name: ClassVar[str] = "O1_mu_field"
    axis: ClassVar[str] = "I_contact_material_field"

    def __init__(
        self,
        region: Rect,
        mu_s: float,
        mu_d: float,
        restitution: float = 0.0,
    ) -> None:
        if mu_d < 0.0 or mu_s < 0.0:
            raise ValueError(f"friction must be non-negative, got mu_s={mu_s}, mu_d={mu_d}")
        self._region = FrictionRegion(rect=region, mu_s=mu_s, mu_d=mu_d, restitution=restitution)

    @property
    def region(self) -> FrictionRegion:
        return self._region

    def on_reset(self, backend: LocomotionBackend) -> None:
        backend.add_friction_regions([self._region])

    def get_privileged_state(self) -> dict[str, float]:
        rect = self._region.rect
        return {
            "mu_s": self._region.mu_s,
            "mu_d": self._region.mu_d,
            "restitution": self._region.restitution,
            "region_cx": rect.cx,
            "region_cy": rect.cy,
            "region_hx": rect.hx,
            "region_hy": rect.hy,
        }
