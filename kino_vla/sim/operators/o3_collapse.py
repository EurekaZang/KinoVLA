"""O3 Triggered Collapse — thin ice that breaks once loaded (spec §8.2, Axis I).

θ = (trigger_dwell, mu_collapsed, region). A region that is intact (``mu_intact``)
until the robot has dwelled on it past ``trigger_dwell_s`` (a time proxy for the
contact-load threshold F_th of the spec's collider-swap mechanism), after which its
friction collapses to ``mu_collapsed`` for the rest of the episode. Spec instance
[B: region-level topology hazard — the whole homogeneous sheet must be marked, not
just the broken cell].

Forms the **constructive ambiguity pair with O1**: both read as a low-friction
slip, but the decision granularity is opposite — O1 uniform ice wants a local
slow-down/lower-CoM, O3 thin ice wants a region-level traversability ban.
"""

from __future__ import annotations

from typing import ClassVar

from kino_vla.sim.backend import LocomotionBackend
from kino_vla.sim.operators.base import FailureOperator
from kino_vla.sim.types import CollapseRegion
from kino_vla.utils.geometry import Rect


class Collapse(FailureOperator):
    """A region whose friction collapses after the robot dwells on it past a threshold."""

    name: ClassVar[str] = "O3_collapse"
    axis: ClassVar[str] = "I_contact_material_field"

    def __init__(
        self,
        region: Rect,
        mu_collapsed: float,
        trigger_dwell_s: float,
        mu_intact: float = 0.8,
    ) -> None:
        if mu_collapsed < 0.0 or mu_intact < 0.0:
            raise ValueError("friction must be non-negative")
        if trigger_dwell_s < 0.0:
            raise ValueError(f"trigger_dwell_s must be non-negative, got {trigger_dwell_s}")
        self._region = region
        self._mu_intact = float(mu_intact)
        self._mu_collapsed = float(mu_collapsed)
        self._trigger_dwell = float(trigger_dwell_s)

    @property
    def region(self) -> Rect:
        return self._region

    def on_reset(self, backend: LocomotionBackend) -> None:
        backend.add_collapse_regions(
            [
                CollapseRegion(
                    rect=self._region,
                    mu_intact=self._mu_intact,
                    mu_collapsed=self._mu_collapsed,
                    trigger_dwell_s=self._trigger_dwell,
                )
            ]
        )

    def get_privileged_state(self) -> dict[str, float]:
        return {
            "mu_intact": self._mu_intact,
            "mu_collapsed": self._mu_collapsed,
            "trigger_dwell_s": self._trigger_dwell,
            "region_cx": self._region.cx,
            "region_cy": self._region.cy,
            "region_hx": self._region.hx,
            "region_hy": self._region.hy,
        }
