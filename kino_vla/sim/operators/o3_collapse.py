"""O3 Triggered Collapse — load-damaged support topology (spec §8.2, Axis I).

The realistic parameterization is θ = (normal-impulse damage threshold, drop, residual support,
region).  It changes real collision topology.  The older dwell/friction swap remains available as
the controlled-core approximation used by historical A0--A7 runs.

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
    """A support region using load damage when configured, otherwise legacy dwell."""

    name: ClassVar[str] = "O3_collapse"
    axis: ClassVar[str] = "I_contact_material_field"

    def __init__(
        self,
        region: Rect,
        mu_collapsed: float,
        trigger_dwell_s: float = 0.0,
        mu_intact: float = 0.8,
        *,
        damage_threshold_ns: float | None = None,
        drop_m: float = 0.0,
        residual_support: float = 0.0,
    ) -> None:
        if mu_collapsed < 0.0 or mu_intact < 0.0:
            raise ValueError("friction must be non-negative")
        if trigger_dwell_s < 0.0:
            raise ValueError(f"trigger_dwell_s must be non-negative, got {trigger_dwell_s}")
        if damage_threshold_ns is not None and damage_threshold_ns <= 0.0:
            raise ValueError("damage_threshold_ns must be positive when provided")
        if drop_m < 0.0:
            raise ValueError("drop_m must be non-negative")
        if not 0.0 <= residual_support <= 1.0:
            raise ValueError("residual_support must be in [0, 1]")
        self._region = region
        self._mu_intact = float(mu_intact)
        self._mu_collapsed = float(mu_collapsed)
        self._trigger_dwell = float(trigger_dwell_s)
        self._damage_threshold_ns = (
            None if damage_threshold_ns is None else float(damage_threshold_ns)
        )
        self._drop_m = float(drop_m)
        self._residual_support = float(residual_support)

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
                    damage_threshold_ns=self._damage_threshold_ns,
                    drop_m=self._drop_m,
                    residual_support=self._residual_support,
                )
            ]
        )

    def get_privileged_state(self) -> dict[str, float]:
        state = {
            "mu_intact": self._mu_intact,
            "mu_collapsed": self._mu_collapsed,
            "trigger_dwell_s": self._trigger_dwell,
            "region_cx": self._region.cx,
            "region_cy": self._region.cy,
            "region_hx": self._region.hx,
            "region_hy": self._region.hy,
            "drop_m": self._drop_m,
            "residual_support": self._residual_support,
        }
        if self._damage_threshold_ns is not None:
            state["damage_threshold_ns"] = self._damage_threshold_ns
        return state
