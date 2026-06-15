"""O2 Compliance-Field — soft sinking ground (mud), spec §8.2 Axis I.

θ = (k_c, c_c, d_sink, region). A compliant-contact patch: as the robot drives through
it, a displacement-dependent tangential resistance ``F = k_c·s + c_c·|v|`` opposes motion
and the chassis sinks by ``d_sink`` (a measurable base-height drop). **Explicitly not FEM
softbody** (infeasible at RL-parallel scale, spec P1) — compliant contact + geometric
sink. A/B boundary is continuous in θ (shallow mud [A] → deep bog [B], Suite-Bound).

Forms the **constructive ambiguity pair with O4** (spec §8.2, P4): tuned to share an
identical tangential-resistance-vs-displacement curve with the O4 adhesive tether, so
proprioception alone cannot tell soft mud from a sticky board — only the visual map
(brown mud vs yellow adhesive) decides high-step-through vs back-off-and-route.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from kino_vla.sim.backend import LocomotionBackend
from kino_vla.sim.operators.base import FailureOperator
from kino_vla.sim.types import ResistanceRegion
from kino_vla.utils.geometry import Rect

if TYPE_CHECKING:
    from kino_vla.map.types import SemanticRegion


class ComplianceField(FailureOperator):
    """A soft, sinking ground patch with displacement-dependent tangential resistance."""

    name: ClassVar[str] = "O2_compliance"
    axis: ClassVar[str] = "I_contact_material_field"

    def __init__(
        self,
        region: Rect,
        k_c: float,
        c_c: float,
        d_sink: float,
        appearance_class: str = "brown_mud",
        visual_cost: float = 0.3,
    ) -> None:
        if k_c < 0.0 or c_c < 0.0:
            raise ValueError("compliance stiffness/damping must be non-negative")
        if d_sink < 0.0:
            raise ValueError(f"sink depth must be non-negative, got {d_sink}")
        self._region = region
        self._k_c = float(k_c)
        self._c_c = float(c_c)
        self._d_sink = float(d_sink)
        self._appearance = appearance_class
        self._visual_cost = float(visual_cost)

    @property
    def region(self) -> Rect:
        return self._region

    def on_reset(self, backend: LocomotionBackend) -> None:
        backend.add_resistance_regions(
            [
                ResistanceRegion(
                    rect=self._region,
                    stiffness_n_per_m=self._k_c,
                    damping_ns_per_m=self._c_c,
                    sink_depth_m=self._d_sink,
                    kind="compliance",
                )
            ]
        )

    def scene_region(self) -> SemanticRegion | None:
        from kino_vla.map.types import SemanticRegion

        return SemanticRegion(
            rect=self._region,
            appearance_class=self._appearance,
            visual_cost=self._visual_cost,
        )

    def get_privileged_state(self) -> dict[str, float]:
        return {
            "k_c": self._k_c,
            "c_c": self._c_c,
            "d_sink": self._d_sink,
            "region_cx": self._region.cx,
            "region_cy": self._region.cy,
            "region_hx": self._region.hx,
            "region_hy": self._region.hy,
        }
