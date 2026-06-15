"""O4 Tether / Adhesion — elastic attachment (sticky board / cable), spec §8.2 Axis II.

θ = (k, d, L_0, F_break, region). On entering the region a spring-damper engages from
the entry anchor: after the free length ``L_0`` is taken up, a restoring force
``F = k·(s − L_0) + d·|v|`` opposes further motion until ``F`` exceeds ``F_break`` and the
tether snaps. **Explicitly not FEM rope** (spec P1): a D6 spring-damper satisfies the
Hooke's-law narrative and stays numerically stable at RL-parallel scale.

**Key design (spec §8.2, P4):** ``(k, d)`` are tuned to match the O2 compliance field's
tangential-resistance-vs-displacement curve, so the two are proprioceptively
indistinguishable — the constructive evidence that pure-proprioception baselines (B2)
must fail and a visual semantic map is irreplaceable. The recovery diverges on appearance:
a yellow adhesive board ⇒ back off / route around; brown mud ⇒ high-step through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from kino_vla.sim.backend import LocomotionBackend
from kino_vla.sim.operators.base import FailureOperator
from kino_vla.sim.types import ResistanceRegion
from kino_vla.utils.geometry import Rect

if TYPE_CHECKING:
    from kino_vla.map.types import SemanticRegion


class Tether(FailureOperator):
    """An elastic adhesion region: a spring-damper from the entry anchor that can break."""

    name: ClassVar[str] = "O4_tether"
    axis: ClassVar[str] = "II_external_wrench_attachment"

    def __init__(
        self,
        region: Rect,
        k: float,
        d: float,
        l0: float,
        f_break: float,
        d_sink: float = 0.0,
        appearance_class: str = "yellow_adhesive",
        visual_cost: float = 0.7,
    ) -> None:
        if k < 0.0 or d < 0.0:
            raise ValueError("spring stiffness/damping must be non-negative")
        if l0 < 0.0:
            raise ValueError(f"free length L_0 must be non-negative, got {l0}")
        if f_break <= 0.0:
            raise ValueError(f"break force must be positive, got {f_break}")
        if d_sink < 0.0:
            raise ValueError(f"adhesive sink depth must be non-negative, got {d_sink}")
        self._region = region
        self._k = float(k)
        self._d = float(d)
        self._l0 = float(l0)
        self._f_break = float(f_break)
        # Foot penetration into the adhesive layer (a glue-trap board sinks the foot too).
        # Defaults to 0; the constructive O2↔O4 ambiguity pair sets it equal to O2's d_sink
        # so even the base-height channel matches — proprioception is then airtight-identical.
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
                    stiffness_n_per_m=self._k,
                    damping_ns_per_m=self._d,
                    sink_depth_m=self._d_sink,
                    slack_length_m=self._l0,
                    break_force_n=self._f_break,
                    kind="tether",
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
            "k": self._k,
            "d": self._d,
            "L_0": self._l0,
            "F_break": self._f_break,
            "d_sink": self._d_sink,
            "region_cx": self._region.cx,
            "region_cy": self._region.cy,
            "region_hx": self._region.hx,
            "region_hy": self._region.hy,
        }
