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
        peel_factor: float = 0.3,
        force_cap_n: float = float("inf"),
        force_offset_n: float = 0.0,
        p0_m: float = 0.0,
        k2_n_per_m: float = 0.0,
    ) -> None:
        if k < 0.0 or d < 0.0:
            raise ValueError("spring stiffness/damping must be non-negative")
        if l0 < 0.0:
            raise ValueError(f"free length L_0 must be non-negative, got {l0}")
        if f_break <= 0.0:
            raise ValueError(f"break force must be positive, got {f_break}")
        if d_sink < 0.0:
            raise ValueError(f"adhesive sink depth must be non-negative, got {d_sink}")
        if not 0.0 <= peel_factor <= 1.0:
            raise ValueError(f"peel_factor must be in [0,1], got {peel_factor}")
        if force_cap_n < 0.0:
            raise ValueError(f"force_cap_n must be non-negative, got {force_cap_n}")
        if force_offset_n < 0.0:
            raise ValueError(f"force_offset_n must be non-negative, got {force_offset_n}")
        if p0_m < 0.0:
            raise ValueError(f"two-phase plateau depth p0_m must be non-negative, got {p0_m}")
        if k2_n_per_m < 0.0:
            raise ValueError(f"two-phase ramp stiffness k2_n_per_m must be non-negative, "
                             f"got {k2_n_per_m}")
        self._region = region
        self._k = float(k)
        self._d = float(d)
        self._l0 = float(l0)
        self._f_break = float(f_break)
        # #49 peel-plateau force shaping (E1; default inf/0 ⇒ unshaped, byte-identical).
        self._force_cap_n = float(force_cap_n)
        self._force_offset_n = float(force_offset_n)
        # A4.1 two-phase delayed-divergence (default 0/0 ⇒ the #49 path, byte-identical).
        self._p0_m = float(p0_m)
        self._k2_n_per_m = float(k2_n_per_m)
        # Bug-1: the adhesive resists going DEEPER at full strength but only peel_factor of that in
        # reverse, so the dog escapes by backing off (peeling), not by pushing through.
        self._peel_factor = float(peel_factor)
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
                    peel_factor=self._peel_factor,
                    force_cap_n=self._force_cap_n,
                    force_offset_n=self._force_offset_n,
                    p0_m=self._p0_m,
                    k2_n_per_m=self._k2_n_per_m,
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
            "force_cap_n": self._force_cap_n,
            "force_offset_n": self._force_offset_n,
            "peel_factor": self._peel_factor,
            "p0_m": self._p0_m,
            "k2_n_per_m": self._k2_n_per_m,
            "region_cx": self._region.cx,
            "region_cy": self._region.cy,
            "region_hx": self._region.hx,
            "region_hy": self._region.hy,
        }
