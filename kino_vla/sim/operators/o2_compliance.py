"""O2 Compliance-Field — soft sinking ground (mud), spec §8.2 Axis I.

θ = (k_c, c_c, d_sink, region). A soft-ground **drag field**: a *bounded* tangential
resistance ``F = drag + c_c·|v|`` (a constant Coulomb-like sinking drag ``drag``=k_c[N] plus a
viscous term) opposes motion, and the chassis sinks by ``d_sink`` (a measurable base-height
drop). Real mud is a drag field — resistance depends on speed, NOT on how far you have
travelled — so it is **crossable at reduced speed**, not an elastic trap. **Explicitly not FEM
softbody** (infeasible at RL-parallel scale, spec P1). A/B boundary is continuous in θ (shallow
mud [A] → deep bog [B], Suite-Bound).

NOTE (§6 #38): O2 was previously an elastic SPRING (``F = k·path_len``) to share an identical
proprioceptive curve with the O4 adhesive tether — the §8.2 P4 ambiguity pair. That spring grew
unboundedly with distance and TRAPPED the robot, contradicting "compliant = traversable". O2 is
now a physically-faithful bounded drag, which RETIRES the O2↔O4 matched-proprioception claim
(mud=drag ≠ tether=spring). Mud and adhesive still differ in appearance (brown vs yellow) and
recovery (high-step-through vs back-off-and-route); the "vision-irreplaceable" leg now needs a
redesigned vision-necessary pair (a human spec decision).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from kino_vla.sim.backend import LocomotionBackend
from kino_vla.sim.operators.base import FailureOperator
from kino_vla.sim.terramechanics import FootTerramechanicsConfig
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
        *,
        realistic_foot_model: bool = False,
        shear_velocity_scale_mps: float = 0.12,
        matched_control: bool = False,
        longitudinal_axis: str = "x",
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
        self._realistic_foot_model = bool(realistic_foot_model)
        self._shear_velocity_scale_mps = float(shear_velocity_scale_mps)
        self._matched_control = bool(matched_control)
        self._longitudinal_axis = str(longitudinal_axis)
        if self._longitudinal_axis not in {"x", "y"}:
            raise ValueError("longitudinal_axis must be 'x' or 'y'")
        if self._matched_control and not self._realistic_foot_model:
            raise ValueError("matched O2 control requires the realistic per-foot model")

    @property
    def region(self) -> Rect:
        return self._region

    def on_reset(self, backend: LocomotionBackend) -> None:
        if self._realistic_foot_model:
            add_foot_compliance = getattr(backend, "add_foot_compliance", None)
            if not callable(add_foot_compliance):
                raise TypeError("realistic O2 requires a backend with add_foot_compliance")
            add_foot_compliance(
                FootTerramechanicsConfig(
                    region=self._region,
                    max_sink_depth_m=self._d_sink,
                    shear_retention=self._c_c,
                    vertical_stiffness_n_per_m=self._k_c,
                    shear_velocity_scale_mps=self._shear_velocity_scale_mps,
                    longitudinal_axis=self._longitudinal_axis,
                ),
                matched_control=self._matched_control,
            )
            return
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
            "realistic_foot_model": float(self._realistic_foot_model),
            "shear_retention": self._c_c,
            "matched_control": float(self._matched_control),
            "longitudinal_axis_is_y": float(self._longitudinal_axis == "y"),
        }
