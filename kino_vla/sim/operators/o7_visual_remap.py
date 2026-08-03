"""O7 Visual-Physics Remap — deceptive appearance + depth corruption, spec §8.2 Axis III.

θ = (mu_s, mu_d, depth_bias_m, region). The rendering material is decoupled from the
physics material: a patch *looks* like ``"solid_ground"`` (low visual cost — the planner
would walk it) but is physically a low-friction hazard (``mu_d`` ice-like), and a depth
channel corruption ``depth_bias_m`` displaces where RGB-D back-projects it, so the clean
D channel can no longer "see through" the deception (the engineering requirement of spec
§8.2 O7). Instance: visually-deceptive cliff [B].

This is the operator that proves the map's *physics-corrects-vision* inversion (spec §7):
the visual prior is wrong, but the kinodynamic failure overwrites the right odometry cell
regardless of the corrupted depth — physics writes the map where vision lied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from kino_vla.sim.backend import LocomotionBackend
from kino_vla.sim.operators.base import FailureOperator
from kino_vla.sim.types import FrictionRegion
from kino_vla.utils.geometry import Rect

if TYPE_CHECKING:
    from kino_vla.map.types import SemanticRegion


class VisualPhysicsRemap(FailureOperator):
    """A patch whose benign appearance (+ corrupted depth) hides a low-friction hazard."""

    name: ClassVar[str] = "O7_visual_remap"
    axis: ClassVar[str] = "III_geometry_visual_decoupling"

    def __init__(
        self,
        region: Rect,
        mu_s: float,
        mu_d: float,
        depth_bias_m: float,
        appearance_class: str = "solid_ground",
        visual_cost: float = 0.0,
    ) -> None:
        if mu_s < 0.0 or mu_d < 0.0:
            raise ValueError("friction must be non-negative")
        self._region = region
        self._mu_s = float(mu_s)
        self._mu_d = float(mu_d)
        self._depth_bias = float(depth_bias_m)
        self._appearance = appearance_class
        self._visual_cost = float(visual_cost)

    @property
    def region(self) -> Rect:
        return self._region

    def on_reset(self, backend: LocomotionBackend) -> None:
        backend.add_friction_regions(
            [FrictionRegion(rect=self._region, mu_s=self._mu_s, mu_d=self._mu_d)]
        )
        # Isaac's realistic path renders the benign-looking patch into the real RTX camera and
        # corrupts only its segmented depth pixels after acquisition.  CPU/legacy backends keep
        # using scene_region() below, so the controlled core remains backwards compatible.
        install = getattr(backend, "add_visual_depth_fault_region", None)
        if callable(install):
            install(self._region, self._appearance, self._depth_bias)

    def scene_region(self) -> SemanticRegion | None:
        from kino_vla.map.types import SemanticRegion

        # Appearance decoupled from physics: benign look + corrupted depth back-projection.
        return SemanticRegion(
            rect=self._region,
            appearance_class=self._appearance,
            depth_bias_m=self._depth_bias,
            visual_cost=self._visual_cost,
        )

    def get_privileged_state(self) -> dict[str, float]:
        return {
            "mu_s": self._mu_s,
            "mu_d": self._mu_d,
            "depth_bias_m": self._depth_bias,
            "region_cx": self._region.cx,
            "region_cy": self._region.cy,
            "region_hx": self._region.hx,
            "region_hy": self._region.hy,
        }
