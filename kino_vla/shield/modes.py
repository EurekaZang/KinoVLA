"""Posture/gait modes and their support polygons (spec §6.1, §6.2, §6.7).

Each discrete mode σ (a posture height + stance footprint) defines its own reduced
support polygon and therefore its own safe set 𝒞_σ. Continuous primitives steer the
ZMP *within* a mode; discrete primitives (Switch_Gait, Adjust_Posture) change σ and
must pass the admission rule (spec §6.7) before they are allowed to change the safe
set the CBF defends.

For the diagonal-pair trot the instantaneous contact set is a degenerate line, so we
use the *virtual support polygon* (spec §6.7): the convex hull of the planned
footfalls over a gait cycle, here modelled as the body footprint with a gait-specific
margin δ_σ. All polygons are body-frame, centred on the CoM ground projection, with
unit outward edge normals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from kino_vla.utils.config import Config


@dataclass(frozen=True)
class Mode:
    """A posture/gait mode σ: CoM height z_c and a rectangular support footprint.

    ``lx`` / ``ly`` are the half-length (forward) / half-width (lateral) of the
    (virtual) support polygon [m]; ``delta`` is the barrier contraction margin δ_σ
    (spec §6.2) and ``delta_u`` the ZMP-realizability margin (spec §6.5).
    """

    name: str
    z_c: float
    lx: float
    ly: float
    delta: float
    delta_u: float

    def omega(self, gravity: float) -> float:
        """LIP natural frequency ω = √(g / z_c) for this posture (spec §6.1)."""
        return math.sqrt(gravity / self.z_c)

    def support_polygon(self) -> tuple[np.ndarray, np.ndarray]:
        """Half-plane form (A, b) of the rectangular support polygon, unit normals."""
        a_mat = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        b_vec = np.array([self.lx, self.lx, self.ly, self.ly])
        return a_mat, b_vec

    def barriers(self, xi: np.ndarray) -> np.ndarray:
        """Per-edge barrier values h_j(x) = (b_j − δ) − a_j·ξ for this mode (spec §6.2)."""
        a_mat, b_vec = self.support_polygon()
        return (b_vec - self.delta) - a_mat @ np.asarray(xi, dtype=np.float64).reshape(2)


def load_modes(cfg: Config) -> dict[str, Mode]:
    """Build the mode table from a shield config's ``modes`` block."""
    modes: dict[str, Mode] = {}
    for name, m in cfg.modes.to_dict().items():
        modes[name] = Mode(
            name=name,
            z_c=float(m["z_c"]),
            lx=float(m["lx"]),
            ly=float(m["ly"]),
            delta=float(m["delta"]),
            delta_u=float(m["delta_u"]),
        )
    return modes
