"""Command-agnostic contact mechanics for peelable adhesive terrain.

This interface deliberately omits the explicit ``force_release`` input supported by the v2
development model.  An adhesive bond can therefore release only from measured foot unloading
and lift.  The controller command is not an input to the contact law, which is required for a
valid Continue/Backstep consequence comparison.
"""

from __future__ import annotations

import numpy as np

from kino_vla.sim.adhesion_v2 import (
    SurfaceAdhesionConfig,
    SurfaceAdhesionState,
    SurfaceAdhesionUpdate,
    step_surface_adhesion as _step_surface_adhesion_v2,
)


def step_surface_adhesion(
    config: SurfaceAdhesionConfig,
    state: SurfaceAdhesionState,
    foot_pos_xyz: np.ndarray,
    foot_vel_xyz: np.ndarray,
    contact_force_n: float,
    *,
    dt_s: float,
    allow_attach: bool = True,
) -> SurfaceAdhesionUpdate:
    """Advance one bond using contact, pose and velocity only.

    Release is governed exclusively by the frozen unload-count and peel-height thresholds in
    ``SurfaceAdhesionConfig``.  In particular, no high-level action can request release.
    """

    return _step_surface_adhesion_v2(
        config,
        state,
        foot_pos_xyz,
        foot_vel_xyz,
        contact_force_n,
        dt_s=dt_s,
        allow_attach=allow_attach,
        force_release=False,
    )


__all__ = [
    "SurfaceAdhesionConfig",
    "SurfaceAdhesionState",
    "SurfaceAdhesionUpdate",
    "step_surface_adhesion",
]
