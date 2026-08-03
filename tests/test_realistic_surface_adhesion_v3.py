from __future__ import annotations

import inspect

import numpy as np

from kino_vla.sim.adhesion import IrregularRegion
from kino_vla.sim.adhesion_v3 import (
    SurfaceAdhesionConfig,
    SurfaceAdhesionState,
    step_surface_adhesion,
)


def _config() -> SurfaceAdhesionConfig:
    return SurfaceAdhesionConfig(
        region=IrregularRegion(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5))),
        peel_height_m=0.02,
        unload_steps_to_peel=2,
    )


def test_v3_contact_law_has_no_command_or_force_release_input() -> None:
    parameters = inspect.signature(step_surface_adhesion).parameters
    assert "command" not in parameters
    assert "force_release" not in parameters


def test_v3_loaded_bond_cannot_be_released_by_high_level_action() -> None:
    config = _config()
    attached = step_surface_adhesion(
        config,
        SurfaceAdhesionState(),
        np.array([0.3, 0.0, 0.0]),
        np.zeros(3),
        20.0,
        dt_s=0.02,
    ).state
    update = step_surface_adhesion(
        config,
        attached,
        np.array([0.31, 0.0, 0.0]),
        np.array([-0.3, 0.0, 0.0]),
        20.0,
        dt_s=0.02,
    )
    assert update.state.attached
    assert update.event == "none"


def test_v3_release_requires_measured_unload_and_lift() -> None:
    config = _config()
    state = step_surface_adhesion(
        config,
        SurfaceAdhesionState(),
        np.array([0.3, 0.0, 0.0]),
        np.zeros(3),
        20.0,
        dt_s=0.02,
    ).state
    for index in range(2):
        update = step_surface_adhesion(
            config,
            state,
            np.array([0.32, 0.0, 0.03]),
            np.array([0.0, 0.0, 0.2]),
            0.0,
            dt_s=0.02,
        )
        state = update.state
    assert update.event == "peeled"
    assert not state.attached
