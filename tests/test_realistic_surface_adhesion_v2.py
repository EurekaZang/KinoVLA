from __future__ import annotations

import numpy as np
import pytest

from kino_vla.sim.adhesion import IrregularRegion
from kino_vla.sim.adhesion_v2 import (
    SurfaceAdhesionConfig,
    SurfaceAdhesionState,
    step_surface_adhesion,
)


def _config(**overrides: object) -> SurfaceAdhesionConfig:
    values: dict[str, object] = {
        "region": IrregularRegion(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5))),
        "surface_z_m": 0.0,
        "attach_contact_force_n": 5.0,
        "detach_contact_force_n": 2.0,
        "attach_height_tolerance_m": 0.04,
        "tangential_stiffness_n_per_m": 100.0,
        "tangential_damping_ns_per_m": 0.0,
        "normal_stiffness_n_per_m": 100.0,
        "normal_damping_ns_per_m": 0.0,
        "tangential_force_cap_n": 8.0,
        "normal_force_cap_n": 6.0,
        "peel_height_m": 0.02,
        "unload_steps_to_peel": 2,
        "reattach_cooldown_steps": 2,
    }
    values.update(overrides)
    return SurfaceAdhesionConfig(**values)  # type: ignore[arg-type]


def _attach(config: SurfaceAdhesionConfig) -> SurfaceAdhesionState:
    update = step_surface_adhesion(
        config,
        SurfaceAdhesionState(),
        np.array([0.3, 0.0, 0.0]),
        np.zeros(3),
        20.0,
        dt_s=0.02,
    )
    assert update.event == "attached"
    return update.state


def test_surface_bond_requires_load_and_region() -> None:
    config = _config()
    outside = step_surface_adhesion(
        config,
        SurfaceAdhesionState(),
        np.array([-0.1, 0.0, 0.0]),
        np.zeros(3),
        20.0,
        dt_s=0.02,
    )
    unloaded = step_surface_adhesion(
        config,
        SurfaceAdhesionState(),
        np.array([0.3, 0.0, 0.0]),
        np.zeros(3),
        1.0,
        dt_s=0.02,
    )
    assert not outside.state.attached
    assert not unloaded.state.attached
    assert _attach(config).attached


def test_stance_loading_is_capped_and_accumulates_tangential_work() -> None:
    config = _config()
    update = step_surface_adhesion(
        config,
        _attach(config),
        np.array([0.5, 0.0, 0.0]),
        np.array([0.2, 0.0, 0.0]),
        20.0,
        dt_s=0.02,
    )
    assert update.state.attached
    assert update.force_world_n == pytest.approx((-8.0, 0.0, 0.0))
    assert update.raw_tangential_force_n == pytest.approx(20.0)
    assert update.state.tangential_work_j > 0.0


def test_unloaded_swing_peels_after_dwell_instead_of_becoming_world_tether() -> None:
    config = _config()
    first_unload = step_surface_adhesion(
        config,
        _attach(config),
        np.array([0.35, 0.0, 0.03]),
        np.array([0.1, 0.0, 0.2]),
        0.0,
        dt_s=0.02,
    )
    assert first_unload.state.attached
    assert first_unload.state.phase == "unloading"
    peeled = step_surface_adhesion(
        config,
        first_unload.state,
        np.array([0.38, 0.0, 0.05]),
        np.array([0.1, 0.0, 0.2]),
        0.0,
        dt_s=0.02,
    )
    assert peeled.event == "peeled"
    assert not peeled.state.attached
    assert peeled.state.peel_count == 1
    assert np.allclose(peeled.force_world_n, 0.0)


def test_brief_contact_force_noise_does_not_peel_a_loaded_foot() -> None:
    config = _config(unload_steps_to_peel=3)
    update = step_surface_adhesion(
        config,
        _attach(config),
        np.array([0.31, 0.0, 0.025]),
        np.zeros(3),
        0.0,
        dt_s=0.02,
    )
    recovered = step_surface_adhesion(
        config,
        update.state,
        np.array([0.31, 0.0, 0.0]),
        np.zeros(3),
        20.0,
        dt_s=0.02,
    )
    assert recovered.state.attached
    assert recovered.state.unload_steps == 0


def test_peel_is_not_terminal_and_later_stance_can_reattach() -> None:
    config = _config(reattach_cooldown_steps=2, unload_steps_to_peel=1)
    peeled = step_surface_adhesion(
        config,
        _attach(config),
        np.array([0.35, 0.0, 0.03]),
        np.zeros(3),
        0.0,
        dt_s=0.02,
    )
    assert peeled.event == "peeled"
    state = peeled.state
    for _ in range(2):
        state = step_surface_adhesion(
            config,
            state,
            np.array([0.4, 0.0, 0.0]),
            np.zeros(3),
            20.0,
            dt_s=0.02,
        ).state
    assert state.attached
    assert state.attachment_count == 2
    assert state.peel_count == 1


def test_explicit_recovery_release_peels_immediately() -> None:
    config = _config()
    update = step_surface_adhesion(
        config,
        _attach(config),
        np.array([0.31, 0.0, 0.0]),
        np.zeros(3),
        20.0,
        dt_s=0.02,
        force_release=True,
    )
    assert update.event == "peeled"
    assert not update.state.attached
