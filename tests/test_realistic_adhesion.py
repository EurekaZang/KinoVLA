from __future__ import annotations

import numpy as np
import pytest

from kino_vla.sim.adhesion import (
    FootAdhesionConfig,
    FootAdhesionState,
    IrregularRegion,
    step_foot_adhesion,
)
from kino_vla.sim.realistic_scene import (
    audit_scene_spec,
    build_indoor_adhesion_scene_spec,
    usd_safe_name,
)


def _config(**overrides) -> FootAdhesionConfig:
    values = {
        "region": IrregularRegion(((0.0, -0.5), (1.2, -0.4), (1.0, 0.6), (0.1, 0.5))),
        "surface_z_m": 0.0,
        "attach_contact_force_n": 5.0,
        "attach_height_tolerance_m": 0.08,
        "stiffness_xy_n_per_m": 100.0,
        "damping_xy_ns_per_m": 0.0,
        "stiffness_z_n_per_m": 100.0,
        "damping_z_ns_per_m": 0.0,
        "force_cap_n": 30.0,
        "break_force_n": 50.0,
        "peel_release_force_n": 10.0,
        "peel_velocity_threshold_mps": 0.03,
    }
    values.update(overrides)
    return FootAdhesionConfig(**values)


def test_irregular_region_contains_interior_boundary_and_exterior() -> None:
    region = _config().region
    assert region.contains((0.5, 0.0))
    assert region.contains((0.0, -0.5))
    assert not region.contains((-0.1, 0.0))
    assert abs(region.signed_area_m2) > 0.9


def test_foot_attaches_only_with_contact_inside_region() -> None:
    cfg = _config()
    idle = FootAdhesionState()
    outside = step_foot_adhesion(
        cfg, idle, np.array([-0.2, 0.0, 0.02]), np.zeros(3), contact_force_n=20.0
    )
    assert not outside.state.attached
    no_contact = step_foot_adhesion(
        cfg, idle, np.array([0.4, 0.0, 0.02]), np.zeros(3), contact_force_n=2.0
    )
    assert not no_contact.state.attached
    attached = step_foot_adhesion(
        cfg, idle, np.array([0.4, 0.0, 0.02]), np.zeros(3), contact_force_n=20.0
    )
    assert attached.event == "attached"
    assert attached.state.attached
    assert attached.state.anchor_xyz == pytest.approx((0.4, 0.0, 0.0))


def test_attachment_authorization_can_disable_new_bonds_during_peel_phase() -> None:
    cfg = _config()
    update = step_foot_adhesion(
        cfg,
        FootAdhesionState(),
        np.array([0.4, 0.0, 0.0]),
        np.zeros(3),
        contact_force_n=20.0,
        allow_attach=False,
        peel_requested=True,
    )

    assert not update.state.attached
    assert update.event == "none"


def test_forward_loading_applies_force_at_explicit_anchor_and_caps_it() -> None:
    cfg = _config()
    attached = step_foot_adhesion(
        cfg,
        FootAdhesionState(),
        np.array([0.4, 0.0, 0.0]),
        np.zeros(3),
        contact_force_n=20.0,
    ).state
    loaded = step_foot_adhesion(
        cfg,
        attached,
        np.array([0.65, 0.0, 0.0]),
        np.array([0.2, 0.0, 0.0]),
        contact_force_n=20.0,
    )
    assert loaded.state.attached
    assert loaded.state.phase == "loaded"
    assert loaded.force_world_n[0] < 0.0
    assert np.linalg.norm(loaded.force_world_n) == pytest.approx(25.0)
    capped = step_foot_adhesion(
        cfg,
        loaded.state,
        np.array([0.75, 0.0, 0.0]),
        np.array([0.2, 0.0, 0.0]),
        contact_force_n=20.0,
    )
    assert np.linalg.norm(capped.force_world_n) == pytest.approx(30.0)


def test_reverse_peel_releases_below_forward_break_and_does_not_reattach() -> None:
    cfg = _config()
    attached = step_foot_adhesion(
        cfg,
        FootAdhesionState(),
        np.array([0.4, 0.0, 0.0]),
        np.zeros(3),
        contact_force_n=20.0,
    ).state
    peeled = step_foot_adhesion(
        cfg,
        attached,
        np.array([0.52, 0.0, 0.0]),
        np.array([-0.2, 0.0, 0.0]),
        contact_force_n=20.0,
        peel_requested=True,
    )
    assert peeled.event == "peeled"
    assert peeled.raw_force_n == pytest.approx(12.0)
    assert peeled.state.terminal_release
    assert not peeled.state.broken
    attempted_reattach = step_foot_adhesion(
        cfg,
        peeled.state,
        np.array([0.4, 0.0, 0.0]),
        np.zeros(3),
        contact_force_n=20.0,
    )
    assert not attempted_reattach.state.attached
    assert attempted_reattach.state.phase == "released"


def test_explicit_reverse_peels_before_tethered_foot_velocity_turns_negative() -> None:
    cfg = _config()
    attached = step_foot_adhesion(
        cfg,
        FootAdhesionState(),
        np.array([0.4, 0.0, 0.0]),
        np.zeros(3),
        contact_force_n=20.0,
    ).state

    peeled = step_foot_adhesion(
        cfg,
        attached,
        np.array([0.52, 0.0, 0.0]),
        np.array([0.2, 0.0, 0.0]),
        contact_force_n=20.0,
        peel_requested=True,
    )

    assert peeled.event == "peeled"
    assert peeled.state.terminal_release
    assert not peeled.state.broken


def test_stance_foot_backward_velocity_is_not_a_peel_without_protocol_reversal() -> None:
    cfg = _config()
    attached = step_foot_adhesion(
        cfg,
        FootAdhesionState(),
        np.array([0.4, 0.0, 0.0]),
        np.zeros(3),
        contact_force_n=20.0,
    ).state
    loaded = step_foot_adhesion(
        cfg,
        attached,
        np.array([0.52, 0.0, 0.0]),
        np.array([-0.2, 0.0, 0.0]),
        contact_force_n=20.0,
        peel_requested=False,
    )
    assert loaded.state.attached
    assert loaded.event == "none"
    assert loaded.state.phase == "loaded"


def test_forward_overload_records_break_event() -> None:
    cfg = _config()
    attached = step_foot_adhesion(
        cfg,
        FootAdhesionState(),
        np.array([0.4, 0.0, 0.0]),
        np.zeros(3),
        contact_force_n=20.0,
    ).state
    broken = step_foot_adhesion(
        cfg,
        attached,
        np.array([0.95, 0.0, 0.0]),
        np.array([0.2, 0.0, 0.0]),
        contact_force_n=20.0,
    )
    assert broken.event == "broken"
    assert broken.state.broken
    assert broken.state.terminal_release
    assert np.allclose(broken.force_world_n, 0.0)


def test_indoor_scene_passes_static_route_and_layer_contract_audit() -> None:
    spec = build_indoor_adhesion_scene_spec(seed=42)
    audit = audit_scene_spec(spec)
    assert audit["passed"], audit
    assert audit["n_primitives"] >= 200
    assert audit["n_materials"] >= 60
    assert audit["corridor_collisions"] == []
    assert audit["adhesion_on_corridor"]
    assert 1.0 < audit["adhesion_area_m2"] < 3.0
    assert "embodiedgen_interface_ready" in spec.source_kind


def test_usd_safe_name_removes_decimal_and_sign_characters() -> None:
    assert usd_safe_name("Shelf_0.45_Side_-0.44") == "Shelf_0_p_45_Side__neg_0_p_44"
    assert usd_safe_name("3rd prop") == "P_3rd_prop"
