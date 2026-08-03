from __future__ import annotations

from kino_vla.sim import isaac_policy_backend as backend_module
from kino_vla.sim.isaac_policy_backend import (
    IsaacPolicyBackend,
    _coplanar_segmented_ground_layout,
    _contact_local_soil_offset_update,
    _triggered_collapse_offset_update,
)
from kino_vla.utils.geometry import Rect

import numpy as np
import pytest


def test_segmented_operator_disables_and_restores_realistic_scene_floor(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_set_collision(path: str, enabled: bool) -> list[str]:
        calls.append((path, enabled))
        return [path] if len(calls) in (1, 2) else []

    monkeypatch.setattr(backend_module, "_set_collision_enabled", fake_set_collision)
    backend = IsaacPolicyBackend.__new__(IsaacPolicyBackend)
    backend._realistic_scene_floor_path = "/World/KinoIndoor/Collision/Floor"
    backend._disabled_realistic_scene_floor_colliders = []

    assert backend._disable_realistic_scene_floor_for_operator_topology() == 1
    assert backend._disable_realistic_scene_floor_for_operator_topology() == 1
    assert calls == [("/World/KinoIndoor/Collision/Floor", False)]

    backend._restore_realistic_scene_floor_collision()
    assert calls[-1] == ("/World/KinoIndoor/Collision/Floor", True)
    assert backend._disabled_realistic_scene_floor_colliders == []


def test_segmented_operator_is_noop_without_realistic_scene_floor(monkeypatch) -> None:
    monkeypatch.setattr(
        backend_module,
        "_set_collision_enabled",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected call")),
    )
    backend = IsaacPolicyBackend.__new__(IsaacPolicyBackend)
    backend._realistic_scene_floor_path = None
    backend._disabled_realistic_scene_floor_colliders = []
    assert backend._disable_realistic_scene_floor_for_operator_topology() == 0


def test_segmented_operator_hides_and_restores_flat_route_appearance(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        backend_module,
        "_set_prim_visible",
        lambda path, visible: calls.append((path, visible)),
    )
    backend = IsaacPolicyBackend.__new__(IsaacPolicyBackend)
    backend._realistic_scene_route_appearance_path = (
        "/World/KinoIndoor/Appearance/RouteSurface"
    )
    backend._realistic_scene_route_appearance_hidden = False
    assert backend._hide_realistic_scene_route_appearance_for_operator_topology() is True
    assert backend._hide_realistic_scene_route_appearance_for_operator_topology() is True
    assert calls == [("/World/KinoIndoor/Appearance/RouteSurface", False)]
    backend._restore_realistic_scene_route_appearance()
    assert calls[-1] == ("/World/KinoIndoor/Appearance/RouteSurface", True)
    assert backend._realistic_scene_route_appearance_hidden is False


def test_contact_local_soil_deformation_has_compact_smooth_support() -> None:
    x = np.asarray([0.0, 0.03, 0.06, 0.075, 0.20])
    y = np.zeros_like(x)
    offset = _contact_local_soil_offset_update(
        np.zeros_like(x),
        x,
        y,
        center_xy_m=(0.0, 0.0),
        sinkage_m=0.09,
        radius_xy_m=(0.075, 0.05),
    )

    assert offset[0] == pytest.approx(-0.09)
    assert offset[0] < offset[1] < offset[2] < 0.0
    assert offset[3] == pytest.approx(0.0)
    assert offset[4] == pytest.approx(0.0)


def test_contact_local_soil_deformation_accumulates_without_global_region_marker() -> None:
    x = np.asarray([0.0, 0.10, 0.20, 1.0])
    y = np.zeros_like(x)
    first = _contact_local_soil_offset_update(
        np.zeros_like(x),
        x,
        y,
        center_xy_m=(0.0, 0.0),
        sinkage_m=0.04,
        radius_xy_m=(0.12, 0.06),
    )
    second = _contact_local_soil_offset_update(
        first,
        x,
        y,
        center_xy_m=(0.20, 0.0),
        sinkage_m=0.08,
        radius_xy_m=(0.12, 0.06),
    )

    assert np.all(second <= first + 1.0e-12)
    assert second[0] == first[0]
    assert second[2] == pytest.approx(-0.08)
    assert second[3] == pytest.approx(0.0)


def test_o1_coplanar_layout_tiles_ground_without_area_overlap() -> None:
    operator = Rect(cx=1.5, cy=0.0, hx=0.65, hy=0.60)
    cells = _coplanar_segmented_ground_layout(operator, 15.0)

    assert cells[-1] == operator
    total_area = sum(4.0 * cell.hx * cell.hy for cell in cells)
    assert total_area == pytest.approx(30.0 * 30.0)
    for first_index, first in enumerate(cells):
        for second in cells[first_index + 1 :]:
            overlap_x = min(first.cx + first.hx, second.cx + second.hx) - max(
                first.cx - first.hx, second.cx - second.hx
            )
            overlap_y = min(first.cy + first.hy, second.cy + second.hy) - max(
                first.cy - first.hy, second.cy - second.hy
            )
            assert overlap_x <= 1.0e-12 or overlap_y <= 1.0e-12


def test_o1_coplanar_layout_rejects_region_outside_ground() -> None:
    with pytest.raises(ValueError, match="inside"):
        _coplanar_segmented_ground_layout(Rect(cx=14.8, cy=0.0, hx=0.4, hy=0.5), 15.0)


def test_o3_visual_collapse_is_compact_irregular_and_trigger_only() -> None:
    region = Rect(cx=1.5, cy=0.0, hx=0.65, hy=0.60)
    x, y = np.meshgrid(np.linspace(0.4, 2.6, 81), np.linspace(-1.1, 1.1, 81))
    initial = np.zeros_like(x)
    collapsed = _triggered_collapse_offset_update(
        initial,
        x,
        y,
        center_xy_m=(1.46, 0.08),
        region=region,
        drop_m=0.12,
    )

    assert np.count_nonzero(initial) == 0
    assert collapsed.min() <= -0.11
    assert np.count_nonzero(collapsed < -1.0e-6) > 100
    assert np.all(collapsed[(x < 0.4 + 1.0e-9) | (x > 2.6 - 1.0e-9)] == 0.0)
    # A rectangular mask would have one x-span at every affected row; the angular perturbation
    # intentionally produces multiple boundary widths.
    row_widths = [int(np.count_nonzero(row < -1.0e-6)) for row in collapsed]
    assert len({width for width in row_widths if width > 0}) >= 8
