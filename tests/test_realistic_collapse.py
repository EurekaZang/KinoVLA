from __future__ import annotations

import numpy as np
import pytest

from kino_vla.sim.collapse import (
    CollapseDamageState,
    collapsed_cell_indices,
    step_collapse_damage,
    support_cell_layout,
)
from kino_vla.sim.operators import Collapse
from kino_vla.sim.surrogate import SurrogateBackend
from kino_vla.utils.config import load_config
from kino_vla.utils.geometry import Rect


def test_damage_integrates_only_loaded_feet_inside_region() -> None:
    region = Rect(0.0, 0.0, 0.5, 0.5)
    state = CollapseDamageState()
    update = step_collapse_damage(
        state,
        region=region,
        foot_xy_m=np.array([[0.1, 0.1], [0.2, -0.1], [1.0, 0.0], [0.0, 1.0]]),
        foot_normal_force_n=np.array([50.0, 70.0, 1000.0, -20.0]),
        dt_s=0.02,
        damage_threshold_ns=5.0,
        step_index=12,
    )
    assert update.loaded_feet == 2
    assert update.step_normal_impulse_ns == pytest.approx(2.4)
    assert not update.state.collapsed
    update = step_collapse_damage(
        update.state,
        region=region,
        foot_xy_m=np.array([[0.1, 0.1], [0.2, -0.1]]),
        foot_normal_force_n=np.array([80.0, 80.0]),
        dt_s=0.02,
        damage_threshold_ns=5.0,
        step_index=13,
    )
    assert update.triggered_now
    assert update.state.normal_impulse_ns == pytest.approx(5.6)
    assert update.state.trigger_step == 13


def test_support_failure_is_deterministic_and_matches_residual_ratio() -> None:
    region = Rect(1.0, 2.0, 0.8, 0.6)
    cells = support_cell_layout(region, cells_xy=(4, 4))
    collapsed = collapsed_cell_indices(cells, region=region, residual_support=0.45)
    assert len(cells) == 16
    assert len(collapsed) == 8  # ceil(0.45 * 16) = 8 remain
    assert collapsed == collapsed_cell_indices(cells, region=region, residual_support=0.45)
    center_distances = [
        (cells[index].cx - region.cx) ** 2 + (cells[index].cy - region.cy) ** 2
        for index in collapsed
    ]
    remaining = set(range(len(cells))) - set(collapsed)
    assert max(center_distances) <= min(
        (cells[index].cx - region.cx) ** 2 + (cells[index].cy - region.cy) ** 2
        for index in remaining
    )


def test_surrogate_realistic_parameter_path_uses_impulse_not_dwell() -> None:
    cfg = load_config("sim/surrogate.yaml")
    backend = SurrogateBackend(cfg, np.array([0.0, 0.0]), 0.0)
    backend.reset(0)
    op = Collapse(
        region=Rect(0.0, 0.0, 1.0, 1.0),
        mu_collapsed=0.08,
        trigger_dwell_s=1000.0,
        damage_threshold_ns=3.0,
        drop_m=0.06,
        residual_support=0.45,
    )
    op.on_reset(backend)
    assert backend.friction_at(np.array([0.0, 0.0])) == pytest.approx(0.8)
    for _ in range(10):
        backend.step(np.zeros(3))
        if backend.friction_at(np.array([0.0, 0.0])) < 0.2:
            break
    assert backend.friction_at(np.array([0.0, 0.0])) == pytest.approx(0.08)
    truth = op.get_privileged_state()
    assert truth["damage_threshold_ns"] == 3.0
    assert truth["drop_m"] == 0.06
