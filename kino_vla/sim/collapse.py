"""Pure damage accumulation and support-cell selection for realistic O3 collapse."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from kino_vla.utils.geometry import Rect


@dataclass(frozen=True)
class CollapseDamageState:
    """Cumulative normal impulse and one-way topology-transition state."""

    normal_impulse_ns: float = 0.0
    collapsed: bool = False
    trigger_step: int | None = None


@dataclass(frozen=True)
class CollapseDamageUpdate:
    """One contact-step update with auditable per-step load."""

    state: CollapseDamageState
    step_normal_impulse_ns: float
    loaded_feet: int
    triggered_now: bool


def step_collapse_damage(
    state: CollapseDamageState,
    *,
    region: Rect,
    foot_xy_m: np.ndarray,
    foot_normal_force_n: np.ndarray,
    dt_s: float,
    damage_threshold_ns: float,
    step_index: int,
) -> CollapseDamageUpdate:
    """Accumulate only positive normal impulse from feet geometrically inside the region."""
    positions = np.asarray(foot_xy_m, dtype=np.float64)
    forces = np.asarray(foot_normal_force_n, dtype=np.float64).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"foot_xy_m must have shape (F, 2), got {positions.shape}")
    if len(positions) != len(forces):
        raise ValueError("foot positions and forces must have the same length")
    if dt_s <= 0.0 or damage_threshold_ns <= 0.0:
        raise ValueError("dt_s and damage_threshold_ns must be positive")
    if not np.isfinite(positions).all() or not np.isfinite(forces).all():
        raise ValueError("foot contact inputs must be finite")
    if state.collapsed:
        return CollapseDamageUpdate(state, 0.0, 0, False)
    inside = np.asarray([region.contains(position) for position in positions], dtype=bool)
    positive_normal = np.maximum(forces, 0.0)
    step_impulse = float(positive_normal[inside].sum() * dt_s)
    total = state.normal_impulse_ns + step_impulse
    triggered = total >= damage_threshold_ns
    next_state = CollapseDamageState(
        normal_impulse_ns=total,
        collapsed=triggered,
        trigger_step=int(step_index) if triggered else None,
    )
    return CollapseDamageUpdate(next_state, step_impulse, int(inside.sum()), triggered)


def support_cell_layout(
    region: Rect,
    *,
    cells_xy: tuple[int, int] = (4, 4),
    gap_m: float = 0.006,
) -> tuple[Rect, ...]:
    """Tessellate a support region into non-overlapping cells with millimetre-scale seams."""
    nx, ny = cells_xy
    if nx <= 0 or ny <= 0:
        raise ValueError("cell counts must be positive")
    cell_w = 2.0 * region.hx / nx
    cell_h = 2.0 * region.hy / ny
    if gap_m < 0.0 or gap_m >= min(cell_w, cell_h):
        raise ValueError("gap_m must be non-negative and smaller than each cell")
    cells: list[Rect] = []
    for ix in range(nx):
        for iy in range(ny):
            cells.append(
                Rect(
                    cx=region.cx - region.hx + (ix + 0.5) * cell_w,
                    cy=region.cy - region.hy + (iy + 0.5) * cell_h,
                    hx=0.5 * (cell_w - gap_m),
                    hy=0.5 * (cell_h - gap_m),
                )
            )
    return tuple(cells)


def collapsed_cell_indices(
    cells: tuple[Rect, ...], *, region: Rect, residual_support: float
) -> tuple[int, ...]:
    """Choose failed cells centre-first to approximate a requested residual support ratio."""
    if not 0.0 <= residual_support <= 1.0:
        raise ValueError("residual_support must be in [0, 1]")
    n_remove = len(cells) - int(math.ceil(residual_support * len(cells)))
    ranked = sorted(
        range(len(cells)),
        key=lambda index: (
            (cells[index].cx - region.cx) ** 2 + (cells[index].cy - region.cy) ** 2,
            index,
        ),
    )
    return tuple(ranked[:n_remove])


def damage_update_to_dict(update: CollapseDamageUpdate) -> dict[str, Any]:
    """Serialize a damage update for privileged QA telemetry."""
    return {
        "normal_impulse_ns": update.state.normal_impulse_ns,
        "step_normal_impulse_ns": update.step_normal_impulse_ns,
        "loaded_feet": update.loaded_feet,
        "collapsed": update.state.collapsed,
        "triggered_now": update.triggered_now,
        "trigger_step": update.state.trigger_step,
    }
