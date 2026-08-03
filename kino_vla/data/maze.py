"""PHASE 1 procedural counterfactual maze (spec §10).

Spec §10 PHASE 1 generates an abstract maze that *shuffles the physics-visual correspondence*
("蓝色平整区域 = 0.15m 下陷软体网格"), forcing the model off visual common-sense shortcuts so it
must reconstruct traversability from the Kino-Tokens/RGB conflict. Here a maze is a seeded
sequence of cells — each one Kino-Fail operator (sampled θ) placed on the walking-skeleton
path — with a configurable fraction of cells whose appearance is *decoupled* from physics (a
visual deception on top of O7's built-in one). The ambiguity pairs (O4↔O2, O5↔O10, O3↔O1) are
weighted up: they are the Suite-Sem core where attribution decides opposite recoveries.

Each cell builds the actual operator + a ``SemanticRegion`` (the visual footprint the snapshot
renders), so one maze cell drives the same closed loop (monitor → recovery → shield) the demo
uses, exactly as ``scripts/record_operators.py`` does for the per-operator videos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kino_vla.map.types import SemanticRegion
from kino_vla.sim.operators import (
    Collapse,
    ComplianceField,
    EffortDecay,
    FailureOperator,
    HighCentering,
    InvisibleCollider,
    MuField,
    Payload,
    Tether,
    VisualPhysicsRemap,
)
from kino_vla.utils.config import Config
from kino_vla.utils.geometry import Rect

# Hazard appearance palette the decoupling shuffle draws from (distinct visual materials).
_DECOUPLE_PALETTE: tuple[str, ...] = ("ice_sheet", "brown_mud", "yellow_adhesive")
# Operators with no characteristic visual footprint (global / invisible): no scene region.
_NO_REGION: frozenset[str] = frozenset(
    {"O5_payload", "O10_effort_decay", "O8_invisible_collider", "O9_high_centering"}
)


@dataclass(frozen=True)
class MazeCell:
    """One maze cell: an operator + sampled θ + its (possibly decoupled) appearance."""

    index: int
    op_name: str
    theta: dict[str, float]
    appearance: str
    depth_bias_m: float = 0.0
    is_decoupled: bool = False
    ambiguity_pair: str | None = None
    seed: int = 0

    def build_operator(self, rect: Rect) -> FailureOperator:
        """Instantiate the failure operator with this cell's sampled θ at ``rect``."""
        t = self.theta
        if self.op_name == "O1_mu_field":
            return MuField(region=rect, mu_s=t["mu"], mu_d=t["mu"])
        if self.op_name == "O3_collapse":
            return Collapse(
                region=rect,
                mu_collapsed=t["mu_collapsed"],
                trigger_dwell_s=t["dwell"],
                mu_intact=t["mu_intact"],
            )
        if self.op_name == "O2_compliance":
            return ComplianceField(
                region=rect,
                k_c=t["k"],
                c_c=t["c"],
                d_sink=t["d_sink"],
                appearance_class=self.appearance,
            )
        if self.op_name == "O4_tether":
            return Tether(
                region=rect,
                k=t["k"],
                d=t["d"],
                l0=0.0,
                f_break=t["f_break"],
                appearance_class=self.appearance,
            )
        if self.op_name == "O5_payload":
            return Payload(mass_kg=t["mass"], com_offset_m=(t["com_m"], 0.0))
        if self.op_name == "O10_effort_decay":
            return EffortDecay(
                decay_rate_per_s=t["rate"], floor=t["floor"], t_start_s=t["t_start_s"]
            )
        if self.op_name == "O7_visual_remap":
            return VisualPhysicsRemap(
                region=rect, mu_s=t["mu"], mu_d=t["mu"], depth_bias_m=self.depth_bias_m
            )
        if self.op_name == "O8_invisible_collider":
            return InvisibleCollider(region=rect)
        if self.op_name == "O9_high_centering":
            return HighCentering(region=rect, residual_support=t["residual"])
        raise ValueError(f"maze cell has no operator builder for {self.op_name!r}")

    def scene_region(self, rect: Rect) -> SemanticRegion | None:
        """The visual footprint the snapshot renders (None for global/invisible operators)."""
        if self.op_name in _NO_REGION:
            return None
        return SemanticRegion(
            Rect(rect.cx, rect.cy, rect.hx, rect.hy),
            self.appearance,
            depth_bias_m=self.depth_bias_m,
        )

    def factory(self, rect: Rect) -> tuple[FailureOperator, SemanticRegion | None]:
        """``operator_factory`` form for ``build_walking_skeleton``."""
        return self.build_operator(rect), self.scene_region(rect)


def generate_maze(cfg: Config, seed: int) -> list[MazeCell]:
    """Seeded procedural maze of ``cfg.maze.n_cells`` operator cells (spec §10 PHASE 1)."""
    maze = cfg.maze
    specs = list(maze.cells)  # raw dicts (Config wraps dict attributes, not list elements)
    weights = np.array([float(s.get("weight", 1)) for s in specs], dtype=np.float64)
    weights /= weights.sum()
    defaults = maze.defaults.to_dict() if "defaults" in maze else {}
    pair_of = _pair_labels(maze)
    decouple_fraction = float(maze.decouple_fraction)
    rng = np.random.default_rng(int(seed))

    cells: list[MazeCell] = []
    for i in range(int(maze.n_cells)):
        spec = specs[int(rng.choice(len(specs), p=weights))]
        op_name = str(spec["op"])
        theta = _sample_theta(op_name, spec.get("params", {}), defaults, rng)
        appearance = str(spec["appearance"])
        is_decoupled = False
        # PHASE 1 shuffle: decouple appearance from physics for the visible-hazard operators
        # (not O7, which already carries a canonical visual deception).
        if (
            op_name != "O7_visual_remap"
            and appearance in _DECOUPLE_PALETTE
            and rng.random() < decouple_fraction
        ):
            alternatives = [m for m in _DECOUPLE_PALETTE if m != appearance]
            appearance = str(rng.choice(alternatives))
            is_decoupled = True
        cells.append(
            MazeCell(
                index=i,
                op_name=op_name,
                theta=theta,
                appearance=appearance,
                depth_bias_m=float(theta.get("depth_bias", 0.0)),
                is_decoupled=is_decoupled,
                ambiguity_pair=pair_of.get(op_name),
                seed=int(seed) * 1000 + i,
            )
        )
    return cells


def _sample_theta(
    op_name: str, params: dict, defaults: dict, rng: np.random.Generator
) -> dict[str, float]:
    """Uniformly sample each θ range, then fold in the fixed per-operator defaults."""
    theta = {name: float(rng.uniform(lo, hi)) for name, (lo, hi) in params.items()}
    if op_name == "O3_collapse":
        theta["mu_intact"] = float(defaults.get("collapse_mu_intact", 0.8))
    elif op_name == "O10_effort_decay":
        theta["t_start_s"] = float(defaults.get("effort_t_start_s", 1.5))
    elif op_name == "O5_payload":
        theta["com_m"] = float(defaults.get("payload_com_m", 0.0))
    return theta


def _pair_labels(maze: Config) -> dict[str, str]:
    """Map each operator that belongs to an ambiguity pair to its canonical pair label."""
    out: dict[str, str] = {}
    for pair in maze.ambiguity_pairs:
        a, b = str(pair[0]), str(pair[1])
        label = "|".join(sorted((a, b)))
        out[a] = label
        out[b] = label
    return out
