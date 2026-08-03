"""Primitive Compiler — VLA recovery primitives → executable commands (spec §5).

v1.2 had only ``Backstep`` and ``Replan_Waypoint``, so the whole reflection loop
degenerated to a binary classifier a 200-line FSM could mimic. v2 compiles the full
Sport-Client primitive library, so an *attribution* (ice vs. thin-ice, mud vs.
adhesive) maps to genuinely different — sometimes opposite — recoveries (spec §5
many-to-many map). Each primitive is validated against a schema; mode-changing
primitives (Switch_Gait, Adjust_Posture) are gated by the CBF admission rule
(spec §6.7) and, if rejected, return a structured code for the next reflection round.

This is the typed boundary the VLA planner (M7) emits into. At M3 it is exercised
standalone (every primitive compiles or is rejected with a code); the demo FSM keeps
emitting velocities directly until M7 swaps it for the planner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kino_vla.shield.cbf_shield import CbfShield
from kino_vla.sim.types import Obs

GAITS = ("high_step", "trot", "crawl")


# --------------------------------------------------------------------- primitives
@dataclass(frozen=True)
class Primitive:
    """Base recovery primitive (spec §5)."""


@dataclass(frozen=True)
class Continue(Primitive):
    """Explicit non-intervention: keep the nominal policy in control."""


@dataclass(frozen=True)
class Backstep(Primitive):
    distance_m: float


@dataclass(frozen=True)
class ReplanWaypoint(Primitive):
    point_xy: tuple[float, float]  # odometry-frame 2D goal (depth back-projected upstream)


@dataclass(frozen=True)
class SwitchGait(Primitive):
    mode: str  # one of GAITS


@dataclass(frozen=True)
class AdjustPosture(Primitive):
    body_height_m: float
    pitch_deg: float = 0.0


@dataclass(frozen=True)
class SetConstraint(Primitive):
    max_speed: float
    stiffness: float


@dataclass(frozen=True)
class UpdateTopology(Primitive):
    region_xy: tuple[float, float]
    radius_m: float
    status: str  # e.g. "untraversable"


@dataclass(frozen=True)
class HoldAndRequest(Primitive):
    reason: str


# ------------------------------------------------------------------ compiled form
@dataclass(frozen=True)
class CompiledCommand:
    """Result of compiling one primitive (spec §5, §6.7).

    ``accepted`` is False only when validation fails or an admission check rejects a
    mode switch; ``code`` then carries the structured reason for the next reflection
    round. ``backstep_m`` rides along so the executor knows the transient distance.
    """

    accepted: bool
    code: str
    v_cmd: np.ndarray | None = None  # (3,) body-frame velocity, when the primitive sets one
    target_mode: str | None = None  # posture/gait to switch to (admitted)
    waypoint_xy: np.ndarray | None = None
    topology_op: tuple[float, float, float, str] | None = None  # (x, y, r, status)
    max_speed: float | None = None
    stiffness: float | None = None
    halt: bool = False
    backstep_m: float | None = None


class PrimitiveCompiler:
    """Compile the §5 primitive library against the shield's modes and admission rule."""

    def __init__(self, shield: CbfShield, backstep_speed_mps: float = 0.25) -> None:
        self._shield = shield
        self._backstep_speed = float(backstep_speed_mps)

    @property
    def shield(self) -> CbfShield:
        """The CBF shield this compiler admits against — the SAME instance the loop filters with
        (rollout wires one shield into both). The planner reads it to ENACT an admitted mode switch
        via :meth:`CbfShield.set_mode` (spec §6.7): admission alone leaves the shield's active mode
        unchanged, so without this the VLA's chosen gait/posture would never reach the support
        polygon. ``compile`` stays side-effect-free; the planner performs the switch on accept."""
        return self._shield

    def _nearest_mode(self, z_c: float) -> str:
        """Map a requested CoM height to the closest configured posture mode."""
        modes = self._shield.modes_table()
        return min(modes.values(), key=lambda m: abs(m.z_c - z_c)).name

    def _admit(self, target_mode: str, obs: Obs) -> CompiledCommand:
        """Gate a mode switch through the live admission rule (spec §6.7)."""
        adm = self._shield.admit_mode_switch(target_mode, obs)
        return CompiledCommand(adm.approved, adm.code, target_mode=target_mode)

    def compile(self, prim: Primitive, obs: Obs) -> CompiledCommand:
        """Compile one primitive into an executable command (or a rejection code)."""
        if isinstance(prim, Continue):
            return CompiledCommand(True, "OK: continue")

        if isinstance(prim, Backstep):
            if not prim.distance_m > 0.0:
                return CompiledCommand(False, f"REJECT: Backstep.distance_m={prim.distance_m} ≤ 0")
            return CompiledCommand(
                True,
                "OK: Backstep",
                v_cmd=np.array([-self._backstep_speed, 0.0, 0.0]),
                backstep_m=float(prim.distance_m),
            )

        if isinstance(prim, ReplanWaypoint):
            return CompiledCommand(
                True,
                "OK: Replan_Waypoint",
                waypoint_xy=np.asarray(prim.point_xy, dtype=np.float64),
            )

        if isinstance(prim, SwitchGait):
            if prim.mode not in GAITS:
                return CompiledCommand(False, f"REJECT: unknown gait {prim.mode!r}")
            return self._admit(prim.mode, obs)

        if isinstance(prim, AdjustPosture):
            if not prim.body_height_m > 0.0:
                return CompiledCommand(
                    False, f"REJECT: AdjustPosture.body_height_m={prim.body_height_m} ≤ 0"
                )
            return self._admit(self._nearest_mode(prim.body_height_m), obs)

        if isinstance(prim, SetConstraint):
            if not (prim.max_speed > 0.0 and prim.stiffness >= 0.0):
                return CompiledCommand(False, "REJECT: Set_Constraint out of range")
            return CompiledCommand(
                True,
                "OK: Set_Constraint",
                max_speed=float(prim.max_speed),
                stiffness=float(prim.stiffness),
            )

        if isinstance(prim, UpdateTopology):
            if not prim.radius_m > 0.0:
                return CompiledCommand(False, "REJECT: Update_Topology.radius_m ≤ 0")
            return CompiledCommand(
                True,
                "OK: Update_Topology",
                topology_op=(
                    float(prim.region_xy[0]),
                    float(prim.region_xy[1]),
                    float(prim.radius_m),
                    str(prim.status),
                ),
            )

        if isinstance(prim, HoldAndRequest):
            return CompiledCommand(
                True,
                f"OK: Hold_and_Request ({prim.reason})",
                v_cmd=np.zeros(3),
                halt=True,
            )

        return CompiledCommand(False, f"REJECT: unknown primitive {type(prim).__name__}")
