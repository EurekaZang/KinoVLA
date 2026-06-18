"""VLA Recovery Planner — the closed-loop drop-in for the M1 scripted FSM stub (spec §1 / §5 / §11).

This is the milestone's headline swap: the ``[STUB: scripted FSM recovery]`` of CLAUDE.md §1
becomes the trained planner. It implements the same :class:`~kino_vla.loop.RecoveryPolicy`
contract (``on_event`` / ``step``) so it slots into the identical monitor→planner→shield→map loop,
but instead of always Backstep-then-replan it: snapshots the failure (the §10 PHASE 2 multimodal
package), asks a :class:`VlaPolicy` to attribute the physical cause and pick ONE §5 primitive,
validates it (:func:`kino_vla.vla.output.parse_vla_decision` — 100% parse-or-reject), compiles it
through the CBF-admission Primitive Compiler, and executes it. Multi-round: every monitor event
re-snapshots and re-decides (the spec's reflection loop that breaks a dead loop).

Two policies:

- :class:`StubVlaPolicy` — a deterministic taxonomy oracle (CI + the "perfect attribution"
  reference). Reads the privileged operator/θ so it always attributes correctly; an ``error_mode``
  makes it choose a sibling's strategy (the wrong-attribution rollout for DPO Rejected pairs).
- :class:`ModelVlaPolicy` — the real :class:`~kino_vla.vla.model.KinoVLA`: reads only RGB +
  proprioception (never the operator label) and reasons. The GPU/model dependency.

The executor reuses the FSM stub's tested waypoint-pursuit + box-detour geometry; the §5 primitives
map to navigation intents: Backstep/Update_Topology/Replan ⇒ escape & route around; Set_Constraint/
Switch_Gait ⇒ push through (capped speed); Adjust_Posture ⇒ lift & continue; Hold_and_Request ⇒
stop and request help. So a *correct* attribution and a *wrong* one drive genuinely different —
sometimes opposite — trajectories, which is exactly what makes success a physical signal (spec §5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np

from kino_vla.data.schema import CoTAnnotation, RecoveryPrimitive, Snapshot
from kino_vla.data.snapshot import SnapshotRecorder
from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.monitor.rule_monitor import MonitorEvent
from kino_vla.shield.primitive_compiler import (
    Backstep,
    CompiledCommand,
    HoldAndRequest,
    Primitive,
    PrimitiveCompiler,
    SetConstraint,
    SwitchGait,
    UpdateTopology,
)
from kino_vla.sim.types import Obs
from kino_vla.utils.config import Config
from kino_vla.utils.geometry import rot90, segment_hits_circle, unit, wrap_angle
from kino_vla.vla.output import ParsedDecision, parse_vla_decision, to_compiler_primitive
from kino_vla.vla.prompt import build_messages, context_from_snapshot


# ---------------------------------------------------------------------- policies
class VlaPolicy(Protocol):
    """Maps a failure snapshot to a parsed atomic recovery decision (the planner's brain)."""

    def decide(self, snapshot: Snapshot) -> ParsedDecision: ...


_CANON_PARAMS = {
    "Backstep": {"distance_m": 0.5},
    "Replan_Waypoint": {"point_px": [480, 360]},
    "Switch_Gait": {"mode": "high_step"},
    "Adjust_Posture": {"body_height_m": 0.25, "pitch_deg": 0.0},
    "Set_Constraint": {"max_speed": 0.4, "stiffness": 0.5},
    "Update_Topology": {"region_xy": [0.0, 0.0], "radius_m": 0.6, "status": "untraversable"},
    "Hold_and_Request": {"reason": "actuator torque saturated; cannot proceed safely"},
}
_GAIT_FOR_CATEGORY = {"effort_decay": "crawl", "compliant_terrain": "high_step"}


class StubVlaPolicy:
    """Deterministic taxonomy oracle (CI + the perfect-attribution reference, spec §10 anchor).

    Derives the true category from the privileged operator/θ and emits its canonical §5 recovery.
    ``error_mode='sibling'`` instead emits the ambiguity sibling's strategy (the wrong-attribution
    trajectory the DPO uses as a Rejected sample, spec §11) — so a single class generates both the
    correct and the failure-mode rollouts without a model.
    """

    def __init__(
        self,
        cfg: Config,
        taxonomy: FailureTaxonomy,
        *,
        error_mode: str | None = None,
    ) -> None:
        self._cfg = cfg
        self._tax = taxonomy
        self._error = error_mode
        self._sibling = _sibling_categories(cfg, taxonomy)

    def decide(self, snapshot: Snapshot) -> ParsedDecision:
        true_cat = self._tax.category_of(snapshot.operator_name)
        category = true_cat
        if self._error == "sibling" and true_cat in self._sibling:
            category = self._sibling[true_cat]
        primitive = str(self._cfg.recovery.canonical.get(category, "Set_Constraint"))
        params = dict(_CANON_PARAMS.get(primitive, {}))
        if primitive == "Switch_Gait":
            params["mode"] = _GAIT_FOR_CATEGORY.get(category, "high_step")
        if primitive == "Update_Topology":
            params["region_xy"] = [float(snapshot.pose_xy[0]), float(snapshot.pose_xy[1])]
        ann = CoTAnnotation(
            thought=f"Attributing {snapshot.monitor_channel} to {category}.",
            attribution=category,
            primitive=RecoveryPrimitive(primitive, params),
            attribution_raw=category,
            raw_text="",
        )
        return ParsedDecision(ok=True, raw_text="", annotation=ann)


class ModelVlaPolicy:
    """Real KinoVLA policy: generate a reflection from RGB + proprioception, then parse it."""

    def __init__(
        self,
        model: object,
        cfg: Config,
        taxonomy: FailureTaxonomy,
        *,
        route: str = "latent",
        n_images: int = 1,
        temperature: float = 0.0,
        reveal_appearance: bool = False,
    ) -> None:
        self._model = model
        self._cfg = cfg
        self._tax = taxonomy
        self._route = route
        self._n_images = int(n_images)
        self._temperature = float(temperature)
        self._reveal = bool(reveal_appearance)

    def decide(self, snapshot: Snapshot) -> ParsedDecision:
        ctx = context_from_snapshot(snapshot, route=self._route, reveal_appearance=self._reveal)
        messages = build_messages(ctx, self._cfg, route=self._route, n_images=self._n_images)
        images = list(snapshot.rgb[-self._n_images :]) if snapshot.rgb.size else []
        text = self._model.generate(
            messages,
            images,
            proprio_window=snapshot.proprio_window if self._route == "latent" else None,
            temperature=self._temperature,
        )
        return parse_vla_decision(
            text, synonyms=self._tax.synonyms, valid_categories=self._tax.valid_categories
        )


# ---------------------------------------------------------------------- the planner
class Phase(Enum):
    NOMINAL = "nominal"
    BACKSTEP = "backstep"
    DETOUR = "detour"
    HALTED = "halted"


@dataclass
class AvoidCircle:
    center: np.ndarray
    radius: float


@dataclass
class RolloutRecord:
    """One round's decision + its compiled outcome (for logging / DPO pair construction)."""

    t: float
    channel: str
    decision: ParsedDecision
    code: str
    attribution: str | None
    primitive: str | None


@dataclass
class VlaPlanner:
    """Closed-loop recovery policy driven by a :class:`VlaPolicy` (RecoveryPolicy drop-in)."""

    cfg: Config  # navigation params (the fsm config), reused for pursuit/backstep/detour
    policy: VlaPolicy
    goal_xy: np.ndarray
    dt: float
    recorder: SnapshotRecorder
    compiler: PrimitiveCompiler | None = None
    max_rounds: int = 6

    phase: Phase = field(init=False, default=Phase.NOMINAL)
    waypoints: list[np.ndarray] = field(init=False, default_factory=list)
    avoid_circles: list[AvoidCircle] = field(init=False, default_factory=list)
    decisions: list[RolloutRecord] = field(init=False, default_factory=list)
    first_snapshot: Snapshot | None = field(init=False, default=None)  # the DPO prompt node
    speed_cap: float | None = field(init=False, default=None)
    _pending_event: MonitorEvent | None = field(init=False, default=None)
    _backstep_until: float = field(init=False, default=0.0)
    _grace_until: float = field(init=False, default=0.0)
    _cmd_prev: np.ndarray = field(init=False, default_factory=lambda: np.zeros(3))
    _prior_texts: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.goal_xy = np.asarray(self.goal_xy, dtype=np.float64).copy()
        self.waypoints = [self.goal_xy.copy()]

    # -- RecoveryPolicy contract -------------------------------------------------
    def on_event(self, event: MonitorEvent) -> bool:
        """Queue a reflection for this event (decided in the next :meth:`step`, given obs)."""
        if self.phase is Phase.HALTED or event.t < self._grace_until:
            return False
        if len(self.decisions) >= self.max_rounds:
            return False
        self._pending_event = event
        return True

    def step(self, obs: Obs) -> np.ndarray:
        self.recorder.buffer(obs)
        if self._pending_event is not None:
            self._reflect(self._pending_event)
            self._pending_event = None
        if self.phase is Phase.HALTED:
            return self._slew(np.zeros(3))
        if self.phase is Phase.BACKSTEP:
            if obs.t >= self._backstep_until:
                self._replan(obs.pos)
                self.phase = Phase.DETOUR
                self._grace_until = obs.t + float(self.cfg.event_grace_s)
                return self._slew(self._pursuit(obs))
            return self._slew(np.array([-float(self.cfg.backstep.speed_mps), 0.0, 0.0]))
        return self._slew(self._pursuit(obs))

    def adopt_map_hazards(self, hazards: list[tuple[np.ndarray, float]]) -> None:
        """Merge semantic-map avoid discs (spec §7 map→planner); idempotent."""
        for center, radius in hazards:
            center = np.asarray(center, dtype=np.float64)
            covered = any(
                float(np.linalg.norm(c.center - center)) + radius <= c.radius
                for c in self.avoid_circles
            )
            if not covered:
                self.avoid_circles.append(AvoidCircle(center.copy(), float(radius)))

    # -- reflection --------------------------------------------------------------
    def _reflect(self, event: MonitorEvent) -> None:
        """Snapshot → policy → parse → compile → apply the chosen recovery (spec §5/§10)."""
        snapshot = self.recorder.capture(event, prior_outputs=list(self._prior_texts))
        if self.first_snapshot is None:
            self.first_snapshot = snapshot
        decision = self.policy.decide(snapshot)
        code = decision.reject_code
        if decision.ok and decision.annotation is not None:
            code = self._apply(decision.annotation, event, obs_pos=event.pos)
            self._prior_texts.append(f"{decision.attribution}:{decision.primitive_name}")
        self.decisions.append(
            RolloutRecord(
                t=float(event.t),
                channel=event.channel,
                decision=decision,
                code=code,
                attribution=decision.attribution,
                primitive=decision.primitive_name,
            )
        )

    def _apply(self, annotation: CoTAnnotation, event: MonitorEvent, obs_pos: np.ndarray) -> str:
        """Compile + execute one primitive into the navigation plan; return a status code."""
        prim = annotation.primitive
        # Map pixel/region coords to the failure-site odometry coords the planner knows.
        region_xy = (float(event.pos[0]), float(event.pos[1]))
        try:
            compiled_prim = to_compiler_primitive(
                prim,
                point_xy=self._waypoint_from_pixel(prim, event),
                region_xy=region_xy,
                region_radius_m=float(self.cfg.avoid.radius_m),
            )
        except Exception as err:  # noqa: BLE001 - a malformed param is a structured reject
            return f"REJECT_COMPILE: {err}"
        if self.compiler is not None:
            cc = self.compiler.compile(compiled_prim, _obs_for_compiler(event))
            if not cc.accepted:
                return cc.code
            self._enact(compiled_prim, cc, event)
            return cc.code
        self._enact(compiled_prim, None, event)
        return "OK"

    def _enact(self, prim: Primitive, cc: CompiledCommand | None, event: MonitorEvent) -> None:
        """Turn a compiled primitive into a navigation intent (escape/route vs push-through).

        Escape primitives (Backstep, Update_Topology) mark the failure site untraversable AND, when
        the robot is on it, back out first before routing around (spec §10 safety iron-rule — you
        cannot detour from inside a trap). Push-through primitives (Set_Constraint, Switch_Gait)
        cap the speed and continue straight across; on a fatal hazard that is the failing strategy
        — which is exactly what makes a wrong attribution physically distinguishable (spec §5)."""
        if isinstance(prim, Backstep):
            self._mark_avoid(event.pos, float(self.cfg.avoid.radius_m))
            self.phase = Phase.BACKSTEP
            self._backstep_until = event.t + float(self.cfg.backstep.duration_s)
        elif isinstance(prim, UpdateTopology):
            center = np.array([prim.region_xy[0], prim.region_xy[1]], dtype=np.float64)
            self._mark_avoid(center, float(prim.radius_m))
            if self._inside_avoid(event.pos):  # at the trap ⇒ escape first, then the detour
                self.phase = Phase.BACKSTEP
                self._backstep_until = event.t + float(self.cfg.backstep.duration_s)
            else:
                self._replan(event.pos)
        elif isinstance(prim, SetConstraint):
            self.speed_cap = float(prim.max_speed)  # push through, but slowly (no detour)
        elif isinstance(prim, SwitchGait):
            # high_step/crawl: push through; modelled as a mild speed cap (gait change), no detour.
            self.speed_cap = float(self.cfg.cruise_speed_mps) * (
                0.6 if prim.mode == "crawl" else 0.8
            )
        elif isinstance(prim, HoldAndRequest):
            self.phase = Phase.HALTED
        # AdjustPosture / ReplanWaypoint handled via cc fields below.
        if cc is not None and cc.waypoint_xy is not None:
            self.waypoints = [np.asarray(cc.waypoint_xy, dtype=np.float64), self.goal_xy.copy()]

    def _mark_avoid(self, center: np.ndarray, radius: float) -> None:
        """Mark a region untraversable (dedup against existing circles)."""
        center = np.asarray(center, dtype=np.float64)
        covered = any(
            float(np.linalg.norm(c.center - center)) + radius <= c.radius
            for c in self.avoid_circles
        )
        if not covered:
            self.avoid_circles.append(AvoidCircle(center.copy(), float(radius)))

    def _inside_avoid(self, pos: np.ndarray) -> bool:
        return any(float(np.linalg.norm(c.center - pos)) < c.radius for c in self.avoid_circles)

    def _waypoint_from_pixel(self, prim: RecoveryPrimitive, event: MonitorEvent) -> tuple | None:
        """A minimal pixel→odometry stand-in for Replan_Waypoint (a lateral detour goal).

        The spec depth-unprojects the 2D pixel; lacking live depth here we route to a lateral
        offset past the failure site toward the goal (enough to test the closed loop). The real
        Isaac path uses the RGB-D unprojection (spec §7)."""
        if prim.name != "Replan_Waypoint":
            return None
        to_goal = unit(self.goal_xy - event.pos)
        lateral = rot90(to_goal) * float(self.cfg.avoid.radius_m)
        return (float(event.pos[0] + lateral[0]), float(event.pos[1] + lateral[1]))

    # -- navigation (reused FSM geometry) ---------------------------------------
    def _replan(self, pos: np.ndarray) -> None:
        self.waypoints = _plan_detour(
            np.asarray(pos, dtype=np.float64),
            self.goal_xy,
            self.avoid_circles,
            float(self.cfg.avoid.clearance_m),
        )

    def _pursuit(self, obs: Obs) -> np.ndarray:
        while self.waypoints:
            to_wp = self.waypoints[0] - obs.pos
            if float(np.linalg.norm(to_wp)) > float(self.cfg.waypoint_tol_m):
                break
            self.waypoints.pop(0)
        if not self.waypoints:
            return np.zeros(3)
        to_wp = self.waypoints[0] - obs.pos
        dist = float(np.linalg.norm(to_wp))
        heading_err = wrap_angle(math.atan2(to_wp[1], to_wp[0]) - obs.heading)
        wz = float(self.cfg.heading_gain) * heading_err
        cruise = float(self.cfg.cruise_speed_mps)
        if self.speed_cap is not None:
            cruise = min(cruise, self.speed_cap)
        speed = cruise * min(1.0, dist / float(self.cfg.approach_slowdown_m))
        vx = (
            0.0
            if abs(heading_err) > float(self.cfg.align_angle_rad)
            else speed * max(0.0, math.cos(heading_err))
        )
        return np.array([vx, 0.0, wz])

    def _slew(self, raw: np.ndarray) -> np.ndarray:
        max_dv = float(self.cfg.cmd_slew_mps2) * self.dt
        max_dw = float(self.cfg.yaw_slew_radps2) * self.dt
        cmd = self._cmd_prev.copy()
        cmd[:2] += np.clip(raw[:2] - cmd[:2], -max_dv, max_dv)
        cmd[2] += float(np.clip(raw[2] - cmd[2], -max_dw, max_dw))
        self._cmd_prev = cmd
        return cmd.copy()


# --------------------------------------------------------------------- helpers
def _sibling_categories(cfg: Config, taxonomy: FailureTaxonomy) -> dict[str, str]:
    op_cat = cfg.attribution.operator_category
    out: dict[str, str] = {}
    for pair in cfg.maze.ambiguity_pairs:
        a, b = str(pair[0]), str(pair[1])
        ca, cb = str(op_cat.get(a)), str(op_cat.get(b))
        if ca and cb and ca != cb:
            out[ca], out[cb] = cb, ca
    return out


def _obs_for_compiler(event: MonitorEvent) -> Obs:
    """A minimal Obs at the failure site for the compiler's admission check (pose only)."""
    return Obs(
        t=float(event.t),
        pos=event.pos.copy(),
        heading=0.0,
        vel_body=np.zeros(2),
        yaw_rate=0.0,
        slip_ratio=0.0,
        effort_ratio=0.0,
        support_ratio=1.0,
        base_height=0.32,
        tilt=0.0,
        cmd_prev=np.zeros(3),
        fallen=False,
    )


def _plan_detour(
    start: np.ndarray,
    goal: np.ndarray,
    circles: list[AvoidCircle],
    clearance_m: float,
    max_legs: int = 6,
) -> list[np.ndarray]:
    """Box detour around avoid circles (same geometry as the FSM stub, spec §5 Replan)."""
    waypoints: list[np.ndarray] = []
    cur = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    for _ in range(max_legs):
        blocking = next(
            (
                c
                for c in circles
                if segment_hits_circle(cur, goal, c.center, c.radius + clearance_m)
            ),
            None,
        )
        if blocking is None:
            waypoints.append(goal)
            return waypoints
        reach = blocking.radius + clearance_m
        u = unit(goal - cur)
        perp = rot90(u)
        cross = float(u[0] * (blocking.center[1] - cur[1]) - u[1] * (blocking.center[0] - cur[0]))
        side = -1.0 if cross > 0.0 else 1.0
        corner1 = blocking.center + side * perp * reach
        corner2 = corner1 + u * reach
        waypoints.extend([corner1, corner2])
        cur = corner2
    waypoints.append(goal)
    return waypoints
