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

There is NO geometric avoid-disc or detour: ALL high-level navigation is the VLA's. In NOMINAL it
picks the next waypoint PIXEL each ~1 Hz tick (back-projected, spec §7) and ROUTES AROUND a hazard
it has discovered (its costmap mark + the visible feature, fed back as a map-note). The primitives
map to intents: Backstep/Update_Topology ⇒ back out of the trap (then the VLA routes around);
Set_Constraint/Switch_Gait ⇒ push through (capped speed); Adjust_Posture ⇒ lift & continue;
Hold_and_Request ⇒ stop and request help. So a *correct* attribution and a *wrong* one drive
genuinely different — sometimes opposite — trajectories: success is a physical signal (spec §5).
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
from kino_vla.monitor.reflex import ActiveProbe
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
from kino_vla.utils.geometry import rot90, unit, wrap_angle
from kino_vla.vla.output import (
    ParsedDecision,
    parse_nav_decision,
    parse_vla_decision,
    to_compiler_primitive,
)
from kino_vla.vla.prompt import (
    build_messages,
    build_nav_messages,
    context_from_snapshot,
    format_map_note,
)


# ---------------------------------------------------------------------- policies
class VlaPolicy(Protocol):
    """Maps a failure snapshot to a parsed atomic recovery decision (the planner's brain)."""

    def decide(self, snapshot: Snapshot, map_note: str = "") -> ParsedDecision: ...


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


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two appearance features (0 if shapes differ or either is degenerate)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return 0.0  # CLIP (512) vs a surrogate fallback (64) ⇒ not comparable, treat as no-match
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na > 0.0 and nb > 0.0 else 0.0


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

    def decide(self, snapshot: Snapshot, map_note: str = "") -> ParsedDecision:  # noqa: ARG002
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
        proprio_detail: str = "binned",
    ) -> None:
        self._model = model
        self._cfg = cfg
        self._tax = taxonomy
        self._route = route
        self._n_images = int(n_images)
        self._temperature = float(temperature)
        self._reveal = bool(reveal_appearance)
        self._proprio_detail = str(proprio_detail)

    def decide(self, snapshot: Snapshot, map_note: str = "") -> ParsedDecision:
        ctx = context_from_snapshot(
            snapshot,
            route=self._route,
            reveal_appearance=self._reveal,
            proprio_detail=self._proprio_detail,
            map_note=map_note,
        )
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

    def decide_nav(
        self, snapshot: Snapshot, goal_bearing_deg: float, map_note: str = ""
    ) -> ParsedDecision:
        """NOMINAL-mode (point 2): the VLA picks the next nav action — a Replan_Waypoint PIXEL
        the goal, OR a Turn (yaw_deg) to reorient toward clear ground (the user directive: the VLA
        owns the turn at ANY tick, not the planner). ``map_note`` is the §7 map crop (Q5)."""
        ctx = context_from_snapshot(
            snapshot,
            route=self._route,
            reveal_appearance=self._reveal,
            proprio_detail=self._proprio_detail,
            map_note=map_note,
        )
        messages = build_nav_messages(
            ctx, self._cfg, goal_bearing_deg, route=self._route, n_images=self._n_images
        )
        images = list(snapshot.rgb[-self._n_images :]) if snapshot.rgb.size else []
        text = self._model.generate(
            messages,
            images,
            proprio_window=snapshot.proprio_window if self._route == "latent" else None,
            temperature=self._temperature,
        )
        return parse_nav_decision(
            text, synonyms=self._tax.synonyms, valid_categories=self._tax.valid_categories
        )


# ---------------------------------------------------------------------- the planner
class Phase(Enum):
    NOMINAL = (
        "nominal"  # the VLA picks the next nav waypoint (and routes around discovered hazards)
    )
    BACKSTEP = "backstep"  # backing out of the adhesive (then back to NOMINAL)
    HALTED = "halted"  # Hold_and_Request


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
    probe: ActiveProbe | None = None  # active-sensing probe (#34c); None ⇒ reflect now

    phase: Phase = field(init=False, default=Phase.NOMINAL)
    _probing: bool = field(init=False, default=False)
    _probed: bool = field(init=False, default=False)  # probe once per hazard, then let recovery run
    waypoints: list[np.ndarray] = field(init=False, default_factory=list)
    _hazard_marked: bool = field(init=False, default=False)  # discovered a hazard ⇒ map_note on
    _forbidden_features: list = field(init=False, default_factory=list)  # CLIP feats the nav avoids
    decisions: list[RolloutRecord] = field(init=False, default_factory=list)
    nav_log: list = field(init=False, default_factory=list)  # (t, pixel, world_wp) per nav-pick
    first_snapshot: Snapshot | None = field(init=False, default=None)  # the DPO prompt node
    speed_cap: float | None = field(init=False, default=None)
    _pending_event: MonitorEvent | None = field(init=False, default=None)
    _backstep_until: float = field(init=False, default=0.0)  # safety cap only (closed-loop on exit)
    _backout_target: np.ndarray | None = field(init=False, default=None)  # drive-to exit point
    _grace_until: float = field(init=False, default=0.0)
    _cmd_prev: np.ndarray = field(init=False, default_factory=lambda: np.zeros(3))
    _prior_texts: list[str] = field(init=False, default_factory=list)
    monitor: object | None = field(init=False, default=None)  # set by the loop ⇒ live anomaly_score
    nav_map: object | None = field(init=False, default=None)  # set by the loop ⇒ stamp/read costmap
    _backstep_origin: np.ndarray | None = field(init=False, default=None)  # back-out ref point
    _backstep_dir: np.ndarray = field(init=False, default_factory=lambda: np.zeros(2))  # rev dir
    _steps: int = field(init=False, default=0)  # control-step counter (the ~1 Hz reflect clock)
    _reflect_every: int = field(init=False, default=1)  # steps between VLA reflections (~1 Hz)
    _recovery_rounds: int = field(
        init=False, default=0
    )  # recovery reflections (the max_rounds cap)

    def __post_init__(self) -> None:
        self.goal_xy = np.asarray(self.goal_xy, dtype=np.float64).copy()
        self.waypoints = [self.goal_xy.copy()]
        self._reflect_every = max(1, round(float(self.cfg.get("reflect_period_s", 1.0)) / self.dt))

    # -- RecoveryPolicy contract -------------------------------------------------
    def on_event(self, event: MonitorEvent) -> bool:
        """Queue a reflection for this event (decided in the next :meth:`step`, given obs).

        With an active probe configured, arm it now and run it for ~one window BEFORE the snapshot
        is captured (Gap-3 #34c) so the VLA reads an in-distribution proprioception window.
        """
        if self.phase is Phase.HALTED or event.t < self._grace_until:
            return False
        if self._recovery_rounds >= self.max_rounds:  # cap RECOVERY rounds, not the 1 Hz nav ticks
            return False
        # Once a hazard has been DISCOVERED, a fire while on CLEAR ground (not inside any §7 region)
        # is a route-around maneuvering artifact (the in-place turns spike the tracking channel) —
        # a new hazard — recovering on it MIS-attributes and can Hold_and_Request → HALT (run #27).
        # The feature-veto already keeps the robot off the known patch; a genuine NEW hazard fires
        # from INSIDE its region (_in_adhesive True) and is still handled.
        if self._forbidden_features and not self._in_adhesive(event.pos):
            return False
        self._pending_event = event
        # Probe ONCE per hazard to get an in-distribution attribution window, then let the chosen
        # recovery run uninterrupted. Re-probing on every monitor fire (persistent ice slip fires
        # repeatedly) traps the robot oscillating in the decel->accel maneuver and it never reaches.
        if self.probe is not None and not self._probing and not self._probed:
            self.probe.start(event.t)
            self._probing = True
            self._probed = True
            self._grace_until = event.t + self.probe.total_s  # don't re-fire mid-probe
        return True

    def step(self, obs: Obs) -> np.ndarray:
        self.recorder.buffer(obs)
        self._steps += 1
        if self._probing:
            cmd = self.probe.command(obs)
            if cmd is not None:
                return self._slew(cmd)  # drive the active-sensing probe; the window fills
            self._probing = False  # probe done ⇒ the window is in-distribution; reflect now
        if self.phase is Phase.BACKSTEP:
            return self._backstep_step(obs)
        if self.phase is Phase.HALTED:
            return self._slew(np.zeros(3))
        # The VLA runs on a ~1 Hz clock (point 2) and drives ALL high-level navigation — there is NO
        # geometric avoid-disc/detour. RECOVERY is entered ONLY by a monitor fire (arm-delay-aware),
        # COMMITTED (no mid-recovery re-attribution, #34c). After a Backstep the dog is in NOMINAL
        # again, where the VLA picks the next waypoint while AVOIDING the hazard it just discovered
        # (its costmap mark + the visible feature) — that is the whole route-around.
        due = self._steps % self._reflect_every == 0
        if self._pending_event is not None:
            # A fire queued DURING a backstep predates the post-backout grace. If we are now within
            # that grace, DROP it (stale — the dog already backed out): else it re-backsteps the
            # instant the backstep ends and the NOMINAL nav-pick never gets to route around (the
            # oscillation bug — the dog ping-pongs in/out of the patch forever).
            if obs.t >= self._grace_until:
                self._reflect(self._pending_event, obs)
            self._pending_event = None
        elif (
            due
            and self.phase is Phase.NOMINAL
            and obs.t >= float(self.cfg.get("nav_warmup_s", 0.0))
            and hasattr(self.policy, "decide_nav")
        ):
            # NOMINAL: the VLA picks the next nav action (waypoint/turn — routes around). Optional
            # nav_warmup_s delays the first pick (a safety knob; 0 by default — the cruise prompt is
            # the stable run-#23 one, so the corrective first pick is wanted, not suppressed).
            self._reflect_nav(obs)
        if self.phase is Phase.BACKSTEP:
            return self._backstep_step(obs)
        if self.phase is Phase.HALTED:
            return self._slew(np.zeros(3))
        return self._slew(self._pursuit(obs))

    def _tick_event(self, obs: Obs, anomaly: float) -> MonitorEvent:
        """A synthetic MonitorEvent for a 1 Hz clock reflection (no real monitor fire)."""
        return MonitorEvent(
            t=float(obs.t),
            pos=obs.pos.copy(),
            channel="clock",
            value=float(anomaly),
            threshold=1.0,
            summary="1Hz tick",
        )

    def _goal_bearing(self, obs: Obs) -> float:
        """Goal bearing (deg) relative to the robot heading — the NOMINAL nav prompt's direction."""
        to_goal = self.goal_xy - obs.pos
        return math.degrees(wrap_angle(math.atan2(to_goal[1], to_goal[0]) - obs.heading))

    def _reflect_nav(self, obs: Obs) -> None:
        """NOMINAL 1 Hz nav: the VLA chooses the next ACTION — a Replan_Waypoint pixel
        (back-projected) OR a Turn (yaw_deg, the VLA reorienting). The hard CLIP-feature veto (the
        user rule) rejects a waypoint OR a turn-target on a surface the robot already failed in and
        re-asks the VLA — the VLA, NOT the planner, then chooses a clear waypoint or a turn angle.
        If after the retries the VLA still gives nothing legal, HOLD (no planner geometry); the next
        tick re-asks. Final approach: once the straight line to the goal is clear, head directly."""
        if self._forbidden_features and self._path_clear(obs.pos, self.goal_xy):
            self.waypoints = [self.goal_xy.copy()]  # rounded the patch ⇒ go straight to the goal
            self.nav_log.append((round(float(obs.t), 2), None, None))
            self._record_history("nav:goal_direct")
            return
        ev = self._tick_event(obs, self._anomaly() or 0.0)
        snapshot = self.recorder.capture(ev, prior_outputs=list(self._prior_texts))
        base_note = self._map_note(obs.pos, obs.heading)
        retries = int(self.cfg.get("nav_forbidden_retries", 2))
        px: list | None = None
        wp: np.ndarray | None = None
        tag = "parse_fail"
        hint = ""
        committed = False
        wp_hint = (
            " Your last pixel landed ON the forbidden hazard surface (the coloured region you got "
            "stuck in). Pick a DIFFERENT pixel on clear ground NOT on that surface, OR a Turn "
            "(yaw_deg in [-90,90]) to rotate toward clear ground that goes around the patch."
        )
        turn_hint = (
            " That turn faces the forbidden hazard. Choose a DIFFERENT yaw_deg ([-90,90]) toward "
            "clear ground, or a clear waypoint pixel."
        )
        for _ in range(retries + 1):
            decision = self.policy.decide_nav(
                snapshot, self._goal_bearing(obs), map_note=base_note + hint
            )
            if (
                decision.ok and decision.nav_turn_deg is not None
            ):  # the VLA chose to ROTATE (user rule)
                yaw = float(decision.nav_turn_deg)
                target = self._turn_target(obs, yaw)
                if not self._is_forbidden(target):
                    self.waypoints = [target, self.goal_xy.copy()]
                    tag, committed = f"turn{yaw:+.0f}", True
                    break
                hint, tag = turn_hint, "turn_forbidden"
                continue
            if decision.ok and decision.primitive_name == "Replan_Waypoint":
                cand_px = decision.annotation.primitive.params.get("point_px")
                cand_wp = None
                if cand_px is not None and len(cand_px) == 2:
                    px = cand_px
                    cand_wp = self.recorder.world_from_pixel(
                        obs.pos, obs.heading, float(cand_px[0]) / 1000.0, float(cand_px[1]) / 1000.0
                    )
                if cand_wp is not None and float(np.linalg.norm(cand_wp - obs.pos)) > float(
                    self.cfg.waypoint_tol_m
                ):
                    if not self._is_forbidden(cand_wp):  # legal clear-feature ground ⇒ accept
                        wp = np.asarray(cand_wp, dtype=np.float64)
                        self.waypoints = [wp, self.goal_xy.copy()]
                        tag, committed = "Replan_Waypoint", True
                        break
                    hint, tag = wp_hint, "wp_forbidden"
                    continue
            hint = wp_hint  # parse-fail / too-close ⇒ re-ask (the hint also offers a Turn)
        if not committed:  # the VLA never produced a legal waypoint/turn ⇒ HOLD; next tick re-asks
            self.waypoints = []  # NO planner-computed geometry (the VLA owns all high-level nav)
        self.nav_log.append(
            (
                round(float(obs.t), 2),
                list(px) if px is not None else None,
                [round(float(wp[0]), 2), round(float(wp[1]), 2)] if wp is not None else None,
            )
        )
        self._record_history(f"nav:{tag}")

    def _turn_target(self, obs: Obs, yaw_deg: float) -> np.ndarray:
        """The short waypoint the robot reaches by rotating yaw_deg (the VLA's Turn) then stepping —
        used to EXECUTE the VLA's chosen turn and to veto a turn that would face the hazard."""
        yaw = math.radians(max(-90.0, min(90.0, float(yaw_deg))))
        ang = obs.heading + yaw
        step = float(self.cfg.get("turn_probe_m", 1.6))
        return obs.pos + step * np.array([math.cos(ang), math.sin(ang)], dtype=np.float64)

    def _record_history(self, tag: str) -> None:
        """Keep the last N VLA outputs as context (point 3: prior outputs feed the next round)."""
        self._prior_texts.append(tag)
        keep = int(self.cfg.get("context_rounds", 5))
        if len(self._prior_texts) > keep:
            self._prior_texts[:] = self._prior_texts[-keep:]

    def _anomaly(self) -> float | None:
        """Live Kino-Monitor anomaly_score (None if the loop didn't wire the monitor in)."""
        m = self.monitor
        return float(m.anomaly_score) if m is not None else None

    def _start_backstep(self, event: MonitorEvent) -> None:
        """Enter BACKSTEP: a body-frame REVERSE — back straight out the way it came while KEEPING
        the goal-facing heading, so the route-around afterwards needs no topple-prone 180-deg
        turn-around (run #16). Free, since the adhesive has zero backward resistance."""
        self._backstep_origin = None
        self.phase = Phase.BACKSTEP
        self._backstep_until = event.t + float(self.cfg.get("backstep.max_duration_s", 8.0))

    def _backstep_step(self, obs: Obs) -> np.ndarray:
        """Reverse straight back (NO heading change — keep facing the goal) until the robot is
        clear of the adhesive by a margin, then hand the nav to the VLA, which routes AROUND the
        hazard (the feature-veto forbids re-entering it). Keeping the heading avoids the 180-deg
        turn-around that topples the dog. The reverse is free; max_duration_s caps it."""
        if self._backstep_origin is None:
            self._backstep_origin = obs.pos.copy()
        backed = float(np.linalg.norm(obs.pos - self._backstep_origin))
        clear = not self._in_adhesive(obs.pos)
        min_back = float(self.cfg.get("backstep.min_backout_m", 1.0))
        if (clear and backed >= min_back) or obs.t >= self._backstep_until:
            self._backstep_origin = None
            self._pending_event = None  # drop the stale during-backstep fire (the dog is clear now)
            self.phase = Phase.NOMINAL
            self._grace_until = obs.t + float(self.cfg.event_grace_s)
            if hasattr(self.policy, "decide_nav"):
                self._reflect_nav(obs)  # the VLA picks a route-around waypoint immediately
            return self._slew(self._pursuit(obs))
        return self._slew(np.array([-float(self.cfg.backstep.speed_mps), 0.0, 0.0]))

    # -- reflection --------------------------------------------------------------
    def _reflect(self, event: MonitorEvent, obs: Obs) -> None:
        """Snapshot → policy → parse → compile → apply the chosen recovery (spec §5/§10)."""
        self._recovery_rounds += 1
        snapshot = self.recorder.capture(event, prior_outputs=list(self._prior_texts))
        if self.first_snapshot is None:
            self.first_snapshot = snapshot
        decision = self.policy.decide(snapshot, map_note=self._map_note(obs.pos, obs.heading))
        code = decision.reject_code
        if decision.ok and decision.annotation is not None:
            code = self._apply(decision.annotation, event, obs_pos=event.pos)
            self._record_history(f"{decision.attribution}:{decision.primitive_name}")
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
            self._stamp_costmap(event.pos)  # mark it so the VLA's map_note routes around it
            self._start_backstep(event)
        elif isinstance(prim, UpdateTopology):
            self._stamp_costmap(event.pos)
            if self._in_adhesive(
                event.pos
            ):  # standing on it ⇒ back out; the VLA then routes around
                self._start_backstep(event)
        elif isinstance(prim, SetConstraint):
            self.speed_cap = float(prim.max_speed)  # push through, but slowly
        elif isinstance(prim, SwitchGait):
            # high_step/crawl: push through (a mild speed cap), no detour.
            self.speed_cap = float(self.cfg.cruise_speed_mps) * (
                0.6 if prim.mode == "crawl" else 0.8
            )
        elif isinstance(prim, HoldAndRequest):
            self.phase = Phase.HALTED
        if cc is not None and cc.waypoint_xy is not None:  # Replan_Waypoint: the VLA's own waypoint
            self.waypoints = [np.asarray(cc.waypoint_xy, dtype=np.float64), self.goal_xy.copy()]

    def _stamp_costmap(self, pos: np.ndarray) -> None:
        """Mark the DISCOVERED hazard on the §7 semantic costmap AND record its CLIP feature, so the
        nav-pick can HARD-FORBID landing on the same-feature surface (the user rule). Flips the
        discovered-hazard flag (before discovery the VLA cruises: step in, feel, THEN avoid)."""
        self._hazard_marked = True
        pos = np.asarray(pos, dtype=np.float64)
        if self.nav_map is not None and hasattr(self.nav_map, "mark_failure"):
            self.nav_map.mark_failure(pos)
        # Record the CLIP feature of the region we just failed in: every subsequent nav waypoint is
        # then forbidden from landing where the appearance matches it (the semantic map as a HARD
        # filter on the pick, not advisory text — the user's directive). Prefer the region-DOMINANT
        # feature (averaged over observed patch cells, always the costmap/CLIP width) over the exact
        # failure cell, which the forward-down camera often never imaged.
        feat = None
        if self.nav_map is not None and hasattr(self.nav_map, "region_feature"):
            feat = self.nav_map.region_feature(pos)
        if feat is None and self.nav_map is not None and hasattr(self.nav_map, "feature_at"):
            feat = self.nav_map.feature_at(pos)
        if feat is not None:
            thresh = float(getattr(self.nav_map, "sim_threshold", 0.8))
            if not any(_cosine(feat, f) >= thresh for f in self._forbidden_features):
                self._forbidden_features.append(np.asarray(feat, dtype=np.float64))

    def _matches_forbidden(self, feat: np.ndarray | None) -> bool:
        if feat is None or not self._forbidden_features:
            return False
        thresh = float(getattr(self.nav_map, "sim_threshold", 0.8))
        return any(_cosine(feat, f) >= thresh for f in self._forbidden_features)

    def _is_forbidden(self, wp: np.ndarray) -> bool:
        """Is a candidate waypoint on a surface whose CLIP feature matches one the robot has already
        failed in? FORBIDDEN (the hard feature-veto, the user rule). Two checks: (1) the perceived
        feature at the cell (covers observed + CLIP-propagated ground); (2) the WHOLE scene region
        it falls in, keyed by that region's dominant feature — so an as-yet-UNobserved part of the
        same failed patch is forbidden too (run #19: the dog drove into an unimaged patch cell)."""
        nm = self.nav_map
        if nm is None or not self._forbidden_features:
            return False
        wp = np.asarray(wp, dtype=np.float64)
        if hasattr(nm, "feature_at") and self._matches_forbidden(nm.feature_at(wp)):
            return True
        if hasattr(nm, "scene") and hasattr(nm, "region_feature"):
            m = float(self.cfg.get("nav_forbidden_margin_m", 0.8))  # keep waypoints clear of the
            # patch by a margin, so the ~0.3-0.5 m control drift while skirting does not clip in
            for region in nm.scene:
                r = region.rect
                if abs(wp[0] - r.cx) <= r.hx + m and abs(wp[1] - r.cy) <= r.hy + m:
                    center = np.array([r.cx, r.cy], dtype=np.float64)
                    if self._matches_forbidden(nm.region_feature(center)):
                        return True
        return False

    def _path_clear(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Is the straight segment a→b free of the forbidden region? Sampled at ~0.4 m spacing."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        n = max(2, int(float(np.linalg.norm(b - a)) / 0.4))
        return not any(self._is_forbidden(a + (b - a) * (i / n)) for i in range(n + 1))

    def _map_note(self, pos: np.ndarray, heading: float) -> str:
        """The §7 map crop for the VLA (Q5): once the hazard is DISCOVERED (felt + costmap-marked),
        tell the VLA WHERE the same-feature region is so it heads toward the goal AROUND it. The
        hard feature-veto enforces avoidance (a pick on it is rejected); the note keeps the picks
        efficient. Empty before discovery — the dog steps in, feels it, THEN remembers."""
        nm = self.nav_map
        if not self._hazard_marked or nm is None or not hasattr(nm, "scene") or not nm.scene:
            return ""
        return format_map_note(pos, heading, [reg.rect for reg in nm.scene])

    def _in_adhesive(self, pos: np.ndarray) -> bool:
        """Is the robot inside the perceived hazard region (the §7 costmap-grounded scene rect)? A
        PERCEPTION read for 'have I backed out / am I on the trap', NOT a routing avoid-disc."""
        nm = self.nav_map
        if nm is None or not hasattr(nm, "scene") or not nm.scene:
            return False
        return any(
            abs(pos[0] - reg.rect.cx) <= reg.rect.hx and abs(pos[1] - reg.rect.cy) <= reg.rect.hy
            for reg in nm.scene
        )

    def _waypoint_from_pixel(self, prim: RecoveryPrimitive, event: MonitorEvent) -> tuple | None:
        """A minimal pixel→odometry stand-in for a recovery Replan_Waypoint (a lateral goal). The
        live NOMINAL nav back-projects the VLA's pixel exactly (world_from_pixel); this corner-case
        recovery primitive routes to a lateral offset past the failure site toward the goal."""
        if prim.name != "Replan_Waypoint":
            return None
        to_goal = unit(self.goal_xy - event.pos)
        lateral = rot90(to_goal) * 1.5
        return (float(event.pos[0] + lateral[0]), float(event.pos[1] + lateral[1]))

    def _pursuit(self, obs: Obs) -> np.ndarray:
        if getattr(self, "_start_xy", None) is None:
            self._start_xy = obs.pos.copy()
        startup = float(self.cfg.get("startup_straight_s", 0.0))
        if obs.t < startup and self.goal_xy is not None:
            # Pure-pursuit the START→GOAL LINE with a GENTLE gain during the push-off: this corrects
            # the Go2's intrinsic +y startup drift (which topples it if uncorrected — fell at y=0.37)
            # WITHOUT the sharp turn a nav-pick can demand (which ALSO topples the marginal gait). The
            # dog tracks the straight line INTO the patch (step-in→feel→recover), not around it. The
            # nav-pick still fires; only its turn waits out this brief warmup.
            d = np.asarray(self.goal_xy, dtype=np.float64) - self._start_xy
            dn = float(np.linalg.norm(d))
            if dn > 1e-6:
                d = d / dn
                proj = self._start_xy + float(np.dot(obs.pos - self._start_xy, d)) * d
                look = proj + 1.0 * d  # a CLOSE look-ahead ON the line (strong cross-track correction)
                to = look - obs.pos
                heading_err = wrap_angle(math.atan2(to[1], to[0]) - obs.heading)
                wz = float(self.cfg.get("startup_heading_gain", 1.8)) * heading_err
                # cruise SLOWLY during the push-off (a gentle gait is more stable from rest) — ramp
                # to full speed once warmed up; the line tracking corrects the +y drift meanwhile.
                spd = float(self.cfg.get("startup_speed_mps", 0.35))
                return np.array([spd * max(0.4, math.cos(heading_err)), 0.0, wz])
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
