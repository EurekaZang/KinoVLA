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
from kino_vla.monitor.event import MonitorEvent
from kino_vla.monitor.reflex import ActiveProbe
from kino_vla.shield.primitive_compiler import (
    AdjustPosture,
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
from kino_vla.utils.geometry import wrap_angle
from kino_vla.vla.nav_planner import occupancy_count, plan_route
from kino_vla.vla.output import (
    TURN_CLAMP_DEG,
    ParsedDecision,
    parse_nav_decision,
    parse_vla_decision,
    read_params_xy,
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


# Canonical §5 params for the StubVlaPolicy (the CI/demo perfect-attribution oracle). The executor
# now obeys these verbatim (no min_backout_m / avoid.radius_m override), so the distance/radius are
# sized to clear a ~2 m hazard patch from its interior — the old override behaviour, but now carried
# IN the primitive where the VLA owns it. The real ModelVlaPolicy emits its own learned values.
_CANON_PARAMS = {
    "Backstep": {"distance_m": 1.2},
    "Replan_Waypoint": {"point_px": [480, 360]},
    "Switch_Gait": {"mode": "high_step"},
    "Adjust_Posture": {"body_height_m": 0.25, "pitch_deg": 0.0},
    "Set_Constraint": {"max_speed": 0.4, "stiffness": 0.5},
    "Update_Topology": {"region_xy": [0.0, 0.0], "radius_m": 1.0, "status": "untraversable"},
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
        mask_proprio: bool = False,
    ) -> None:
        self._model = model
        self._cfg = cfg
        self._tax = taxonomy
        self._route = route
        self._n_images = int(n_images)
        self._temperature = float(temperature)
        self._reveal = bool(reveal_appearance)
        self._proprio_detail = str(proprio_detail)
        # B-V (A2 vision-only arm): mask the proprioception channel. On the latent route this drops
        # the projected Kino-Tokens (the <|kino|> placeholder embeddings are left un-overwritten, so
        # the model reads vision + a constant, proprio-independent placeholder); on the text route it
        # forces proprio_detail="none". Additive (default off ⇒ the deployed policy is unchanged).
        self._mask_proprio = bool(mask_proprio)

    def decide(self, snapshot: Snapshot, map_note: str = "") -> ParsedDecision:
        detail = "none" if (self._mask_proprio and self._route == "text") else self._proprio_detail
        ctx = context_from_snapshot(
            snapshot,
            route=self._route,
            reveal_appearance=self._reveal,
            proprio_detail=detail,
            map_note=map_note,
        )
        messages = build_messages(ctx, self._cfg, route=self._route, n_images=self._n_images)
        images = list(snapshot.rgb[-self._n_images :]) if snapshot.rgb.size else []
        pass_proprio = self._route == "latent" and not self._mask_proprio
        text = self._model.generate(
            messages,
            images,
            proprio_window=snapshot.proprio_window if pass_proprio else None,
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
    TURNING = "turning"  # pure in-place yaw rotation (Turn primitive); then back to NOMINAL
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
    _route_occ: int = field(init=False, default=-1)  # marked-cell count the route was planned for
    first_snapshot: Snapshot | None = field(init=False, default=None)  # the DPO prompt node
    speed_cap: float | None = field(init=False, default=None)
    # Commanded BODY POSTURE the low-level backend physically tracks (the closed-loop height
    # controller, the M2 Reflex made real). None ⇒ nominal trot (policy owns the body). Set by the
    # posture/gait primitives so Switch_Gait/Adjust_Posture/Set_Constraint each produce a DISTINCT,
    # dog-executed physical response, not just a speed cap (#41). The loop forwards these to
    # backend.set_posture each step (getattr-guarded, like set_reflex).
    posture_height: float | None = field(init=False, default=None)
    posture_stiffness: float = field(init=False, default=1.0)
    _pending_event: MonitorEvent | None = field(init=False, default=None)
    _backstep_until: float = field(init=False, default=0.0)  # safety cap only (closed-loop on exit)
    _backstep_distance: float = field(init=False, default=0.0)  # VLA-commanded reverse distance (m)
    _backout_target: np.ndarray | None = field(init=False, default=None)  # drive-to exit point
    _turn_target_heading: float = field(init=False, default=0.0)  # in-place Turn target heading
    _turn_until: float = field(init=False, default=0.0)  # Turn safety-cap deadline
    _grace_until: float = field(init=False, default=0.0)
    _cmd_prev: np.ndarray = field(init=False, default_factory=lambda: np.zeros(3))
    _prior_texts: list[str] = field(init=False, default_factory=list)
    monitor: object | None = field(init=False, default=None)  # set by the loop ⇒ live anomaly_score
    nav_map: object | None = field(init=False, default=None)  # set by the loop ⇒ stamp/read costmap
    # READ-ONLY on-policy nav tap (#43 DAgger collector): when set, called once per NOMINAL nav tick
    # with the visited state + the planner's verdict. It NEVER alters a decision (a data sink, not a
    # plugin) — None in the deployed loop ⇒ behaviour byte-identical. Set via run_vla_rollout.
    nav_trace_sink: object | None = field(init=False, default=None)
    _backstep_origin: np.ndarray | None = field(init=False, default=None)  # back-out ref point
    _backstep_dir: np.ndarray = field(init=False, default_factory=lambda: np.zeros(2))  # rev dir
    _steps: int = field(init=False, default=0)  # control-step counter (the ~1 Hz reflect clock)
    _reflect_every: int = field(init=False, default=1)  # steps between VLA reflections (~1 Hz)
    _recovery_rounds: int = field(
        init=False, default=0
    )  # recovery reflections since the last NEW hazard (the dead-loop cap; reset on progress, #1)
    # Multi-patch / long-distance readiness (#47): the last ATTRIBUTION site (so max_rounds bounds
    # re-attribution at ONE spot, never the lifetime number of hazards on a long course) and the
    # location the post-maneuver grace protects (so the grace suppresses SAME-spot re-fires but lets
    # a fire at a NEW, distant patch through — dense/sequential multi-patch).
    _last_reflect_pos: np.ndarray | None = field(init=False, default=None)
    _grace_center: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.goal_xy = np.asarray(self.goal_xy, dtype=np.float64).copy()
        self.waypoints = [self.goal_xy.copy()]
        self._reflect_every = max(1, round(float(self.cfg.get("reflect_period_s", 1.0)) / self.dt))
        # NOMINAL nav mode (#44 decoupling): "decoupled" ⇒ a deployed grid planner commits the route
        # around the persistent costmap (the VLA only ATTRIBUTES on contact); "vla_reactive" ⇒ the
        # VLA picks the waypoint/Turn per frame (the ablation baseline that oscillates at the edge).
        self._nav_mode = str(self.cfg.get("nav_mode", "decoupled"))

    # -- RecoveryPolicy contract -------------------------------------------------
    def on_event(self, event: MonitorEvent) -> bool:
        """Queue a reflection for this event (decided in the next :meth:`step`, given obs).

        With an active probe configured, arm it now and run it for ~one window BEFORE the snapshot
        is captured (Gap-3 #34c) so the VLA reads an in-distribution proprioception window.
        """
        if self.phase is Phase.HALTED:
            return False
        # #1 progress reset: a fire FAR from the last attribution is a NEW, distinct hazard down the
        # course, not a dead loop: reset the counter so max_rounds bounds re-attribution at one spot
        # (a real dead loop), never the lifetime hazard count (long / multi-patch courses).
        reset_dist = float(self.cfg.get("recovery_reset_dist_m", 1.5))
        moved = self._last_reflect_pos is None or (
            float(np.linalg.norm(event.pos - self._last_reflect_pos)) > reset_dist
        )
        if moved:
            self._recovery_rounds = 0
        # #2 location-aware grace: the post-maneuver grace suppresses re-fires on the SAME hazard,
        # but a fire at a NEW, distant patch is not that maneuver — let it through (dense/sequential
        # multi-patch). _grace_suppresses = within the grace window AND near the protected centre.
        if self._grace_suppresses(event.pos, event.t):
            return False
        if self._recovery_rounds >= self.max_rounds:  # cap re-attribution AT ONE SPOT (dead loop)
            return False
        # Once a hazard has been DISCOVERED, a fire while on CLEAR ground (not inside any §7 region)
        # is a route-around maneuvering artifact — recovering on it MIS-attributes and can
        # Hold_and_Request → HALT (run #27). A genuine NEW hazard fires from INSIDE its region
        # (_in_adhesive True) and is still handled.
        if self._forbidden_features and not self._in_adhesive(event.pos):
            return False
        self._pending_event = event
        # Probe at each NEW hazard (#5: ``moved`` ⇒ a fresh in-distribution window), or the first
        # time. NOT on every re-fire of the SAME hazard (re-probing a persistent slip traps the dog
        # oscillating in the decel→accel maneuver). The grace it sets is centred on this fire.
        if self.probe is not None and not self._probing and (moved or not self._probed):
            self.probe.start(event.t)
            self._probing = True
            self._probed = True
            self._grace_until = event.t + self.probe.total_s  # don't re-fire mid-probe
            self._grace_center = np.asarray(event.pos, dtype=np.float64).copy()
        return True

    def _grace_suppresses(self, pos: np.ndarray, t: float) -> bool:
        """#2: is a fire at ``pos`` at time ``t`` suppressed by the post-maneuver grace? Only within
        the grace window AND near the protected centre (the maneuver site). A fire at a NEW, distant
        patch (dense / long multi-patch) is NOT the same maneuver, so it is not suppressed."""
        if t >= self._grace_until:
            return False
        if self._grace_center is None:
            return True  # grace active without a located centre (startup) ⇒ suppress
        reset_dist = float(self.cfg.get("recovery_reset_dist_m", 1.5))
        return bool(
            float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - self._grace_center))
            <= reset_dist
        )

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
        if self.phase is Phase.TURNING:
            return self._turn_step(obs)
        if self.phase is Phase.HALTED:
            return self._slew(np.zeros(3))
        # The VLA runs on a ~1 Hz clock (point 2) and drives ALL high-level navigation — there is NO
        # geometric avoid-disc/detour. RECOVERY is entered ONLY by a monitor fire (arm-delay-aware),
        # COMMITTED (no mid-recovery re-attribution, #34c). After a Backstep the dog is in NOMINAL
        # again, where the VLA picks the next waypoint while AVOIDING the hazard it just discovered
        # (its costmap mark + the visible feature) — that is the whole route-around.
        due = self._steps % self._reflect_every == 0
        if self._pending_event is not None:
            # A fire queued DURING a backstep predates the post-backout grace. If it is a SAME-spot
            # re-fire within the grace, DROP it (stale — the dog already backed out): else it
            # re-backsteps the instant the backstep ends and never routes around (the oscillation
            # bug). Location-aware (#2): a fire at a NEW, distant patch is NOT stale ⇒ reflected.
            if not self._grace_suppresses(self._pending_event.pos, obs.t):
                self._reflect(self._pending_event, obs)
            self._pending_event = None
        elif (
            due
            and self.phase is Phase.NOMINAL
            and obs.t >= float(self.cfg.get("nav_warmup_s", 0.0))
        ):
            # NOMINAL nav tick: the decoupled grid planner commits a route around the persistent
            # costmap (default), or the VLA picks per frame (the vla_reactive ablation).
            # nav_warmup_s delays the first pick (a safety knob; 0 by default).
            self._nav_tick(obs)
        if self.phase is Phase.BACKSTEP:
            return self._backstep_step(obs)
        if self.phase is Phase.TURNING:  # a nav-pick just chose Turn ⇒ rotate in place this step
            return self._turn_step(obs)
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

    def _nav_tick(self, obs: Obs) -> None:
        """One NOMINAL nav decision in the configured mode (#44): the decoupled geometric planner
        (default) or the VLA-reactive per-frame pick (the ablation baseline). The decoupled planner
        needs no VlaPolicy; the reactive path is skipped when the policy has no ``decide_nav``."""
        if self._nav_mode == "decoupled":
            self._plan_nav(obs)
        elif hasattr(self.policy, "decide_nav"):
            self._reflect_nav(obs)

    def _plan_nav(self, obs: Obs) -> None:
        """DECOUPLED nominal nav (#44): a deployed grid planner COMMITS a route around the marked
        costmap hazards the VLA attributed on contact. The VLA owns open-set attribution (recovery);
        the planner owns routing. It RE-PLANS only when the marked-cell count changes (commit-and-
        hold ⇒ no per-frame oscillation); otherwise it holds the committed waypoints the pursuit
        controller is already driving. With no costmap, it commits the straight line to the goal."""
        nm = self.nav_map
        cm = getattr(nm, "costmap", None) if nm is not None else None
        if cm is None or not hasattr(cm, "cost_grid"):
            if not self.waypoints:
                self.waypoints = [self.goal_xy.copy()]
            return
        threshold = float(self.cfg.get("nav_cost_threshold", 0.5))
        occ = occupancy_count(cm.cost_grid, threshold)
        if self.waypoints and occ == self._route_occ:
            return  # commit-and-hold: no new hazard ⇒ keep the committed route (no re-plan churn)
        self._route_occ = occ
        self.waypoints = plan_route(
            cm.cost_grid,
            cm.origin_xy,
            cm.resolution_m,
            obs.pos,
            self.goal_xy,
            threshold=threshold,
            inflation_m=float(self.cfg.get("nav_inflation_m", 0.5)),
        )
        self.nav_log.append(
            (
                round(float(obs.t), 2),
                None,
                [[round(float(w[0]), 2), round(float(w[1]), 2)] for w in self.waypoints],
            )
        )
        self._record_history(f"nav:route({len(self.waypoints)})")

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
        for _ in range(retries + 1):
            decision = self.policy.decide_nav(
                snapshot, self._goal_bearing(obs), map_note=base_note + hint
            )
            if (
                decision.ok and decision.nav_turn_deg is not None
            ):  # the VLA chose to ROTATE IN PLACE (user rule) — a PURE yaw turn, not a translation.
                # NO forbidden-veto: an in-place rotation never moves the CoM onto the hazard; it
                # only re-aims the camera so the NEXT nav-pick can find clear ground (the point of
                # Turn). The dog physically executes the commanded yaw velocity (low-level policy).
                self._start_turn(obs, float(decision.nav_turn_deg))
                tag, committed = f"turn{decision.nav_turn_deg:+.0f}", True
                break
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
        # READ-ONLY tap (after the decision is finalized ⇒ cannot influence it): hand the visited
        # state + verdict to the DAgger collector for on-policy relabeling (#43). No-op when unset.
        if self.nav_trace_sink is not None:
            self.nav_trace_sink(
                {
                    "t": float(obs.t),
                    "pose_xy": obs.pos.copy(),
                    "heading": float(obs.heading),
                    "goal_bearing_deg": self._goal_bearing(obs),
                    "map_note": base_note,
                    "prior_outputs": list(self._prior_texts),
                    "rgb": snapshot.rgb,
                    "proprio_window": snapshot.proprio_window,
                    "tag": tag,
                    "committed": bool(committed),
                }
            )

    def _start_turn(self, obs: Obs, yaw_deg: float) -> None:
        """Enter TURNING: a PURE in-place yaw rotation by ``yaw_deg`` (clamped to [-90, 90]) the dog
        physically executes via a commanded yaw velocity (vx = vy = 0). It does not translate, so it
        cannot move the CoM onto the hazard — it only re-aims the camera. On completion the planner
        returns to NOMINAL and re-picks a nav action from the NEW heading (the route-around the old
        projected-waypoint Turn could not do because every forward pixel was feature-vetoed)."""
        yaw = math.radians(max(-TURN_CLAMP_DEG, min(TURN_CLAMP_DEG, float(yaw_deg))))
        self._turn_target_heading = wrap_angle(float(obs.heading) + yaw)
        self._turn_until = float(obs.t) + float(self.cfg.get("turn_max_s", 4.0))
        self.posture_height = None  # turn at the nominal trot posture (no held crouch/raise)
        self.phase = Phase.TURNING

    def _turn_step(self, obs: Obs) -> np.ndarray:
        """Rotate in place toward ``_turn_target_heading`` at the configured yaw rate (NO linear
        velocity), then hand back to NOMINAL and re-pick from the new heading. ``turn_max_s`` is the
        only (safety) cap — bounds a turn the low-level policy cannot complete."""
        err = wrap_angle(self._turn_target_heading - float(obs.heading))
        if abs(err) <= float(self.cfg.get("turn_tol_rad", 0.12)) or obs.t >= self._turn_until:
            self.phase = Phase.NOMINAL
            self._grace_until = max(self._grace_until, obs.t + float(self.cfg.event_grace_s))
            self._grace_center = obs.pos.copy()  # the grace protects THIS turn site (#2)
            self._nav_tick(obs)  # re-pick a route-around action from the new heading
            return self._slew(self._pursuit(obs))
        rate = float(self.cfg.get("turn_rate_radps", 1.0))
        return self._slew(np.array([0.0, 0.0, math.copysign(rate, err)], dtype=np.float64))

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

    def _start_backstep(self, event: MonitorEvent, distance_m: float) -> None:
        """Enter BACKSTEP: a body-frame REVERSE — back straight out the way it came while KEEPING
        the goal-facing heading, so the route-around afterwards needs no topple-prone 180-deg
        turn-around (run #16). Free, since the adhesive has zero backward resistance.

        ``distance_m`` is the VLA-commanded reverse distance (Backstep.distance_m, or the
        Update_Topology hazard radius for the iron-rule back-out) — the executor OBEYS it; the only
        early stop is the ``max_duration_s`` safety cap."""
        self._backstep_origin = None
        self._backstep_distance = max(0.0, float(distance_m))
        self.posture_height = None  # back out at the nominal trot posture (no held crouch/raise)
        self.phase = Phase.BACKSTEP
        self._backstep_until = event.t + float(self.cfg.get("backstep.max_duration_s", 8.0))

    def _backstep_step(self, obs: Obs) -> np.ndarray:
        """Reverse straight back (NO heading change — keep facing the goal) the VLA-COMMANDED
        distance, then hand the nav to the VLA, which routes AROUND the hazard (the feature-veto
        forbids re-entering it). Keeping the heading avoids the 180-deg turn-around that topples the
        dog. STRICT OBEDIENCE: the reverse stops at the commanded ``distance_m`` (odometry), not a
        hardcoded ``min_backout_m`` / 'clear of the patch' heuristic; ``max_duration_s`` is the only
        (safety) cap. If the commanded distance was too short to clear the patch, the next NOMINAL
        nav-pick re-decides — the VLA owns 'how far', as it owns every other recovery param."""
        if self._backstep_origin is None:
            self._backstep_origin = obs.pos.copy()
        backed = float(np.linalg.norm(obs.pos - self._backstep_origin))
        if backed >= self._backstep_distance or obs.t >= self._backstep_until:
            self._backstep_origin = None
            self._pending_event = None  # drop the stale during-backstep fire (the dog is clear now)
            self.phase = Phase.NOMINAL
            self._grace_until = obs.t + float(self.cfg.event_grace_s)
            self._grace_center = obs.pos.copy()  # the grace protects THIS backout site (#2)
            self._nav_tick(obs)  # re-plan around the freshly-marked hazard (or VLA re-pick)
            return self._slew(self._pursuit(obs))
        return self._slew(np.array([-float(self.cfg.backstep.speed_mps), 0.0, 0.0]))

    # -- reflection --------------------------------------------------------------
    def _reflect(self, event: MonitorEvent, obs: Obs) -> None:
        """Snapshot → policy → parse → compile → apply the chosen recovery (spec §5/§10)."""
        self._recovery_rounds += 1
        self._last_reflect_pos = np.asarray(event.pos, dtype=np.float64).copy()  # #1 attrib. site
        snapshot = self.recorder.capture(event, prior_outputs=list(self._prior_texts))
        if self.first_snapshot is None:
            self.first_snapshot = snapshot
        decision = self.policy.decide(snapshot, map_note=self._map_note(obs.pos, obs.heading))
        code = decision.reject_code
        if decision.ok and decision.annotation is not None:
            code = self._apply(decision.annotation, event, heading=float(obs.heading))
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

    def _apply(self, annotation: CoTAnnotation, event: MonitorEvent, heading: float) -> str:
        """Compile + execute one primitive into the navigation plan; return a status code.

        STRICT PRIMITIVE OBEDIENCE (the goal): the VLA's OWN spatial params are authoritative.
        Replan_Waypoint's pixel is back-projected through the same camera the VLA saw
        (:meth:`world_from_pixel`), NOT a hardcoded lateral; Update_Topology's region_xy/radius_m
        are taken from the model when present. The planner only supplies a fallback (the failure
        site / configured radius) when the model omits a coordinate — never an override of what the
        model emitted (see :func:`kino_vla.vla.output.to_compiler_primitive`)."""
        prim = annotation.primitive
        # Fallback ONLY when the model omits the coordinate (then the felt failure site / configured
        # radius stand in). A model-supplied value always wins (to_compiler_primitive precedence).
        model_region = read_params_xy(prim.params, "region_xy")
        region_xy = (
            model_region
            if model_region is not None
            else (
                float(event.pos[0]),
                float(event.pos[1]),
            )
        )
        model_radius = prim.params.get("radius_m")
        region_radius = (
            float(model_radius) if model_radius is not None else float(self.cfg.avoid.radius_m)
        )
        try:
            compiled_prim = to_compiler_primitive(
                prim,
                point_xy=self._waypoint_from_pixel(prim, event, heading),
                region_xy=region_xy,
                region_radius_m=region_radius,
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

        STRICT PRIMITIVE OBEDIENCE: every parameter the VLA emitted is executed, never silently
        dropped. Backstep reverses the VLA's ``distance_m`` (carried on ``cc.backstep_m``);
        Update_Topology marks the VLA's ``region_xy``/``radius_m`` (carried on ``cc.topology_op``);
        an admitted Switch_Gait/Adjust_Posture is ENACTED on the shield (``set_mode``), not merely
        speed-capped. Safety still bounds execution: the §10 iron-rule backs out of a trap the robot
        is standing in before routing around, and ``max_duration_s`` caps the reverse.

        Escape primitives (Backstep, Update_Topology) mark the failure site untraversable; push-
        through primitives (Set_Constraint, Switch_Gait, Adjust_Posture) cap the speed and continue
        straight across; on a fatal hazard that is the failing strategy — which is exactly what
        makes a wrong attribution physically distinguishable (spec §5)."""
        if isinstance(prim, Backstep):
            self._stamp_costmap(event.pos)  # mark it so the VLA's map_note routes around it
            dist = (
                float(cc.backstep_m)
                if cc is not None and cc.backstep_m is not None
                else float(prim.distance_m)
            )
            self._start_backstep(event, dist)  # reverse the VLA-commanded distance (obeyed)
        elif isinstance(prim, UpdateTopology):
            # Mark at the VLA's chosen region + radius (cc.topology_op = (x, y, r, status)).
            if cc is not None and cc.topology_op is not None:
                tx, ty, tr, _ = cc.topology_op
                self._stamp_costmap(np.array([tx, ty], dtype=np.float64), radius_m=float(tr))
            else:
                self._stamp_costmap(
                    np.array(prim.region_xy, dtype=np.float64), radius_m=prim.radius_m
                )
            if self._in_adhesive(
                event.pos
            ):  # standing on it ⇒ back out (iron-rule); the VLA then routes around
                self._start_backstep(event, float(prim.radius_m))  # clear the marked hazard extent
        elif isinstance(prim, SetConstraint):
            self.speed_cap = float(prim.max_speed)  # push through, but slowly
            # stiffness now has a REAL physical channel: a stiffer constraint lowers the stance for
            # stability (trot z_c → crawl z_c, interpolated by stiffness), tracked by the backend
            # posture controller — the dog slows AND physically crouches (#41).
            s = float(np.clip(prim.stiffness, 0.0, 1.0))
            trot_z, crawl_z = self._mode_height("trot"), self._mode_height("crawl")
            if trot_z is not None and crawl_z is not None:
                self.posture_height = trot_z - s * (trot_z - crawl_z)
            self.posture_stiffness = s
        elif isinstance(prim, SwitchGait):
            mode = self._switch_mode(cc, prim.mode)  # ENACT the admitted gait on the shield polygon
            self.speed_cap = float(self.cfg.cruise_speed_mps) * (
                0.6 if prim.mode == "crawl" else 0.8
            )
            self.posture_height = self._mode_height(mode)  # + the dog physically holds the gait z_c
        elif isinstance(prim, AdjustPosture):
            # Map height→nearest shield mode (the admitted target_mode, the safety polygon) AND
            # command the backend to physically reach body_height_m — was a NO-OP before (#40→#41).
            mode = self._switch_mode(cc, None)
            self.speed_cap = float(self.cfg.cruise_speed_mps) * (0.7 if mode == "crawl" else 0.8)
            self.posture_height = float(prim.body_height_m)  # the dog tracks the commanded height
        elif isinstance(prim, HoldAndRequest):
            self.phase = Phase.HALTED
        if cc is not None and cc.waypoint_xy is not None:  # Replan_Waypoint: the VLA's own waypoint
            self.waypoints = [np.asarray(cc.waypoint_xy, dtype=np.float64), self.goal_xy.copy()]

    def _switch_mode(self, cc: CompiledCommand | None, fallback_mode: str | None) -> str | None:
        """ENACT an admitted gait/posture switch on the SHARED shield (spec §6.7): admission alone
        leaves the shield's active mode unchanged, so without this the VLA's chosen mode never
        reaches the support polygon. Returns the mode actually set (``cc.target_mode`` when the CBF
        admitted it, else ``fallback_mode``). Honours safety: a rejected switch never calls
        ``set_mode`` (cc.accepted is False ⇒ _apply already returned the reject code)."""
        mode = cc.target_mode if cc is not None and cc.target_mode is not None else fallback_mode
        if mode is not None and self.compiler is not None:
            self.compiler.shield.set_mode(mode)
        return mode

    def _mode_height(self, mode: str | None) -> float | None:
        """The CoM/body height z_c the shield assigns to a posture mode — the ABSOLUTE target the
        backend posture controller drives the trunk to (the backend clamps to its achievable range).
        None when there is no compiler/mode (then the backend keeps the nominal trot posture)."""
        if mode is None or self.compiler is None:
            return None
        m = self.compiler.shield.modes_table().get(mode)
        return float(m.z_c) if m is not None else None

    def _stamp_costmap(self, pos: np.ndarray, radius_m: float | None = None) -> None:
        """Mark the DISCOVERED hazard on the §7 semantic costmap AND record its CLIP feature, so the
        nav-pick can HARD-FORBID landing on the same-feature surface (the user rule). Flips the
        discovered-hazard flag (before discovery the VLA cruises: step in, feel, THEN avoid).

        ``radius_m`` (when given) honours the VLA's Update_Topology.radius_m — the model's own
        hazard extent — instead of the default failure_radius_m (strict primitive obedience)."""
        self._hazard_marked = True
        pos = np.asarray(pos, dtype=np.float64)
        if self.nav_map is not None and hasattr(self.nav_map, "mark_failure"):
            if radius_m is not None:
                self.nav_map.mark_failure(pos, radius_m=float(radius_m))
            else:
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

    def _waypoint_from_pixel(
        self, prim: RecoveryPrimitive, event: MonitorEvent, heading: float
    ) -> tuple | None:
        """Back-project a recovery Replan_Waypoint's VLA-chosen PIXEL to an odometry-frame point
        through the SAME camera the VLA saw (:meth:`SnapshotRecorder.world_from_pixel`, spec §7) —
        identical to the NOMINAL nav path. STRICT OBEDIENCE: this used to discard ``point_px`` and
        route to a hardcoded 1.5 m lateral offset; now the VLA's pixel is honoured. Returns ``None``
        when the model gave no pixel or it is not ground-projectable (above the horizon), so the
        caller surfaces a structured reject (``REJECT_COMPILE``) — never a silent substitution."""
        if prim.name != "Replan_Waypoint":
            return None
        px = read_params_xy(prim.params, "point_px")
        if px is None:
            return None  # no pixel ⇒ to_compiler_primitive raises ⇒ structured reject upstream
        wp = self.recorder.world_from_pixel(
            np.asarray(event.pos, dtype=np.float64),
            float(heading),
            float(px[0]) / 1000.0,
            float(px[1]) / 1000.0,
        )
        return (float(wp[0]), float(wp[1])) if wp is not None else None

    def _pursuit(self, obs: Obs) -> np.ndarray:
        if getattr(self, "_start_xy", None) is None:
            self._start_xy = obs.pos.copy()
        startup = float(self.cfg.get("startup_straight_s", 0.0))
        if obs.t < startup and self.goal_xy is not None:
            # Pure-pursuit the START→GOAL LINE with a GENTLE gain during the push-off: this
            # corrects the Go2's intrinsic +y startup drift (topples it if uncorrected — fell at
            # y=0.37) WITHOUT the sharp turn a nav-pick can demand (which ALSO topples the marginal
            # gait). The dog tracks the straight line INTO the patch (step-in→feel→recover), not
            # around it. The nav-pick still fires; only its turn waits out this brief warmup.
            d = np.asarray(self.goal_xy, dtype=np.float64) - self._start_xy
            dn = float(np.linalg.norm(d))
            if dn > 1e-6:
                d = d / dn
                proj = self._start_xy + float(np.dot(obs.pos - self._start_xy, d)) * d
                look = proj + 1.0 * d  # a CLOSE look-ahead ON the line (strong cross-track fix)
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
