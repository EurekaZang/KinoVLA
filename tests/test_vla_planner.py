"""VLA closed-loop planner + rollout sampler on the surrogate (CI; spec §5/§11)."""

from __future__ import annotations

import pytest

from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.utils.config import load_config
from kino_vla.vla import scenarios as S
from kino_vla.vla.planner import StubVlaPolicy
from kino_vla.vla.rollout import run_vla_rollout, sample_rollouts


@pytest.fixture(scope="module")
def cfg():
    return load_config("data/hindsight.yaml")


@pytest.fixture(scope="module")
def tax(cfg):
    return FailureTaxonomy(cfg)


def test_planner_is_recovery_policy_drop_in(cfg, tax):
    """The planner implements on_event/step and runs the standard monitor→planner→shield loop."""
    r = run_vla_rollout(S.o3_collapse(), StubVlaPolicy(cfg, tax), seed=0, backend="surrogate")
    assert r.n_rounds >= 1  # the monitor fired and the planner reflected
    assert r.first_attribution == "region_collapse"  # the stub oracle attributes correctly
    assert r.first_primitive in {"Update_Topology", "Backstep", "Replan_Waypoint"}


def test_correct_vs_wrong_attribution_drive_opposite_recoveries(cfg, tax):
    """The headline §5 claim at the DECISION level: a correct vs a sibling-wrong attribution pick
    OPPOSITE recoveries (escape vs push-through). The closed-loop OUTCOME divergence (correct routes
    around + survives, wrong topples) is a VLA-route-around capability measured on the REAL Isaac
    stack (§0); the surrogate stub has no VLA nav, so a closed-loop outcome here is not meaningful
    (both fall without a route-around). O3 thin-ice: correct ⇒ Update_Topology (escape), sibling ⇒
    Set_Constraint (slow & continue across the give-way)."""
    correct = run_vla_rollout(S.o3_collapse(), StubVlaPolicy(cfg, tax), seed=0, backend="surrogate")
    wrong = run_vla_rollout(
        S.o3_collapse(), StubVlaPolicy(cfg, tax, error_mode="sibling"), seed=0, backend="surrogate"
    )
    assert correct.first_attribution == "region_collapse"
    assert correct.first_primitive == "Update_Topology"  # escape (mark untraversable + back out)
    assert wrong.first_attribution == "low_friction"  # mis-attributed to the sibling
    assert wrong.first_primitive == "Set_Constraint"  # the opposite push-through (slow & continue)


def test_invisible_collider_attributed(cfg, tax):
    """O8 invisible collider: the stub attributes it and picks an escape primitive. The route-around
    to the goal is a VLA-nav capability verified on the real stack (not the surrogate stub)."""
    r = run_vla_rollout(S.o8_invisible(), StubVlaPolicy(cfg, tax), seed=0, backend="surrogate")
    assert r.first_attribution == "invisible_obstacle"
    assert r.first_primitive in {"Update_Topology", "Backstep"}


def test_sample_rollouts_varied_decisions(cfg, tax):
    """Sampling correct vs wrong policies at one node yields DIFFERENT recovery decisions (the DPO
    preference signal). The physical-outcome divergence is the real-stack dichotomy, not the
    surrogate (which has no VLA route-around)."""
    scn = S.o3_collapse()

    def factory(temp: float):
        # temp encodes the policy here (CI surrogate has no real sampling): >0 ⇒ the wrong sibling.
        return StubVlaPolicy(cfg, tax, error_mode="sibling" if temp > 0 else None)

    results = sample_rollouts(scn, factory, temperatures=[0.0, 1.0], seed=0, backend="surrogate")
    attrs = {r.first_attribution for r in results}
    assert "region_collapse" in attrs and "low_friction" in attrs  # correct vs sibling mis-attr


def test_probe_defers_reflect_until_window_refilled(cfg, tax):
    """With an active probe (Gap-3 #34c), the planner runs the decel→accel maneuver and only
    captures the snapshot / calls the VLA AFTER the probe has refilled the 500 ms window."""
    import numpy as np

    from kino_vla.monitor.event import MonitorEvent
    from kino_vla.monitor.reflex import ActiveProbe
    from kino_vla.sim.types import Obs
    from kino_vla.vla.output import ParsedDecision
    from kino_vla.vla.planner import VlaPlanner

    decided: list = []

    class SpyPolicy:
        def decide(self, snapshot, map_note=""):
            decided.append(snapshot)
            return ParsedDecision(ok=False, raw_text="", reject_code="spy")

    class FakeRecorder:
        def buffer(self, obs):
            pass

        def capture(self, event, prior_outputs):
            return "snapshot"

    probe = ActiveProbe(decel_s=0.1, accel_s=0.1, settle_window_ms=0.0)  # total_s = 0.2 s
    planner = VlaPlanner(
        cfg=load_config("recovery/fsm_isaac.yaml"),
        policy=SpyPolicy(),
        goal_xy=np.array([5.0, 0.0]),
        dt=0.02,
        recorder=FakeRecorder(),
        compiler=None,
        probe=probe,
    )

    def _obs(t):
        return Obs(
            t=t,
            pos=np.array([1.0, 0.0]),
            heading=0.0,
            vel_body=np.zeros(2),
            yaw_rate=0.0,
            cmd_prev=np.zeros(3),
            slip_ratio=0.0,
            base_height=0.32,
            tilt=0.0,
            fallen=False,
        )

    ev = MonitorEvent(
        t=1.0, pos=np.array([1.0, 0.0]), channel="slip_ratio", value=0.5, threshold=0.4, summary="x"
    )
    assert planner.on_event(ev)
    planner.step(_obs(1.02))  # decel phase
    assert decided == [], "must NOT reflect during the probe's decel phase"
    planner.step(_obs(1.12))  # accel phase
    assert decided == [], "must NOT reflect during the probe's accel phase"
    planner.step(_obs(1.24))  # probe done (elapsed 0.24 > 0.2) ⇒ reflect on the refilled window
    assert len(decided) == 1, "reflect exactly once, after the probe refills the window"


# --------------------------------------------------------------------------------------------
# STRICT PRIMITIVE OBEDIENCE (the goal): the low-level executes EXACTLY the VLA's emitted params —
# distance_m, point_px, region_xy/radius_m, and the admitted gait — never a hardcoded substitute.
# --------------------------------------------------------------------------------------------
import numpy as np  # noqa: E402

from kino_vla.data.schema import CoTAnnotation, RecoveryPrimitive  # noqa: E402
from kino_vla.monitor.event import MonitorEvent  # noqa: E402
from kino_vla.shield.cbf_shield import CbfShield  # noqa: E402
from kino_vla.shield.primitive_compiler import PrimitiveCompiler  # noqa: E402
from kino_vla.sim.types import Obs  # noqa: E402
from kino_vla.vla.planner import Phase, VlaPlanner  # noqa: E402


class _NoNavPolicy:
    """A VlaPolicy with no decide_nav (so NOMINAL nav ticks are skipped in unit tests)."""

    def decide(self, snapshot, map_note=""):  # pragma: no cover - not exercised here
        return None


class _RecorderSpy:
    """Records world_from_pixel calls and returns a fixed odometry point."""

    def __init__(self, ret):
        self.ret = ret
        self.calls = []

    def buffer(self, obs):
        pass

    def capture(self, event, prior_outputs=None):  # pragma: no cover
        return "snapshot"

    def world_from_pixel(self, pose_xy, heading, u_frac, v_frac):
        self.calls.append((float(heading), float(u_frac), float(v_frac)))
        return None if self.ret is None else np.asarray(self.ret, dtype=np.float64)


class _NavMapSpy:
    """Captures mark_failure(pos, radius_m) — and deliberately exposes NO scene/feature API."""

    def __init__(self):
        self.calls = []

    def mark_failure(self, world_xy, embedding=None, radius_m=None):
        self.calls.append((np.asarray(world_xy, dtype=np.float64), radius_m))
        return {"stamped": 1, "propagated": 0}


def _event(t=1.0, pos=(1.0, 0.0)):
    return MonitorEvent(
        t=t,
        pos=np.array(pos, dtype=np.float64),
        channel="slip_ratio",
        value=0.5,
        threshold=0.4,
        summary="x",
    )


def _make_planner(*, compiler=None, recorder=None, nav_map=None):
    planner = VlaPlanner(
        cfg=load_config("recovery/fsm_isaac.yaml"),
        policy=_NoNavPolicy(),
        goal_xy=np.array([6.0, 0.0]),
        dt=0.02,
        recorder=recorder if recorder is not None else _RecorderSpy(None),
        compiler=compiler,
    )
    planner.nav_map = nav_map
    return planner


def _obs_at(t, x):
    return Obs(
        t=t,
        pos=np.array([x, 0.0], dtype=np.float64),
        heading=0.0,
        vel_body=np.zeros(2),
        yaw_rate=0.0,
        cmd_prev=np.zeros(3),
        slip_ratio=0.0,
        base_height=0.32,
        tilt=0.0,
        fallen=False,
    )


def test_backstep_obeys_commanded_distance_not_min_backout():
    """Backstep reverses the VLA's distance_m (odometry), NOT the old hardcoded min_backout_m=1.0.

    Two decisive sub-cases at distance < and > the old 1.0 m default prove the param is obeyed:
    a 0.5 m command exits at ~0.5 m (so it is NOT clamped up to 1.0); a 1.5 m command is STILL
    backing at 1.0 m (so it is NOT clamped down to 1.0)."""
    # distance 0.5 < old min_backout 1.0 ⇒ must exit at ~0.5, proving obedience (not min_backout).
    planner = _make_planner()
    planner._start_backstep(_event(t=1.0), distance_m=0.5)
    assert planner.phase is Phase.BACKSTEP
    planner._backstep_step(_obs_at(1.0, 1.0))  # origin set at x=1.0
    planner._backstep_step(_obs_at(1.2, 0.65))  # backed 0.35 < 0.5 ⇒ still reversing
    assert planner.phase is Phase.BACKSTEP
    planner._backstep_step(_obs_at(1.4, 0.45))  # backed 0.55 ≥ 0.5 ⇒ obeyed, exit
    assert planner.phase is Phase.NOMINAL

    # distance 1.5 > old min_backout 1.0 ⇒ must STILL be reversing at 1.0 (not clamped down).
    planner2 = _make_planner()
    planner2._start_backstep(_event(t=1.0), distance_m=1.5)
    planner2._backstep_step(_obs_at(1.0, 1.0))
    planner2._backstep_step(_obs_at(2.0, 0.0))  # backed 1.0 < 1.5 ⇒ still reversing (obeyed)
    assert planner2.phase is Phase.BACKSTEP


def test_recovery_replan_waypoint_backprojects_the_vla_pixel():
    """A recovery Replan_Waypoint is back-projected through world_from_pixel from the VLA's
    point_px — NOT routed to the old hardcoded 1.5 m lateral offset."""
    recorder = _RecorderSpy(ret=[4.2, -3.1])
    planner = _make_planner(recorder=recorder)
    prim = RecoveryPrimitive("Replan_Waypoint", {"point_px": [700, 200]})
    wp = planner._waypoint_from_pixel(prim, _event(pos=(1.0, 0.0)), heading=0.3)
    assert recorder.calls == [(0.3, 0.7, 0.2)], "the VLA's pixel (÷1000) is back-projected, obeyed"
    assert wp == (4.2, -3.1), "the waypoint is the back-projected point, not a lateral stub"
    # An unprojectable pixel returns None ⇒ surfaced as a structured reject upstream, not faked.
    planner_none = _make_planner(recorder=_RecorderSpy(ret=None))
    assert planner_none._waypoint_from_pixel(prim, _event(), heading=0.0) is None


# --------------------------------------------------------------------------------------------
# IN-PLACE TURN (user directive): the Turn primitive is a PURE yaw rotation (commanded yaw velocity,
# vx=vy=0) the dog physically executes — NOT a "turn then walk a waypoint", and it is NEVER feature-
# vetoed (an in-place rotation cannot move the CoM onto the hazard; it only re-aims the camera).
# --------------------------------------------------------------------------------------------
import math  # noqa: E402


def _obs_th(t, heading, x=0.0, y=0.0):
    return Obs(
        t=t,
        pos=np.array([x, y], dtype=np.float64),
        heading=float(heading),
        vel_body=np.zeros(2),
        yaw_rate=0.0,
        cmd_prev=np.zeros(3),
        slip_ratio=0.0,
        base_height=0.32,
        tilt=0.0,
        fallen=False,
    )


def test_turn_is_pure_in_place_rotation():
    """A Turn commands ONLY a yaw velocity (vx=vy=0) until the heading has rotated by the commanded
    angle, then hands back to NOMINAL — it never projects/drives a waypoint (the old bug)."""
    planner = _make_planner()  # _NoNavPolicy ⇒ completion does not re-pick (clean mechanics test)
    obs = _obs_th(0.0, 0.0)
    planner._start_turn(obs, 90.0)
    assert planner.phase is Phase.TURNING
    assert abs(planner._turn_target_heading - math.pi / 2) < 1e-6

    heading, t, saw = 0.0, 0.0, False
    for _ in range(1000):
        obs = _obs_th(t, heading)
        cmd = planner._turn_step(obs)
        if planner.phase is not Phase.TURNING:
            break
        saw = True
        assert abs(cmd[0]) < 1e-9 and abs(cmd[1]) < 1e-9, "Turn must NOT translate (pure yaw)"
        assert cmd[2] > 0.0, "rotating toward +90° ⇒ positive yaw rate"
        nh = heading + float(cmd[2]) * 0.02  # mock the backend integrating the yaw command
        heading = math.atan2(math.sin(nh), math.cos(nh))
        t += 0.02
    assert saw, "the planner issued at least one in-place rotation command"
    assert planner.phase is Phase.NOMINAL, "the turn completed and handed back to NOMINAL"
    assert abs(heading - math.pi / 2) < 0.15, "rotated to ≈ the commanded +90° in place"


def test_turn_is_never_feature_vetoed():
    """The Turn must commit even where ALL ground is feature-forbidden — the in-place rotation is
    NOT subject to the waypoint feature-veto (the fix for the tether-freeze: every forward pixel was
    forbidden, so the OLD projected-waypoint Turn was rejected and the robot froze)."""
    from kino_vla.vla.output import ParsedDecision

    class _TurnNav:
        def decide(self, snapshot, map_note=""):  # pragma: no cover - recovery path unused
            return ParsedDecision(ok=False, raw_text="", reject_code="x")

        def decide_nav(self, snapshot, goal_bearing_deg, map_note=""):
            return ParsedDecision(ok=True, raw_text="", nav_turn_deg=30.0)

    class _AllForbiddenMap:
        sim_threshold = 0.8
        scene: list = []

        def feature_at(self, wp):
            return np.ones(8, dtype=np.float64)  # every cell matches the forbidden feature

        def region_feature(self, c):
            return np.ones(8, dtype=np.float64)

    planner = VlaPlanner(
        cfg=load_config("recovery/fsm_isaac.yaml"),
        policy=_TurnNav(),
        goal_xy=np.array([6.0, 0.0]),
        dt=0.02,
        recorder=_RecorderSpy(None),
        compiler=None,
    )
    planner.nav_map = _AllForbiddenMap()
    planner._forbidden_features = [np.ones(8, dtype=np.float64)]
    planner._hazard_marked = True  # past discovery ⇒ the all-forbidden veto is live
    planner._reflect_nav(_obs_th(0.0, 0.0))
    assert planner.phase is Phase.TURNING, "an in-place Turn commits even on all-forbidden ground"


def test_nav_trace_sink_is_a_read_only_tap():
    """The DAgger collector's nav_trace_sink (#43) observes each NOMINAL nav decision + verdict but
    NEVER changes it: with the sink installed the planner still commits the same Turn, and the sink
    receives exactly one record with the visited-state keys. None ⇒ no-op (deployed loop)."""
    import types

    from kino_vla.vla.output import ParsedDecision

    class _TurnNav:
        def decide(self, snapshot, map_note=""):  # pragma: no cover - recovery path unused
            return ParsedDecision(ok=False, raw_text="", reject_code="x")

        def decide_nav(self, snapshot, goal_bearing_deg, map_note=""):
            return ParsedDecision(ok=True, raw_text="", nav_turn_deg=30.0)

    class _Rec:
        def buffer(self, obs):
            pass

        def capture(self, event, prior_outputs=None):
            return types.SimpleNamespace(
                rgb=np.zeros((1, 4, 4, 3), dtype=np.float32),
                proprio_window=np.zeros((4, 11), dtype=np.float32),
            )

        def world_from_pixel(self, *a):  # pragma: no cover - Turn path needs no projection
            return None

    def _mk():
        return VlaPlanner(
            cfg=load_config("recovery/fsm_isaac.yaml"),
            policy=_TurnNav(),
            goal_xy=np.array([6.0, 0.0]),
            dt=0.02,
            recorder=_Rec(),
            compiler=None,
        )

    no_sink = _mk()
    no_sink._reflect_nav(_obs_th(0.0, 0.0))
    assert no_sink.phase is Phase.TURNING  # baseline decision

    captured: list = []
    with_sink = _mk()
    with_sink.nav_trace_sink = lambda rec: captured.append(rec)
    with_sink._reflect_nav(_obs_th(0.0, 0.0))
    assert with_sink.phase is Phase.TURNING, "the tap does not alter the decision (read-only)"
    assert len(captured) == 1, "fires exactly once per nav tick"
    rec = captured[0]
    for k in (
        "t",
        "pose_xy",
        "heading",
        "goal_bearing_deg",
        "map_note",
        "prior_outputs",
        "rgb",
        "proprio_window",
        "tag",
        "committed",
    ):
        assert k in rec, f"missing visited-state key {k!r}"
    assert rec["committed"] is True and str(rec["tag"]).startswith("turn")


def test_update_topology_obeys_region_xy_and_radius():
    """Update_Topology marks the costmap at the VLA's OWN region_xy + radius_m — not the monitor
    event position and not the configured avoid.radius_m."""
    shield = CbfShield(load_config("shield/cbf_v0.yaml"))
    spy = _NavMapSpy()
    planner = _make_planner(compiler=PrimitiveCompiler(shield), nav_map=spy)
    ann = CoTAnnotation(
        thought="mark it",
        attribution="region_collapse",
        primitive=RecoveryPrimitive(
            "Update_Topology",
            {"region_xy": [3.0, -1.0], "radius_m": 0.8, "status": "untraversable"},
        ),
        attribution_raw="region_collapse",
        raw_text="",
    )
    code = planner._apply(ann, _event(pos=(1.0, 0.0)), heading=0.0)
    assert code.startswith("OK")
    assert len(spy.calls) == 1
    pos, radius = spy.calls[0]
    assert np.allclose(pos, [3.0, -1.0]), "marked at the VLA's region_xy, not the event pos (1,0)"
    assert radius == pytest.approx(0.8), "marked with the VLA's radius_m, not cfg.avoid.radius_m"


def test_admitted_switch_gait_enacts_mode_on_the_shield():
    """An admitted Switch_Gait actually switches the shared shield's active mode (set_mode), so the
    VLA's chosen gait reaches the support polygon — not merely a speed cap."""
    shield = CbfShield(load_config("shield/cbf_v0.yaml"))
    assert shield.mode == "trot"  # default
    planner = _make_planner(compiler=PrimitiveCompiler(shield))
    ann = CoTAnnotation(
        thought="switch gait",
        attribution="effort_decay",
        primitive=RecoveryPrimitive("Switch_Gait", {"mode": "crawl"}),
        attribution_raw="effort_decay",
        raw_text="",
    )
    code = planner._apply(ann, _event(), heading=0.0)
    assert code.startswith("ADMIT"), f"a standing-obs switch should admit, got {code!r}"
    assert shield.mode == "crawl", "the admitted gait is enacted on the shield"
    assert planner.speed_cap is not None and planner.speed_cap > 0.0


# --------------------------------------------------------------------------------------------
# EVERY PRIMITIVE GROUNDS TO A LOW-LEVEL RESPONSE (the goal): the posture/gait primitives emit a
# commanded BODY HEIGHT that the backend physically tracks — not just a speed cap or a shield mode.
# --------------------------------------------------------------------------------------------
def _posture_ann(primitive: str, params: dict) -> CoTAnnotation:
    return CoTAnnotation(
        thought="posture",
        attribution="effort_decay",
        primitive=RecoveryPrimitive(primitive, params),
        attribution_raw="effort_decay",
        raw_text="",
    )


def test_each_posture_primitive_emits_a_commanded_body_height():
    """Switch_Gait/Adjust_Posture/Set_Constraint each set planner.posture_height to a concrete
    target the backend tracks — the real low-level response (#41). Backstep clears it (nominal)."""
    shield = CbfShield(load_config("shield/cbf_v0.yaml"))
    modes = shield.modes_table()
    trot_z, crawl_z = modes["trot"].z_c, modes["crawl"].z_c

    p = _make_planner(compiler=PrimitiveCompiler(shield))
    p._apply(_posture_ann("Switch_Gait", {"mode": "crawl"}), _event(), heading=0.0)
    assert p.posture_height == pytest.approx(crawl_z), "Switch_Gait holds the gait's z_c"

    p = _make_planner(compiler=PrimitiveCompiler(shield))
    adj = _posture_ann("Adjust_Posture", {"body_height_m": 0.25, "pitch_deg": 0.0})
    p._apply(adj, _event(), heading=0.0)
    assert p.posture_height == pytest.approx(0.25), "Adjust_Posture tracks the commanded height"

    # Set_Constraint: stiffness now has a real channel (trot→crawl crouch scaled by stiffness).
    p = _make_planner(compiler=PrimitiveCompiler(shield))
    p._apply(_posture_ann("Set_Constraint", {"max_speed": 0.4, "stiffness": 1.0}), _event(), 0.0)
    assert p.posture_height == pytest.approx(crawl_z), "stiffness=1 ⇒ full crouch to crawl z_c"
    p = _make_planner(compiler=PrimitiveCompiler(shield))
    p._apply(_posture_ann("Set_Constraint", {"max_speed": 0.4, "stiffness": 0.0}), _event(), 0.0)
    assert p.posture_height == pytest.approx(trot_z), "stiffness=0 ⇒ nominal trot height"

    # Backstep resets posture to nominal (the escape backs out at the nominal trot).
    p = _make_planner(compiler=PrimitiveCompiler(shield))
    p.posture_height = crawl_z
    p._apply(_posture_ann("Backstep", {"distance_m": 1.0}), _event(), 0.0)
    assert p.posture_height is None, "Backstep releases any held posture"


def test_surrogate_backend_physically_tracks_commanded_height():
    """The backend executes a commanded posture: base_height converges to the target and releases
    back to nominal — the dog-executed body-height response (Isaac mirror verified on GPU, #41)."""
    from kino_vla.sim.surrogate import SurrogateBackend

    cfg = load_config("sim/surrogate.yaml")
    backend = SurrogateBackend(cfg, np.array([0.0, 0.0]), 0.0)
    backend.reset(0)
    nominal = float(cfg.base_height_m)
    backend.set_posture(0.22, stiffness=1.0)
    for _ in range(40):
        obs = backend.step(np.zeros(3))
    assert obs.base_height == pytest.approx(0.22, abs=0.01), "tracks the commanded crouch height"
    backend.set_posture(None)
    for _ in range(40):
        obs = backend.step(np.zeros(3))
    assert obs.base_height == pytest.approx(nominal, abs=0.01), "releases back to nominal"


def test_loop_forwards_planner_posture_to_backend():
    """The episode loop forwards the planner's commanded posture to backend.set_posture every step
    (the wiring that makes a posture primitive reach the low-level controller, #41)."""
    from kino_vla.loop import run_episode
    from kino_vla.shield.passthrough import PassThroughShield
    from kino_vla.sim.operators import OperatorStack
    from tests._monitor_stub import StubMonitor

    seen: list = []

    def _benign_obs(t):
        return Obs(
            t=t,
            pos=np.array([0.1 * t, 0.0]),
            heading=0.0,
            vel_body=np.zeros(2),
            yaw_rate=0.0,
            cmd_prev=np.zeros(3),
            slip_ratio=0.0,
            base_height=0.31,
            tilt=0.0,
            fallen=False,
        )

    class FakeBackend:
        dt = 0.02
        mass_kg = 15.0

        def __init__(self):
            self._t = 0.0

        def reset(self, seed):
            self._t = 0.0
            return _benign_obs(0.0)

        def step(self, cmd):
            self._t += self.dt
            return _benign_obs(self._t)

        def set_posture(self, height_m, stiffness=1.0):
            seen.append(height_m)

    class PosturePolicy:
        posture_height = 0.22
        posture_stiffness = 1.0

        def on_event(self, event):
            return False

        def step(self, obs):
            return np.zeros(3)

    run_episode(
        FakeBackend(),
        OperatorStack([]),
        StubMonitor(),
        PosturePolicy(),
        PassThroughShield(),
        seed=0,
        goal_xy=np.array([100.0, 0.0]),  # far ⇒ never "reached"; run several steps
        goal_tol_m=0.5,
        max_time_s=0.1,  # ~5 steps at dt=0.02
    )
    assert seen, "the loop must forward posture to the backend"
    assert all(h == 0.22 for h in seen), "the planner's commanded height reaches the backend"


# --------------------------------------------------------------------------------------------
# #47 MULTI-PATCH / LONG-DISTANCE READINESS: max_rounds bounds re-attribution AT ONE SPOT (a dead
# loop), reset by PROGRESS; the post-maneuver grace suppresses SAME-spot re-fires but lets a NEW,
# distant patch through; the probe re-arms at each new hazard.
# --------------------------------------------------------------------------------------------
def test_recovery_rounds_cap_is_per_spot_not_lifetime():
    """A long / multi-patch course is never capped at max_rounds total hazards: max_rounds bounds
    re-attribution AT ONE SPOT (a real dead loop), but a fire FAR from the last attribution (a new
    patch) RESETS the counter."""
    p = _make_planner()
    p._recovery_rounds = p.max_rounds  # exhausted at the current spot
    p._last_reflect_pos = np.array([1.0, 0.0])
    assert not p.on_event(_event(t=10.0, pos=(1.0, 0.0))), "same spot stays capped (dead loop)"
    assert p.on_event(_event(t=11.0, pos=(6.0, 0.0))), "a new, distant patch resets the cap"
    assert p._recovery_rounds == 0


def test_grace_suppresses_same_spot_but_not_a_new_distant_patch():
    """The post-maneuver grace suppresses a SAME-spot re-fire (mid-maneuver), but a fire at a NEW,
    distant patch within the same window is a different hazard and is NOT suppressed (dense/long
    multi-patch)."""
    p = _make_planner()
    p._grace_until = 100.0
    p._grace_center = np.array([1.0, 0.0])
    assert not p.on_event(_event(t=5.0, pos=(1.2, 0.0))), "near the grace centre ⇒ suppressed"
    assert p.on_event(_event(t=5.0, pos=(6.0, 0.0))), "a new, distant patch ⇒ not suppressed"


def test_probe_rearms_at_a_new_hazard():
    """The active-sensing probe re-arms at each NEW (distant) hazard so its attribution window is
    in-distribution — not once-per-episode."""
    from kino_vla.monitor.reflex import ActiveProbe

    p = _make_planner()
    p.probe = ActiveProbe(decel_s=0.1, accel_s=0.1, settle_window_ms=0.0)
    assert p.on_event(_event(t=1.0, pos=(1.0, 0.0))) and p._probing and p._probed
    p._probing = False  # the first probe finishes
    p._last_reflect_pos = np.array([1.0, 0.0])
    p._grace_until = 0.0
    assert p.on_event(_event(t=5.0, pos=(6.0, 0.0))) and p._probing, "re-armed at the new hazard"
