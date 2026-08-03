"""Real-Go2 (Isaac) Hindsight-CoT rollout: drive into one operator's failure + snapshot it.

Shared by ``scripts/isaac_hindsight_collect.py`` (build the dataset over real-Go2 snapshots) and
``scripts/isaac_hindsight_check.py`` (the GPU gate). The spec builds the data pipeline on Isaac
(§1/§8.1/§10), so the snapshot's proprioception + privileged θ come from the physically-simulated
Go2 — not the surrogate point-robot. The ``backend`` is injected (an ``IsaacPolicyBackend``); this
module imports no Isaac itself, so it stays CPU-importable.

Each operator runs in its own lateral lane (Isaac is one-episode/process, #21a), driven straight
across with the M4 bang-bang excitation. Region operators (O1/O2/O3/O4/O7) are intercepted
in-region by the high-recall collection monitor. The embodiment operators (O5/O10) have no spatial
locus, so they run a dedicated capture that STRADDLES the fault onset (the #31 fix): a healthy
baseline fills the window, the fault is installed, and the snapshot is taken a fixed delay later so
the proprio trace shows the step that names the cause — O10's effort spikes into its derated cap
and the trunk SAGS (a give-way leaves the motors UNLOADED, so effort activity + sag = actuators,
not collapse); O5's heavier-but-healthy trunk is the overload sibling. ``clear_payload`` keeps each
lane's payload absolute (reset does not strip it, #22), so O5 composes in any shard.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from kino_vla.data.schema import Snapshot
from kino_vla.data.snapshot import SnapshotRecorder
from kino_vla.map.types import SemanticRegion
from kino_vla.monitor.event import MonitorEvent
from kino_vla.monitor.learned_monitor import load_deployed_monitor
from kino_vla.monitor.reflex import ActiveProbe
from kino_vla.sim.types import CollapseRegion, FrictionRegion, Obs, ResistanceRegion
from kino_vla.utils.config import Config
from kino_vla.utils.geometry import Rect

REGION_OPS: frozenset[str] = frozenset(
    {"O1_mu_field", "O7_visual_remap", "O3_collapse", "O2_compliance", "O4_tether"}
)


def random_lanes(cfg: Config, seed: int, n_lanes: int) -> list[dict]:
    """Generate ``n_lanes`` randomized-θ lanes over the OBSERVABLE operators (scale collection).

    Samples each lane's operator by weight and its θ uniformly from the configured range
    (configs/data/hindsight.yaml isaac_collect.random), and spreads the lanes in y (alternating
    ±) so they don't overlap. O5 is excluded (effort unobservable, the #15 floor)."""
    rc = cfg.isaac_collect.random
    specs = list(rc.ops)
    weights = np.array([float(s.get("weight", 1)) for s in specs], dtype=np.float64)
    weights /= weights.sum()
    spacing = float(rc.y_spacing_m)
    rng = np.random.default_rng(int(seed))
    lanes: list[dict] = []
    for i in range(int(n_lanes)):
        spec = specs[int(rng.choice(len(specs), p=weights))]
        lane: dict = {
            "op": str(spec["op"]),
            "appearance": str(spec["appearance"]),
            "y": spacing * ((i // 2) + 1) * (1.0 if i % 2 == 0 else -1.0),
        }
        for k, v in spec.items():
            if k not in ("op", "weight", "appearance"):
                lane[k] = float(rng.uniform(float(v[0]), float(v[1])))
        if lane["op"] == "O3_collapse":
            lane["mu_intact"] = 0.8
        lanes.append(lane)
    return lanes


def theta_for(lane: dict) -> dict[str, float]:
    """The operator's θ for the taxonomy A/B refinement (boundary operators only)."""
    op = lane["op"]
    if op == "O2_compliance":
        return {"d_sink": float(lane["d_sink"])}
    if op == "O5_payload":
        return {"mass_kg": float(lane["mass"])}
    if op == "O10_effort_decay":
        return {"floor": float(lane["floor"])}
    return {}


def apply_operator(backend: Any, lane: dict, rect: Rect) -> None:
    """Install a region operator on the real Go2 via its PhysX hooks (spec §8.2)."""
    op = lane["op"]
    if op in ("O1_mu_field", "O7_visual_remap"):
        mu = float(lane["mu"])
        backend.add_friction_regions([FrictionRegion(rect, mu, mu)])
    elif op == "O3_collapse":
        backend.add_collapse_regions(
            [
                CollapseRegion(
                    rect=rect,
                    mu_intact=float(lane.get("mu_intact", 0.8)),
                    mu_collapsed=float(lane["mu_collapsed"]),
                    trigger_dwell_s=float(lane["dwell"]),
                )
            ]
        )
    elif op == "O2_compliance":
        backend.add_resistance_regions(
            [
                ResistanceRegion(
                    rect,
                    float(lane["k"]),
                    float(lane["c"]),
                    sink_depth_m=float(lane["d_sink"]),
                    kind="compliance",
                )
            ]
        )
    elif op == "O4_tether":
        backend.add_resistance_regions(
            [
                ResistanceRegion(
                    rect,
                    float(lane["k"]),
                    float(lane["c"]),
                    break_force_n=float(lane["f_break"]),
                    kind="tether",
                )
            ]
        )


def _onset_event(obs: Obs, op: str, channel: str) -> MonitorEvent:
    """A capture trigger for an embodiment op, labelled with its HONEST observable channel.

    O10 fires the effort channel (sustained actuator saturation); O5 fires the tracking channel
    (a velocity deficit). Neither leaks privileged θ — both are Sport-Client-visible signals; the
    trunk-height sag that names O5 'overload' is read from the proprio window, not from here.
    """
    value = (
        float(obs.effort_ratio)
        if op == "O10_effort_decay"
        else float(np.linalg.norm(obs.cmd_prev[:2] - obs.vel_body))
    )
    return MonitorEvent(
        t=obs.t, pos=obs.pos.copy(), channel=channel, value=value, threshold=0.0, summary="onset"
    )


def _collect_probe(cc: Config) -> ActiveProbe | None:
    """The active-sensing probe for data collection (Gap-3 #34c, Path B), or None.

    Loaded from the SAME deploy config (``recovery/fsm_isaac.yaml``) the planner uses, so the
    training window is filled by the identical decel→accel maneuver as the deployment window —
    the two are IID by construction (the standardizer then normalizes both into the same band).
    Gated on ``isaac_collect.use_probe`` so the legacy bang-bang collection stays reproducible.
    """
    if not bool(cc.get("use_probe", False)):
        return None
    from kino_vla.utils.config import load_config

    return ActiveProbe.from_config(load_config("recovery/fsm_isaac.yaml"))


def _probe_capture(
    backend: Any, recorder: SnapshotRecorder, probe: ActiveProbe, obs: Obs, event: MonitorEvent
) -> Obs:
    """At the failure locus, run the probe maneuver (buffering the response) then capture.

    Mirrors the deployment planner (kino_vla/vla/planner.py): the monitor fires, the probe runs
    decel→accel for ~one window, and the snapshot is taken on the now probe-dominated (and so
    in-distribution) window. ``event`` carries the original onset metadata (channel/time)."""
    probe.start(float(obs.t))
    while not bool(obs.fallen):
        cmd = probe.command(obs)
        if cmd is None:
            break
        recorder.observe(obs, None)  # buffer the probe response; do not capture yet
        obs = backend.step(cmd)
    recorder.observe(obs, event)  # the window is the probe transient ⇒ capture the snapshot
    return obs


def _collect_embodiment_lane(
    backend: Any,
    cc: Config,
    lane: dict,
    recorder: SnapshotRecorder,
    obs: Obs,
    speed_fn: Callable[[int], float],
    rng: np.random.Generator,
    probe: ActiveProbe | None = None,
) -> tuple[Snapshot | None, dict[str, float], bool]:
    """O5 (overload) / O10 (effort-decay): no spatial locus, no slip/visual tell — the #31 fix.

    Fill a HEALTHY baseline window, install the embodiment fault, then capture so the window
    STRADDLES the onset and the proprio trace shows the step that names the cause. BOTH failures
    CROUCH the trunk (a base_height sag), so the sag alone is the ambiguity; they split on the feet
    and effort: O10's derated actuator hits its cap (the effort trace SPIKES) and the feet SLIP, so
    we wait for the first effort bind; O5's healthy motors never cap (effort ~0) and the feet keep
    GRIP (low slip), so we capture a fixed delay after the load onset. (The base_height sag is
    exposed to the Oracle in oracle._proprio_summary; before that, every O10 read as region_collapse
    — slip with the sag invisible — and was filtered out, the gap this closes.)
    """
    op = lane["op"]
    baseline = int(cc.get("embodiment_baseline_steps", 13))
    delay = int(cc.get("effort_capture_delay_steps", 12))
    effort_min = float(cc.get("effort_capture_min", 0.25))
    channel = "effort_ratio" if op == "O10_effort_decay" else "tracking_err"
    # 1) healthy baseline so the window holds the nominal trunk height / un-saturated effort
    #    (the pre-onset context the Oracle reads as the step the fault introduces).
    for k in range(baseline):
        if obs.fallen:
            break
        recorder.observe(obs, None)
        jit = 0.03 * rng.standard_normal(2)
        obs = backend.step(np.array([speed_fn(k), jit[0], jit[1]]))
    # 2) install the embodiment fault AFTER the healthy baseline.
    if op == "O10_effort_decay":
        backend.set_effort_scale(float(lane["floor"]))
    else:
        backend.add_payload(float(lane["mass"]), np.zeros(2))
    # 3) drive on, then capture so the window straddles the developed failure. O10's effort SPIKES
    #    when its derated cap is hit (it needs a few accel/decel cycles), so wait for the first bind
    #    then +delay to land on the saturation. O5's healthy motors never bind effort, so capture a
    #    fixed delay after the load onset to land on the developed sag (the overload tell).
    countdown = delay if op == "O5_payload" else -1  # -1 = wait for the effort channel to bind
    for k in range(int(cc.n_steps)):
        if obs.fallen:
            break
        if countdown < 0 and float(obs.effort_ratio) >= effort_min:
            countdown = delay
        if countdown == 0:  # the onset: capture (under the probe maneuver when enabled)
            event = _onset_event(obs, op, channel)
            if probe is not None:
                obs = _probe_capture(backend, recorder, probe, obs, event)
            else:
                recorder.observe(obs, event)
            break
        if countdown > 0:
            countdown -= 1
        recorder.observe(obs, None)
        if recorder.snapshot is not None:
            break
        jit = 0.03 * rng.standard_normal(2)
        obs = backend.step(np.array([speed_fn(baseline + k), jit[0], jit[1]]))
    # Effort never bound (O5 below the strain floor) or it fell first — still snapshot the last
    # state so the lane is represented (a truly un-straining O5 will be filter-dropped, honestly).
    if recorder.snapshot is None:
        event = _onset_event(obs, op, channel)
        if probe is not None:
            _probe_capture(backend, recorder, probe, obs, event)
        else:
            recorder.observe(obs, event)
    if op == "O10_effort_decay":
        backend.set_effort_scale(1.0)  # restore before the next lane
    return recorder.snapshot, theta_for(lane), bool(obs.fallen)


def collect_lane(
    backend: Any, cfg: Config, monitor_cfg: Config, lane: dict, seed: int,
    monitor: object | None = None,
) -> tuple[Snapshot | None, dict[str, float], bool]:
    """Drive one real-Go2 lane into its failure; return ``(snapshot_or_None, op_theta, fell)``."""
    cc = cfg.isaac_collect
    op, y, appr = lane["op"], float(lane["y"]), str(lane["appearance"])
    dt = backend.dt
    rect = Rect(float(cc.patch_cx), y, float(cc.patch_hx), float(cc.patch_hy))
    if op in REGION_OPS:
        apply_operator(backend, lane, rect)
    backend._start_pos = np.array([0.0, y])
    backend._start_heading = 0.0
    obs = backend.reset(seed)
    # Start each lane from the nominal robot: reset does NOT strip a prior lane's payload (#22) or
    # restore a derated actuator, so clear both — else add_payload (+=) compounds across lanes.
    backend.clear_payload()
    backend.set_effort_scale(1.0)

    period, duty, lo = float(cc.speed_period_s), float(cc.speed_duty), float(cc.speed_lo)
    # Per-op bang-bang demand (the embodiment failures surface under different excitation). O10
    # (decay) takes the FAST demand so the accel/decel drives its DERATED actuator into the cap
    # (effort spikes) and stumbles the gait (the trunk sags + feet SLIP). O5 (overload) takes a
    # gentler demand with a HEAVY load: the healthy motors never cap (effort≈0), but the load
    # CROUCHES the trunk (a sag) and bogs the speed without the fast-accel topple, and the feet
    # keep their GRIP (low slip). So both crouch — the ambiguity — split by effort+slip. Region ops
    # keep the standard bang-bang. (A give-way leaves the motors UNLOADED — effort≈0, slip STEP.)
    if op == "O5_payload":
        hi = float(cc.o5_speed)
    elif op == "O10_effort_decay":
        hi = float(cc.effort_speed_hi)
    else:
        hi = float(cc.speed_hi)

    def speed_fn(k: int) -> float:
        return hi if (((k * dt) / period) % 1.0) < duty else lo

    for _ in range(int(cc.settle_steps)):
        obs = backend.step(np.array([speed_fn(0), 0.0, 0.0]))

    scene = (
        [SemanticRegion(Rect(rect.cx, rect.cy, rect.hx, rect.hy), appr)] if op in REGION_OPS else []
    )
    # Region ops gate to the patch (snapshot the in-region slip); embodiment ops (O5/O10) have no
    # spatial locus, so they take no gate and are handled by _collect_embodiment_lane.
    gate = rect if op in REGION_OPS else None
    recorder = SnapshotRecorder(
        cfg,
        scene=scene,
        operator_name=op,
        appearance_class=appr,
        privileged_fn=backend.privileged_physics,
        gate_rect=gate,
    )
    rng = np.random.default_rng(seed + 7)
    probe = _collect_probe(cc)  # Path B: capture under the same maneuver the planner deploys (#34c)
    if op in ("O10_effort_decay", "O5_payload"):
        return _collect_embodiment_lane(
            backend, cc, lane, recorder, obs, speed_fn, rng, probe=probe
        )

    monitor = monitor if monitor is not None else load_deployed_monitor(dt, threshold=0.3)
    monitor.reset()
    for k in range(int(cc.n_steps)):
        if obs.fallen:
            break
        event = monitor.step(obs)
        # At the in-region failure locus, run the probe then capture (the high-recall monitor also
        # fires on the clean-ground bang-bang spike, so gate the probe to the patch — as the
        # snapshot recorder gates its capture).
        if event is not None and probe is not None and rect.contains(obs.pos):
            obs = _probe_capture(backend, recorder, probe, obs, event)
            break
        recorder.observe(obs, event)
        if recorder.snapshot is not None:
            break
        jit = 0.03 * rng.standard_normal(2)
        obs = backend.step(np.array([speed_fn(k), jit[0], jit[1]]))
    return recorder.snapshot, theta_for(lane), bool(obs.fallen)
