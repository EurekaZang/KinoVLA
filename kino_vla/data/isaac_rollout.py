"""Real-Go2 (Isaac) Hindsight-CoT rollout: drive into one operator's failure + snapshot it.

Shared by ``scripts/isaac_hindsight_collect.py`` (build the dataset over real-Go2 snapshots) and
``scripts/isaac_hindsight_check.py`` (the GPU gate). The spec builds the data pipeline on Isaac
(§1/§8.1/§10), so the snapshot's proprioception + privileged θ come from the physically-simulated
Go2 — not the surrogate point-robot. The ``backend`` is injected (an ``IsaacPolicyBackend``); this
module imports no Isaac itself, so it stays CPU-importable.

Each operator runs in its own lateral lane (Isaac is one-episode/process, #21a): the operator is
installed, the Go2 is driven straight across with the M4 bang-bang excitation (so O5/O10 effort
binds, #15), and the high-recall collection monitor intercepts the anomaly. Capture gates the
failure locus — region operators strictly in-region (the slip/resistance lands in the window),
global operators (O5/O10) anywhere (no spatial locus).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kino_vla.data.schema import Snapshot
from kino_vla.data.snapshot import SnapshotRecorder
from kino_vla.map.types import SemanticRegion
from kino_vla.monitor.rule_monitor import MonitorEvent, RuleMonitor
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


def collect_lane(
    backend: Any, cfg: Config, monitor_cfg: Config, lane: dict, seed: int
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

    period, duty = float(cc.speed_period_s), float(cc.speed_duty)
    lo = float(cc.speed_lo)
    hi = float(cc.effort_speed_hi if op == "O10_effort_decay" else cc.speed_hi)

    def speed_fn(k: int) -> float:
        return hi if (((k * dt) / period) % 1.0) < duty else lo

    if op == "O10_effort_decay":
        backend.set_effort_scale(1.0)
    for _ in range(int(cc.settle_steps)):
        obs = backend.step(np.array([speed_fn(0), 0.0, 0.0]))
    if op == "O10_effort_decay":
        backend.set_effort_scale(float(lane["floor"]))  # cut after settling healthy
    if op == "O5_payload":
        backend.add_payload(float(lane["mass"]), np.zeros(2))

    scene = (
        [SemanticRegion(Rect(rect.cx, rect.cy, rect.hx, rect.hy), appr)] if op in REGION_OPS else []
    )
    # Region ops gate to the patch (snapshot the in-region slip); global ops (O5/O10) have no
    # spatial locus, so they take no gate. (Method A's A2 — gating them to the patch to capture a
    # developed effort signature — was tried and reverted: the payload/decay torque does NOT exceed
    # the saturation floor at safe drive speeds, so effort_ratio≈0 regardless of WHEN we capture —
    # the genuine #15 observability floor on the real Go2; gating O10 to the patch also lost it.)
    gate = rect if op in REGION_OPS else None
    recorder = SnapshotRecorder(
        cfg,
        scene=scene,
        operator_name=op,
        appearance_class=appr,
        privileged_fn=backend.privileged_physics,
        gate_rect=gate,
    )
    monitor = RuleMonitor(monitor_cfg, dt=dt)
    monitor.reset()
    # Embodiment ops (O5/O10) have no slip/visual signature; their tell is actuator saturation,
    # which binds only intermittently and AFTER the early bang-bang tracking spike. So force the
    # capture to the window where effort actually binds, not the first monitor event.
    effort_min = float(cc.get("effort_capture_min", 0.25))
    is_effort_op = op in ("O10_effort_decay", "O5_payload")

    def effort_event(o: Obs) -> MonitorEvent:
        return MonitorEvent(
            t=o.t,
            pos=o.pos.copy(),
            channel="effort_ratio",
            value=float(o.effort_ratio),
            threshold=effort_min,
            summary="effort-bind capture",
        )

    delay = int(cc.get("effort_capture_delay_steps", 12))
    countdown = -1  # -1 = effort has not bound yet
    rng = np.random.default_rng(seed + 7)
    for k in range(int(cc.n_steps)):
        if obs.fallen:
            break
        if is_effort_op:
            # Ignore the bang-bang tracking spike; on the FIRST effort bind, wait ~half a window
            # so the capture window fills with the (intermittent) effort spikes, then capture.
            if countdown < 0 and float(obs.effort_ratio) >= effort_min:
                countdown = delay
            event = effort_event(obs) if countdown == 0 else None
            if countdown > 0:
                countdown -= 1
        else:
            event = monitor.step(obs)
        recorder.observe(obs, event)
        if recorder.snapshot is not None:
            break
        jit = 0.03 * rng.standard_normal(2)
        obs = backend.step(np.array([speed_fn(k), jit[0], jit[1]]))
    # An effort op whose effort never bound (O5 — load spreads, never saturates) still gets a
    # snapshot so it stays in the dataset (it will be filter-dropped — the honest #15 floor).
    if is_effort_op and recorder.snapshot is None:
        recorder.observe(obs, effort_event(obs))
    if op == "O10_effort_decay":
        backend.set_effort_scale(1.0)  # restore before the next lane
    return recorder.snapshot, theta_for(lane), bool(obs.fallen)
