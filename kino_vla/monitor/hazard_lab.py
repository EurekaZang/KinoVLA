"""Shared hazard-scenario builder + per-step hazard labels for the learning-based monitor.

The learning-based Kino-Monitor (replacing the rule-based ``RuleMonitor``) is a supervised
anomaly/attribution detector over the 500 ms proprioception window. This module is the single,
CPU-importable source of the 11-operator scenarios it is trained and evaluated on, so the data
collector (``scripts/isaac_monitor_data_collect.py``), the trainer, and the online eval all agree
on (a) how each operator is instantiated in a lane, (b) the deployment drive profile, and (c) the
ground-truth hazard label at each control step.

Label convention (the supervision target): ``hazard=1`` iff the operator's physical effect is
manifest at that step — in-region for material/geometry operators (O1/O2/O4/O7/O8/O9), in-region
*after the collapse dwell* for O3, post-onset for the embodiment/transient operators (O5 payload,
O10 effort-decay, O6 push), and throughout for O11 (the IMU fault is always present). ``clean`` and
``maneuver`` lanes are all-normal negatives — ``maneuver`` deliberately drives turns + accel/decel
on clean ground so the detector learns NOT to fire on the maneuver artifacts the rule monitor
false-fired on (#46 turn slip, the bang-bang tracking spike).

No torch / no Isaac here — only the operator classes (CPU) and geometry, so every consumer imports
it freely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from kino_vla.sim.operators import (
    Collapse,
    ComplianceField,
    EffortDecay,
    HighCentering,
    InvisibleCollider,
    MuField,
    ObsBias,
    Push,
    Tether,
    VisualPhysicsRemap,
)
from kino_vla.sim.types import Obs
from kino_vla.tokens.features import FEATURE_SCHEMA, obs_to_features
from kino_vla.utils.geometry import Rect

# The learned monitor's feature vector: the M4 proprio schema PLUS ``support_ratio`` — the
# foot-support fraction the rule monitor never thresholded, O9 high-centering's defining channel.
MON_FEATURE_SCHEMA: tuple[str, ...] = (*FEATURE_SCHEMA, "support_ratio")
MON_N_FEATURES: int = len(MON_FEATURE_SCHEMA)


def monitor_features(obs: Obs) -> np.ndarray:
    """The (MON_N_FEATURES,) monitor feature vector: M4 proprio features + support_ratio."""
    return np.append(obs_to_features(obs), float(obs.support_ratio))


# The 11 operators (the detector's positive classes) + two negative regimes.
OP_IDS: tuple[str, ...] = (
    "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "O10", "O11",
)
NEG_IDS: tuple[str, ...] = ("clean", "maneuver")
ALL_IDS: tuple[str, ...] = OP_IDS + NEG_IDS
# Multi-class attribution id: 0 = normal, then 1..11 = O1..O11 (a learned regime/channel head).
CLASS_OF: dict[str, int] = {"clean": 0, "maneuver": 0, **{op: i + 1 for i, op in enumerate(OP_IDS)}}
N_CLASSES: int = len(OP_IDS) + 1

PATCH_CX = 2.5  # hazard centre on the +x path [m]
ONSET_T = 3.0  # embodiment/transient fault onset [s] (after the policy reaches steady cruise)
PUSH_WINDOW_S = 1.5  # O6: the disturbance is "manifest" for this long after the impulse
CRUISE_MPS = 0.6  # deployment cruise (configs/recovery/fsm_isaac.yaml)


# θ ranges per operator (B-class / decision-relevant; from configs/data/hindsight.yaml).
_THETA_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "O1": {"mu": (0.06, 0.16)},
    "O2": {"k": (16.0, 26.0), "c": (7.0, 11.0), "d_sink": (0.04, 0.20)},
    "O3": {"mu_collapsed": (0.06, 0.12), "dwell": (0.35, 0.6)},
    "O4": {"k": (16.0, 26.0), "c": (7.0, 11.0), "f_break": (20.0, 32.0)},
    "O5": {"mass": (14.0, 18.0)},
    "O6": {"impulse": (14.0, 22.0)},
    "O7": {"mu": (0.06, 0.13)},
    "O8": {},
    # ridge_h spans a band that beaches the chassis (support drop) but mostly does NOT topple, so
    # O9 produces a SUSTAINED high-centering signature (more, stronger hazard windows) — the 0.10 m
    # default ridge toppled it within ~1 s, starving the detector of O9 positives.
    "O9": {"residual": (0.15, 0.35), "ridge_h": (0.055, 0.085)},
    "O10": {"floor": (0.12, 0.16)},
    "O11": {"tilt_bias": (0.25, 0.60)},
    # Negatives sample a per-lane operating point so the "normal" manifold is dense across the
    # speed/turn envelope the planner uses — this is what suppresses the clean-cruise false fires.
    "clean": {"cruise": (0.3, 0.9)},
    "maneuver": {"base": (0.25, 0.6), "turn_amp": (0.6, 1.4), "turn_freq": (0.25, 0.9)},
}


def sample_theta(op_id: str, rng: np.random.Generator) -> dict[str, float]:
    """Sample this lane's θ uniformly from the operator's range (empty for O8/clean/maneuver)."""
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in _THETA_RANGES[op_id].items()}


@dataclass
class Scenario:
    op_id: str
    theta: dict[str, float]
    operator: object | None  # a FailureOperator, or None for clean/maneuver/payload-at-onset
    rect: Rect | None
    onset_kind: str  # region | payload | effort | push | obsbias | clean | maneuver
    dwell_s: float = 0.0  # O3 collapse trigger dwell
    label_rect: Rect | None = None  # hazard-label zone (defaults to ``rect``); O8 widens it


def build_scenario(op_id: str, y: float, theta: dict[str, float]) -> Scenario:
    """Instantiate one operator lane at ``y`` with sampled θ. Region ops carry their Rect."""
    wide = Rect(PATCH_CX, y, 1.0, 1.0)
    if op_id == "O1":
        return Scenario(op_id, theta, MuField(wide, theta["mu"], theta["mu"]), wide, "region")
    if op_id == "O2":
        op = ComplianceField(wide, theta["k"], theta["c"], theta["d_sink"])
        return Scenario(op_id, theta, op, wide, "region")
    if op_id == "O3":
        op = Collapse(wide, theta["mu_collapsed"], theta["dwell"], 0.8)
        return Scenario(op_id, theta, op, wide, "region", dwell_s=theta["dwell"])
    if op_id == "O4":
        op = Tether(wide, theta["k"], theta["c"], 0.0, theta["f_break"])
        return Scenario(op_id, theta, op, wide, "region")
    if op_id == "O5":
        return Scenario(op_id, theta, None, None, "payload")  # mass added at onset
    if op_id == "O6":
        op = Push(np.array([0.0, theta["impulse"]]), ONSET_T)
        return Scenario(op_id, theta, op, None, "push")
    if op_id == "O7":
        op = VisualPhysicsRemap(wide, theta["mu"], theta["mu"], 0.5)
        return Scenario(op_id, theta, op, wide, "region")
    if op_id == "O8":
        wall = Rect(PATCH_CX, y, 0.15, 1.5)
        # The robot BODY blocks at origin x≈2.0 (before its origin enters the thin wall rect at
        # x≥2.35), so the hazard-label zone covers the stuck-against-the-wall band, not the wall.
        label = Rect(2.15, y, 0.55, 1.5)  # x in [1.6, 2.7]
        return Scenario(op_id, theta, InvisibleCollider(wall), wall, "region", label_rect=label)
    if op_id == "O9":
        ridge = Rect(PATCH_CX, y, 0.5, 0.8)
        return Scenario(op_id, theta, HighCentering(ridge, theta["residual"]), ridge, "region")
    if op_id == "O10":
        return Scenario(op_id, theta, EffortDecay(2.0, theta["floor"], ONSET_T), None, "effort")
    if op_id == "O11":
        return Scenario(op_id, theta, ObsBias({"tilt": theta["tilt_bias"]}), None, "obsbias")
    if op_id in ("clean", "maneuver"):
        return Scenario(op_id, theta, None, None, op_id)
    raise ValueError(f"unknown op {op_id}")


def install(sc: Scenario, backend: object) -> None:
    """Install a region operator's physics on the backend. O9 overrides the ridge height per lane
    (sustained beaching without topple); the others use the operator's own ``on_reset``."""
    if sc.operator is None or sc.onset_kind != "region":
        return
    if sc.op_id == "O9":
        from kino_vla.sim.types import SupportLossRegion

        backend.add_support_loss_regions(
            [SupportLossRegion(sc.rect, float(sc.theta["residual"]))],
            height_m=float(sc.theta["ridge_h"]),
        )
    else:
        sc.operator.on_reset(backend)


def hazard_label(sc: Scenario, pos: np.ndarray, t: float, t_in_region: float) -> int:
    """Ground-truth hazard (1) / normal (0) at this step for scenario ``sc`` (the supervision)."""
    k = sc.onset_kind
    if k in ("clean", "maneuver"):
        return 0
    if k == "payload" or k == "effort":
        return 1 if t >= ONSET_T else 0
    if k == "push":
        return 1 if ONSET_T <= t <= ONSET_T + PUSH_WINDOW_S else 0
    if k == "obsbias":
        return 1  # the IMU fault is present throughout the lane
    if k == "region":
        zone = sc.label_rect if sc.label_rect is not None else sc.rect
        in_region = zone is not None and zone.contains(np.asarray(pos, dtype=np.float64))
        if not in_region:
            return 0
        if sc.op_id == "O3":  # thin ice is only hazardous AFTER it collapses (post-dwell)
            return 1 if t_in_region >= sc.dwell_s else 0
        return 1
    return 0


def drive_cmd(sc: Scenario, k: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    """The +x deployment drive. Operator lanes cruise straight at the per-lane speed; ``clean``
    lanes cruise at a sampled speed (dense normal manifold); ``maneuver`` lanes add turns +
    accel/decel on CLEAN ground so the detector learns the hard negatives (a commanded turn / a
    speed change is NOT a hazard — the rule monitor's #46 turn-slip / bang-bang false fires)."""
    if sc.op_id == "clean":
        return np.array([float(sc.theta.get("cruise", CRUISE_MPS)), 0.0, 0.0])
    if sc.op_id == "maneuver":
        t = k * dt
        base = float(sc.theta.get("base", 0.45))
        amp = float(sc.theta.get("turn_amp", 0.9))
        freq = float(sc.theta.get("turn_freq", 0.5))
        vx = base + 0.35 * (0.5 + 0.5 * math.sin(0.8 * t))  # accel/decel sweep
        wz = amp * math.sin(freq * t + 0.5)  # turns above the #46 0.6 turn-gate
        vy = 0.05 * float(rng.standard_normal())
        return np.array([vx, vy, wz])
    return np.array([CRUISE_MPS, 0.0, 0.0])  # operator lane: steady deployment cruise
