"""Proprioceptive-statistics matching for the O4↔O2 ambiguity pair (spec §8.1 P4).

This is a **paper artifact** (per the kino-fail-operators skill, never delete it): the
constructive proof that O4 (elastic adhesion) and O2 (compliance sink) are, by parameter
tuning, indistinguishable from proprioception alone — so a pure-proprioception baseline
(B2) is *principled*ly unable to choose the right recovery, and the visual semantic map
(M5) is irreplaceable.

The matched statistic the spec names is the **tangential-resistance-vs-displacement
curve**. We estimate it from observables only: at a steady velocity command the demanded
tracking acceleration is ``(v_cmd − v)/τ`` and the achieved acceleration is ``dv/dt``; the
shortfall, times mass, is the resistive force the ground applied. Binned against the path
length travelled inside the patch it gives the resistance curve. We also compare the
base-height and slip traces (the other proprioceptive channels). If all match while the
appearance embeddings are well separated, the pair is constructively ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kino_vla.map.appearance import cosine_similarity
from kino_vla.sim.operators.base import FailureOperator, OperatorStack
from kino_vla.sim.operators.o2_compliance import ComplianceField
from kino_vla.sim.operators.o4_tether import Tether
from kino_vla.sim.surrogate import SurrogateBackend
from kino_vla.utils.config import Config, load_config
from kino_vla.utils.geometry import Rect


def build_matched_pair(
    cfg: Config | None = None,
) -> tuple[ComplianceField, Tether]:
    """Construct the matched O2/O4 ambiguity pair from config (spec §8.1 P4).

    The two share stiffness, damping, and sink so their proprioceptive traces coincide;
    they differ only in appearance class. This is the single source of the matched
    parameters — the script and the gate both build from here.
    """
    cfg = cfg if cfg is not None else load_config("operators/m5_instances.yaml")
    p = cfg.ambiguity_o4_o2
    region = Rect(
        cx=float(p.region.cx),
        cy=float(p.region.cy),
        hx=float(p.region.hx),
        hy=float(p.region.hy),
    )
    o2 = ComplianceField(
        region=region,
        k_c=float(p.stiffness),
        c_c=float(p.damping),
        d_sink=float(p.sink_depth),
        appearance_class=str(p.o2_appearance),
    )
    o4 = Tether(
        region=region,
        k=float(p.stiffness),
        d=float(p.damping),
        l0=float(p.o4_l0),
        f_break=float(p.o4_f_break),
        d_sink=float(p.sink_depth),
        appearance_class=str(p.o4_appearance),
    )
    return o2, o4


@dataclass(frozen=True)
class _Trace:
    s: np.ndarray  # path length travelled inside the region [m]
    resistance_n: np.ndarray  # estimated tangential resistance force [N]
    base_height: np.ndarray  # measured base height [m]
    slip: np.ndarray  # measured slip ratio


def _rollout(
    op: FailureOperator, region: Rect, cmd: np.ndarray, n_steps: int, seed: int, sim_cfg: Config
) -> _Trace:
    """Drive a constant command through ``op``'s region; return proprioceptive traces."""
    backend = SurrogateBackend(sim_cfg, np.array([0.0, 0.0]), 0.0)
    backend.reset(seed)
    stack = OperatorStack([op])
    stack.on_reset(backend)
    tau = float(sim_cfg.tau_track_s)
    mass = backend.mass_kg
    s = 0.0
    prev_vx = 0.0
    prev_pos = np.array([0.0, 0.0])
    ss, res, bh, sl = [], [], [], []
    obs = None
    for _ in range(n_steps):
        stack.on_step(backend, 0.0 if obs is None else obs.t)
        obs = backend.step(cmd)
        vx = float(obs.vel_body[0])
        inside = region.contains(obs.pos)
        if inside:
            s += float(np.linalg.norm(obs.pos - prev_pos))
            a_demand = (float(cmd[0]) - prev_vx) / tau
            a_achieved = (vx - prev_vx) / backend.dt
            ss.append(s)
            res.append(mass * (a_demand - a_achieved))
            bh.append(obs.base_height)
            sl.append(obs.slip_ratio)
        prev_vx = vx
        prev_pos = obs.pos.copy()
    return _Trace(
        s=np.asarray(ss),
        resistance_n=np.asarray(res),
        base_height=np.asarray(bh),
        slip=np.asarray(sl),
    )


def _binned_curve(trace: _Trace, bins: np.ndarray) -> np.ndarray:
    """Mean resistance per displacement bin (NaN where a bin has no samples)."""
    idx = np.digitize(trace.s, bins)
    out = np.full(len(bins) + 1, np.nan)
    for b in range(len(bins) + 1):
        sel = idx == b
        if np.any(sel):
            out[b] = float(np.mean(trace.resistance_n[sel]))
    return out


@dataclass(frozen=True)
class MatchResult:
    resistance_curve_max_abs_diff_n: float
    base_height_max_abs_diff_m: float
    slip_max_abs_diff: float
    appearance_similarity: float
    appearance_separation: float
    s_bins: np.ndarray
    o2_curve: np.ndarray
    o4_curve: np.ndarray


def match_ambiguity_pair(
    o2: FailureOperator,
    o4: FailureOperator,
    *,
    cmd: tuple[float, float, float] = (0.7, 0.0, 0.0),
    n_steps: int = 200,
    seed: int = 0,
    sim_cfg: Config | None = None,
    n_bins: int = 12,
) -> MatchResult:
    """Compare the proprioceptive signatures of the matched O2/O4 pair (spec §8.1 P4)."""
    sim_cfg = sim_cfg if sim_cfg is not None else load_config("sim/surrogate.yaml")
    cmd_arr = np.asarray(cmd, dtype=np.float64)
    t2 = _rollout(o2, o2.region, cmd_arr, n_steps, seed, sim_cfg)
    t4 = _rollout(o4, o4.region, cmd_arr, n_steps, seed, sim_cfg)
    s_max = float(max(t2.s.max(initial=0.0), t4.s.max(initial=0.0)))
    bins = np.linspace(0.0, max(s_max, 1e-6), n_bins + 1)[1:-1]
    c2, c4 = _binned_curve(t2, bins), _binned_curve(t4, bins)
    both = ~np.isnan(c2) & ~np.isnan(c4)
    res_diff = float(np.max(np.abs(c2[both] - c4[both]))) if np.any(both) else float("inf")
    n = min(len(t2.base_height), len(t4.base_height))
    bh_diff = float(np.max(np.abs(t2.base_height[:n] - t4.base_height[:n]))) if n else float("inf")
    slip_diff = float(np.max(np.abs(t2.slip[:n] - t4.slip[:n]))) if n else float("inf")
    sim = cosine_similarity(o2.scene_region().embedding, o4.scene_region().embedding)
    return MatchResult(
        resistance_curve_max_abs_diff_n=res_diff,
        base_height_max_abs_diff_m=bh_diff,
        slip_max_abs_diff=slip_diff,
        appearance_similarity=sim,
        appearance_separation=1.0 - sim,
        s_bins=bins,
        o2_curve=c2,
        o4_curve=c4,
    )
