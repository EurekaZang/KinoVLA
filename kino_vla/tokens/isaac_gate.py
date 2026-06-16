"""Strict four-channel θ gate for the Kino-Tokens extractor on the real Go2 (spec §4).

CLAUDE.md §6 #22 closes the M4 strict-GPU gap: the Isaac gate must regress *all four*
privileged channels (μ, payload, effort-scale, support) below their
``configs/tolerances.yaml`` bars on the physically-simulated Go2 — not μ alone at a
relaxed bar. Each channel is excited in its own single-operator lane/phase on Isaac
(``scripts/isaac_tokens_check.py``), the per-step rollouts are saved to an ``.npz``, and
this module trains the extractor and gates **per channel on the windows where that
channel actually varies** (a constant channel would pass an averaged MAE trivially —
the honest gate scores μ only on the μ lanes, payload only on the payload phase, etc.).

Decoupled from the GPU collection on purpose (machine has ~14 GB RAM and has frozen on
heavy interactive runs): collect once on Isaac → ``.npz`` → train+gate on CPU here, so
the gate iterates offline (``scripts/isaac_tokens_gate.py``) without re-touching the GPU.

Pure numpy for the data path; torch is imported lazily inside ``train_and_gate`` (the
extractor), so the collection script and the npz round-trip stay torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from kino_vla.tokens.dataset import TokenDataset

if TYPE_CHECKING:  # torch-only; imported lazily at runtime to keep this module torch-free
    from kino_vla.tokens.extractor import Extractor
from kino_vla.tokens.features import MU_INDEX, SUPPORT_INDEX, TARGET_SCHEMA
from kino_vla.tokens.window import slice_rollout_to_windows
from kino_vla.utils.config import Config

# Which privileged-target column each excitation channel varies (and is gated on).
CHANNEL_IDX: dict[str, int] = {
    "mu": TARGET_SCHEMA.index("mu"),
    "payload": TARGET_SCHEMA.index("payload_kg"),
    "effort": TARGET_SCHEMA.index("effort_scale"),
    "support": TARGET_SCHEMA.index("support_ratio"),
}
SUPPORT_IDX: int = SUPPORT_INDEX


@dataclass(frozen=True)
class Rollout:
    """One single-operator drive on the real Go2: which channel it excites, the channel
    level, the contrastive regime id, and the per-step (features, privileged target)."""

    channel: str
    level: float
    regime: int
    feats: np.ndarray  # (S, N_FEATURES)
    tgts: np.ndarray  # (S, N_TARGETS)


def save_rollouts(path: str | Path, rollouts: list[Rollout]) -> None:
    """Persist rollouts to a pickle-free ``.npz`` (concatenated + length index)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        feats=np.concatenate([r.feats for r in rollouts], axis=0),
        tgts=np.concatenate([r.tgts for r in rollouts], axis=0),
        lengths=np.array([len(r.feats) for r in rollouts], dtype=np.int64),
        channels=np.array([r.channel for r in rollouts]),
        levels=np.array([r.level for r in rollouts], dtype=np.float64),
        regimes=np.array([r.regime for r in rollouts], dtype=np.int64),
    )


def load_rollouts(path: str | Path) -> list[Rollout]:
    """Inverse of :func:`save_rollouts`."""
    d = np.load(Path(path), allow_pickle=False)
    feats, tgts, lengths = d["feats"], d["tgts"], d["lengths"]
    channels, levels, regimes = d["channels"], d["levels"], d["regimes"]
    out: list[Rollout] = []
    off = 0
    for i, n in enumerate(int(x) for x in lengths):
        out.append(
            Rollout(
                channel=str(channels[i]),
                level=float(levels[i]),
                regime=int(regimes[i]),
                feats=feats[off : off + n],
                tgts=tgts[off : off + n],
            )
        )
        off += n
    return out


def smooth_support(tgts: np.ndarray, win: int) -> np.ndarray:
    """Causal trailing moving-average of the support column (spec §6 / CLAUDE.md §6 #22).

    On a real trot the *instantaneous* foot-support fraction is dominated by gait phase
    (only the diagonal stance pair bears load at any tick), so the last-step window
    target would be pure phase noise. High-centering is a slowly-varying *property*, so
    the supervised target is the trailing window-average; the other channels are left
    untouched.
    """
    tgts = np.asarray(tgts, dtype=np.float64).copy()
    s = tgts[:, SUPPORT_IDX]
    win = max(1, int(win))
    csum = np.cumsum(np.insert(s, 0, 0.0))
    out = np.empty_like(s)
    for i in range(len(s)):
        lo = max(0, i - win + 1)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    tgts[:, SUPPORT_IDX] = out
    return tgts


def build_split(
    rollouts: list[Rollout],
    win_len: int,
    stride: int,
    *,
    train_frac: float,
    gap: int,
    support_smooth: int,
) -> tuple[TokenDataset, dict[str, TokenDataset]]:
    """Window every rollout and split each by time into disjoint train / eval halves.

    Train windows are sliced from ``feats[:n_tr]`` and eval windows from ``feats[n_tr +
    gap:]`` — two disjoint step ranges, so no window crosses the cut and no eval window
    shares a step with any train window. ``gap`` adds a small extra buffer to decorrelate
    the eval tail from the train head (windows are overlapping, so adjacent ones are
    correlated even when they share no step). Eval windows are grouped *by channel* so each
    θ component can be scored on the lane where it actually varies.
    """
    train_w: list[np.ndarray] = []
    train_t: list[np.ndarray] = []
    train_r: list[np.ndarray] = []
    eval_w: dict[str, list[np.ndarray]] = {c: [] for c in CHANNEL_IDX}
    eval_t: dict[str, list[np.ndarray]] = {c: [] for c in CHANNEL_IDX}
    for r in rollouts:
        tgts = smooth_support(r.tgts, support_smooth)
        n = len(r.feats)
        n_tr = int(train_frac * n)
        tr_w, tr_t = slice_rollout_to_windows(r.feats[:n_tr], tgts[:n_tr], win_len, stride=stride)
        ev_start = n_tr + gap
        ev_w, ev_t = slice_rollout_to_windows(
            r.feats[ev_start:], tgts[ev_start:], win_len, stride=stride
        )
        if len(tr_w):
            train_w.append(tr_w)
            train_t.append(tr_t)
            train_r.append(np.full(len(tr_w), r.regime, dtype=np.int64))
        if len(ev_w):
            eval_w[r.channel].append(ev_w)
            eval_t[r.channel].append(ev_t)
    train = TokenDataset(np.concatenate(train_w), np.concatenate(train_t), np.concatenate(train_r))
    eval_by_channel = {
        c: TokenDataset(
            np.concatenate(eval_w[c]),
            np.concatenate(eval_t[c]),
            np.zeros(sum(len(a) for a in eval_w[c]), dtype=np.int64),
        )
        for c in CHANNEL_IDX
        if eval_w[c]
    }
    return train, eval_by_channel


@dataclass(frozen=True)
class GateReport:
    """Outcome of the strict four-channel Isaac gate."""

    passed: bool
    mae: dict[str, float]
    bars: dict[str, float]
    mu_firm: float
    mu_ice: float
    friction_firm_m: float
    friction_ice_m: float
    latency_ms: float
    checks: dict[str, bool]
    dropped: list[tuple[str, float]]  # (channel, level) excluded as an unobservable floor
    level_breakdown: dict[str, list[tuple[float, float, float]]]  # ch → [(true, pred_mean, mae)]


def filter_observable(rollouts: list[Rollout], ig: Config) -> tuple[list[Rollout], list[Rollout]]:
    """Split rollouts into (observable, unobservable-floor) by the config regime thresholds.

    μ is recoverable only on deep ice (slipping) or firm ground; effort only once the cut
    binds. The mid bands leave no proprioceptive trace (measured), so training on them would
    feed contradictory labels (identical features, different θ) and poison the head — they
    are excluded from BOTH train and eval, and returned so the caller can log them.
    """
    mu_ice_max, mu_firm_min = float(ig.mu_ice_max), float(ig.mu_firm_min)
    eff_bind, eff_healthy = float(ig.effort_bind_max), float(ig.effort_healthy_min)
    kept: list[Rollout] = []
    dropped: list[Rollout] = []
    for r in rollouts:
        if r.channel == "mu":
            ok = r.level <= mu_ice_max or r.level >= mu_firm_min
        elif r.channel == "effort":
            ok = r.level <= eff_bind or r.level >= eff_healthy
        else:
            ok = True
        (kept if ok else dropped).append(r)
    return kept, dropped


def _level_breakdown(
    extractor: Extractor, eval_ds: TokenDataset, idx: int
) -> list[tuple[float, float, float]]:
    """Per-true-level (true, pred_mean, MAE) so observability is visible, not hidden."""
    pred = extractor.predict(eval_ds.windows).theta_hat[:, idx]
    true = eval_ds.targets[:, idx]
    out: list[tuple[float, float, float]] = []
    for v in sorted({round(float(t), 2) for t in true}):
        m = np.abs(np.round(true, 2) - v) < 1e-6
        out.append((v, float(pred[m].mean()), float(np.abs(pred[m] - true[m]).mean())))
    return out


def train_and_gate(rollouts: list[Rollout], tok_cfg: Config, tol: Config) -> GateReport:
    """Train the extractor on the Isaac rollouts and gate all four θ channels.

    Per-channel MAE is computed on the eval windows of the channel's own lane (μ on the
    friction lanes, payload on the payload phase, …) so a channel that is constant
    elsewhere cannot dilute the score, and only over each channel's *observable regime*
    (the unobservable floor is excluded and reported, never silently averaged in or
    tolerated by a weakened bar). Also verifies the μ̂→CBF-shield coupling (spec §6.5) on
    the real-Go2 estimates and the extractor inference budget (spec §6.9).
    """
    from kino_vla.shield.cbf_shield import CbfShield
    from kino_vla.tokens.extractor import Extractor
    from kino_vla.utils.config import load_config

    kept, dropped_rolls = filter_observable(rollouts, tok_cfg.isaac_gate)
    dropped = [(r.channel, r.level) for r in dropped_rolls]

    win_len = int(round(float(tok_cfg.window.window_ms) * 1e-3 * float(tok_cfg.window.control_hz)))
    stride = int(tok_cfg.window.stride)
    train, eval_by_channel = build_split(
        kept,
        win_len,
        stride,
        train_frac=float(tok_cfg.isaac_gate.train_frac),
        gap=win_len,
        support_smooth=win_len,
    )

    extractor = Extractor(tok_cfg)
    extractor.fit(train, log=False)

    bars = {
        "mu": float(tol.regression.mu_mae),
        "payload": float(tol.regression.payload_kg_mae),
        "effort": float(tol.regression.effort_scale_mae),
        "support": float(tol.regression.support_ratio_mae),
    }
    mae: dict[str, float] = {}
    checks: dict[str, bool] = {}
    level_breakdown: dict[str, list[tuple[float, float, float]]] = {}
    for ch, idx in CHANNEL_IDX.items():
        ev = eval_by_channel[ch]
        theta = extractor.predict(ev.windows).theta_hat
        mae[ch] = float(np.mean(np.abs(theta[:, idx] - ev.targets[:, idx])))
        checks[f"{ch} MAE {mae[ch]:.3f} < {bars[ch]}"] = mae[ch] < bars[ch]
        level_breakdown[ch] = _level_breakdown(extractor, ev, idx)

    # μ̂→shield coupling (spec §6.5) on the real-Go2 friction estimates: firm vs ice.
    mu_ev = eval_by_channel["mu"]
    mu_hat = extractor.predict(mu_ev.windows).mu_hat
    mu_true = mu_ev.targets[:, MU_INDEX]
    firm = float(np.mean(mu_hat[mu_true >= 0.55]))
    ice = float(np.mean(mu_hat[mu_true <= 0.20]))
    shield = CbfShield(load_config("shield/cbf_v0.yaml"))
    shield.set_mu_estimate(firm)
    r_firm = shield.friction_radius()
    shield.set_mu_estimate(ice)
    r_ice = shield.friction_radius()
    checks[f"μ̂ firm {firm:.2f} > {float(tol.coupling.min_mu_hat_on_firm)}"] = firm > float(
        tol.coupling.min_mu_hat_on_firm
    )
    checks[f"μ̂ ice {ice:.2f} < {float(tol.coupling.max_mu_hat_on_ice)}"] = ice < float(
        tol.coupling.max_mu_hat_on_ice
    )
    checks[f"μ̂→friction cone tightens on ice ({r_firm:.3f}->{r_ice:.3f} m)"] = r_ice < r_firm

    latency = extractor.inference_latency_ms()
    checks[f"inference p99 {latency:.2f} ms < {float(tol.latency.inference_ms_p99)}"] = (
        latency < float(tol.latency.inference_ms_p99)
    )

    passed = all(checks.values())
    return GateReport(
        passed=passed,
        mae=mae,
        bars=bars,
        mu_firm=firm,
        mu_ice=ice,
        friction_firm_m=r_firm,
        friction_ice_m=r_ice,
        latency_ms=latency,
        checks=checks,
        dropped=dropped,
        level_breakdown=level_breakdown,
    )


def print_report(report: GateReport) -> None:
    """Print the per-channel gate verdict (shared by the Isaac and offline entry points)."""
    if report.dropped:
        floor = ", ".join(f"{ch}={lvl:g}" for ch, lvl in report.dropped)
        print(f"[isaac_tokens] excluded unobservable-floor levels (documented, not gated): {floor}")
    for ch, rows in report.level_breakdown.items():
        cells = " ".join(f"{t:g}→{p:.2f}(±{e:.2f})" for t, p, e in rows)
        print(f"[isaac_tokens] {ch:8s} true→pred(MAE): {cells}")
    print(
        "[isaac_tokens] per-channel MAE "
        + ", ".join(f"{c}={report.mae[c]:.3f} (<{report.bars[c]})" for c in CHANNEL_IDX)
    )
    for name, ok in report.checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("PASS: M4 Kino-Tokens on Isaac" if report.passed else "FAIL: M4 Kino-Tokens on Isaac")
