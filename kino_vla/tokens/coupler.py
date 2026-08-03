"""Online μ̂ → CBF-shield coupler with anomaly-gated injection (spec §4 #4, §6.5).

This is the M4 perception↔safety coupling the M3 shield left as a hook
(``CbfShield.set_mu_estimate``). Each control step the coupler pushes the new
observation into the 500 ms rolling window and, *only when the Kino-Monitor is
aroused* (anomaly-gated injection — steady-state extractor compute is skipped, spec
§4 #4), runs the extractor and feeds an EMA-smoothed μ̂ to the shield so a detected
ice patch tightens the friction-cone constraint. When the monitor quiets past a hold
window, μ̂ relaxes back to nominal.

Torch-free at import: it consumes an already-built extractor by duck-typed protocol
(``predict``), so ``run_episode`` can take an optional coupler without the CI / demo
path ever importing torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from kino_vla.monitor.event import Monitor
from kino_vla.sim.types import Obs
from kino_vla.tokens.features import MU_INDEX
from kino_vla.tokens.semantics import REGIMES
from kino_vla.tokens.window import RollingWindow, window_length
from kino_vla.utils.config import Config

if TYPE_CHECKING:  # torch-free at runtime: the type is only needed for static checking
    from kino_vla.tokens.extractor import ExtractResult


class _ExtractorLike(Protocol):
    def predict(self, windows: np.ndarray) -> ExtractResult: ...


class _ShieldLike(Protocol):
    def set_mu_estimate(self, mu: float) -> None: ...


@dataclass(frozen=True)
class CouplerRecord:
    """One step's coupling telemetry (spec §6.8 logging / demo artifact)."""

    t: float
    aroused: bool  # monitor gate open (extractor ran this step)
    mu_hat: float  # EMA-smoothed friction estimate pushed to the shield
    mu_raw: float  # this step's raw extractor μ̂ (NaN when gate closed)
    ood_score: float  # reconstruction-residual OOD score (NaN when gate closed)
    regime: str  # predicted physical regime (Kino-Text head), "" when gate closed


class MuEstimateCoupler:
    """Feeds anomaly-gated μ̂ from the extractor into the CBF shield (spec §6.5)."""

    def __init__(self, cfg: Config, extractor: _ExtractorLike, shield: _ShieldLike) -> None:
        self._cfg = cfg
        self._ex = extractor
        self._shield = shield
        cc = cfg.coupler
        self._gate_threshold = float(cc.gate_threshold)
        self._alpha = float(cc.mu_ema_alpha)
        self._hold_s = float(cc.hold_s)
        self._nominal_mu = float(cc.nominal_mu)
        self._win = RollingWindow(
            window_length(float(cfg.window.window_ms), float(cfg.window.control_hz))
        )
        self.records: list[CouplerRecord] = []
        self._mu_ema = self._nominal_mu
        self._hold_until = -1.0

    def reset(self) -> None:
        self._win.reset()
        self.records = []
        self._mu_ema = self._nominal_mu
        self._hold_until = -1.0
        self._shield.set_mu_estimate(self._nominal_mu)

    def step(self, obs: Obs, monitor: Monitor) -> CouplerRecord:
        """Update μ̂ from the window when aroused; push it to the shield."""
        self._win.push(obs)
        aroused = monitor.anomaly_score >= self._gate_threshold or obs.t < self._hold_until
        mu_raw = float("nan")
        ood = float("nan")
        regime = ""
        if aroused and self._win.ready:
            res = self._ex.predict(self._win.window())
            mu_raw = float(res.theta_hat[0, MU_INDEX])
            ood = float(res.ood_score[0])
            regime = REGIMES[int(np.argmax(res.regime_logits[0]))]
            # Hold the aroused state briefly so μ̂ does not flap as the robot crosses
            # the patch (the monitor anomaly is intermittent across the gait cycle).
            if monitor.anomaly_score >= self._gate_threshold:
                self._hold_until = obs.t + self._hold_s
            self._mu_ema += self._alpha * (mu_raw - self._mu_ema)
        else:
            # Steady state: relax μ̂ back to nominal (extractor compute skipped).
            self._mu_ema += self._alpha * (self._nominal_mu - self._mu_ema)
        self._shield.set_mu_estimate(self._mu_ema)
        record = CouplerRecord(
            t=float(obs.t),
            aroused=bool(aroused and self._win.ready),
            mu_hat=float(self._mu_ema),
            mu_raw=mu_raw,
            ood_score=ood,
            regime=regime,
        )
        self.records.append(record)
        return record
