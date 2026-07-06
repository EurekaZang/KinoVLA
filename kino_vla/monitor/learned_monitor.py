"""Learning-based Kino-Monitor — the supervised hazard detector that replaces the rule monitor.

A small 1D-CNN + Perceiver-Resampler over the 500 ms proprioception window (the SAME backbone as
the M4 Kino-Extractor, spec §4 — reused here for the monitor task) with two heads:

  - ``det`` (1 logit) — hazard vs normal. Its sigmoid is the firing/anomaly score; this is the
    ~50 Hz online trigger that wakes the VLA reflection (replacing RuleMonitor's threshold/EMA).
  - ``attr`` (N_CLASSES logits) — normal + O1..O11 attribution. Its argmax names the firing
    ``MonitorEvent.channel`` (the inferred physical cause), so the event already carries a grounded
    attribution instead of "whatever raw channel crossed" (the wrong-channel failure measured on
    the rule monitor).

``LearnedMonitor`` is the online wrapper: same contract as ``RuleMonitor`` (``step(obs) ->
MonitorEvent | None``, ``anomaly_score``, ``reset()``, ``events``) so it is a drop-in across the
loop / coupler / planner. Torch lives only here and in the trainer; the wrapper is fed an
already-loaded model.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from kino_vla.monitor.event import MonitorEvent
from kino_vla.monitor.hazard_lab import (
    MON_N_FEATURES,
    N_CLASSES,
    OP_IDS,
    monitor_features,
)
from kino_vla.sim.types import Obs
from kino_vla.tokens.features import Standardizer

# Attribution class id -> human/cause name (0 = normal, 1..11 = O1..O11).
CLASS_NAMES: tuple[str, ...] = ("normal", *OP_IDS)


class MonitorWindow:
    """Online ``(T, MON_N_FEATURES)`` rolling buffer of monitor feature vectors."""

    def __init__(self, length: int) -> None:
        self._length = int(length)
        self._buf: deque[np.ndarray] = deque(maxlen=self._length)

    @property
    def ready(self) -> bool:
        return len(self._buf) == self._length

    def reset(self) -> None:
        self._buf.clear()

    def push(self, obs: Obs) -> None:
        self._buf.append(monitor_features(obs))

    def window(self) -> np.ndarray:
        rows = list(self._buf)
        if rows and len(rows) < self._length:
            rows = [rows[0]] * (self._length - len(rows)) + rows
        return np.stack(rows, axis=0)


@dataclass(frozen=True)
class MonitorNetConfig:
    """Backbone hyper-params (mirrors configs/tokens/extractor_v0.yaml ``model``)."""

    window_len: int = 25
    conv_channels: tuple[int, ...] = (48, 48)
    conv_kernels: tuple[int, ...] = (5, 3)
    latent_dim: int = 96
    n_latents: int = 6
    n_heads: int = 4
    mlp_hidden: int = 96
    dropout: float = 0.2  # regularize the heads — generalize to held-out clean lanes (lower FPR)


class MonitorNet(nn.Module):
    """1D-CNN + Perceiver Resampler + (detection, attribution) heads over a (T, F) window."""

    def __init__(self, cfg: MonitorNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        conv: list[nn.Module] = []
        in_ch = MON_N_FEATURES
        for out_ch, k in zip(cfg.conv_channels, cfg.conv_kernels, strict=True):
            conv.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2))
            conv.append(nn.GELU())
            in_ch = out_ch
        self.conv = nn.Sequential(*conv)
        d = cfg.latent_dim
        self.kv_proj = nn.Linear(in_ch, d)
        self.latents = nn.Parameter(torch.randn(cfg.n_latents, d) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d, cfg.n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, cfg.mlp_hidden), nn.GELU(), nn.Linear(cfg.mlp_hidden, d)
        )
        self.ffn_norm = nn.LayerNorm(d)
        p = float(cfg.dropout)
        self.det_head = nn.Sequential(
            nn.Linear(d, cfg.mlp_hidden), nn.GELU(), nn.Dropout(p), nn.Linear(cfg.mlp_hidden, 1)
        )
        self.attr_head = nn.Sequential(
            nn.Linear(d, cfg.mlp_hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(cfg.mlp_hidden, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (B, T, F) standardized window.
        b = x.shape[0]
        feats = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T, C)
        kv = self.kv_proj(feats)
        q = self.latents.unsqueeze(0).expand(b, -1, -1)
        attn, _ = self.cross_attn(q, kv, kv, need_weights=False)
        tokens = self.attn_norm(q + attn)
        tokens = self.ffn_norm(tokens + self.ffn(tokens))
        pooled = tokens.mean(dim=1)  # (B, D)
        return {"det": self.det_head(pooled).squeeze(-1), "attr": self.attr_head(pooled)}


class LearnedMonitorModel:
    """Trainable/loadable wrapper with numpy I/O + the feature standardizer (no online state)."""

    def __init__(
        self, cfg: MonitorNetConfig, standardizer: Standardizer, device: str = "cpu"
    ) -> None:
        self.cfg = cfg
        self.std = standardizer
        self.device = torch.device(device)
        self.net = MonitorNet(cfg).to(self.device)

    @torch.no_grad()
    def predict(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (hazard_prob (B,), attr_probs (B, N_CLASSES)) for raw ``(B, T, F)`` windows."""
        self.net.eval()
        w = np.asarray(windows, dtype=np.float64)
        if w.ndim == 2:
            w = w[None]
        x = torch.tensor(self.std.transform(w), dtype=torch.float32, device=self.device)
        out = self.net(x)
        prob = torch.sigmoid(out["det"]).cpu().numpy()
        attr = torch.softmax(out["attr"], dim=-1).cpu().numpy()
        return prob, attr

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path.with_suffix(".pt"))
        meta = {"cfg": self.cfg.__dict__, "std": self.std.to_dict()}
        path.with_suffix(".meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> LearnedMonitorModel:
        path = Path(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        c = meta["cfg"]
        cfg = MonitorNetConfig(
            window_len=int(c["window_len"]),
            conv_channels=tuple(c["conv_channels"]),
            conv_kernels=tuple(c["conv_kernels"]),
            latent_dim=int(c["latent_dim"]),
            n_latents=int(c["n_latents"]),
            n_heads=int(c["n_heads"]),
            mlp_hidden=int(c["mlp_hidden"]),
            dropout=float(c.get("dropout", 0.2)),
        )
        model = cls(cfg, Standardizer.from_dict(meta["std"]), device=device)
        state = torch.load(path.with_suffix(".pt"), map_location=model.device)
        model.net.load_state_dict(state)
        return model


class LearnedMonitor:
    """Online ~50 Hz hazard monitor — drop-in for ``RuleMonitor`` (same step/anomaly_score/reset).

    Pushes each step's measured-proprioception feature into the rolling 500 ms window, runs the
    detector, and fires a ``MonitorEvent`` when the hazard probability holds above ``threshold`` for
    ``debounce_steps`` (after an ``arm_s`` warmup, with a ``cooldown_s`` after a fire). The event's
    ``channel`` is the attribution head's argmax cause (e.g. ``"O5"``), ``value`` the hazard prob.
    """

    def __init__(
        self,
        model: LearnedMonitorModel,
        dt: float,
        *,
        threshold: float = 0.6,
        debounce_steps: int = 5,
        arm_s: float = 2.5,  # skip the startup accel ramp (deployment startup_straight_s); all
        cooldown_s: float = 6.0,  # operator hazards onset >= 3.0 s, so no detection is missed
        ema_alpha: float = 0.4,
    ) -> None:
        self._model = model
        self._dt = float(dt)
        self._threshold = float(threshold)
        self._debounce = int(debounce_steps)
        self._arm_s = float(arm_s)
        self._cooldown_s = float(cooldown_s)
        self._ema_alpha = float(ema_alpha)
        self._win = MonitorWindow(model.cfg.window_len)
        self.reset()

    def reset(self) -> None:
        self._win.reset()
        self._above = 0
        self._cooldown_until = 0.0
        self._anomaly = 0.0  # EMA-smoothed hazard probability (the firing statistic)
        self._raw = 0.0  # latest raw hazard probability
        self._last_attr = np.zeros(N_CLASSES, dtype=np.float64)
        self.events: list[MonitorEvent] = []

    @property
    def anomaly_score(self) -> float:
        """Latest hazard probability in [0, 1] — the firing statistic (≥ threshold fires)."""
        return float(self._anomaly)

    @property
    def channel_emas(self) -> dict[str, float]:
        """Telemetry: latest per-cause attribution probabilities (analogue of the rule EMAs)."""
        return {CLASS_NAMES[i]: float(self._last_attr[i]) for i in range(N_CLASSES)}

    @property
    def attr_probs(self) -> np.ndarray:
        """Latest (N_CLASSES,) attribution probabilities — argmax is the inferred cause."""
        return self._last_attr

    @property
    def raw_prob(self) -> float:
        """Latest RAW (un-EMA'd) hazard probability for this window."""
        return float(self._raw)

    def step(self, obs: Obs) -> MonitorEvent | None:
        self._win.push(obs)
        if not self._win.ready:
            return None
        prob, attr = self._model.predict(self._win.window())
        self._raw = float(prob[0])
        # EMA-smooth the hazard probability: an isolated false-positive window on clean ground is
        # damped, while a sustained hazard signal accumulates — the temporal half of the FPR fix.
        self._anomaly += self._ema_alpha * (self._raw - self._anomaly)
        self._last_attr = attr[0].astype(np.float64)
        if obs.t < self._arm_s or obs.t < self._cooldown_until:
            self._above = 0
            return None
        # Fire only when BOTH heads agree it is a hazard: the detector probability is over threshold
        # AND the attribution argmax is NOT "normal" (class 0). The agreement gate suppresses the
        # det/attr-disagreement false positives (a confident det spike the attribution calls clean).
        cls = int(np.argmax(self._last_attr))
        if self._anomaly >= self._threshold and cls != 0:
            self._above += 1
        else:
            self._above = 0
        if self._above >= self._debounce:
            channel = CLASS_NAMES[cls]
            event = MonitorEvent(
                t=float(obs.t),
                pos=obs.pos.copy(),
                channel=channel,
                value=self._anomaly,
                threshold=self._threshold,
                summary=(
                    f"[anomaly] t={obs.t:.2f}s pos=({obs.pos[0]:.2f},{obs.pos[1]:.2f}): "
                    f"hazard p={self._anomaly:.2f}≥{self._threshold:.2f}; attribution={channel}"
                ),
            )
            self.events.append(event)
            self._cooldown_until = obs.t + self._cooldown_s
            self._above = 0
            return event
        return None


DEPLOYED_MODEL_PATH = "outputs/monitor_learned/monitor"


def load_deployed_monitor(dt: float, *, model_path: str = DEPLOYED_MODEL_PATH,
                          device: str = "cpu", **overrides: object) -> LearnedMonitor:
    """Construct the deployed Kino-Monitor: load the trained checkpoint + the calibrated
    real-Go2 operating point (threshold 0.6, EMA 0.4, debounce 5, arm 2.5 s, det-attr gate;
    TPR 1.0 / FPR 0 over 132 positive + 80 negative real-Go2 lanes, outputs/monitor_learned/
    RESULTS.md). ``overrides`` adjust the operating point (e.g. for a high-recall data-collection
    pass). Raises FileNotFoundError if the checkpoint is missing — there is NO rule fallback."""
    from pathlib import Path

    if not Path(model_path).with_suffix(".pt").exists():
        raise FileNotFoundError(
            f"learned monitor checkpoint not found at {model_path}.pt — train it with "
            f"scripts/train_learned_monitor.py (the rule-based monitor has been removed)"
        )
    model = LearnedMonitorModel.load(model_path, device=device)
    return LearnedMonitor(model, dt=dt, **overrides)  # type: ignore[arg-type]
