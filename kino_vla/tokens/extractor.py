"""Kino-Tokens extractor: privileged physics distillation (spec §4).

Backbone (spec §4 #3): a 1D-CNN over the 500 ms proprioception window captures local
mutations, then a Perceiver Resampler (``K`` learned latent queries cross-attend to
the CNN feature sequence) compresses it to ``K`` fixed-length tokens. Three heads hang
off the pooled latent:

  #1 regression head → privileged θ (μ, payload, effort-scale, support) — the *main*
     supervision (teacher-student distillation); μ̂ feeds the CBF shield (spec §6.5).
  #2 Kino-Text contrastive head → InfoNCE alignment to a learned per-regime text
     prototype — *auxiliary* semantic readability only (spec §4 #2).
  #9 reconstruction head → re-decodes the (standardized) window; its residual is the
     OOD score (spec §9: "manifold deviation" made a concrete physics-prediction
     residual, computable online without ground truth).

Anomaly-gated injection (spec §4 #4) is realized by the coupler, which only runs the
extractor when the Kino-Monitor is aroused (steady-state tokens are effectively zero —
the compute is simply skipped); ``gate_tokens`` exposes the explicit zeroing for the
M7 VLA injection route.

torch-only module: imported lazily so the torch-free CI / demo path never touches it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from kino_vla.tokens.dataset import TokenDataset
from kino_vla.tokens.features import (
    MU_INDEX,
    N_FEATURES,
    N_TARGETS,
    Standardizer,
)
from kino_vla.tokens.semantics import N_REGIMES
from kino_vla.utils.config import Config


@dataclass(frozen=True)
class ExtractResult:
    """One window's extractor outputs (numpy, batched on the leading axis)."""

    theta_hat: np.ndarray  # (B, n_targets) physics in raw units
    mu_hat: np.ndarray  # (B,) friction estimate (convenience view of theta_hat[:, MU_INDEX])
    ood_score: np.ndarray  # (B,) reconstruction-residual OOD score (spec §9)
    regime_logits: np.ndarray  # (B, n_regimes) Kino-Text prototype similarities
    tokens: np.ndarray  # (B, K, D) fixed-length Kino-Tokens (pre-gate)


class _KinoNet(nn.Module):
    """1D-CNN + Perceiver Resampler + three heads (the raw torch module)."""

    def __init__(self, window_len: int, model_cfg: Config) -> None:
        super().__init__()
        channels = [int(c) for c in model_cfg.conv_channels]
        kernels = [int(k) for k in model_cfg.conv_kernels]
        if len(channels) != len(kernels):
            raise ValueError("conv_channels and conv_kernels must have equal length")
        self.window_len = int(window_len)
        self.dim = int(model_cfg.latent_dim)
        self.n_latents = int(model_cfg.n_latents)
        n_heads = int(model_cfg.n_heads)
        hidden = int(model_cfg.mlp_hidden)

        conv: list[nn.Module] = []
        in_ch = N_FEATURES
        for out_ch, k in zip(channels, kernels, strict=True):
            conv.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2))
            conv.append(nn.GELU())
            in_ch = out_ch
        self.conv = nn.Sequential(*conv)

        # Perceiver Resampler: K learned queries cross-attend to the CNN sequence.
        self.kv_proj = nn.Linear(in_ch, self.dim)
        self.latents = nn.Parameter(torch.randn(self.n_latents, self.dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(self.dim, n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, hidden), nn.GELU(), nn.Linear(hidden, self.dim)
        )
        self.ffn_norm = nn.LayerNorm(self.dim)

        self.reg_head = nn.Sequential(
            nn.Linear(self.dim, hidden), nn.GELU(), nn.Linear(hidden, N_TARGETS)
        )
        self.recon_head = nn.Sequential(
            nn.Linear(self.dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.window_len * N_FEATURES),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.dim, hidden), nn.GELU(), nn.Linear(hidden, self.dim)
        )
        self.text_prototypes = nn.Parameter(torch.randn(N_REGIMES, self.dim) * 0.02)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (B, T, F) standardized window.
        b = x.shape[0]
        feats = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T, C)
        kv = self.kv_proj(feats)  # (B, T, D)
        q = self.latents.unsqueeze(0).expand(b, -1, -1)  # (B, K, D)
        attn, _ = self.cross_attn(q, kv, kv, need_weights=False)
        tokens = self.attn_norm(q + attn)
        tokens = self.ffn_norm(tokens + self.ffn(tokens))  # (B, K, D)
        pooled = tokens.mean(dim=1)  # (B, D)

        theta_std = self.reg_head(pooled)  # (B, n_targets), standardized
        recon = self.recon_head(pooled).reshape(b, self.window_len, N_FEATURES)
        text_embed = nn.functional.normalize(self.text_proj(pooled), dim=-1)
        protos = nn.functional.normalize(self.text_prototypes, dim=-1)
        regime_logits = text_embed @ protos.t()  # (B, n_regimes) cosine similarity
        return {
            "tokens": tokens,
            "theta_std": theta_std,
            "recon": recon,
            "regime_logits": regime_logits,
        }


class Extractor:
    """Trainable wrapper around ``_KinoNet`` with numpy I/O, standardizers, and the
    spec §9 OOD machinery (reconstruction residual + a latent-Mahalanobis fallback)."""

    def __init__(self, cfg: Config, device: str = "cpu") -> None:
        self.cfg = cfg
        self.window_len = int(
            round(float(cfg.window.window_ms) * 1e-3 * float(cfg.window.control_hz))
        )
        self.device = torch.device(device)
        self.net = _KinoNet(self.window_len, cfg.model).to(self.device)
        self.feat_std: Standardizer | None = None
        self.target_std: Standardizer | None = None
        # Latent Gaussian for the Mahalanobis OOD fallback (fit at end of training).
        self._lat_mean: np.ndarray | None = None
        self._lat_prec: np.ndarray | None = None

    # ------------------------------------------------------------------ training
    def fit(self, train: TokenDataset, log: bool = False) -> dict[str, list[float]]:
        tc = self.cfg.train
        eps = float(tc.std_eps)
        self.feat_std = Standardizer.fit(train.windows, eps)
        self.target_std = Standardizer.fit(train.targets, eps)

        x = torch.tensor(self.feat_std.transform(train.windows), dtype=torch.float32)
        y = torch.tensor(self.target_std.transform(train.targets), dtype=torch.float32)
        r = torch.tensor(train.regimes, dtype=torch.long)

        torch.manual_seed(int(tc.seed))
        opt = torch.optim.Adam(
            self.net.parameters(), lr=float(tc.lr), weight_decay=float(tc.weight_decay)
        )
        n = x.shape[0]
        batch = int(tc.batch_size)
        gen = torch.Generator().manual_seed(int(tc.seed))
        history: dict[str, list[float]] = {"loss": [], "reg": [], "recon": [], "contrast": []}
        self.net.train()
        for epoch in range(int(tc.epochs)):
            perm = torch.randperm(n, generator=gen)
            tot = {"loss": 0.0, "reg": 0.0, "recon": 0.0, "contrast": 0.0}
            for s in range(0, n, batch):
                idx = perm[s : s + batch]
                xb = x[idx].to(self.device)
                yb = y[idx].to(self.device)
                rb = r[idx].to(self.device)
                out = self.net(xb)
                reg = nn.functional.mse_loss(out["theta_std"], yb)
                recon = nn.functional.mse_loss(out["recon"], xb)
                contrast = nn.functional.cross_entropy(
                    out["regime_logits"] / float(tc.contrastive_temp), rb
                )
                loss = (
                    float(tc.w_regression) * reg
                    + float(tc.w_reconstruction) * recon
                    + float(tc.w_contrastive) * contrast
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                bs = xb.shape[0]
                tot["loss"] += float(loss) * bs
                tot["reg"] += float(reg) * bs
                tot["recon"] += float(recon) * bs
                tot["contrast"] += float(contrast) * bs
            for key in tot:
                history[key].append(tot[key] / n)
            if log and (epoch % 20 == 0 or epoch == int(tc.epochs) - 1):
                print(
                    f"  epoch {epoch:3d}  loss={history['loss'][-1]:.4f}  "
                    f"reg={history['reg'][-1]:.4f}  recon={history['recon'][-1]:.4f}  "
                    f"contrast={history['contrast'][-1]:.4f}"
                )
        self._fit_latent_gaussian(x)
        return history

    def _fit_latent_gaussian(self, x_std_all: torch.Tensor) -> None:
        """Fit a Gaussian over training pooled latents (Mahalanobis OOD fallback)."""
        self.net.eval()
        with torch.no_grad():
            lat = []
            for s in range(0, x_std_all.shape[0], 1024):
                out = self.net(x_std_all[s : s + 1024].to(self.device))
                lat.append(out["tokens"].mean(dim=1).cpu().numpy())
        lat_all = np.concatenate(lat, axis=0)
        self._lat_mean = lat_all.mean(axis=0)
        cov = np.cov(lat_all, rowvar=False) + 1e-3 * np.eye(lat_all.shape[1])
        self._lat_prec = np.linalg.inv(cov)

    # ----------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, windows: np.ndarray) -> ExtractResult:
        """Run the extractor on a batch of raw ``(B, T, F)`` windows."""
        if self.feat_std is None or self.target_std is None:
            raise RuntimeError("extractor is not fitted/loaded (no standardizers)")
        self.net.eval()
        windows = np.asarray(windows, dtype=np.float64)
        if windows.ndim == 2:
            windows = windows[None]
        x_std = self.feat_std.transform(windows)
        xt = torch.tensor(x_std, dtype=torch.float32, device=self.device)
        out = self.net(xt)
        theta_std = out["theta_std"].cpu().numpy()
        theta_hat = self.target_std.inverse(theta_std)
        recon = out["recon"].cpu().numpy()
        ood = np.mean((recon - x_std) ** 2, axis=(1, 2))  # reconstruction residual
        return ExtractResult(
            theta_hat=theta_hat,
            mu_hat=theta_hat[:, MU_INDEX],
            ood_score=ood,
            regime_logits=out["regime_logits"].cpu().numpy(),
            tokens=out["tokens"].cpu().numpy(),
        )

    def predict_mu(self, window: np.ndarray) -> float:
        """μ̂ for a single window — the value handed to ``CbfShield.set_mu_estimate``."""
        return float(self.predict(window).mu_hat[0])

    def ood_mahalanobis(self, windows: np.ndarray) -> np.ndarray:
        """Latent-space Mahalanobis OOD distance (fallback detector, spec §9)."""
        if self._lat_mean is None or self._lat_prec is None:
            raise RuntimeError("latent Gaussian not fitted")
        tokens = self.predict(windows).tokens
        pooled = tokens.mean(axis=1)
        d = pooled - self._lat_mean
        return np.einsum("bi,ij,bj->b", d, self._lat_prec, d)

    @staticmethod
    def gate_tokens(tokens: np.ndarray, gate: float) -> np.ndarray:
        """Anomaly-gated injection (spec §4 #4): scale tokens by the monitor gate
        (0 at steady state ⇒ no reflection wakeup; 1 when aroused)."""
        return tokens * float(np.clip(gate, 0.0, 1.0))

    def inference_latency_ms(self, n: int = 200) -> float:
        """p99 single-window forward latency on this device (spec §6.9 budget)."""
        if self.feat_std is None:
            raise RuntimeError("extractor not fitted")
        w = np.zeros((1, self.window_len, N_FEATURES), dtype=np.float64)
        self.predict(w)  # warm up
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.predict(w)
            times.append((time.perf_counter() - t0) * 1e3)
        return float(np.percentile(times, 99))

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.feat_std is None or self.target_std is None:
            raise RuntimeError("refusing to save an unfitted extractor")
        torch.save(self.net.state_dict(), path.with_suffix(".pt"))
        meta = {
            "feat_std": self.feat_std.to_dict(),
            "target_std": self.target_std.to_dict(),
            "lat_mean": None if self._lat_mean is None else self._lat_mean.tolist(),
            "lat_prec": None if self._lat_prec is None else self._lat_prec.tolist(),
        }
        path.with_suffix(".meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, cfg: Config, path: str | Path, device: str = "cpu") -> Extractor:
        path = Path(path)
        ex = cls(cfg, device=device)
        state = torch.load(path.with_suffix(".pt"), map_location=ex.device)
        ex.net.load_state_dict(state)
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        ex.feat_std = Standardizer.from_dict(meta["feat_std"])
        ex.target_std = Standardizer.from_dict(meta["target_std"])
        if meta.get("lat_mean") is not None:
            ex._lat_mean = np.asarray(meta["lat_mean"], dtype=np.float64)
            ex._lat_prec = np.asarray(meta["lat_prec"], dtype=np.float64)
        return ex
