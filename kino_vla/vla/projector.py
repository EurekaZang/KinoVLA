"""Kino-Projector: 500 ms proprioception → continuous Kino-Tokens in the VLM embedding space.

The latent route (spec §3 route B, §4) injects the high-frequency physics the text route quantizes
away. This module is the §4 tokenizer (1D-CNN over the window + a Perceiver-Resampler with K
learned queries → K fixed-length Kino-Tokens) followed by the **Kino-Projector** MLP that lifts
each token into the VLM hidden size (Qwen3-VL: 2560). The model splices these K soft-token
embeddings in at the ``<kino_tokens>`` placeholder (:mod:`kino_vla.vla.model`).

A privileged-θ regression head keeps the latent physically grounded (spec §4 main supervision):
during Kino-SFT a small auxiliary MSE on the snapshot's privileged θ anchors the tokens to real
physics rather than letting them drift to whatever minimizes next-token loss — so the latent stays
interpretable (the spec's payoff for route B over a black-box adapter). The architecture mirrors
``kino_vla.tokens.extractor._KinoNet`` and can be initialized from a trained M4 extractor.

Torch, but CPU-runnable and tiny (~0.2 M params) — its forward is unit-tested on the CI machine.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from kino_vla.tokens.features import N_FEATURES, N_TARGETS


class KinoProjector(nn.Module):
    """1D-CNN + Perceiver-Resampler tokenizer + projection MLP into the VLM embedding space."""

    def __init__(
        self,
        vlm_dim: int,
        *,
        conv_channels: tuple[int, ...] = (48, 48),
        conv_kernels: tuple[int, ...] = (5, 3),
        latent_dim: int = 96,
        n_latents: int = 6,
        n_heads: int = 4,
        mlp_hidden: int = 96,
        proj_hidden: int = 512,
    ) -> None:
        super().__init__()
        if len(conv_channels) != len(conv_kernels):
            raise ValueError("conv_channels and conv_kernels must have equal length")
        self.vlm_dim = int(vlm_dim)
        self.latent_dim = int(latent_dim)
        self.n_latents = int(n_latents)

        conv: list[nn.Module] = []
        in_ch = N_FEATURES
        for out_ch, k in zip(conv_channels, conv_kernels, strict=True):
            conv.append(nn.Conv1d(in_ch, int(out_ch), kernel_size=int(k), padding=int(k) // 2))
            conv.append(nn.GELU())
            in_ch = int(out_ch)
        self.conv = nn.Sequential(*conv)

        self.kv_proj = nn.Linear(in_ch, self.latent_dim)
        self.latents = nn.Parameter(torch.randn(self.n_latents, self.latent_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(self.latent_dim, int(n_heads), batch_first=True)
        self.attn_norm = nn.LayerNorm(self.latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.latent_dim, int(mlp_hidden)),
            nn.GELU(),
            nn.Linear(int(mlp_hidden), self.latent_dim),
        )
        self.ffn_norm = nn.LayerNorm(self.latent_dim)

        # Privileged-θ head (spec §4 #1) — keeps the latent grounded; trained on target_theta.
        self.theta_head = nn.Sequential(
            nn.Linear(self.latent_dim, int(mlp_hidden)),
            nn.GELU(),
            nn.Linear(int(mlp_hidden), N_TARGETS),
        )
        # The Kino-Projector proper: each Kino-Token → one VLM-space soft token.
        self.proj = nn.Sequential(
            nn.Linear(self.latent_dim, int(proj_hidden)),
            nn.GELU(),
            nn.Linear(int(proj_hidden), self.vlm_dim),
        )
        # Feature standardizer (set from the dataset before training; spec §4 — shared at
        # train/eval/online so the window normalization is identical). Buffers so they move with
        # ``.to(device)`` and serialize with the state dict.
        self.register_buffer("feat_mean", torch.zeros(N_FEATURES))
        self.register_buffer("feat_std", torch.ones(N_FEATURES))

    def set_standardizer(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-3) -> None:
        """Install the per-channel window normalization (from the training-set statistics).

        The new buffers are placed on the module's current device so the standardizer can be set
        after ``.to(device)`` (the trainer fits stats post-load) without a device mismatch."""
        dev = self.feat_mean.device
        std_floored = np.maximum(np.asarray(std), eps)
        self.feat_mean = torch.tensor(np.asarray(mean), dtype=torch.float32, device=dev)
        self.feat_std = torch.tensor(std_floored, dtype=torch.float32, device=dev)

    def tokens(self, window_std: torch.Tensor) -> torch.Tensor:
        """(B, T, F) standardized window → (B, K, latent_dim) Kino-Tokens (the §4 tokenizer)."""
        b = window_std.shape[0]
        feats = self.conv(window_std.transpose(1, 2)).transpose(1, 2)  # (B, T, C)
        kv = self.kv_proj(feats)
        q = self.latents.unsqueeze(0).expand(b, -1, -1)
        attn, _ = self.cross_attn(q, kv, kv, need_weights=False)
        tok = self.attn_norm(q + attn)
        return self.ffn_norm(tok + self.ffn(tok))

    def forward(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, F) RAW window → (soft_tokens (B, K, vlm_dim), theta_pred (B, n_targets)).

        Standardization is applied internally (with the installed buffers), so callers pass the
        raw proprioception window exactly as logged.
        """
        window_std = (window - self.feat_mean) / self.feat_std
        tok = self.tokens(window_std)
        theta = self.theta_head(tok.mean(dim=1))
        soft = self.proj(tok)
        return soft, theta

    @property
    def n_soft_tokens(self) -> int:
        return self.n_latents
