"""Descriptor-level implementations of established multimodal fusion families.

The modules in this file consume one visual descriptor and one proprioceptive
descriptor per registered observation.  They intentionally do not accept any
benchmark metadata, such as scene, material, operator, battery, or conflict
cell identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FusionDimensions:
    """Input and output dimensions needed to reconstruct a fusion model."""

    visual: int
    proprio: int
    classes: int


class _Dock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class ConcatMLP(nn.Module):
    """Neural concatenation control sharing the common descriptor interface."""

    def __init__(self, dims: FusionDimensions, hidden: int = 128) -> None:
        super().__init__()
        self.visual = _Dock(dims.visual, hidden)
        self.proprio = _Dock(dims.proprio, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, dims.classes),
        )

    def forward(self, visual: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat([self.visual(visual), self.proprio(proprio)], dim=1))


class TensorFusionNetwork(nn.Module):
    """Two-modality TFN using an augmented outer product."""

    def __init__(self, dims: FusionDimensions, projection: int = 32, hidden: int = 128) -> None:
        super().__init__()
        self.visual = nn.Sequential(nn.Linear(dims.visual, projection), nn.Tanh())
        self.proprio = nn.Sequential(nn.Linear(dims.proprio, projection), nn.Tanh())
        fused = (projection + 1) ** 2
        self.classifier = nn.Sequential(
            nn.Linear(fused, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, dims.classes),
        )

    def forward(self, visual: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        visual_h = self.visual(visual)
        proprio_h = self.proprio(proprio)
        ones = torch.ones((len(visual_h), 1), dtype=visual_h.dtype, device=visual_h.device)
        visual_h = torch.cat([visual_h, ones], dim=1)
        proprio_h = torch.cat([proprio_h, ones], dim=1)
        outer = torch.bmm(visual_h.unsqueeze(2), proprio_h.unsqueeze(1)).flatten(1)
        return self.classifier(outer)


class LowRankMultimodalFusion(nn.Module):
    """LMF factorization of the augmented visual--proprio tensor."""

    def __init__(
        self,
        dims: FusionDimensions,
        rank: int = 4,
        fusion_dim: int = 128,
    ) -> None:
        super().__init__()
        self.visual_factors = nn.Parameter(torch.empty(rank, dims.visual + 1, fusion_dim))
        self.proprio_factors = nn.Parameter(torch.empty(rank, dims.proprio + 1, fusion_dim))
        self.fusion_bias = nn.Parameter(torch.zeros(fusion_dim))
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, dims.classes),
        )
        nn.init.xavier_normal_(self.visual_factors)
        nn.init.xavier_normal_(self.proprio_factors)

    def forward(self, visual: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((len(visual), 1), dtype=visual.dtype, device=visual.device)
        visual_augmented = torch.cat([visual, ones], dim=1)
        proprio_augmented = torch.cat([proprio, ones], dim=1)
        visual_factor = torch.einsum("nd,rdh->nrh", visual_augmented, self.visual_factors)
        proprio_factor = torch.einsum(
            "nd,rdh->nrh", proprio_augmented, self.proprio_factors
        )
        fused = (visual_factor * proprio_factor).sum(dim=1) + self.fusion_bias
        return self.classifier(fused)


class GatedMultimodalUnit(nn.Module):
    """Two-input GMU with a learned multiplicative authority gate."""

    def __init__(self, dims: FusionDimensions, hidden: int = 128) -> None:
        super().__init__()
        self.visual = nn.Linear(dims.visual, hidden)
        self.proprio = nn.Linear(dims.proprio, hidden)
        self.gate = nn.Linear(dims.visual + dims.proprio, hidden)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, dims.classes),
        )

    def forward(self, visual: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        visual_h = torch.tanh(self.visual(visual))
        proprio_h = torch.tanh(self.proprio(proprio))
        gate = torch.sigmoid(self.gate(torch.cat([visual, proprio], dim=1)))
        fused = gate * visual_h + (1.0 - gate) * proprio_h
        return self.classifier(fused)


class EmbraceNet(nn.Module):
    """Two-modality EmbraceNet with stochastic coordinate-wise embracement."""

    def __init__(self, dims: FusionDimensions, hidden: int = 128) -> None:
        super().__init__()
        self.visual = _Dock(dims.visual, hidden)
        self.proprio = _Dock(dims.proprio, hidden)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, dims.classes),
        )

    def forward(
        self,
        visual: torch.Tensor,
        proprio: torch.Tensor,
        *,
        stochastic: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        visual_h = self.visual(visual)
        proprio_h = self.proprio(proprio)
        use_stochastic = self.training if stochastic is None else bool(stochastic)
        if use_stochastic:
            selection = torch.rand(
                visual_h.shape,
                dtype=visual_h.dtype,
                device=visual_h.device,
                generator=generator,
            ) < 0.5
            fused = torch.where(selection, visual_h, proprio_h)
        else:
            fused = 0.5 * (visual_h + proprio_h)
        return self.classifier(fused)


class ModDropFusion(nn.Module):
    """ModDrop training over visual and proprioceptive descriptor streams."""

    def __init__(
        self,
        dims: FusionDimensions,
        hidden: int = 128,
        drop_probability: float = 0.2,
    ) -> None:
        super().__init__()
        self.visual = _Dock(dims.visual, hidden)
        self.proprio = _Dock(dims.proprio, hidden)
        self.drop_probability = float(drop_probability)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, dims.classes),
        )

    def forward(self, visual: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        visual_h = self.visual(visual)
        proprio_h = self.proprio(proprio)
        if self.training and self.drop_probability > 0.0:
            keep_visual = (
                torch.rand((len(visual_h), 1), device=visual_h.device)
                >= self.drop_probability
            )
            keep_proprio = (
                torch.rand((len(proprio_h), 1), device=proprio_h.device)
                >= self.drop_probability
            )
            neither = ~(keep_visual | keep_proprio)
            rescue_visual = torch.rand((len(visual_h), 1), device=visual_h.device) < 0.5
            keep_visual = keep_visual | (neither & rescue_visual)
            keep_proprio = keep_proprio | (neither & ~rescue_visual)
            visual_h = visual_h * keep_visual.to(visual_h.dtype)
            proprio_h = proprio_h * keep_proprio.to(proprio_h.dtype)
        return self.classifier(torch.cat([visual_h, proprio_h], dim=1))


METHODS = (
    "concat_mlp",
    "tfn",
    "lmf",
    "gmu",
    "embracenet",
    "moddrop",
)


def build_fusion_model(name: str, dims: FusionDimensions) -> nn.Module:
    """Construct a registered model without benchmark-specific metadata."""

    if name == "concat_mlp":
        return ConcatMLP(dims)
    if name == "tfn":
        return TensorFusionNetwork(dims)
    if name == "lmf":
        return LowRankMultimodalFusion(dims)
    if name == "gmu":
        return GatedMultimodalUnit(dims)
    if name == "embracenet":
        return EmbraceNet(dims)
    if name == "moddrop":
        return ModDropFusion(dims)
    raise KeyError(f"unknown multimodal fusion method: {name}")
