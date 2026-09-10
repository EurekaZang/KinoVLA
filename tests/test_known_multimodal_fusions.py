from __future__ import annotations

import pytest
import torch

from kino_vla.eval.known_multimodal_fusions import (
    METHODS,
    EmbraceNet,
    FusionDimensions,
    build_fusion_model,
)


@pytest.mark.parametrize("method", METHODS)
def test_fusion_model_shape_and_gradient(method: str) -> None:
    torch.manual_seed(7)
    dims = FusionDimensions(visual=13, proprio=7, classes=4)
    model = build_fusion_model(method, dims)
    visual = torch.randn(8, dims.visual)
    proprio = torch.randn(8, dims.proprio)
    logits = model(visual, proprio)
    assert logits.shape == (8, dims.classes)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_embracenet_deterministic_inference() -> None:
    torch.manual_seed(11)
    dims = FusionDimensions(visual=9, proprio=5, classes=3)
    model = EmbraceNet(dims).eval()
    visual = torch.randn(6, dims.visual)
    proprio = torch.randn(6, dims.proprio)
    first = model(visual, proprio)
    second = model(visual, proprio)
    torch.testing.assert_close(first, second)


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(KeyError):
        build_fusion_model("battery_oracle", FusionDimensions(4, 4, 2))
