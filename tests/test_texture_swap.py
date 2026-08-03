from __future__ import annotations

import torch

from kino_vla.eval.texture_swap import (
    audit_texture_swap_predictions,
    texture_swap_consistency_loss,
)


def _rows(*, flip: bool = False):
    rows = []
    for group_index, target in enumerate(("O1", "O2")):
        for view_index, material in enumerate(("tile", "concrete", "gravel")):
            prediction = "O3" if flip and group_index == 1 and view_index == 2 else target
            probabilities = {target: 0.95, "O3": 0.05}
            if prediction == "O3":
                probabilities = {target: 0.25, "O3": 0.75}
            rows.append(
                {
                    "texture_swap_group_id": f"group_{group_index}",
                    "appearance_view_id": f"view_{view_index}",
                    "target_label": target,
                    "predicted_label": prediction,
                    "material_family": material,
                    "probabilities": probabilities,
                }
            )
    return rows


def test_texture_swap_prediction_audit_passes_stable_predictions() -> None:
    audit = audit_texture_swap_predictions(_rows())
    assert audit.passed, audit
    assert audit.metrics["group_hard_prediction_consistency"] == 1.0
    assert audit.metrics["texture_flip_rate"] == 0.0
    assert audit.metrics["attribution_accuracy"] == 1.0


def test_texture_swap_prediction_audit_rejects_visual_flip() -> None:
    audit = audit_texture_swap_predictions(_rows(flip=True))
    assert not audit.passed
    assert "group_hard_consistency_below_threshold" in audit.issues
    assert audit.metrics["texture_flip_rate"] == 0.5


def test_texture_swap_consistency_loss_is_zero_only_for_equal_group_distributions() -> None:
    identical = torch.tensor([[4.0, 0.0], [4.0, 0.0], [4.0, 0.0]])
    changed = torch.tensor([[4.0, 0.0], [0.0, 4.0], [4.0, 0.0]])
    groups = ["same", "same", "same"]
    assert texture_swap_consistency_loss(identical, groups).item() == 0.0
    assert texture_swap_consistency_loss(changed, groups).item() > 0.1
