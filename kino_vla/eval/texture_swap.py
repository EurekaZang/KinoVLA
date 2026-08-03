"""Texture-swap consistency loss and publication audit for Kino-Fail.

Rows in one ``texture_swap_group_id`` share physics, proprioception, telemetry, camera pose and
timestamp; only the terrain appearance changes.  Prediction changes inside such a group therefore
measure a visual shortcut rather than legitimate sensitivity to a different physical trajectory.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class TextureSwapAudit:
    """Audited metrics and hard-gate result for texture-swapped predictions."""

    passed: bool
    issues: tuple[str, ...]
    metrics: dict[str, Any]


def texture_swap_consistency_loss(logits: Tensor, group_ids: Sequence[str]) -> Tensor:
    """Return the mean generalized-JS loss across same-physics appearance views.

    ``logits`` is a ``[batch, classes]`` torch tensor.  Groups represented by fewer than two
    views contribute nothing, which makes the helper safe for ordinary mixed mini-batches.
    """
    import torch

    if logits.ndim != 2 or logits.shape[0] != len(group_ids):
        raise ValueError("logits must be [batch, classes] and align with group_ids")
    probabilities = torch.softmax(logits, dim=-1)
    losses = []
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        by_group[str(group_id)].append(index)
    epsilon = torch.finfo(probabilities.dtype).eps
    for indices in by_group.values():
        if len(indices) < 2:
            continue
        group = probabilities[indices]
        mean = group.mean(dim=0, keepdim=True)
        losses.append(
            (group * ((group + epsilon).log() - (mean + epsilon).log())).sum(dim=1).mean()
        )
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def _normalized_distribution(value: object) -> dict[str, float] | None:
    if not isinstance(value, Mapping) or not value:
        return None
    try:
        result = {str(key): float(probability) for key, probability in value.items()}
    except (TypeError, ValueError):
        return None
    total = sum(result.values())
    if (
        not math.isfinite(total)
        or total <= 0.0
        or any(not math.isfinite(item) or item < 0.0 for item in result.values())
    ):
        return None
    return {key: probability / total for key, probability in result.items()}


def _jsd(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    labels = sorted(set(left) | set(right))
    p = np.asarray([left.get(label, 0.0) for label in labels], dtype=np.float64)
    q = np.asarray([right.get(label, 0.0) for label in labels], dtype=np.float64)
    middle = 0.5 * (p + q)

    def kl(value: np.ndarray) -> float:
        mask = value > 0.0
        return float(np.sum(value[mask] * np.log(value[mask] / middle[mask])))

    return (0.5 * kl(p) + 0.5 * kl(q)) / math.log(2.0)


def audit_texture_swap_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_group_hard_consistency: float = 0.95,
    min_pairwise_agreement: float = 0.95,
    max_mean_pairwise_jsd: float = 0.05,
) -> TextureSwapAudit:
    """Measure prediction stability under appearance-only terrain interventions."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    malformed_rows: list[int] = []
    for index, row in enumerate(rows):
        required = (
            "texture_swap_group_id",
            "appearance_view_id",
            "target_label",
            "predicted_label",
            "material_family",
        )
        if any(key not in row for key in required):
            malformed_rows.append(index)
            continue
        groups[str(row["texture_swap_group_id"])].append(row)

    malformed_groups: list[str] = []
    group_consistent = 0
    group_correct_consistent = 0
    correct = 0
    pair_agreements = 0
    pair_count = 0
    probability_jsds: list[float] = []
    probability_pairs = 0
    material_results: dict[str, list[bool]] = defaultdict(list)
    for group_id, group in groups.items():
        view_ids = [str(row["appearance_view_id"]) for row in group]
        targets = {str(row["target_label"]) for row in group}
        if len(group) < 2 or len(set(view_ids)) != len(view_ids) or len(targets) != 1:
            malformed_groups.append(group_id)
            continue
        target = next(iter(targets))
        predictions = [str(row["predicted_label"]) for row in group]
        is_consistent = len(set(predictions)) == 1
        group_consistent += int(is_consistent)
        group_correct_consistent += int(is_consistent and predictions[0] == target)
        for row, prediction in zip(group, predictions, strict=True):
            is_correct = prediction == target
            correct += int(is_correct)
            material_results[str(row["material_family"])].append(is_correct)
        for left_index, right_index in combinations(range(len(group)), 2):
            pair_count += 1
            pair_agreements += int(predictions[left_index] == predictions[right_index])
            left = _normalized_distribution(group[left_index].get("probabilities"))
            right = _normalized_distribution(group[right_index].get("probabilities"))
            if left is not None and right is not None:
                probability_pairs += 1
                probability_jsds.append(_jsd(left, right))

    valid_groups = len(groups) - len(malformed_groups)
    valid_rows = sum(len(group) for key, group in groups.items() if key not in malformed_groups)
    group_hard_consistency = group_consistent / valid_groups if valid_groups else 0.0
    group_correct_consistency = group_correct_consistent / valid_groups if valid_groups else 0.0
    pairwise_agreement = pair_agreements / pair_count if pair_count else 0.0
    accuracy = correct / valid_rows if valid_rows else 0.0
    mean_jsd = float(np.mean(probability_jsds)) if probability_jsds else None
    per_material_accuracy = {
        material: sum(values) / len(values) for material, values in sorted(material_results.items())
    }
    worst_material_accuracy = min(per_material_accuracy.values(), default=0.0)
    material_accuracy_gap = (
        max(per_material_accuracy.values()) - worst_material_accuracy
        if per_material_accuracy
        else 0.0
    )
    issues: list[str] = []
    if malformed_rows:
        issues.append("malformed_prediction_rows")
    if malformed_groups:
        issues.append("malformed_texture_swap_groups")
    if not valid_groups:
        issues.append("no_valid_texture_swap_groups")
    if group_hard_consistency < min_group_hard_consistency:
        issues.append("group_hard_consistency_below_threshold")
    if pairwise_agreement < min_pairwise_agreement:
        issues.append("pairwise_agreement_below_threshold")
    if probability_pairs and mean_jsd is not None and mean_jsd > max_mean_pairwise_jsd:
        issues.append("probability_jsd_above_threshold")
    metrics = {
        "rows": len(rows),
        "valid_rows": valid_rows,
        "groups": len(groups),
        "valid_groups": valid_groups,
        "malformed_row_indices": malformed_rows[:100],
        "malformed_group_ids": malformed_groups[:100],
        "group_hard_prediction_consistency": group_hard_consistency,
        "group_correct_consistency": group_correct_consistency,
        "texture_flip_rate": 1.0 - group_hard_consistency,
        "pairwise_prediction_agreement": pairwise_agreement,
        "attribution_accuracy": accuracy,
        "probability_pairs": probability_pairs,
        "mean_pairwise_probability_jsd": mean_jsd,
        "per_material_accuracy": per_material_accuracy,
        "worst_material_accuracy": worst_material_accuracy,
        "material_accuracy_gap": material_accuracy_gap,
        "thresholds": {
            "min_group_hard_consistency": min_group_hard_consistency,
            "min_pairwise_agreement": min_pairwise_agreement,
            "max_mean_pairwise_jsd": max_mean_pairwise_jsd,
        },
    }
    return TextureSwapAudit(passed=not issues, issues=tuple(issues), metrics=metrics)
