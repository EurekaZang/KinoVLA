"""Evidence-routed structured recovery policy used by the A3 successor.

The model keeps the successful bidirectional-conflict VLA as a frozen expert and learns three
small observable experts around it: RGB material evidence, temporal proprioceptive evidence, and
joint evidence.  A router combines their category logits while a separate intervention head owns
the nominal/``continue`` decision.  Scenario ids, operator ids, appearance ids, taxonomy cells,
and privileged physics are never inputs at deployment.

The module deliberately separates the neural decision from serialization.  The network predicts a
closed category and an intervention bit; :class:`RoutedEvidencePolicy` applies the frozen recovery
registry and emits the same schema-valid :class:`~kino_vla.vla.output.ParsedDecision` as the VLA.
This prevents free-form rationale tokens from dominating the safety-critical decision loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kino_vla.data.schema import CoTAnnotation, RecoveryPrimitive, Snapshot
from kino_vla.eval.material_support import (
    dominant_chromatic_rgb,
    matched_material_pair_logits,
    material_evidence_features,
)
from kino_vla.eval.suite_sem import _default_params
from kino_vla.tokens.features import TARGET_SCHEMA
from kino_vla.vla.output import ParsedDecision

EXPERT_NAMES: tuple[str, ...] = (
    "visual",
    "material_pair",
    "proprio",
    "joint",
    "conflict_vla",
)
ROUTER_REGIMES: tuple[str, ...] = EXPERT_NAMES
FORBIDDEN_DEPLOYMENT_INPUTS: tuple[str, ...] = (
    "operator",
    "operator_name",
    "scenario",
    "scenario_id",
    "appearance_id",
    "appearance_split",
    "taxonomy_cell",
    "truth",
    "theta",
    "privileged_theta",
)


def category_params(category: str, primitive: str) -> dict[str, Any]:
    """Registry-compatible deterministic parameters for a structured category decision."""
    params = dict(_default_params(primitive))
    if primitive == "Switch_Gait":
        params["mode"] = "crawl" if category == "effort_decay" else "high_step"
    if primitive == "Hold_and_Request" and category == "overload":
        params["reason"] = "payload exceeds safe capacity"
    if primitive == "continue":
        params = {}
    return params


def regime_target(cell: str, t3_sub: str, truth: str) -> int:
    """Training-only expert target derived from the pre-registered evidence regime.

    This label supervises the router during fitting and is not available to the deployed policy.
    """
    # All T3 rows share the same bidirectional-conflict regime.  The separate intervention gate
    # owns nominal/continue, so the category router need not (and cannot) distinguish a reverse
    # row from a looks-safe row using their deliberately matched proprioception.
    if cell == "T3":
        return EXPERT_NAMES.index("conflict_vla")
    if truth == "nominal":
        return EXPERT_NAMES.index("joint")
    if cell == "T2":
        return EXPERT_NAMES.index("material_pair")
    if cell == "T1" and truth == "compliant_terrain":
        return EXPERT_NAMES.index("material_pair")
    if cell == "T4":
        return EXPERT_NAMES.index("joint")
    return EXPERT_NAMES.index("joint")


@dataclass(frozen=True)
class ObservableFeatures:
    """Only the tensors that are legal inputs to the deployed router."""

    visual: np.ndarray
    proprio: np.ndarray
    material_logits: np.ndarray
    conflict_logits: np.ndarray


class EvidenceRoutedNetwork:  # constructed lazily as a torch.nn.Module below
    """Factory namespace that keeps importing this module possible without torch installed."""

    @staticmethod
    def build(
        *,
        visual_dim: int,
        proprio_dim: int,
        n_categories: int,
        hidden: int = 128,
        material_feature_dim: int = 16,
    ) -> Any:
        import torch
        from torch import nn

        class _TemporalEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(proprio_dim, 64, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.Conv1d(64, 64, kernel_size=3, padding=1),
                    nn.GELU(),
                )
                self.project = nn.Sequential(
                    nn.Linear(128, hidden), nn.GELU(), nn.LayerNorm(hidden)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.conv(x.transpose(1, 2))
                pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
                return self.project(pooled)

        class _Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.visual_encoder = nn.Sequential(
                    nn.Linear(visual_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
                )
                self.proprio_encoder = _TemporalEncoder()
                self.visual_head = nn.Linear(hidden, n_categories)
                self.proprio_head = nn.Linear(hidden, n_categories)
                self.joint_encoder = nn.Sequential(
                    nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden)
                )
                self.joint_head = nn.Linear(hidden, n_categories)
                self.router = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, len(EXPERT_NAMES)),
                )
                self.intervention_head = nn.Sequential(
                    nn.Linear(hidden + material_feature_dim, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, 1),
                )
                self.theta_head = nn.Linear(hidden, len(TARGET_SCHEMA))

            def forward(
                self,
                visual: torch.Tensor,
                proprio: torch.Tensor,
                material_logits: torch.Tensor,
                conflict_logits: torch.Tensor,
            ) -> dict[str, torch.Tensor]:
                zv = self.visual_encoder(visual)
                zp = self.proprio_encoder(proprio)
                zj = self.joint_encoder(torch.cat([zv, zp], dim=-1))
                visual_logits = self.visual_head(zv)
                proprio_logits = self.proprio_head(zp)
                joint_logits = self.joint_head(zj)
                # Evidence responsibility is determined by dynamics, not by appearance.  In the
                # matched construction the train/test proprioception is identical while colour is
                # intentionally shifted; feeding visual features to this router would make expert
                # selection itself appearance-OOD.  Visual evidence remains fully available inside
                # the visual, material-pair, and joint experts.
                router_logits = self.router(zp)
                soft_router_weights = torch.softmax(router_logits, dim=-1)
                if self.training:
                    router_weights = soft_router_weights
                else:
                    router_weights = torch.nn.functional.one_hot(
                        soft_router_weights.argmax(dim=-1), num_classes=len(EXPERT_NAMES)
                    ).to(dtype=soft_router_weights.dtype)
                expert_logits = torch.stack(
                    [
                        visual_logits,
                        material_logits,
                        proprio_logits,
                        joint_logits,
                        conflict_logits,
                    ],
                    dim=1,
                )
                category_logits = (router_weights.unsqueeze(-1) * expert_logits).sum(dim=1)
                # A reverse conflict is deliberately proprio-indistinguishable from looks-safe.
                # Give the gate only the temporal state and the auditable low-dimensional material
                # summary, rather than the high-capacity CLIP embedding or category rationale.
                material_features = visual[:, -material_feature_dim:]
                intervention_logit = self.intervention_head(
                    torch.cat([zp, material_features], dim=-1)
                ).squeeze(-1)
                return {
                    "category_logits": category_logits,
                    "intervention_logit": intervention_logit,
                    "router_logits": router_logits,
                    "router_weights": router_weights,
                    "visual_logits": visual_logits,
                    "proprio_logits": proprio_logits,
                    "joint_logits": joint_logits,
                    "theta": self.theta_head(zp),
                }

        return _Network()


def predict_category(
    outputs: dict[str, Any],
    categories: list[str],
    *,
    continue_threshold: float,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Convert network outputs into explicit continue/intervene category decisions."""
    import torch

    logits = outputs["category_logits"].detach().float().cpu().clone()
    p_intervene = torch.sigmoid(outputs["intervention_logit"].detach().float().cpu()).numpy()
    nominal_idx = categories.index("nominal")
    predictions: list[str] = []
    for i, probability in enumerate(p_intervene):
        if float(probability) < float(continue_threshold):
            predictions.append("nominal")
            continue
        logits[i, nominal_idx] = -torch.inf
        predictions.append(categories[int(torch.argmax(logits[i]).item())])
    return predictions, p_intervene, outputs["router_weights"].detach().float().cpu().numpy()


class RoutedEvidencePolicy:
    """Deployment wrapper for a trained router and the frozen conflict-VLA expert."""

    def __init__(
        self,
        model: Any,
        visual_encoder: Any,
        conflict_policy: Any,
        *,
        categories: list[str],
        canonical: dict[str, str],
        proprio_mean: np.ndarray,
        proprio_std: np.ndarray,
        continue_threshold: float,
        device: str = "cpu",
        conflict_logit_scale: float = 8.0,
        material_models: dict[str, dict[str, Any]] | None = None,
        material_classes: list[str] | None = None,
        material_pair_logit_scale: float = 8.0,
    ) -> None:
        import torch

        self._model = model.to(device).eval()
        self._visual_encoder = visual_encoder
        self._conflict_policy = conflict_policy
        self._categories = list(categories)
        self._canonical = dict(canonical)
        self._mean = np.asarray(proprio_mean, dtype=np.float32)
        self._std = np.maximum(np.asarray(proprio_std, dtype=np.float32), 1.0e-3)
        self._threshold = float(continue_threshold)
        self._device = torch.device(device)
        self._scale = float(conflict_logit_scale)
        self._material_models = material_models
        self._material_classes = tuple(material_classes or ())
        self._material_pair_scale = float(material_pair_logit_scale)

    def observable_features(self, snapshot: Snapshot) -> ObservableFeatures:
        rgb = np.asarray(snapshot.rgb[-1], dtype=np.float32)
        visual = np.asarray(self._visual_encoder.embed(rgb), dtype=np.float32)
        if self._material_models is not None:
            material_rgb = dominant_chromatic_rgb(rgb)
            material = material_evidence_features(
                material_rgb,
                self._material_models,
                classes=self._material_classes,
            )
            visual = np.concatenate([visual, material]).astype(np.float32)
            material_logits = matched_material_pair_logits(
                material_rgb,
                self._material_models,
                self._categories,
                scale=self._material_pair_scale,
            )
        else:
            material_logits = np.zeros(len(self._categories), dtype=np.float32)
        proprio = (
            (np.asarray(snapshot.proprio_window, dtype=np.float32) - self._mean) / self._std
        )
        decision = self._conflict_policy.decide(snapshot)
        conflict = np.zeros(len(self._categories), dtype=np.float32)
        if decision.ok and decision.attribution in self._categories:
            conflict[self._categories.index(str(decision.attribution))] = self._scale
        return ObservableFeatures(
            visual=visual,
            proprio=proprio,
            material_logits=material_logits,
            conflict_logits=conflict,
        )

    def decide(self, snapshot: Snapshot, map_note: str = "") -> ParsedDecision:  # noqa: ARG002
        import torch

        features = self.observable_features(snapshot)
        with torch.no_grad():
            out = self._model(
                torch.from_numpy(features.visual).unsqueeze(0).to(self._device),
                torch.from_numpy(features.proprio).unsqueeze(0).to(self._device),
                torch.from_numpy(features.material_logits).unsqueeze(0).to(self._device),
                torch.from_numpy(features.conflict_logits).unsqueeze(0).to(self._device),
            )
        categories, _, weights = predict_category(
            out, self._categories, continue_threshold=self._threshold
        )
        category = categories[0]
        fallback = "continue" if category == "nominal" else "Set_Constraint"
        primitive = self._canonical.get(category, fallback)
        expert = EXPERT_NAMES[int(np.argmax(weights[0]))]
        annotation = CoTAnnotation(
            thought=f"observable evidence router selected {expert}; attributed {category}.",
            attribution=category,
            primitive=RecoveryPrimitive(primitive, category_params(category, primitive)),
            attribution_raw=category,
            raw_text="",
        )
        return ParsedDecision(ok=True, raw_text="", annotation=annotation)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        visual_encoder: Any,
        conflict_policy: Any,
        device: str = "cpu",
    ) -> RoutedEvidencePolicy:
        import torch

        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = EvidenceRoutedNetwork.build(
            visual_dim=int(blob["visual_dim"]),
            proprio_dim=int(blob["proprio_dim"]),
            n_categories=len(blob["categories"]),
            hidden=int(blob["hidden"]),
            material_feature_dim=int(blob.get("material_feature_dim", 16)),
        )
        model.load_state_dict(blob["state_dict"])
        return cls(
            model,
            visual_encoder,
            conflict_policy,
            categories=list(blob["categories"]),
            canonical=dict(blob["canonical"]),
            proprio_mean=np.asarray(blob["proprio_mean"]),
            proprio_std=np.asarray(blob["proprio_std"]),
            continue_threshold=float(blob["continue_threshold"]),
            device=device,
            conflict_logit_scale=float(blob.get("conflict_logit_scale", 8.0)),
            material_models=blob.get("material_models"),
            material_classes=blob.get("material_classes"),
            material_pair_logit_scale=float(blob.get("material_pair_logit_scale", 8.0)),
        )
