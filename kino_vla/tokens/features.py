"""Proprioceptive feature schema and per-step extraction (spec §4 — the extractor's
input stream and its regression targets).

The Kino-Tokens extractor consumes a 500 ms window of the *measured* Sport-Client
proprioception — exactly what a real Go2 controller sees — and regresses the
*privileged* physics truth the controller cannot observe (teacher-student
distillation, spec §4). This module fixes both schemas as deterministic ordered
tuples so the dataset, the model, and the online coupler agree on layout, and
provides the Obs → feature-vector map plus a numpy standardizer.

Pure numpy on purpose: features/windows/datasets stay importable on the torch-free
CI machine (only ``kino_vla.tokens.extractor`` needs torch).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kino_vla.sim.types import Obs

# Input channels the 1D-CNN sees, in order. These are *measured* (controller-visible)
# signals only — never the privileged targets — so the regression is a genuine
# inverse-dynamics inference, not a copy. ``base_height``/``tilt`` are flat on the
# surrogate but vary on the Isaac Go2; kept for backend portability (a zero-variance
# channel is harmless — the standardizer floors its std).
FEATURE_SCHEMA: tuple[str, ...] = (
    "vx",
    "vy",
    "yaw_rate",
    "cmd_vx",
    "cmd_vy",
    "cmd_wz",
    "tracking_err",
    "slip_ratio",
    "effort_ratio",
    "base_height",
    "tilt",
)
N_FEATURES: int = len(FEATURE_SCHEMA)

# Privileged physics the extractor regresses (spec §4). ``mu`` is the headline channel
# wired into the CBF shield's friction cone (spec §6.5 coupling point).
TARGET_SCHEMA: tuple[str, ...] = ("mu", "payload_kg", "effort_scale", "support_ratio")
N_TARGETS: int = len(TARGET_SCHEMA)
MU_INDEX: int = TARGET_SCHEMA.index("mu")
SUPPORT_INDEX: int = TARGET_SCHEMA.index("support_ratio")


def obs_to_features(obs: Obs) -> np.ndarray:
    """Map one observation to the fixed ``(N_FEATURES,)`` feature vector."""
    tracking_err = float(np.linalg.norm(obs.cmd_prev[:2] - obs.vel_body))
    return np.array(
        [
            float(obs.vel_body[0]),
            float(obs.vel_body[1]),
            float(obs.yaw_rate),
            float(obs.cmd_prev[0]),
            float(obs.cmd_prev[1]),
            float(obs.cmd_prev[2]),
            tracking_err,
            float(obs.slip_ratio),
            float(obs.effort_ratio),
            float(obs.base_height),
            float(obs.tilt),
        ],
        dtype=np.float64,
    )


def physics_to_target(physics: dict[str, float]) -> np.ndarray:
    """Map a backend ``privileged_physics()`` dict to the fixed target vector."""
    return np.array([float(physics[name]) for name in TARGET_SCHEMA], dtype=np.float64)


@dataclass(frozen=True)
class Standardizer:
    """Per-channel zero-mean/unit-std normalizer with a floored std (spec QA: no magic
    numbers — the floor lives in the extractor config). Stored with the checkpoint so
    train/eval/online inference share identical normalization."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, eps: float) -> Standardizer:
        """Fit over the leading axes of ``x`` (last axis is the channel)."""
        flat = np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])
        mean = flat.mean(axis=0)
        std = np.maximum(flat.std(axis=0), float(eps))
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) * self.std + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> Standardizer:
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float64),
            std=np.asarray(data["std"], dtype=np.float64),
        )
