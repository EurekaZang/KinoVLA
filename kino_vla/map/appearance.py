"""Appearance embeddings — a deterministic surrogate for SAM+CLIP open-vocab features.

Spec §7 grounds RGB regions with open-vocabulary segmentation (SAM) and CLIP
features, then propagates traversability labels to *visually homogeneous* neighbours
via CLIP-feature cosine similarity. On the dev/CI tier there is no RGB camera and no
CLIP encoder (mirrors the M4 surrogate scope), so this module
stands in a **named-appearance-class embedding**: a class name (``"brown_mud"``,
``"yellow_adhesive"``, ``"ice"``, ``"solid_ground"``) maps deterministically to a unit
vector, with optional small per-instance jitter modelling within-class texture
variation. Two patches of the same material are near-collinear (cosine ≈ 1); different
materials are near-orthogonal. The map's similarity-propagation logic is therefore
exercised exactly as it would be on real CLIP features — only the encoder is swapped.

The contract a real CLIP encoder must satisfy to drop in here: return an L2-normalised
feature vector per region whose cosine similarity tracks visual homogeneity.
"""

from __future__ import annotations

import hashlib

import numpy as np

EMBED_DIM = 64  # larger than a toy dim so distinct material classes are near-orthogonal


def _seed_from_name(class_name: str) -> int:
    """Stable 32-bit seed from a class name (hashlib, not Python's salted ``hash``)."""
    digest = hashlib.sha256(class_name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def appearance_embedding(class_name: str, jitter: float = 0.0, instance: int = 0) -> np.ndarray:
    """Deterministic L2-normalised appearance feature for a named material class.

    Same ``class_name`` ⇒ same base direction (cosine 1.0 at ``jitter=0``). ``jitter``
    in [0, 1) adds a seeded per-``instance`` perturbation (within-class texture noise):
    neighbours of the same class stay highly similar but not bit-identical, so the
    propagation threshold is a real decision, not a tautology.
    """
    base_rng = np.random.default_rng(_seed_from_name(class_name))
    vec = base_rng.standard_normal(EMBED_DIM)
    vec /= np.linalg.norm(vec)
    if jitter > 0.0:
        inst_rng = np.random.default_rng(_seed_from_name(f"{class_name}#{instance}"))
        noise = inst_rng.standard_normal(EMBED_DIM)
        noise /= np.linalg.norm(noise)
        vec = vec + float(jitter) * noise
        vec /= np.linalg.norm(vec)
    return vec.astype(np.float64)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two appearance vectors, clipped to [-1, 1]."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.clip(float(a @ b) / (na * nb), -1.0, 1.0))
