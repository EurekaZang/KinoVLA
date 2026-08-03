"""Observable RGB support model for selective material-dependent recovery.

The support model is intentionally small and auditable.  It learns prototypes from frozen
calibration appearances and, at deployment, consumes only the RGB tensor that is already given to
the VLA.  Appearance ids and semantic labels are used to fit/audit prototypes, never to decide on
an evaluation row.

This is an *in-support* detector, not a material classifier.  Its purpose is to prevent a recovery
whose benefit depends on a visible material cue from being executed on an appearance far from all
calibrated examples of that material.  Out-of-support inputs fall back to the consequence-derived
safe action.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

MATERIAL_CLASSES: tuple[str, ...] = (
    "adhesion",
    "compliant_terrain",
    "low_friction",
    "solid_ground",
)


def dominant_chromatic_rgb(
    rgb: np.ndarray,
    *,
    chroma_min: float = 0.04,
    quantization: float = 0.01,
) -> np.ndarray:
    """Return a deterministic three-channel material-colour summary.

    The procedural camera includes a neutral ground background and a smoothly shaded material
    patch.  Selecting chromatic pixels suppresses that background for coloured materials.  Nearly
    achromatic materials use all pixels, which keeps grey calibration appearances well-defined.
    Quantisation makes the mode stable to lighting gradients and float round-off.
    """
    array = np.asarray(rgb, dtype=np.float64)
    if array.ndim == 4:
        pixels = array.reshape(-1, 3)
    elif array.ndim == 3 and array.shape[-1] == 3:
        pixels = array.reshape(-1, 3)
    else:
        raise ValueError(f"RGB must have shape (T,H,W,3) or (H,W,3), got {array.shape}")
    pixels = pixels[np.isfinite(pixels).all(axis=1)]
    if not len(pixels):
        raise ValueError("RGB tensor contains no finite pixels")
    chroma = np.ptp(pixels, axis=1)
    selected = pixels[chroma > float(chroma_min)]
    # A tiny number of chromatic antialiasing pixels is not a material observation.
    if len(selected) < max(32, int(0.01 * len(pixels))):
        selected = pixels
    quantized = np.round(selected / float(quantization)) * float(quantization)
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    return colors[int(np.argmax(counts))].astype(np.float64)


def fit_material_prototypes(
    rows: list[dict[str, Any]],
    *,
    target_class: str,
    feature_field: str = "material_rgb",
) -> dict[str, Any]:
    """Fit one robust RGB prototype per calibration appearance of ``target_class``."""
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        if str(row.get("semantic_class")) != str(target_class):
            continue
        feature = np.asarray(row[feature_field], dtype=np.float64)
        if feature.shape != (3,) or not np.isfinite(feature).all():
            raise ValueError(f"invalid material feature for {row.get('sample_id')}: {feature}")
        grouped[str(row["appearance_id"])].append(feature)
    if not grouped:
        raise ValueError(f"no calibration appearances for target class {target_class!r}")
    prototypes = {
        appearance: np.median(np.stack(features), axis=0).round(6).tolist()
        for appearance, features in sorted(grouped.items())
    }
    return {
        "schema_version": 1,
        "feature": "dominant_chromatic_rgb",
        "feature_field": feature_field,
        "target_class": str(target_class),
        "distance": "nearest_prototype_l2",
        "n_appearances": len(prototypes),
        "prototypes_by_calibration_appearance": prototypes,
        "deployment_forbidden_inputs": [
            "appearance_id",
            "semantic_class",
            "truth",
            "scenario",
            "theta",
        ],
    }


def material_support_distance(feature: np.ndarray | list[float], model: dict[str, Any]) -> float:
    """Distance from an observable RGB feature to the nearest frozen target prototype."""
    value = np.asarray(feature, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"material feature must be a finite RGB triple, got {value}")
    prototypes = np.asarray(
        list(model["prototypes_by_calibration_appearance"].values()), dtype=np.float64
    )
    if prototypes.ndim != 2 or prototypes.shape[1] != 3 or not len(prototypes):
        raise ValueError("material support model has no valid prototypes")
    return float(np.linalg.norm(prototypes - value[None, :], axis=1).min())


def material_evidence_features(
    feature: np.ndarray | list[float],
    models: dict[str, dict[str, Any]],
    *,
    classes: tuple[str, ...] = MATERIAL_CLASSES,
) -> np.ndarray:
    """Return an auditable material-evidence vector from frozen train-only prototypes.

    The vector contains the observable RGB summary, one nearest-prototype distance per class,
    distance-to-best margins, and a soft nearest-prototype assignment.  All terms are functions of
    the current RGB observation and frozen prototypes; no appearance id or evaluation label is an
    input.  Margins make the useful relative statement (for example, *closer to calibrated
    adhesion than calibrated mud*) directly linearly accessible to the downstream visual expert.
    """
    value = np.asarray(feature, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"material feature must be a finite RGB triple, got {value}")
    missing = [name for name in classes if name not in models]
    if missing:
        raise ValueError(f"missing material support models: {missing}")
    chroma = np.asarray([float(np.ptp(value))], dtype=np.float64)
    distances = np.asarray(
        [material_support_distance(value, models[name]) for name in classes], dtype=np.float64
    )
    margins = distances - distances.min()
    # The temperature is fixed a priori in RGB-distance units.  It is not tuned on held-out rows.
    temperature = 0.15
    assignment = np.exp(-distances / temperature)
    assignment /= assignment.sum()
    return np.concatenate([value, chroma, distances, margins, assignment]).astype(np.float32)


def matched_material_pair_logits(
    feature: np.ndarray | list[float],
    models: dict[str, dict[str, Any]],
    categories: list[str],
    *,
    scale: float = 8.0,
) -> np.ndarray:
    """Frozen expert for the pre-registered adhesion-vs-compliance matched construction.

    Only the two load-bearing material hypotheses receive finite support.  The expert therefore
    answers the narrow question justified by the construction; it does not pretend that colour
    alone can distinguish arbitrary anomaly categories.  A learned observable router decides
    whether this expert is applicable.
    """
    required = {"adhesion", "compliant_terrain"}
    if not required.issubset(categories):
        raise ValueError(f"categories omit matched material hypotheses: {sorted(required)}")
    adhesion = material_support_distance(feature, models["adhesion"])
    compliant = material_support_distance(feature, models["compliant_terrain"])
    margin = float(scale) * (compliant - adhesion)
    logits = np.full(len(categories), -float(scale), dtype=np.float32)
    logits[categories.index("adhesion")] = margin
    logits[categories.index("compliant_terrain")] = -margin
    return logits
