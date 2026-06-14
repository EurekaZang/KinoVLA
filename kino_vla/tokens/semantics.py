"""Physical-regime semantics for the Kino-Text contrastive head (spec §4, aux loss).

The contrastive head aligns each window's latent with a learned *text prototype* of
its physical regime (InfoNCE), so the Kino-Tokens carry an open-vocabulary-ready,
human-readable semantic — without (yet) a heavyweight text encoder. The regime of a
window is derived deterministically from its privileged target at the window's last
step, so labels are self-consistent with the regression supervision (a window early
in an ice episode, still on good ground, is correctly ``normal``).

M7 upgrade path: swap the prototype table for a frozen sentence/CLIP encoder over
``REGIME_TEXT`` to get true open-vocabulary alignment; the InfoNCE objective is
unchanged. The text strings below are that swap's anchors.
"""

from __future__ import annotations

import numpy as np

from kino_vla.tokens.features import TARGET_SCHEMA

# Ordered regime vocabulary; index is the prototype-table row and the class id.
REGIMES: tuple[str, ...] = (
    "normal",
    "low_friction",
    "overload",
    "actuator_decay",
    "high_centered",
)
N_REGIMES: int = len(REGIMES)

# Canonical natural-language description per regime — the contrastive anchors and the
# demo's human-readable readout (spec §4 "latent 天然物理可解释").
REGIME_TEXT: dict[str, str] = {
    "normal": "the ground is firm and the robot moves freely",
    "low_friction": "the ground is slippery with low friction, the feet are sliding",
    "overload": "the robot is carrying a heavy payload and the actuators are saturating",
    "actuator_decay": "an actuator is weakening and cannot deliver its commanded torque",
    "high_centered": "a foot has lost ground contact, the belly is bearing the load",
}


def regime_of(
    target: np.ndarray,
    *,
    mu_nominal: float,
    mu_low: float,
    payload_min_kg: float,
    effort_low: float,
    support_low: float,
) -> int:
    """Classify a privileged target vector into a regime id (thresholds from config).

    Single-operator training data (spec §8.3: Suite-Comp is test-only) makes this a
    clean one-of-five label; the most physically salient deviation wins when several
    are borderline.
    """
    idx = {name: i for i, name in enumerate(TARGET_SCHEMA)}
    mu = float(target[idx["mu"]])
    payload = float(target[idx["payload_kg"]])
    effort = float(target[idx["effort_scale"]])
    support = float(target[idx["support_ratio"]])
    # Order by salience: friction loss and lost support are the most acute.
    if mu <= mu_low:
        return REGIMES.index("low_friction")
    if support <= support_low:
        return REGIMES.index("high_centered")
    if payload >= payload_min_kg:
        return REGIMES.index("overload")
    if effort <= effort_low:
        return REGIMES.index("actuator_decay")
    _ = mu_nominal  # documents the nominal anchor; normal is the residual class
    return REGIMES.index("normal")
