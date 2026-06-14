"""Kino-Fail v2 failure operators O1–O11 (spec §8). M1 shipped O1, O6, O11; M2 adds
O3, O5, O8, O9, O10 (the remaining cheap operators)."""

from kino_vla.sim.operators.base import FailureOperator, OperatorStack
from kino_vla.sim.operators.o1_mu_field import MuField
from kino_vla.sim.operators.o3_collapse import Collapse
from kino_vla.sim.operators.o5_payload import Payload
from kino_vla.sim.operators.o6_push import Push
from kino_vla.sim.operators.o8_invisible_collider import InvisibleCollider
from kino_vla.sim.operators.o9_high_centering import HighCentering
from kino_vla.sim.operators.o10_effort_decay import EffortDecay
from kino_vla.sim.operators.o11_obs_bias import ObsBias

# Registry for suite construction (spec §8.3); later milestones extend this.
OPERATORS: dict[str, type[FailureOperator]] = {
    MuField.name: MuField,
    Collapse.name: Collapse,
    Payload.name: Payload,
    Push.name: Push,
    InvisibleCollider.name: InvisibleCollider,
    HighCentering.name: HighCentering,
    EffortDecay.name: EffortDecay,
    ObsBias.name: ObsBias,
}

__all__ = [
    "OPERATORS",
    "Collapse",
    "EffortDecay",
    "FailureOperator",
    "HighCentering",
    "InvisibleCollider",
    "MuField",
    "ObsBias",
    "OperatorStack",
    "Payload",
    "Push",
]
