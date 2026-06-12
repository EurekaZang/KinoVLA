"""Kino-Fail v2 failure operators O1–O11 (spec §8). M1 ships O1, O6, O11."""

from kino_vla.sim.operators.base import FailureOperator, OperatorStack
from kino_vla.sim.operators.o1_mu_field import MuField
from kino_vla.sim.operators.o6_push import Push
from kino_vla.sim.operators.o11_obs_bias import ObsBias

# Registry for suite construction (spec §8.3); later milestones extend this.
OPERATORS: dict[str, type[FailureOperator]] = {
    MuField.name: MuField,
    Push.name: Push,
    ObsBias.name: ObsBias,
}

__all__ = ["OPERATORS", "FailureOperator", "MuField", "ObsBias", "OperatorStack", "Push"]
