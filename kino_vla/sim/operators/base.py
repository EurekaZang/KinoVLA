"""Failure-operator framework: parameterized operators with privileged θ (spec §8.2).

Principle P2: every failure is an operator over a *continuous* parameter vector θ,
and θ itself is the privileged ground truth — it supervises the Kino-Tokens
distillation (spec §4), anchors the truth-consistency filter (spec §10 PHASE 3),
and defines the Suite-Bound sweeps. The uniform ``get_privileged_state()`` API is
the single access path to θ; downstream code never reads operator internals.

Composition (groundwork for Suite-Comp, spec §8.3): an ``OperatorStack`` is just
an ordered list — parameter vectors concatenate, hooks apply in order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, ClassVar

from kino_vla.sim.backend import LocomotionBackend
from kino_vla.sim.types import Obs

if TYPE_CHECKING:
    from kino_vla.map.types import SemanticRegion


class FailureOperator(ABC):
    """Base class for Kino-Fail v2 operators O1–O11.

    Lifecycle hooks (all optional except θ access):
    - ``on_reset``: install static world modifications (materials, colliders).
    - ``on_step``: time/event-triggered physics actions (impulses, schedules).
    - ``transform_obs``: corrupt the *measured* observation stream (Axis IV);
      the backend's internal truth is never touched.
    """

    name: ClassVar[str]
    axis: ClassVar[str]  # one of the four mechanism axes, spec §8.2

    @abstractmethod
    def get_privileged_state(self) -> dict[str, float]:
        """Flat ``{parameter_name: value}`` view of θ (uniform API, principle P2)."""

    # Deliberately empty optional hooks (not abstract): most operators override
    # exactly one of them; forcing all three on every operator adds boilerplate.
    def on_reset(self, backend: LocomotionBackend) -> None:  # noqa: B027
        """Apply setup-time world modifications. Default: none."""

    def on_step(self, backend: LocomotionBackend, t: float) -> None:  # noqa: B027
        """Apply time-triggered actions before the physics step at sim time ``t``."""

    def transform_obs(self, obs: Obs) -> Obs:
        """Corrupt the measured observation. Default: identity."""
        return obs

    def scene_region(self) -> SemanticRegion | None:
        """Visual signature this operator contributes to the semantic map (spec §7).

        Default: none (operator has no characteristic appearance the map should ground).
        Operators that exercise the map (O2/O4/O7, M5) return a ``SemanticRegion`` so the
        segmenter can paint their appearance — crucially, O2 and O4 return *different*
        appearances over *identical* proprioception (the P4 ambiguity pair), and O7 returns
        an appearance decoupled from its physics (visual-physics remap).
        """
        return None


class OperatorStack:
    """Ordered composition of operators; θ vectors concatenate under slot-indexed keys."""

    def __init__(self, operators: list[FailureOperator]) -> None:
        self._operators = list(operators)

    def __len__(self) -> int:
        return len(self._operators)

    def __iter__(self) -> Iterator[FailureOperator]:
        return iter(self._operators)

    def on_reset(self, backend: LocomotionBackend) -> None:
        for op in self._operators:
            op.on_reset(backend)

    def on_step(self, backend: LocomotionBackend, t: float) -> None:
        for op in self._operators:
            op.on_step(backend, t)

    def transform_obs(self, obs: Obs) -> Obs:
        for op in self._operators:
            obs = op.transform_obs(obs)
        return obs

    def scene_regions(self) -> list[SemanticRegion]:
        """Collect the visual signatures of all operators that contribute one (spec §7)."""
        regions: list[SemanticRegion] = []
        for op in self._operators:
            region = op.scene_region()
            if region is not None:
                regions.append(region)
        return regions

    def get_privileged_state(self) -> dict[str, float]:
        """Concatenated θ: keys are ``op{i}.{operator_name}.{param}`` (deterministic schema)."""
        state: dict[str, float] = {}
        for i, op in enumerate(self._operators):
            for key, value in op.get_privileged_state().items():
                state[f"op{i}.{op.name}.{key}"] = value
        return state
