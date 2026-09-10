"""Pass-through Safety Shield — walking-skeleton stub, replaced by CBF-QP at M3.

This is the [STUB: pass-through shield]. The interface already
matches the M3 contract (spec §6.8: every command yields a decision with an
intervention flag and structured codes) so swapping in the CBF-QP changes one
constructor in the demo, nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kino_vla.sim.types import Obs


@dataclass(frozen=True)
class ShieldDecision:
    """Filtered command plus intervention metadata (spec §6.8 structured codes)."""

    cmd: np.ndarray  # (3,) command actually forwarded to the backend
    intervened: bool
    codes: tuple[str, ...] = field(default_factory=tuple)


class PassThroughShield:
    """Forwards every command unmodified; never intervenes."""

    def filter(self, cmd: np.ndarray, obs: Obs) -> ShieldDecision:
        return ShieldDecision(cmd=np.asarray(cmd, dtype=np.float64).copy(), intervened=False)
