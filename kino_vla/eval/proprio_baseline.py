"""B1 — the strongest pure-proprioception baseline as a ``VlaPolicy`` (E2).

E2's fair strong opponent: a proprioception-only attributor that, given a failure node, reads ONLY
the proprioceptive history window (ignores RGB/depth), attributes the operator, and selects the
canonical recovery primitive for that attribution. It drops into the same evaluation paths the agent
uses — ``eval.suite_sem.evaluate_attribution`` (open-loop attribution) and
``vla.rollout.run_vla_rollout`` (closed-loop recovery) — so B1 vs B5 (agent) is a controlled
head-to-head on the identical scenes.

The attribution head is an RMA-style proprio history encoder + a multi-class attribution head (the
``LearnedMonitorModel`` backbone, real-Go2-trained; for E2 optionally retrained at a longer RMA
window). E1's C2ST already proved the Bayes-optimal proprio discriminator is at chance on the
matched O4/O2, so B1's chance result there is a property of the construction, not under-training.

The model is INJECTED (duck-typed: ``.predict(window)`` + ``.std.mean``), so this module imports no
torch — the CPU unit test drives it with a fake monitor; ``from_deployed`` lazy-loads the real
``LearnedMonitorModel``.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from kino_vla.data.schema import CoTAnnotation, RecoveryPrimitive, Snapshot
from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.eval.suite_sem import _default_params
from kino_vla.monitor.hazard_lab import OP_IDS
from kino_vla.utils.config import Config
from kino_vla.vla.output import REJECT_SCHEMA, ParsedDecision

# Attribution class id -> name (0 = normal, 1..11 = O1..O11) — the LearnedMonitor attr-head order
# (mirrors learned_monitor.CLASS_NAMES, defined here to keep this module torch-free).
CLASS_NAMES: tuple[str, ...] = ("normal", *OP_IDS)


class _AttrModel(Protocol):
    """The minimal interface B1 needs from an attribution model (LearnedMonitorModel-compatible)."""

    std: Any  # has .mean (np.ndarray over the model's feature channels)

    def predict(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (hazard_prob (B,), attr_probs (B, N_CLASSES)) for raw (T,F) or (B,T,F) windows."""
        ...


class ProprioBaselinePolicy:
    """B1: proprioception-only attribution → canonical recovery, as a ``VlaPolicy``."""

    def __init__(
        self,
        model: _AttrModel,
        taxonomy: FailureTaxonomy,
        *,
        canonical: dict[str, str],
        operator_category: dict[str, str],
        class_names: tuple[str, ...] = CLASS_NAMES,
        expected_features: int = 12,
    ) -> None:
        self._model = model
        self._tax = taxonomy
        self._canonical = dict(canonical)
        self._class_names = tuple(class_names)
        self._expected_features = int(expected_features)
        # short op id ("O4") -> full operator name ("O4_tether"), config-driven (no hardcoded dict).
        self._opname_of = {name.split("_")[0]: name for name in operator_category}

    @classmethod
    def from_deployed(
        cls,
        pcfg: Config,
        taxonomy: FailureTaxonomy,
        *,
        model_path: str | None = None,
        device: str = "cpu",
    ) -> ProprioBaselinePolicy:
        """Load the deployed (or a config-pointed) LearnedMonitor as B1's attribution head."""
        from kino_vla.monitor.learned_monitor import (  # lazy: keeps the module torch-free
            DEPLOYED_MODEL_PATH,
            LearnedMonitorModel,
        )

        model = LearnedMonitorModel.load(model_path or DEPLOYED_MODEL_PATH, device=device)
        return cls(
            model,
            taxonomy,
            canonical=pcfg.recovery.canonical.to_dict(),
            operator_category=pcfg.attribution.operator_category.to_dict(),
            expected_features=int(getattr(model.std.mean, "shape", [12])[0]),
        )

    def _bridge_features(self, window: np.ndarray) -> np.ndarray:
        """Match the window's channel count to the model's expected features. If the snapshot
        carries the 11-dim M4 window but the monitor wants 12 (+support_ratio), append the channel's
        training mean so it standardizes to ~0 (no-information; immaterial for the ambiguity pairs —
        E1 showed the full 12-dim is at chance too). Wider/narrower mismatches are an error."""
        f = window.shape[-1]
        if f == self._expected_features:
            return window
        if f == self._expected_features - 1:
            fill = float(np.asarray(self._model.std.mean)[-1])
            pad = np.full((window.shape[0], 1), fill, dtype=window.dtype)
            return np.concatenate([window, pad], axis=1)
        raise ValueError(
            f"proprio window has {f} features; model expects {self._expected_features} (no bridge)"
        )

    def decide(self, snapshot: Snapshot, map_note: str = "") -> ParsedDecision:  # noqa: ARG002
        """Attribute from the proprioception window alone and pick the canonical recovery."""
        window = np.asarray(snapshot.proprio_window, dtype=np.float64)
        if window.ndim != 2 or window.shape[0] == 0:
            return ParsedDecision(ok=False, raw_text="", reject_code=f"{REJECT_SCHEMA}: no proprio")
        try:
            window = self._bridge_features(window)
        except ValueError as err:
            return ParsedDecision(ok=False, raw_text="", reject_code=f"{REJECT_SCHEMA}: {err}")
        _, attr = self._model.predict(window)
        # argmax over operator classes (skip class 0 "normal"; a failure node always has a cause)
        cls = int(np.argmax(attr[0, 1:])) + 1
        op_id = self._class_names[cls]
        op_name = self._opname_of.get(op_id)
        if op_name is None:
            return ParsedDecision(
                ok=False, raw_text="", reject_code=f"{REJECT_SCHEMA}: no operator for {op_id!r}"
            )
        category = self._tax.category_of(op_name)
        primitive = self._canonical.get(category, "Set_Constraint")
        ann = CoTAnnotation(
            thought=f"proprio monitor attributes {op_id} ({category}); default recovery.",
            attribution=category,
            primitive=RecoveryPrimitive(primitive, _default_params(primitive)),
            attribution_raw=op_id,
            raw_text="",
        )
        return ParsedDecision(ok=True, raw_text="", annotation=ann)
