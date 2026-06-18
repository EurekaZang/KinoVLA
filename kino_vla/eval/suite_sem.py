"""Suite-Sem attribution-accuracy evaluation: VLA vs the rule-FSM baseline (M7 exit criterion 1).

Suite-Sem is the B-class semantic套件 — the three constructive ambiguity pairs (spec §8.3): O4↔O2,
O5↔O10, O3↔O1. Its decisive question (spec §2.5 / §5 / §12 B2): can a planner *attribute* the
physical cause well enough to pick the right — sometimes opposite — recovery, where a rule-FSM that
ignores the cause cannot? This module scores attribution accuracy on the held-out Suite-Sem
snapshots for the trained VLA vs the FSM baseline (B2), the M7 exit-1 comparison.

The literal FSM stub emits no attribution at all (it always Backsteps), so its attribution accuracy
is 0; to be *fair* (not a straw man) the primary baseline is the strongest non-semantic predictor —
the majority class of the training split (a constant). The VLA must beat that. Pure-logic scoring;
the VLA policy is injected (stub for CI, the real model for the deliverable).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kino_vla.data.schema import CoTAnnotation, RecoveryPrimitive, Snapshot
from kino_vla.utils.config import Config
from kino_vla.vla.dataset_build import _snapshot_from_record
from kino_vla.vla.output import ParsedDecision
from kino_vla.vla.planner import VlaPolicy


@dataclass(frozen=True)
class SemItem:
    """One held-out Suite-Sem evaluation node."""

    snapshot: Snapshot
    attribution_truth: str
    ab_class: str
    ambiguity_pair: str | None
    primitive_truth: str


def load_suite_sem(
    dataset_dir: str | Path, sample_ids: list[str], *, ambiguity_only: bool = True
) -> list[SemItem]:
    """Reconstruct the held-out Suite-Sem nodes (ambiguity-pair snapshots) for the given ids."""
    out = Path(dataset_dir)
    records = {
        json.loads(line)["sample_id"]: json.loads(line)
        for line in (out / "samples.jsonl").read_text().splitlines()
        if line
    }
    npz = np.load(out / "frames.npz")
    items: list[SemItem] = []
    for sid in sample_ids:
        rec = records.get(sid)
        if rec is None:
            continue
        if ambiguity_only and rec.get("ambiguity_pair") is None:
            continue
        frames = {k: npz[f"{sid}__{k}"] for k in ("rgb", "depth", "proprio")}
        items.append(
            SemItem(
                snapshot=_snapshot_from_record(rec, frames),
                attribution_truth=rec["ground_truth"]["category"],
                ab_class=rec["ground_truth"]["ab_class"],
                ambiguity_pair=rec.get("ambiguity_pair"),
                primitive_truth=rec["annotation"]["action"]["primitive"],
            )
        )
    return items


class MajorityBaselinePolicy:
    """B2 (fair): a constant predictor of the train-split majority attribution (no perception)."""

    def __init__(self, cfg: Config, category: str, primitive: str) -> None:
        self._category = category
        self._primitive = primitive

    def decide(self, snapshot: Snapshot) -> ParsedDecision:  # noqa: ARG002 - constant by design
        ann = CoTAnnotation(
            thought="rule baseline: anomaly detected, default recovery.",
            attribution=self._category,
            primitive=RecoveryPrimitive(self._primitive, _default_params(self._primitive)),
            attribution_raw=self._category,
            raw_text="",
        )
        return ParsedDecision(ok=True, raw_text="", annotation=ann)


def majority_attribution(train_categories: list[str]) -> str:
    """The most common ground-truth attribution in the train split (the constant baseline)."""
    return Counter(train_categories).most_common(1)[0][0]


def evaluate_attribution(
    policy: VlaPolicy, items: list[SemItem], *, feasible_sets: dict | None = None
) -> dict:
    """Score a policy's attribution accuracy (and feasible-recovery rate) over Suite-Sem items."""
    n = len(items)
    correct = 0
    feasible_ok = 0
    parsed = 0
    per_op_correct: Counter = Counter()
    per_op_total: Counter = Counter()
    for it in items:
        op = it.snapshot.operator_name
        per_op_total[op] += 1
        decision = policy.decide(it.snapshot)
        if not decision.ok or decision.annotation is None:
            continue
        parsed += 1
        if decision.attribution == it.attribution_truth:
            correct += 1
            per_op_correct[op] += 1
        if feasible_sets is not None:
            feas = feasible_sets.get(it.attribution_truth, set())
            if decision.primitive_name in feas:
                feasible_ok += 1
    return {
        "n": n,
        "parse_rate": parsed / max(1, n),
        "attribution_accuracy": correct / max(1, n),
        "feasible_recovery_rate": feasible_ok / max(1, n),
        "per_operator_accuracy": {op: per_op_correct[op] / per_op_total[op] for op in per_op_total},
    }


def compare_vla_vs_fsm(
    vla_policy: VlaPolicy,
    items: list[SemItem],
    *,
    cfg: Config,
    train_categories: list[str],
    feasible_sets: dict | None = None,
) -> dict:
    """Run the exit-1 comparison: VLA attribution accuracy vs the majority-class FSM baseline."""
    maj_cat = majority_attribution(train_categories)
    maj_prim = str(cfg.recovery.canonical.get(maj_cat, "Set_Constraint"))
    fsm = MajorityBaselinePolicy(cfg, maj_cat, maj_prim)
    vla = evaluate_attribution(vla_policy, items, feasible_sets=feasible_sets)
    fsm_eval = evaluate_attribution(fsm, items, feasible_sets=feasible_sets)
    return {
        "n_items": len(items),
        "vla": vla,
        "fsm_baseline": fsm_eval,
        "fsm_majority_category": maj_cat,
        "vla_beats_fsm": vla["attribution_accuracy"] > fsm_eval["attribution_accuracy"],
        "margin": vla["attribution_accuracy"] - fsm_eval["attribution_accuracy"],
    }


def _default_params(primitive: str) -> dict:
    return {
        "Backstep": {"distance_m": 0.5},
        "Replan_Waypoint": {"point_px": [480, 360]},
        "Switch_Gait": {"mode": "high_step"},
        "Adjust_Posture": {"body_height_m": 0.25, "pitch_deg": 0.0},
        "Set_Constraint": {"max_speed": 0.4, "stiffness": 0.5},
        "Update_Topology": {"region_xy": [0.0, 0.0], "radius_m": 0.6, "status": "untraversable"},
        "Hold_and_Request": {"reason": "default"},
    }.get(primitive, {})
