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


def ambiguous_appearances(items: list[SemItem]) -> set[str]:
    """The appearance classes whose look does NOT determine the category (purity < 1).

    Data-driven (not hard-coded): an appearance is *appearance-ambiguous* when the same surface
    look co-occurs with more than one privileged attribution in the set (e.g. ``ice_sheet`` =
    low_friction OR region_collapse; ``solid_ground`` = low_friction/overload/effort_decay). On
    these, vision alone cannot attribute — only the proprioception can — so they are the regime
    where the latent Kino-Tokens route is claimed to earn its keep (spec §3/§4). The complement is
    *appearance-solvable* (``brown_mud`` = compliant, ``yellow_adhesive`` = adhesion).
    """
    by_app: dict[str, set[str]] = {}
    for it in items:
        by_app.setdefault(it.snapshot.appearance_class, set()).add(it.attribution_truth)
    return {app for app, cats in by_app.items() if len(cats) > 1}


def evaluate_by_regime(
    policy: VlaPolicy,
    items: list[SemItem],
    *,
    ambiguous_apps: set[str],
    feasible_sets: dict | None = None,
) -> dict:
    """Attribution accuracy split by appearance regime (the §3 fidelity ablation's decisive cut).

    ``overall`` is the whole Suite-Sem set; ``ambiguous`` is the proprio-decided subset (appearance
    in ``ambiguous_apps``); ``solvable`` is the vision-decided complement. The latent route's claim
    lives in ``ambiguous`` — ``overall`` is diluted/saturated by the vision-solvable half.
    """
    amb = [it for it in items if it.snapshot.appearance_class in ambiguous_apps]
    solv = [it for it in items if it.snapshot.appearance_class not in ambiguous_apps]
    return {
        "overall": evaluate_attribution(policy, items, feasible_sets=feasible_sets),
        "ambiguous": evaluate_attribution(policy, amb, feasible_sets=feasible_sets),
        "solvable": evaluate_attribution(policy, solv, feasible_sets=feasible_sets),
    }


def evaluate_attribution(
    policy: VlaPolicy, items: list[SemItem], *, feasible_sets: dict | None = None
) -> dict:
    """Score a policy's attribution + recovery over Suite-Sem items.

    Three recovery readings (the un-gated one is a known footgun — see ``correct_recovery_rate``):

    - ``attribution_accuracy`` — the share whose attributed cause == the privileged truth.
    - ``feasible_recovery_rate`` — the share whose chosen primitive lands in the TRUE cause's
      feasible set. **NOT gated on attribution**: a wrong cause can still emit a primitive that
      happens to be feasible for the truth (e.g. a constant-Backstep baseline scores >0 here with
      0 attribution). Kept for backward-compat / diagnostics; do NOT headline it.
    - ``correct_recovery_rate`` (the JOINT, load-bearing metric) — the share that BOTH attributed
      the cause correctly AND chose a feasible primitive. This is the recovery number that actually
      requires understanding the cause, so it is the one to report alongside attribution.
    """
    n = len(items)
    correct = 0
    feasible_ok = 0
    joint_ok = 0
    parsed = 0
    per_op_correct: Counter = Counter()
    per_op_joint: Counter = Counter()
    per_op_total: Counter = Counter()
    for it in items:
        op = it.snapshot.operator_name
        per_op_total[op] += 1
        decision = policy.decide(it.snapshot)
        if not decision.ok or decision.annotation is None:
            continue
        parsed += 1
        attr_ok = decision.attribution == it.attribution_truth
        if attr_ok:
            correct += 1
            per_op_correct[op] += 1
        prim_feasible = (
            feasible_sets is not None
            and decision.primitive_name in feasible_sets.get(it.attribution_truth, set())
        )
        if prim_feasible:
            feasible_ok += 1
        if attr_ok and prim_feasible:
            joint_ok += 1
            per_op_joint[op] += 1
    return {
        "n": n,
        "parse_rate": parsed / max(1, n),
        "attribution_accuracy": correct / max(1, n),
        "feasible_recovery_rate": feasible_ok / max(1, n),
        "correct_recovery_rate": (joint_ok / max(1, n)) if feasible_sets is not None else None,
        "per_operator_accuracy": {op: per_op_correct[op] / per_op_total[op] for op in per_op_total},
        "per_operator_counts": {
            op: {
                "n": per_op_total[op],
                "attr_correct": per_op_correct[op],
                "joint_correct": per_op_joint[op],
            }
            for op in per_op_total
        },
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
