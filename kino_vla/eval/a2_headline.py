"""A2 — headline three-row, hardened (Paper-A §4 A2, serves C2 + C5).

E2 established the conflict-resolution ablation at n=30 on the yellow/brown matched construction.
A2 SCALES it to the frozen A0.3 corpus (matched-O4 n=120 + matched-O2 n=120) and FORTIFIES it with
the pre-registered **appearance-held-out split** (A0.4) — the shortcut-killer for the strongest
predictable review attack ("B5-conflict learned yellow->Backstep, not conflict resolution"). This
module is the torch-free eval core: it loads the matched corpus (carrying the appearance split),
scores any ``VlaPolicy`` per item, and computes the publication statistics (Wilson 95% CIs, paired
McNemar between adjacent rows on the shared frozen snapshots — §7 protocol).

Headline metric (R6): the JOINT, attribution-gated ``correct_recovery`` — attribution correct AND
the chosen primitive lies in the TRUE cause's pre-registered admissible set (A0.5). The ungated
feasible-rate is kept as a logged diagnostic only (E2's footgun fix — a proprio-blind baseline
scores >0 on it). No Isaac at eval time; the snapshots are frozen (content hash in the manifest).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kino_vla.data.schema import Snapshot
from kino_vla.vla.dataset_build import _snapshot_from_record
from kino_vla.vla.planner import VlaPolicy

# The two matched-construction operators (spec §3 T2 conflict + T1 control). O4_tether = adhesion
# (the vision-decisive conflict); O2_compliance = compliant_terrain (the agree control).
MATCHED_OPERATORS = ("O4_tether", "O2_compliance")


@dataclass(frozen=True)
class A2Item:
    """One matched-corpus evaluation node, carrying the pre-registered appearance split."""

    sample_id: str
    snapshot: Snapshot
    operator: str  # "O4_tether" | "O2_compliance"
    truth_category: str  # "adhesion" | "compliant_terrain"
    ab_class: str
    appearance_id: str
    appearance_split: str  # "train" | "test" (A0.4 pre-registered)
    admissible_set: frozenset[str]  # the TRUE cause's admissible recoveries (A0.5)
    pair_id: str


@dataclass(frozen=True)
class ItemResult:
    """One agent's decision on one item + the derived correctness flags (McNemar-ready)."""

    sample_id: str
    operator: str
    appearance_id: str
    appearance_split: str
    truth_category: str
    parsed: bool
    attribution: str | None
    attr_ok: bool
    primitive: str | None
    prim_feasible: bool  # primitive in the TRUE cause's admissible set (NOT attribution-gated)
    joint_ok: bool  # attr_ok AND prim_feasible — the headline correct-recovery


def load_matched_corpus(
    dataset_dir: str | Path, *, operators: tuple[str, ...] = MATCHED_OPERATORS
) -> list[A2Item]:
    """Load the frozen matched-pair snapshots (ambiguity_pair set) with their appearance split.

    Unlike :func:`kino_vla.eval.suite_sem.load_suite_sem`, this keeps the ``sample_id``,
    ``appearance_split`` and per-item ``admissible_recovery_set`` so A2 can slice by the A0.4 split
    and gate the correct-recovery on the A0.5 registry set.
    """
    out = Path(dataset_dir)
    records = [
        json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines() if line
    ]
    npz = np.load(out / "frames.npz")
    items: list[A2Item] = []
    for rec in records:
        if rec.get("ambiguity_pair") is None:
            continue
        op = rec["snapshot"]["operator_name"]
        if op not in operators:
            continue
        sid = rec["sample_id"]
        frames = {k: npz[f"{sid}__{k}"] for k in ("rgb", "depth", "proprio")}
        gt = rec["ground_truth"]
        # ``admissible_recovery_set`` / ``appearance_*`` are A0.3-schema fields; fall back to the
        # ground-truth feasible set + the snapshot appearance for older (E2-schema) datasets so the
        # same loader can score both corpora.
        admissible = rec.get("admissible_recovery_set") or gt.get("feasible") or []
        items.append(
            A2Item(
                sample_id=sid,
                snapshot=_snapshot_from_record(rec, frames),
                operator=op,
                truth_category=gt["category"],
                ab_class=gt["ab_class"],
                appearance_id=rec.get("appearance_id", rec["snapshot"].get("appearance_class", "")),
                appearance_split=rec.get("appearance_split", "train"),
                admissible_set=frozenset(admissible),
                pair_id=rec.get("pair_id", rec.get("ambiguity_pair", "")),
            )
        )
    return items


def eval_policy(policy: VlaPolicy, items: list[A2Item]) -> list[ItemResult]:
    """Score a policy over the matched items — one open-loop single decision per snapshot.

    ``prim_feasible`` mirrors :func:`kino_vla.eval.suite_sem.evaluate_attribution`: it checks the
    chosen primitive against the TRUE cause's admissible set (NOT the predicted cause), so the
    ungated diagnostic can expose a proprio-blind baseline's coincidental hits, while the headline
    ``joint_ok`` additionally requires the attribution to be right.
    """
    out: list[ItemResult] = []
    for it in items:
        decision = policy.decide(it.snapshot)
        parsed = bool(decision.ok and decision.annotation is not None)
        attribution = decision.attribution if parsed else None
        primitive = decision.primitive_name if parsed else None
        attr_ok = parsed and attribution == it.truth_category
        prim_feasible = parsed and primitive in it.admissible_set
        out.append(
            ItemResult(
                sample_id=it.sample_id,
                operator=it.operator,
                appearance_id=it.appearance_id,
                appearance_split=it.appearance_split,
                truth_category=it.truth_category,
                parsed=parsed,
                attribution=attribution,
                attr_ok=attr_ok,
                primitive=primitive,
                prim_feasible=prim_feasible,
                joint_ok=attr_ok and prim_feasible,
            )
        )
    return out


# ----------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score CI for a binomial proportion (each matched snapshot is an independent trial)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, center - half), 3), round(min(1.0, center + half), 3))


def _binom_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided binomial tail p for k successes in n trials at prob 0.5 (McNemar exact)."""
    if n == 0:
        return 1.0
    from math import comb

    probs = [comb(n, i) * 0.5**n for i in range(n + 1)]
    p_obs = probs[k]
    return float(min(1.0, sum(pi for pi in probs if pi <= p_obs + 1e-12)))


def mcnemar(a_correct: list[bool], b_correct: list[bool]) -> dict:
    """Paired McNemar test between two agents on the SAME ordered items (§7 protocol).

    Reports the discordant counts b = (a right, b wrong), c = (a wrong, b right), the exact
    two-sided binomial p on the discordant pairs (correct for small samples), and the
    continuity-corrected chi-square (for reference). ``a`` is the lower row, ``b`` the higher one,
    so ``c - b > 0`` and a small p means b is significantly better than a on the shared snapshots.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError(
            f"paired McNemar needs equal-length vectors ({len(a_correct)} != {len(b_correct)})"
        )
    b = sum(1 for ai, bi in zip(a_correct, b_correct, strict=True) if ai and not bi)
    c = sum(1 for ai, bi in zip(a_correct, b_correct, strict=True) if bi and not ai)
    disc = b + c
    p_exact = _binom_two_sided_p(min(b, c), disc)
    chi2 = (abs(b - c) - 1) ** 2 / disc if disc > 0 else 0.0
    return {
        "n_pairs": len(a_correct),
        "b_lo_right_hi_wrong": b,
        "c_lo_wrong_hi_right": c,
        "discordant": disc,
        "p_exact_two_sided": round(p_exact, 5),
        "chi2_cc": round(chi2, 4),
        "higher_row_better": c > b,
        "significant_05": p_exact < 0.05,
    }


# ----------------------------------------------------------------------- aggregation
def _cell(results: list[ItemResult]) -> dict:
    """Attribution + gated correct-recovery + ungated diagnostic for one slice, with Wilson CIs."""
    n = len(results)
    attr_k = sum(1 for r in results if r.attr_ok)
    joint_k = sum(1 for r in results if r.joint_ok)
    feas_k = sum(1 for r in results if r.prim_feasible)
    parsed_k = sum(1 for r in results if r.parsed)
    lo_a, hi_a = wilson(attr_k, n)
    lo_j, hi_j = wilson(joint_k, n)
    return {
        "n": n,
        "parse_rate": round(parsed_k / max(1, n), 3),
        "attribution_acc": round(attr_k / max(1, n), 3),
        "attribution_ci": [lo_a, hi_a],
        "correct_recovery_rate": round(joint_k / max(1, n), 3),
        "correct_recovery_ci": [lo_j, hi_j],
        "feasible_recovery_rate_UNGATED": round(feas_k / max(1, n), 3),
    }


def _balanced(results: list[ItemResult], *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Both-directions (macro over O4+O2) accuracy — the CONSTANT-PROOF headline metric.

    The matched pair is proprio-indistinguishable (A1/A2 C2ST=0.5), so a proprio-only agent must
    emit ONE answer across it => balanced accuracy <= 0.5 BY CONSTRUCTION (it is right on the side
    its constant matches, wrong on the other). Reporting a single column is gameable by which
    constant the agent lands on; the balanced number is not. CI = stratified (per-operator)
    bootstrap over items — independent snapshot sets, resampled within each, then macro-averaged.
    """
    o4 = [r for r in results if r.operator == "O4_tether"]
    o2 = [r for r in results if r.operator == "O2_compliance"]
    if not o4 or not o2:
        return {}

    def _macro(rs4: list[ItemResult], rs2: list[ItemResult], field: str) -> float:
        a4 = sum(getattr(r, field) for r in rs4) / len(rs4)
        a2 = sum(getattr(r, field) for r in rs2) / len(rs2)
        return 0.5 * (a4 + a2)

    rng = np.random.default_rng(seed)
    attr_bs, cr_bs = [], []
    i4 = np.arange(len(o4))
    i2 = np.arange(len(o2))
    for _ in range(n_boot):
        s4 = [o4[i] for i in rng.choice(i4, size=len(o4), replace=True)]
        s2 = [o2[i] for i in rng.choice(i2, size=len(o2), replace=True)]
        attr_bs.append(_macro(s4, s2, "attr_ok"))
        cr_bs.append(_macro(s4, s2, "joint_ok"))
    return {
        "n_O4": len(o4),
        "n_O2": len(o2),
        "attribution_balanced": round(_macro(o4, o2, "attr_ok"), 3),
        "attribution_balanced_ci": [round(float(np.percentile(attr_bs, 2.5)), 3),
                                    round(float(np.percentile(attr_bs, 97.5)), 3)],
        "correct_recovery_balanced": round(_macro(o4, o2, "joint_ok"), 3),
        "correct_recovery_balanced_ci": [round(float(np.percentile(cr_bs, 2.5)), 3),
                                        round(float(np.percentile(cr_bs, 97.5)), 3)],
        "o4_attr": round(sum(r.attr_ok for r in o4) / len(o4), 3),
        "o2_attr": round(sum(r.attr_ok for r in o2) / len(o2), 3),
    }


def aggregate(results: list[ItemResult]) -> dict:
    """Slice one agent's results into the A2 cells: per operator x appearance split (+ all)."""
    by_op = {op: [r for r in results if r.operator == op] for op in MATCHED_OPERATORS}
    out: dict = {
        "overall": _cell(results),
        "balanced": {
            "all": _balanced(results),
            "train_appearance": _balanced([r for r in results if r.appearance_split == "train"]),
            "test_appearance": _balanced([r for r in results if r.appearance_split == "test"]),
        },
    }
    for op, rs in by_op.items():
        col = op.split("_")[0]  # "O4" / "O2"
        out[col] = {
            "all": _cell(rs),
            "train_appearance": _cell([r for r in rs if r.appearance_split == "train"]),
            "test_appearance": _cell([r for r in rs if r.appearance_split == "test"]),
            "per_appearance": {
                app: _cell([r for r in rs if r.appearance_id == app])
                for app in sorted({r.appearance_id for r in rs})
            },
        }
    return out


def confusion(results: list[ItemResult], operator: str) -> dict:
    """Predicted-attribution histogram for one operator (what a blind agent falls back to)."""
    return dict(
        Counter(r.attribution or "REJECT" for r in results if r.operator == operator).most_common()
    )
