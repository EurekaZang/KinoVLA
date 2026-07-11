"""A8 metrics helpers (Paper-A external validity).

Torch-free reducers for official Guardian/FailCoT scores and A8b conflict-balanced
attribution. Reuses Wilson/McNemar conventions from A7.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

from kino_vla.eval.a7_ablation import mcnemar, proportion_cell, wilson


def accuracy_ci(rows: Sequence[dict[str, Any]], field: str = "correct") -> dict[str, Any]:
    """Wilson CI over boolean/int correctness rows."""
    return proportion_cell(list(rows), field=field)


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Unweighted mean of per-class F1; undefined labels yield 0 contribution."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    if not y_true:
        return 0.0
    labels = sorted(set(y_true) | set(y_pred))
    scores: list[float] = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        if prec + rec == 0.0:
            scores.append(0.0)
        else:
            scores.append(2.0 * prec * rec / (prec + rec))
    return float(sum(scores) / len(scores)) if scores else 0.0


def per_category_recall(
    y_true: Sequence[str], y_pred: Sequence[str]
) -> dict[str, dict[str, float | int]]:
    """Recall per ground-truth category with support counts."""
    by_lab: dict[str, list[bool]] = defaultdict(list)
    for t, p in zip(y_true, y_pred):
        by_lab[str(t)].append(bool(p == t))
    out: dict[str, dict[str, float | int]] = {}
    for lab, oks in sorted(by_lab.items()):
        k = sum(1 for x in oks if x)
        n = len(oks)
        lo, hi = wilson(k, n)
        out[lab] = {
            "n": n,
            "k": k,
            "recall": round(k / n, 3) if n else 0.0,
            "ci": [round(lo, 3), round(hi, 3)],
        }
    return out


def cba(acc_e2: float, acc_e3: float) -> float:
    """Conflict-balanced attribution: mean of vision-true and proprio-true accuracies."""
    return round(0.5 * (float(acc_e2) + float(acc_e3)), 6)


def delta_fusion(cba_vp: float, cba_v: float, cba_p: float) -> float:
    """Fusion gain over the stronger unimodal branch."""
    return round(float(cba_vp) - max(float(cba_v), float(cba_p)), 6)


def delta_conflict(cba_conflict: float, cba_unshaped: float) -> float:
    """Conflict-training gain over ordinary fusion."""
    return round(float(cba_conflict) - float(cba_unshaped), 6)


def stratum_accuracy(
    rows: Sequence[dict[str, Any]],
    *,
    stratum_field: str = "stratum",
    correct_field: str = "correct",
) -> dict[str, dict[str, Any]]:
    """Per-stratum Wilson accuracy cells."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[str(r.get(stratum_field, "unassigned"))].append(r)
    return {k: accuracy_ci(v, field=correct_field) for k, v in sorted(buckets.items())}


def method_conflict_summary(
    rows_by_arm: dict[str, Sequence[dict[str, Any]]],
    *,
    e2: str = "E2",
    e3: str = "E3",
    correct_field: str = "correct",
    stratum_field: str = "stratum",
) -> dict[str, Any]:
    """Compute CBA / Δ_fusion / Δ_conflict from per-arm prediction rows."""

    def _acc(rows: Sequence[dict[str, Any]], stratum: str) -> float:
        sub = [r for r in rows if str(r.get(stratum_field)) == stratum]
        if not sub:
            return float("nan")
        return float(accuracy_ci(sub, field=correct_field)["rate"])

    arm_stats: dict[str, Any] = {}
    for arm, rows in rows_by_arm.items():
        a2 = _acc(rows, e2)
        a3 = _acc(rows, e3)
        arm_stats[arm] = {
            "acc_e2": a2,
            "acc_e3": a3,
            "cba": cba(a2, a3) if a2 == a2 and a3 == a3 else float("nan"),
            "by_stratum": stratum_accuracy(rows, stratum_field=stratum_field, correct_field=correct_field),
        }

    def _cba(arm: str) -> float:
        return float(arm_stats[arm]["cba"])

    out: dict[str, Any] = {"arms": arm_stats}
    if all(k in arm_stats for k in ("v_only", "p_only", "latent")):
        out["delta_fusion_latent"] = delta_fusion(
            _cba("latent"), _cba("v_only"), _cba("p_only")
        )
    if all(k in arm_stats for k in ("v_only", "p_only", "concat")):
        out["delta_fusion_concat"] = delta_fusion(
            _cba("concat"), _cba("v_only"), _cba("p_only")
        )
    if "latent_conflict" in arm_stats and "latent" in arm_stats:
        out["delta_conflict"] = delta_conflict(_cba("latent_conflict"), _cba("latent"))
    return out


def paired_mcnemar(
    rows_a: Sequence[dict[str, Any]],
    rows_b: Sequence[dict[str, Any]],
    *,
    id_field: str = "sample_id",
    correct_field: str = "correct",
) -> dict[str, Any]:
    """McNemar on shared sample IDs."""
    a_map = {r[id_field]: bool(r[correct_field]) for r in rows_a if id_field in r}
    b_map = {r[id_field]: bool(r[correct_field]) for r in rows_b if id_field in r}
    shared = sorted(set(a_map) & set(b_map))
    if not shared:
        return {"n_shared": 0, "status": "no_shared_ids"}
    return {
        "n_shared": len(shared),
        **mcnemar([a_map[i] for i in shared], [b_map[i] for i in shared]),
    }


def label_counts(values: Iterable[str]) -> dict[str, int]:
    return dict(Counter(str(v) for v in values))
