"""Cluster-aware statistics and hard acceptance gates for the A3 successor."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

CELLS: tuple[str, ...] = ("T1", "T2", "T3", "T4", "T5")
T3_SUBCELLS: tuple[str, ...] = ("looks_safe", "O8", "reverse")


def appearance_cluster(row: dict[str, Any]) -> str:
    """Independent appearance-level unit, stratified by task and true physical case."""
    cell = str(row["cell"])
    case = str(row.get("t3_sub")) if cell == "T3" else str(row["truth"])
    return f"{cell}|{case}|{row['appearance_id']}"


def accuracy_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Snapshot rates used by the preregistered gates, plus T3 subcells."""
    if not rows:
        raise ValueError("accuracy metrics need non-empty rows")

    def rate(items: list[dict[str, Any]]) -> float:
        if not items:
            raise ValueError("required evaluation stratum is empty")
        return float(np.mean([bool(item["attr_ok"]) for item in items]))

    per_cell = {cell: rate([row for row in rows if row["cell"] == cell]) for cell in CELLS}
    t3 = {
        sub: rate(
            [row for row in rows if row["cell"] == "T3" and row.get("t3_sub") == sub]
        )
        for sub in T3_SUBCELLS
    }
    values = list(per_cell.values())
    return {
        "n": len(rows),
        "per_cell": {key: round(value, 6) for key, value in per_cell.items()},
        "t3_subcells": {key: round(value, 6) for key, value in t3.items()},
        "worst_cell": round(min(values), 6),
        "macro": round(float(np.mean(values)), 6),
    }


def acceptance_audit(
    seed_rows: dict[int, list[dict[str, Any]]], acceptance: dict[str, Any]
) -> dict[str, Any]:
    """Require every requested training seed to pass every explicit gate."""
    expected = int(acceptance["n_train_seeds"])
    if len(seed_rows) != expected:
        raise ValueError(f"acceptance requires {expected} seeds, received {len(seed_rows)}")
    per_seed = []
    for seed, rows in sorted(seed_rows.items()):
        metrics = accuracy_metrics(rows)
        checks = {
            "T1_ge": metrics["per_cell"]["T1"] >= float(acceptance["T1"]),
            "T2_eq": metrics["per_cell"]["T2"] == float(acceptance["T2"]),
            "T3_gt": metrics["per_cell"]["T3"]
            > float(acceptance["T3_strictly_greater_than"]),
            "T4_eq": metrics["per_cell"]["T4"] == float(acceptance["T4"]),
            "T5_ge": metrics["per_cell"]["T5"] >= float(acceptance["T5"]),
            "worst_ge": metrics["worst_cell"] >= float(acceptance["worst_cell"]),
            "macro_ge": metrics["macro"] >= float(acceptance["macro"]),
        }
        for sub, threshold in acceptance["t3_subcells"].items():
            checks[f"T3_{sub}_ge"] = metrics["t3_subcells"][sub] >= float(threshold)
        per_seed.append(
            {"seed": seed, "passed": all(checks.values()), "checks": checks, "metrics": metrics}
        )
    require_every = bool(acceptance.get("require_every_seed", True))
    passed = all(item["passed"] for item in per_seed) if require_every else any(
        item["passed"] for item in per_seed
    )
    return {
        "passed": passed,
        "require_every_seed": require_every,
        "n_seeds": len(per_seed),
        "per_seed": per_seed,
    }


def _cluster_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[appearance_cluster(row)].append(bool(row["attr_ok"]))
    return {key: float(np.mean(values)) for key, values in sorted(grouped.items())}


def appearance_cluster_bootstrap(
    seed_rows: dict[int, list[dict[str, Any]]], *, reps: int, seed: int
) -> dict[str, Any]:
    """Two-level bootstrap: training seed, then appearance clusters within each cell."""
    if reps <= 0:
        raise ValueError("bootstrap reps must be positive")
    scores = {training_seed: _cluster_scores(rows) for training_seed, rows in seed_rows.items()}
    reference = set(next(iter(scores.values())))
    if any(set(values) != reference for values in scores.values()):
        raise ValueError("training seeds do not share identical appearance clusters")
    clusters_by_cell = {
        cell: sorted(key for key in reference if key.startswith(f"{cell}|")) for cell in CELLS
    }
    if any(not keys for keys in clusters_by_cell.values()):
        raise ValueError("bootstrap is missing a required cell")
    rng = np.random.default_rng(seed)
    seed_ids = np.asarray(sorted(scores))
    draws: dict[str, list[float]] = {**{cell: [] for cell in CELLS}, "macro": [], "worst": []}
    for _ in range(reps):
        selected_seed = int(rng.choice(seed_ids))
        cells = []
        for cell, keys in clusters_by_cell.items():
            sampled = rng.choice(keys, size=len(keys), replace=True)
            value = float(np.mean([scores[selected_seed][str(key)] for key in sampled]))
            draws[cell].append(value)
            cells.append(value)
        draws["macro"].append(float(np.mean(cells)))
        draws["worst"].append(float(min(cells)))

    def interval(values: list[float]) -> list[float]:
        return [
            round(float(np.percentile(values, 2.5)), 6),
            round(float(np.percentile(values, 97.5)), 6),
        ]

    observed_by_seed = {
        training_seed: {
            cell: float(
                np.mean(
                    [
                        value
                        for key, value in cluster_scores.items()
                        if key.startswith(f"{cell}|")
                    ]
                )
            )
            for cell in CELLS
        }
        for training_seed, cluster_scores in scores.items()
    }
    observed = {
        cell: float(np.mean([values[cell] for values in observed_by_seed.values()]))
        for cell in CELLS
    }
    return {
        "unit": "appearance_cluster_within_cell",
        "training_seed_resampled": True,
        "n_training_seeds": len(seed_rows),
        "n_clusters": len(reference),
        "clusters_per_cell": {cell: len(keys) for cell, keys in clusters_by_cell.items()},
        "reps": reps,
        "seed": seed,
        "cluster_balanced_point": {
            **{cell: round(value, 6) for cell, value in observed.items()},
            "macro": round(float(np.mean(list(observed.values()))), 6),
            "worst": round(float(min(observed.values())), 6),
        },
        "ci95": {key: interval(values) for key, values in draws.items()},
    }


def _exact_sign_p(positive: int, negative: int) -> float:
    n = positive + negative
    if n == 0:
        return 1.0
    lower = min(positive, negative)
    tail = sum(math.comb(n, index) * 0.5**n for index in range(lower + 1))
    return float(min(1.0, 2.0 * tail))


def paired_cluster_test(
    seed_rows: dict[int, list[dict[str, Any]]],
    baseline_rows: list[dict[str, Any]],
    *,
    reps: int,
    seed: int,
    clusterer: Callable[[dict[str, Any]], str] = appearance_cluster,
    unit: str = "paired_appearance_cluster",
) -> dict[str, Any]:
    """Paired sign test and paired cluster bootstrap on identical sample ids."""
    baseline_by_sid = {str(row["sid"]): row for row in baseline_rows}
    seed_by_sid = {
        training_seed: {str(row["sid"]): row for row in rows}
        for training_seed, rows in seed_rows.items()
    }
    sid_sets = [set(baseline_by_sid), *(set(rows) for rows in seed_by_sid.values())]
    if any(values != sid_sets[0] for values in sid_sets[1:]):
        raise ValueError("paired comparison requires identical sample ids")
    cluster_sids: dict[str, list[str]] = defaultdict(list)
    for sid, row in baseline_by_sid.items():
        cluster_sids[clusterer(row)].append(sid)
    deltas = {}
    for cluster, sids in sorted(cluster_sids.items()):
        ours = float(
            np.mean(
                [
                    bool(seed_by_sid[training_seed][sid]["attr_ok"])
                    for training_seed in sorted(seed_by_sid)
                    for sid in sids
                ]
            )
        )
        baseline = float(np.mean([bool(baseline_by_sid[sid]["attr_ok"]) for sid in sids]))
        deltas[cluster] = ours - baseline
    tolerance = 1.0e-12
    positive = sum(value > tolerance for value in deltas.values())
    negative = sum(value < -tolerance for value in deltas.values())
    tied = len(deltas) - positive - negative
    keys = np.asarray(sorted(deltas))
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(reps):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        bootstrap.append(float(np.mean([deltas[str(key)] for key in sampled])))
    point = float(np.mean(list(deltas.values())))
    return {
        "unit": unit,
        "n_clusters": len(deltas),
        "n_positive": positive,
        "n_negative": negative,
        "n_tied": tied,
        "exact_sign_p_two_sided": _exact_sign_p(positive, negative),
        "ours_minus_baseline_cluster_balanced": round(point, 6),
        "paired_bootstrap_ci95": [
            round(float(np.percentile(bootstrap, 2.5)), 6),
            round(float(np.percentile(bootstrap, 97.5)), 6),
        ],
        "bootstrap_reps": reps,
        "bootstrap_seed": seed,
        "snapshot_independence_assumed": False,
    }
