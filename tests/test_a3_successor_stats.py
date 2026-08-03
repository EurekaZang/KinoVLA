from __future__ import annotations

from kino_vla.eval.a3_successor_stats import (
    acceptance_audit,
    appearance_cluster_bootstrap,
    paired_cluster_test,
)
from scripts.a3_successor_v2_publication_report import _canonical_recovery_certificate


def _rows(*, wrong: tuple[str, str] | None = None):
    rows = []
    truth = {
        "T1": "low_friction",
        "T2": "adhesion",
        "T3": "low_friction",
        "T4": "overload",
        "T5": "nominal",
    }
    for cell in ("T1", "T2", "T3", "T4", "T5"):
        subs = ("looks_safe", "O8", "reverse") if cell == "T3" else ("other",)
        for sub in subs:
            for app in ("a", "b"):
                sid = f"{cell}-{sub}-{app}"
                rows.append(
                    {
                        "sid": sid,
                        "cell": cell,
                        "t3_sub": sub,
                        "truth": "nominal" if sub == "reverse" else truth[cell],
                        "appearance_id": app,
                        "attr_ok": wrong != (cell, app),
                    }
                )
    return rows


def _acceptance():
    return {
        "T1": 0.9,
        "T2": 1.0,
        "T3_strictly_greater_than": 0.917,
        "T4": 1.0,
        "T5": 0.9,
        "t3_subcells": {"looks_safe": 0.9, "O8": 0.9, "reverse": 0.9},
        "worst_cell": 0.9,
        "macro": 0.95,
        "n_train_seeds": 5,
        "require_every_seed": True,
    }


def test_hard_gate_requires_every_seed_and_all_t3_subcells():
    seeds = {seed: _rows() for seed in range(5)}
    assert acceptance_audit(seeds, _acceptance())["passed"]
    seeds[4] = _rows(wrong=("T2", "a"))
    audit = acceptance_audit(seeds, _acceptance())
    assert not audit["passed"]
    assert not audit["per_seed"][4]["checks"]["T2_eq"]


def test_cluster_bootstrap_resamples_clusters_not_snapshots():
    result = appearance_cluster_bootstrap(
        {seed: _rows() for seed in range(5)}, reps=100, seed=7
    )
    assert result["n_clusters"] == 14
    assert result["clusters_per_cell"]["T3"] == 6
    assert result["ci95"]["macro"] == [1.0, 1.0]
    assert result["training_seed_resampled"]


def test_paired_test_aligns_sids_and_uses_cluster_signs():
    ours = {seed: _rows() for seed in range(5)}
    baseline = _rows(wrong=("T2", "a"))
    result = paired_cluster_test(ours, baseline, reps=100, seed=3)
    assert result["n_positive"] == 1
    assert result["n_negative"] == 0
    assert result["n_tied"] == 13
    assert result["snapshot_independence_assumed"] is False

    coarsened = paired_cluster_test(
        ours,
        baseline,
        reps=100,
        seed=4,
        clusterer=lambda row: str(row["appearance_id"]),
        unit="paired_appearance_id",
    )
    assert coarsened["unit"] == "paired_appearance_id"
    assert coarsened["n_clusters"] == 2
    assert coarsened["n_positive"] == 1


def test_recovery_certificate_applies_t5_decision_overlay_without_mutating_physics():
    categories = {
        "T1": ("low_friction", "Set_Constraint"),
        "T2": ("adhesion", "Backstep"),
        "T3": ("invisible_obstacle", "Update_Topology"),
        "T4": ("overload", "Hold_and_Request"),
        "T5": ("nominal", "continue"),
    }
    rows = []
    records = []
    canonical = {}
    for cell, (truth, primitive) in categories.items():
        sid = f"{cell}-sample"
        rows.append(
            {
                "sid": sid,
                "cell": cell,
                "t3_sub": "O8" if cell == "T3" else "other",
                "truth": truth,
                "appearance_id": f"{cell}-appearance",
                "attribution": truth,
                "attr_ok": True,
            }
        )
        records.append(
            {
                "sample_id": sid,
                "admissible_recovery_set": (
                    ["Set_Constraint"] if cell == "T5" else [primitive]
                ),
            }
        )
        canonical[truth] = primitive
    result = _canonical_recovery_certificate({0: rows}, records, canonical)
    assert result["n_rows_with_t5_decision_overlay"] == 1
    assert result["per_seed"]["0"]["attribution_and_action_rate"] == 1.0
    assert result["strict_cluster_certificate"]["overall"]["rate"] == 1.0
