"""Suite-Sem attribution eval: VLA (oracle stub) beats the FSM/majority baseline (M7 exit 1)."""

from __future__ import annotations

import numpy as np
import pytest

from kino_vla.data.schema import Snapshot
from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.eval.suite_sem import (
    SemItem,
    ambiguous_appearances,
    compare_vla_vs_fsm,
    evaluate_attribution,
    evaluate_by_regime,
    majority_attribution,
)
from kino_vla.utils.config import load_config
from kino_vla.vla.planner import StubVlaPolicy


@pytest.fixture(scope="module")
def cfg():
    return load_config("data/hindsight.yaml")


@pytest.fixture(scope="module")
def tax(cfg):
    return FailureTaxonomy(cfg)


def _snap(op, appr):
    return Snapshot(
        operator_name=op,
        appearance_class=appr,
        t=2.0,
        pose_xy=np.array([3.0, 0.0]),
        heading=0.0,
        rgb=np.zeros((5, 8, 8, 3), dtype=np.float32),
        depth=np.ones((5, 8, 8), dtype=np.float32),
        proprio_window=np.zeros((25, 11), dtype=np.float32),
        prior_outputs=[],
        privileged_theta={},
        monitor_channel="slip",
    )


def _suite(tax):
    """Suite-Sem items for the O1↔O3 ambiguity pair (both look like ice)."""
    items = []
    for op, cat in [("O1_mu_field", "low_friction"), ("O3_collapse", "region_collapse")]:
        for _ in range(5):
            items.append(
                SemItem(
                    snapshot=_snap(op, "ice_sheet"),
                    attribution_truth=cat,
                    ab_class="A" if op == "O1_mu_field" else "B",
                    ambiguity_pair="O1_mu_field|O3_collapse",
                    primitive_truth=str(tax._canonical.get(cat, "")),
                )
            )
    return items


def test_oracle_vla_perfect_attribution(cfg, tax):
    items = _suite(tax)
    res = evaluate_attribution(StubVlaPolicy(cfg, tax), items)
    assert res["attribution_accuracy"] == 1.0  # the oracle stub attributes every node correctly
    assert res["parse_rate"] == 1.0


def test_vla_beats_fsm_majority_baseline(cfg, tax):
    """Exit criterion 1: the per-snapshot VLA beats the best constant (majority) FSM baseline."""
    items = _suite(tax)
    # train-split categories: skewed O1/O3 ⇒ a constant baseline scores < 1.0 on the pair
    train_cats = ["low_friction"] * 6 + ["region_collapse"] * 4
    cmp = compare_vla_vs_fsm(
        StubVlaPolicy(cfg, tax),
        items,
        cfg=cfg,
        train_categories=train_cats,
        feasible_sets={k: set(v) for k, v in tax._feasible.items()},
    )
    assert cmp["vla_beats_fsm"]
    assert cmp["vla"]["attribution_accuracy"] == 1.0
    assert cmp["fsm_baseline"]["attribution_accuracy"] < 1.0
    assert cmp["margin"] > 0.0


def test_majority_attribution_helper():
    assert majority_attribution(["a", "a", "b"]) == "a"


def _mixed_suite(tax):
    """Ambiguous (ice_sheet → 2 categories) + solvable (mud/adhesive → 1 each) — the §3 regimes."""
    items = []
    for op, cat, appr in [
        ("O1_mu_field", "low_friction", "ice_sheet"),
        ("O3_collapse", "region_collapse", "ice_sheet"),
        ("O2_compliance", "compliant_terrain", "brown_mud"),
        ("O4_tether", "adhesion", "yellow_adhesive"),
    ]:
        for _ in range(3):
            items.append(
                SemItem(
                    snapshot=_snap(op, appr),
                    attribution_truth=cat,
                    ab_class="B",
                    ambiguity_pair="pair",
                    primitive_truth=str(tax._canonical.get(cat, "")),
                )
            )
    return items


def test_ambiguous_appearances_is_data_derived(tax):
    """ice_sheet maps to 2 categories ⇒ appearance-ambiguous; mud/adhesive ⇒ solvable (purity 1)."""
    assert ambiguous_appearances(_mixed_suite(tax)) == {"ice_sheet"}


def test_evaluate_by_regime_partitions_correctly(cfg, tax):
    """The Gap-1 cut: the decisive metric is the appearance-ambiguous (proprio-decided) subset."""
    items = _mixed_suite(tax)
    amb = ambiguous_appearances(items)
    br = evaluate_by_regime(StubVlaPolicy(cfg, tax), items, ambiguous_apps=amb)
    assert br["overall"]["n"] == 12
    assert br["ambiguous"]["n"] == 6  # the two ice_sheet operators
    assert br["solvable"]["n"] == 6  # mud + adhesive
    assert br["ambiguous"]["attribution_accuracy"] == 1.0  # oracle stub attributes both correctly
