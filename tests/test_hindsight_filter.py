"""Truth-consistency filter — golden-file gate + taxonomy unit tests (spec §10 PHASE 3, QA 5.2).

QA 5.2 requires the filter to be tested against frozen synthetic CoTs with known verdicts.
``tests/data/hindsight_golden/cases.json`` is that golden file: it pins every verdict — kept
correct reflections, dropped confabulations (wrong attribution), dropped ambiguity-pair
wrong-sibling strategies (primitive not feasible), dropped same-round detours on sudden traps
(safety), and dropped malformed/non-atomic/off-vocabulary outputs (schema). The filter can
never silently regress without flipping a pinned verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kino_vla.data import FailureTaxonomy, TruthConsistencyFilter
from kino_vla.utils.config import load_config

_CASES = json.loads(
    (Path(__file__).parent / "data" / "hindsight_golden" / "cases.json").read_text()
)["cases"]


@pytest.fixture(scope="module")
def taxonomy() -> FailureTaxonomy:
    return FailureTaxonomy(load_config("data/hindsight.yaml"))


@pytest.fixture(scope="module")
def filt(taxonomy: FailureTaxonomy) -> TruthConsistencyFilter:
    return TruthConsistencyFilter(taxonomy)


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_golden_filter_verdicts(
    case: dict, filt: TruthConsistencyFilter, taxonomy: FailureTaxonomy
):
    gt = taxonomy.ground_truth(case["operator"], case["op_theta"])
    _, verdict = filt.evaluate(case["oracle_text"], gt, prior_outputs=case["prior_outputs"])
    assert verdict.keep == case["expect_keep"], (case["name"], verdict.reason, verdict.detail)
    assert verdict.reason == case["expect_reason"], (case["name"], verdict.reason, verdict.detail)


def test_every_reason_code_is_covered_by_golden():
    """The golden file must exercise all five verdict reasons (no silent gap in coverage)."""
    reasons = {c["expect_reason"] for c in _CASES}
    assert reasons == {
        "keep",
        "attribution_mismatch",
        "primitive_not_feasible",
        "safety_rule_violation",
        "schema_invalid",
    }


def test_taxonomy_operator_categories(taxonomy: FailureTaxonomy):
    assert taxonomy.category_of("O1_mu_field") == "low_friction"
    assert taxonomy.category_of("O3_collapse") == "region_collapse"
    assert taxonomy.category_of("O4_tether") == "adhesion"
    assert taxonomy.category_of("O2_compliance") == "compliant_terrain"
    assert taxonomy.category_of("O5_payload") == "overload"
    assert taxonomy.category_of("O10_effort_decay") == "effort_decay"
    # O7's privileged truth is its PHYSICS (ice), not its deceptive appearance.
    assert taxonomy.category_of("O7_visual_remap") == "low_friction"


def test_taxonomy_ab_boundary_is_theta_refined(taxonomy: FailureTaxonomy):
    """The A/B class flips at the θ threshold for the boundary operators (spec §8.3 Suite-Bound)."""
    assert taxonomy.ground_truth("O2_compliance", {"d_sink": 0.05}).ab_class == "A"
    assert taxonomy.ground_truth("O2_compliance", {"d_sink": 0.18}).ab_class == "B"
    assert taxonomy.ground_truth("O5_payload", {"mass_kg": 3.0}).ab_class == "A"
    assert taxonomy.ground_truth("O5_payload", {"mass_kg": 7.0}).ab_class == "B"
    assert taxonomy.ground_truth("O10_effort_decay", {"floor": 0.5}).ab_class == "A"
    assert taxonomy.ground_truth("O10_effort_decay", {"floor": 0.2}).ab_class == "B"


def test_ambiguity_pairs_have_disjoint_discriminating_primitive(taxonomy: FailureTaxonomy):
    """The methodological crux: a sibling's canonical recovery is INFEASIBLE for the other.

    This is what makes "right vision story, wrong-sibling strategy" droppable by the filter.
    """
    adhesion = taxonomy.ground_truth("O4_tether", {"k": 150.0})
    compliant = taxonomy.ground_truth("O2_compliance", {"k_c": 18.0})
    assert "Backstep" in adhesion.feasible and "Backstep" not in compliant.feasible
    assert "Switch_Gait" in compliant.feasible and "Switch_Gait" not in adhesion.feasible

    overload = taxonomy.ground_truth("O5_payload", {"mass_kg": 7.0})
    decay = taxonomy.ground_truth("O10_effort_decay", {"floor": 0.2})
    assert "Hold_and_Request" in overload.feasible and "Hold_and_Request" not in decay.feasible
    assert "Switch_Gait" in decay.feasible and "Switch_Gait" not in overload.feasible


def test_canonical_primitive_is_always_feasible(taxonomy: FailureTaxonomy):
    """Every category's canonical recovery must lie in its own feasible set (consistency)."""
    for op in (
        "O1_mu_field",
        "O2_compliance",
        "O3_collapse",
        "O4_tether",
        "O5_payload",
        "O10_effort_decay",
        "O8_invisible_collider",
        "O9_high_centering",
    ):
        gt = taxonomy.ground_truth(op, {"mass_kg": 7.0, "floor": 0.2, "d_sink": 0.18})
        assert gt.canonical_primitive in gt.feasible, op
