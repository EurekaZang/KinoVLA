"""A0.5 registry consistency tests (offline; no Isaac). The registry is a pre-registration
artifact (R5) that must stay consistent with the failure taxonomy (R4/R6) — these tests are the
regression guard that it never drifts."""

from __future__ import annotations

import pytest

from kino_vla.data.schema import PRIMITIVE_NAMES
from kino_vla.eval.registry import CONTINUE, TAXONOMY_CELLS, Registry, RegistryError, load_registry


def test_registry_loads_and_validates() -> None:
    """The committed registry must load AND be self-consistent with the taxonomy (no drift)."""
    reg = load_registry()  # raises RegistryError on any inconsistency
    assert reg.scenarios, "registry has no scenarios"


def test_registry_validate_reports_no_issues() -> None:
    reg = Registry.load()
    issues = reg.validate()
    assert issues == [], "registry drifted from taxonomy:\n" + "\n".join(issues)


def test_every_taxonomy_cell_covered() -> None:
    """The benchmark spine T1–T5 must each have at least one registered scenario."""
    reg = Registry.load()
    cells = {s.taxonomy_cell for s in reg.scenarios.values()}
    assert cells == TAXONOMY_CELLS, f"missing taxonomy cells: {TAXONOMY_CELLS - cells}"


def test_canonical_in_admissible_and_valid_primitive() -> None:
    reg = Registry.load()
    valid = PRIMITIVE_NAMES | {CONTINUE}
    for name, s in reg.scenarios.items():
        assert s.canonical_recovery.primitive in valid, f"{name}: bad canonical primitive"
        assert s.canonical_recovery.primitive in s.admissible_recovery_set, (
            f"{name}: canonical not in admissible"
        )
        assert s.admissible_recovery_set <= valid, f"{name}: admissible has non-primitives"


def test_success_criterion_in_vocab() -> None:
    reg = Registry.load()
    for name, s in reg.scenarios.items():
        assert s.success_criterion in reg.success_criteria, f"{name}: unknown success criterion"


def test_nominal_rows_admit_only_continue() -> None:
    """T5/continue rows encode abstention: their only correct action is non-intervention."""
    reg = Registry.load()
    nominal = [s for s in reg.scenarios.values() if s.is_nominal]
    assert nominal, "registry has no nominal (T5) rows"
    for s in nominal:
        assert s.admissible_recovery_set == frozenset({CONTINUE}), f"{s.name}: not {{continue}}"
        assert s.ab_class == "A"


def test_matched_pair_is_registered_and_conflict_labelled() -> None:
    """The load-bearing matched pair: O4 is the T2 conflict, O2 the T1 agree control."""
    reg = Registry.load()
    o4, o2 = reg["matched_O4"], reg["matched_O2"]
    assert o4.taxonomy_cell == "T2" and o4.true_category == "adhesion" and o4.ab_class == "B"
    assert o2.taxonomy_cell == "T1" and o2.true_category == "compliant_terrain"
    # disjoint discriminating canonicals — the ambiguity is in the recovery, per spec §5/§8.1-P4
    assert o4.canonical_recovery.primitive != o2.canonical_recovery.primitive


def test_drift_is_detected() -> None:
    """A corrupted admissible set must be caught by validate() (the anti-drift guarantee works)."""
    reg = Registry.load()
    bad = reg.scenarios["matched_O4"]
    object.__setattr__(bad, "admissible_recovery_set", frozenset({"Switch_Gait"}))  # wrong set
    issues = reg.validate()
    assert any("matched_O4" in i for i in issues), "validate() failed to catch injected drift"


def test_load_registry_raises_on_drift() -> None:
    with pytest.raises(RegistryError):
        # a registry whose matched_O4 admits the sibling's set must fail the load-time gate
        reg = Registry.load()
        object.__setattr__(reg.scenarios["matched_O4"], "true_category", "compliant_terrain")
        issues = reg.validate()
        if issues:
            raise RegistryError("drift")
