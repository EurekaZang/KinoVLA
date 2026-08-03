"""Class-balanced nav re-sampling (kino_vla.vla.sft.balance_train) — the data-level Turn fix (#43).

Pure logic (no torch / no GPU): the band-aid `nav_turn_oversample` is replaced by a principled
balance that lifts the nav-Turn class to parity WITHOUT perturbing the recovery distribution.
"""

from __future__ import annotations

import numpy as np

from kino_vla.vla.dataset_build import VlaExample
from kino_vla.vla.sft import balance_train


def _ex(sid: str, operator: str, primitive: str) -> VlaExample:
    return VlaExample(
        sample_id=sid,
        operator_name=operator,
        appearance_class="x",
        ambiguity_pair=None,
        ab_class="nav" if operator == "navigation" else "A",
        attribution_truth="nominal" if operator == "navigation" else "low_friction",
        primitive_truth=primitive,
        messages=[],
        target_text="",
        rgb=np.zeros((1, 2, 2, 3), dtype=np.float32),
        proprio_window=np.zeros((4, 11), dtype=np.float32),
        target_theta=None if operator == "navigation" else [0.1, 0.0, 1.0, 1.0],
    )


def _make(n_recovery: int, n_waypoint: int, n_turn: int) -> list[VlaExample]:
    train = [_ex(f"rec_{i}", "O1_mu_field", "Set_Constraint") for i in range(n_recovery)]
    train += [_ex(f"wp_{i}", "navigation", "waypoint") for i in range(n_waypoint)]
    train += [_ex(f"tn_{i}", "navigation", "turn") for i in range(n_turn)]
    return train


def _counts(train):
    rec = sum(e.operator_name != "navigation" for e in train)
    wp = sum(e.operator_name == "navigation" and e.primitive_truth == "waypoint" for e in train)
    tn = sum(e.operator_name == "navigation" and e.primitive_truth == "turn" for e in train)
    return rec, wp, tn


def test_turn_lifted_to_parity_with_waypoint():
    """The collapse fix: 73 turn vs 157 waypoint ⇒ after balancing, turn ≈ waypoint."""
    out = balance_train(_make(300, 157, 73), nav_weight=1.0)
    rec, wp, tn = _counts(out)
    assert tn == wp, "Turn is oversampled to parity with the majority nav kind"
    assert tn >= 73, "never DOWNsamples below the observed count"


def test_recovery_distribution_is_untouched():
    """Recovery examples are kept verbatim — protects the M7 attribution ceiling (0.974)."""
    train = _make(300, 157, 73)
    out = balance_train(train, nav_weight=1.0)
    rec_before = [e for e in train if e.operator_name != "navigation"]
    rec_after = [e for e in out if e.operator_name != "navigation"]
    assert len(rec_after) == len(rec_before)
    assert [e.sample_id for e in rec_after] == [e.sample_id for e in rec_before]


def test_nav_weight_scales_the_nav_stratum():
    """nav_weight floors the nav budget at nav_weight × recovery (split across kinds)."""
    small = balance_train(_make(300, 50, 20), nav_weight=1.0)
    big = balance_train(_make(300, 50, 20), nav_weight=2.0)
    assert _counts(big)[1] >= _counts(small)[1]  # heavier nav_weight ⇒ ≥ nav weight


def test_deterministic_and_capped():
    """Same input ⇒ same output (reproducible); per-kind oversampling is capped."""
    a = balance_train(_make(50, 10, 2), nav_weight=1.0, max_factor=3)
    b = balance_train(_make(50, 10, 2), nav_weight=1.0, max_factor=3)
    assert [e.sample_id for e in a] == [e.sample_id for e in b]
    _, _, tn = _counts(a)
    assert tn <= 2 * 3, "turn oversampling respects max_factor (2 originals × 3)"


def test_nav_turn_frac_damps_over_turning():
    """#44 round-2 lever: nav_turn_frac<1 yields fewer turns than waypoints (curbs over-turning)."""
    out = balance_train(_make(300, 157, 73), nav_weight=1.0, nav_turn_frac=0.6)
    _, wp, tn = _counts(out)
    assert tn < wp, "turn fraction < 1 ⇒ fewer turns than waypoints"
    assert tn == round(wp * 0.6), "turn count ≈ waypoint × nav_turn_frac"


def test_no_nav_examples_is_a_noop():
    train = _make(100, 0, 0)
    assert balance_train(train) == train  # recovery-only ⇒ unchanged
