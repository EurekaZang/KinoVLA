"""A2 eval-core unit tests (torch-free, CPU): the constant-predictor ceiling + the paired stats.

The load-bearing invariant of A2 is that a proprio-only agent — which must emit ONE answer across
the proprio-indistinguishable matched pair — is capped at 0.5 BALANCED accuracy by construction, so
the headline metric cannot be gamed by which constant the agent lands on. These tests pin that
invariant (a constant policy scores 1.0 on one operator, 0.0 on the other, 0.5 balanced), plus the
Wilson CI and the exact paired McNemar used in the publication table.
"""

from __future__ import annotations

import numpy as np

from kino_vla.data.schema import CoTAnnotation, RecoveryPrimitive, Snapshot
from kino_vla.eval.a2_headline import (
    A2Item,
    aggregate,
    eval_policy,
    mcnemar,
    wilson,
)
from kino_vla.eval.suite_sem import _default_params
from kino_vla.vla.output import ParsedDecision


def _snap(op: str) -> Snapshot:
    return Snapshot(
        operator_name=op,
        appearance_class="x",
        t=0.0,
        pose_xy=np.zeros(2),
        heading=0.0,
        rgb=np.zeros((1, 2, 2, 3), dtype=np.float32),
        depth=np.zeros((1, 2, 2), dtype=np.float32),
        proprio_window=np.zeros((5, 11), dtype=np.float32),
        prior_outputs=[],
        privileged_theta={},
        monitor_channel="oracle",
    )


def _items(n_each: int = 10) -> list[A2Item]:
    """A balanced matched set: n_each O4 + n_each O2, split half train / half test appearance."""
    out: list[A2Item] = []
    for i in range(n_each):
        split = "train" if i < n_each // 2 else "test"
        out.append(
            A2Item(
                f"o4_{i}",
                _snap("O4_tether"),
                "O4_tether",
                "adhesion",
                "B",
                "yellow_board" if split == "train" else "gray_tape",
                split,
                frozenset({"Backstep", "Update_Topology"}),
                "O4|O2",
            )
        )
        out.append(
            A2Item(
                f"o2_{i}",
                _snap("O2_compliance"),
                "O2_compliance",
                "compliant_terrain",
                "A",
                "brown_mud" if split == "train" else "reddish_mud",
                split,
                frozenset({"Set_Constraint", "Switch_Gait"}),
                "O4|O2",
            )
        )
    return out


class _Constant:
    """A proprio-blind constant predictor (the B1 pathology on the matched pair)."""

    def __init__(self, category: str, primitive: str) -> None:
        self._c, self._p = category, primitive

    def decide(self, snapshot: Snapshot, map_note: str = "") -> ParsedDecision:  # noqa: ARG002
        prim = RecoveryPrimitive(self._p, _default_params(self._p))
        ann = CoTAnnotation(
            thought="", attribution=self._c, primitive=prim, attribution_raw=self._c, raw_text=""
        )
        return ParsedDecision(ok=True, raw_text="", annotation=ann)


class _Perfect:
    """A perfect attributor (the B5-conflict target): reads the privileged operator."""

    _CANON = {"adhesion": "Backstep", "compliant_terrain": "Switch_Gait"}

    def decide(self, snapshot: Snapshot, map_note: str = "") -> ParsedDecision:  # noqa: ARG002
        cat = "adhesion" if snapshot.operator_name == "O4_tether" else "compliant_terrain"
        prim = RecoveryPrimitive(self._CANON[cat], _default_params(self._CANON[cat]))
        ann = CoTAnnotation(
            thought="", attribution=cat, primitive=prim, attribution_raw=cat, raw_text=""
        )
        return ParsedDecision(ok=True, raw_text="", annotation=ann)


def test_constant_predictor_capped_at_half_balanced() -> None:
    items = _items(20)
    res = eval_policy(_Constant("adhesion", "Backstep"), items)
    agg = aggregate(res)
    # constant "adhesion" ⇒ perfect on O4, zero on O2, 0.5 balanced — the C1/C5 ceiling.
    assert agg["O4"]["all"]["attribution_acc"] == 1.0
    assert agg["O2"]["all"]["attribution_acc"] == 0.0
    assert agg["balanced"]["all"]["attribution_balanced"] == 0.5
    assert agg["balanced"]["all"]["correct_recovery_balanced"] == 0.5
    # and it is capped at 0.5 on the held-out appearance split too.
    assert agg["balanced"]["test_appearance"]["attribution_balanced"] == 0.5


def test_perfect_attributor_reaches_one() -> None:
    items = _items(20)
    agg = aggregate(eval_policy(_Perfect(), items))
    assert agg["balanced"]["all"]["attribution_balanced"] == 1.0
    assert agg["balanced"]["all"]["correct_recovery_balanced"] == 1.0
    # Switch_Gait is admissible for compliant_terrain; Backstep for adhesion ⇒ joint == attr here.
    assert agg["O2"]["all"]["correct_recovery_rate"] == 1.0


def test_wilson_interval() -> None:
    lo, hi = wilson(20, 20)
    assert hi == 1.0 and lo < 1.0 and lo > 0.8  # 20/20 ⇒ upper 1.0, lower Wilson bound ~0.83
    lo0, hi0 = wilson(0, 20)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.2


def test_mcnemar_exact() -> None:
    # lower row right where higher wrong: b; higher right where lower wrong: c.
    a = [True] * 2 + [False] * 10 + [True] * 8  # a correct on 10
    b = [False] * 2 + [True] * 10 + [True] * 8  # b correct on 18; discordant b=2, c=10
    m = mcnemar(a, b)
    assert m["b_lo_right_hi_wrong"] == 2
    assert m["c_lo_wrong_hi_right"] == 10
    assert m["higher_row_better"] is True
    assert m["p_exact_two_sided"] < 0.05  # 2 vs 10 discordant ⇒ significant
    # fully concordant ⇒ p = 1.0
    assert mcnemar([True] * 5, [True] * 5)["p_exact_two_sided"] == 1.0
    # A fully discordant comparison must retain its non-zero exact p-value.
    assert mcnemar([True] * 24, [False] * 24)["p_exact_two_sided"] == 2.0**-23
