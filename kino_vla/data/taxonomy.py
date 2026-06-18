"""Failure taxonomy: privileged ground-truth attribution + feasible recovery sets (spec §10).

This is the *anchor* the truth-consistency filter checks Oracle CoTs against (spec §10 PHASE
3 "CoT 中的物理归因结论必须与仿真特权真值一致，且其选择的原语必须属于该失效类别的可行恢复集").
It is the single, config-driven source (``configs/data/hindsight.yaml``) for three mappings:

1. **operator + θ → canonical attribution category** (e.g. ``O1_mu_field`` → ``low_friction``);
2. **category → A/B class** (spec §2.5 recoverability dichotomy), θ-refined at the A/B
   boundary for O2/O5/O10 (the Suite-Bound θ\\* idea, spec §8.3);
3. **category → feasible recovery set + canonical primitive** (the §5 many-to-many map).

The ambiguity pairs are encoded *structurally*: ``adhesion`` and ``compliant_terrain`` have
disjoint discriminating primitives (Backstep vs Switch_Gait), as do ``overload`` and
``effort_decay`` (Hold_and_Request vs Switch_Gait) — so a CoT that tells the right vision
story but picks the *sibling's* strategy lands outside the feasible set and is dropped.
"""

from __future__ import annotations

from kino_vla.data.schema import GroundTruth
from kino_vla.utils.config import Config


class FailureTaxonomy:
    """Config-driven privileged-truth attribution and feasible-recovery knowledge."""

    def __init__(self, cfg: Config) -> None:
        attr = cfg.attribution
        rec = cfg.recovery
        self._operator_category: dict[str, str] = dict(attr.operator_category.to_dict())
        self._default_ab: dict[str, str] = dict(attr.default_ab_class.to_dict())
        self._ab_thresholds: dict[str, dict] = (
            attr.ab_thresholds.to_dict() if "ab_thresholds" in attr else {}
        )
        self._feasible: dict[str, frozenset[str]] = {
            cat: frozenset(prims) for cat, prims in rec.feasible.to_dict().items()
        }
        self._canonical: dict[str, str] = dict(rec.canonical.to_dict())
        self._sudden_trap: frozenset[str] = frozenset(rec.sudden_trap)
        self._escape: frozenset[str] = frozenset(rec.escape_primitives)
        self._synonyms: dict[str, str] = dict(cfg.synonyms.to_dict())
        # The Oracle's category vocabulary: every category with a feasible set, plus "nominal"
        # (the "nothing is wrong" attribution, which is a mismatch on any real failure).
        self._valid_categories: frozenset[str] = frozenset(self._feasible) | {"nominal"}

    # ----------------------------------------------------------------- accessors
    @property
    def valid_categories(self) -> frozenset[str]:
        return self._valid_categories

    @property
    def synonyms(self) -> dict[str, str]:
        return dict(self._synonyms)

    @property
    def escape_primitives(self) -> frozenset[str]:
        return self._escape

    def is_sudden_trap(self, category: str) -> bool:
        return category in self._sudden_trap

    def category_of(self, operator_name: str) -> str:
        if operator_name not in self._operator_category:
            raise KeyError(f"operator {operator_name!r} has no attribution category in config")
        return self._operator_category[operator_name]

    # ------------------------------------------------------------- ground truth
    def ground_truth(self, operator_name: str, op_theta: dict[str, float]) -> GroundTruth:
        """Privileged ground truth for a failure caused by ``operator_name`` with θ ``op_theta``.

        ``op_theta`` is the operator's own ``get_privileged_state()`` (unprefixed keys), used
        to refine the A/B class at the boundary (deep sink / saturating mass / severe derate).
        """
        category = self.category_of(operator_name)
        ab_class = self._ab_class(operator_name, category, op_theta)
        feasible = self._feasible.get(category, frozenset())
        canonical = self._canonical.get(category, "")
        return GroundTruth(
            category=category,
            ab_class=ab_class,
            theta={k: float(v) for k, v in op_theta.items()},
            feasible=feasible,
            canonical_primitive=canonical,
            is_sudden_trap=self.is_sudden_trap(category),
        )

    def _ab_class(self, operator_name: str, category: str, op_theta: dict[str, float]) -> str:
        """A/B class. For a boundary operator (O2/O5/O10) the θ threshold is *authoritative*
        (the A/B split is θ-dependent, spec §8.3): below it the failure is A, above it B. For
        every other operator the class is the category default (intrinsic A or B)."""
        rule = self._ab_thresholds.get(operator_name)
        if rule is not None:
            value = op_theta.get(rule["param"])
            if value is not None:
                if "b_above" in rule:
                    return "B" if float(value) > float(rule["b_above"]) else "A"
                if "b_below" in rule:
                    return "B" if float(value) < float(rule["b_below"]) else "A"
        return self._default_ab.get(category, "A")
