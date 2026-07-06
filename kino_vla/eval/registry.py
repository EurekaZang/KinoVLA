"""A0.5 — Success-criterion + admissible-recovery-set registry loader (Paper-A §2 A0.5).

Loads ``configs/eval/a0_registry.yaml`` into typed, validated scenario specs. The registry is a
THIN pre-registration layer over the failure taxonomy (``configs/data/hindsight.yaml`` via
:class:`kino_vla.data.taxonomy.FailureTaxonomy`): it adds the per-scenario success criterion (R5),
taxonomy cell, appearance, and canonical scripted recovery, and **cross-checks** the recovery
semantics against the taxonomy so the two can never silently drift (R4/R6).
:meth:`Registry.validate` is the guarantee — it recomputes ``(true_category, ab_class,
admissible_recovery_set)`` from the taxonomy for every non-nominal scenario and asserts the YAML
matches, and requires nominal/`continue` rows to admit exactly ``{continue}``. A frozen, committed,
self-consistent registry is what makes the A4 attribution-gated correct-recovery metric (R6) prior
to any agent evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kino_vla.data.schema import PRIMITIVE_NAMES
from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.utils.config import Config, load_config

# The non-intervention pseudo-primitive: the correct action on a benign/nominal (T5) row. Not a §5
# library primitive (there is nothing to compile) — the planner issues no semantic override.
CONTINUE = "continue"
TAXONOMY_CELLS = frozenset({"T1", "T2", "T3", "T4", "T5"})
_VALID_PRIMS = PRIMITIVE_NAMES | {CONTINUE}


def _default_taxonomy() -> FailureTaxonomy:
    return FailureTaxonomy(load_config("data/hindsight.yaml"))


@dataclass(frozen=True)
class CanonicalRecovery:
    """The scripted controller A4's label-swap matrix executes for a scenario's true cause."""

    primitive: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioSpec:
    """One pre-registered scenario: privileged ground truth + success criterion + admissible set."""

    name: str
    builder: str
    operator: str
    theta: dict[str, float]
    taxonomy_cell: str
    true_category: str
    ab_class: str
    appearance_class: str
    success_criterion: str
    admissible_recovery_set: frozenset[str]
    canonical_recovery: CanonicalRecovery
    used_by: tuple[str, ...] = ()
    o4_preset: dict[str, float] | None = None
    note: str = ""

    @property
    def is_nominal(self) -> bool:
        """A benign/T5 row whose correct action is non-intervention (`continue`)."""
        return self.success_criterion == "continue"


class RegistryError(ValueError):
    """Raised when the registry is internally inconsistent or drifts from the taxonomy."""


@dataclass(frozen=True)
class Registry:
    """The parsed A0.5 registry: scenarios + success-criterion + forced-label vocabularies."""

    scenarios: dict[str, ScenarioSpec]
    success_criteria: dict[str, dict]
    forced_labels: dict[str, dict]
    meta: dict

    @classmethod
    def load(cls, path: str = "eval/a0_registry.yaml") -> Registry:
        cfg = load_config(path)
        d = cfg.to_dict()
        scenarios: dict[str, ScenarioSpec] = {}
        for name, s in d["scenarios"].items():
            cr = s["canonical_recovery"]
            scenarios[name] = ScenarioSpec(
                name=name,
                builder=str(s["builder"]),
                operator=str(s["operator"]),
                theta={k: float(v) for k, v in s.get("theta", {}).items()},
                taxonomy_cell=str(s["taxonomy_cell"]),
                true_category=str(s["true_category"]),
                ab_class=str(s["ab_class"]),
                appearance_class=str(s["appearance_class"]),
                success_criterion=str(s["success_criterion"]),
                admissible_recovery_set=frozenset(s["admissible_recovery_set"]),
                canonical_recovery=CanonicalRecovery(
                    primitive=str(cr["primitive"]), params=dict(cr.get("params", {}))
                ),
                used_by=tuple(s.get("used_by", [])),
                o4_preset=(dict(s["o4_preset"]) if "o4_preset" in s else None),
                note=str(s.get("note", "")),
            )
        return cls(
            scenarios=scenarios,
            success_criteria=dict(d["success_criteria"]),
            forced_labels=dict(d["forced_labels"]),
            meta=dict(d.get("meta", {})),
        )

    # ------------------------------------------------------------------- validation
    def validate(self, taxonomy: FailureTaxonomy | None = None) -> list[str]:
        """Return a list of consistency issues ([] ⇒ valid). Recomputes the recovery semantics from
        the taxonomy for every non-nominal scenario and asserts the YAML matches (R4/R6 anti-drift).
        """
        tax = taxonomy if taxonomy is not None else _default_taxonomy()
        issues: list[str] = []
        for name, sc in self.scenarios.items():
            if sc.success_criterion not in self.success_criteria:
                issues.append(f"{name}: unknown success_criterion {sc.success_criterion!r}")
            if sc.taxonomy_cell not in TAXONOMY_CELLS:
                issues.append(f"{name}: bad taxonomy_cell {sc.taxonomy_cell!r}")
            bad_prims = sc.admissible_recovery_set - _VALID_PRIMS
            if bad_prims:
                issues.append(f"{name}: admissible set has non-primitives {sorted(bad_prims)}")
            if sc.canonical_recovery.primitive not in _VALID_PRIMS:
                issues.append(f"{name}: canonical prim {sc.canonical_recovery.primitive!r} bad")
            if sc.canonical_recovery.primitive not in sc.admissible_recovery_set:
                issues.append(
                    f"{name}: canonical {sc.canonical_recovery.primitive!r} ∉ admissible "
                    f"{sorted(sc.admissible_recovery_set)}"
                )
            if sc.is_nominal:
                # A benign/T5 row: correct action is non-intervention; admit exactly {continue}.
                if sc.admissible_recovery_set != frozenset({CONTINUE}):
                    issues.append(f"{name}: nominal row must admit exactly {{continue}}")
                if sc.ab_class != "A":
                    issues.append(f"{name}: nominal row must be ab_class A, got {sc.ab_class}")
                continue
            # Non-nominal: the recovery semantics MUST be the taxonomy's, recomputed from op + θ.
            try:
                gt = tax.ground_truth(sc.operator, sc.theta)
            except KeyError as e:  # unknown operator
                issues.append(f"{name}: {e}")
                continue
            if sc.true_category != gt.category:
                issues.append(f"{name}: true_category {sc.true_category!r} != tax {gt.category!r}")
            if sc.ab_class != gt.ab_class:
                issues.append(f"{name}: ab_class {sc.ab_class!r} != tax {gt.ab_class!r}")
            if sc.admissible_recovery_set != gt.feasible:
                issues.append(
                    f"{name}: admissible {sorted(sc.admissible_recovery_set)} ≠ taxonomy feasible "
                    f"{sorted(gt.feasible)} for {gt.category}"
                )
        return issues

    # ------------------------------------------------------------------- accessors
    def by_cell(self, cell: str) -> list[ScenarioSpec]:
        return [s for s in self.scenarios.values() if s.taxonomy_cell == cell]

    def by_experiment(self, exp: str) -> list[ScenarioSpec]:
        return [s for s in self.scenarios.values() if exp in s.used_by]

    def __getitem__(self, name: str) -> ScenarioSpec:
        return self.scenarios[name]


def load_registry(
    path: str = "eval/a0_registry.yaml", *, taxonomy_cfg: Config | None = None
) -> Registry:
    """Load AND validate the A0.5 registry; raise :class:`RegistryError` on any inconsistency."""
    reg = Registry.load(path)
    tax = FailureTaxonomy(taxonomy_cfg) if taxonomy_cfg is not None else _default_taxonomy()
    issues = reg.validate(tax)
    if issues:
        raise RegistryError("A0.5 registry inconsistent with taxonomy:\n  " + "\n  ".join(issues))
    return reg
