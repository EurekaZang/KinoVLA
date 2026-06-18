"""Truth-consistency filter — the spec §10 PHASE 3 methodological contribution.

The Oracle (PHASE 3) only looks at a snapshot and tells a story; its CoT may be a fluent
*confabulation*. This filter turns "what the big model guessed" into "physically grounded"
by dropping any sample whose reflection is not consistent with the privileged simulation
truth. A sample is **kept** only if all of:

1. it **parses** into one atomic §5 primitive (the spec's "严格原子动作 / 2D 像素坐标" output
   constraints, enforced as schema validation in ``schema.CoTAnnotation``); else ``DROP_SCHEMA``;
2. its **physical attribution matches the privileged θ** (``DROP_ATTRIBUTION`` otherwise — this
   is the confabulation catch: a vision-led story over a deceptive/ambiguous snapshot);
3. its **chosen primitive lies in the feasible recovery set** for that failure class
   (``DROP_PRIMITIVE`` otherwise — the ambiguity-pair catch: right story, wrong-sibling
   strategy, e.g. attributing adhesion but choosing the mud recovery ``Switch_Gait``);
4. it does not violate the **safety iron-rule** (``DROP_SAFETY``): a sudden trap must be
   escaped before a detour — a same-round ``Replan_Waypoint`` with no prior escape is dropped.

The checks run in that order and the first failure wins, so the ``Verdict.reason`` is a
single, golden-testable code.
"""

from __future__ import annotations

from kino_vla.data.schema import (
    DROP_ATTRIBUTION,
    DROP_PRIMITIVE,
    DROP_SAFETY,
    DROP_SCHEMA,
    KEEP,
    CoTAnnotation,
    CoTParseError,
    GroundTruth,
    Verdict,
)
from kino_vla.data.taxonomy import FailureTaxonomy


class TruthConsistencyFilter:
    """The spec §10 PHASE 3 automated truth-consistency verifier."""

    def __init__(self, taxonomy: FailureTaxonomy) -> None:
        self._tax = taxonomy

    def evaluate(
        self,
        oracle_text: str,
        ground_truth: GroundTruth,
        *,
        prior_outputs: list[str] | None = None,
    ) -> tuple[CoTAnnotation | None, Verdict]:
        """Parse raw Oracle text then judge it; returns ``(annotation_or_None, verdict)``.

        A parse failure yields ``(None, DROP_SCHEMA)`` so the caller can still record the
        dropped sample (and *why*) without a valid annotation.
        """
        try:
            annotation = CoTAnnotation.from_oracle_text(
                oracle_text,
                synonyms=self._tax.synonyms,
                valid_categories=self._tax.valid_categories,
            )
        except CoTParseError as err:
            return None, Verdict(keep=False, reason=DROP_SCHEMA, detail=str(err))
        return annotation, self.check(annotation, ground_truth, prior_outputs=prior_outputs)

    def check(
        self,
        annotation: CoTAnnotation,
        ground_truth: GroundTruth,
        *,
        prior_outputs: list[str] | None = None,
    ) -> Verdict:
        """Judge an already-parsed annotation against the privileged ground truth."""
        # (2) physical attribution must match the privileged truth.
        if annotation.attribution != ground_truth.category:
            return Verdict(
                keep=False,
                reason=DROP_ATTRIBUTION,
                detail=(
                    f"attributed {annotation.attribution!r} but privileged truth is "
                    f"{ground_truth.category!r}"
                ),
            )
        # (3) chosen primitive must be a feasible recovery for that failure class.
        primitive = annotation.primitive.name
        if primitive not in ground_truth.feasible:
            return Verdict(
                keep=False,
                reason=DROP_PRIMITIVE,
                detail=(
                    f"{primitive!r} not in feasible recovery set "
                    f"{sorted(ground_truth.feasible)} for {ground_truth.category!r}"
                ),
            )
        # (4) safety iron-rule: a sudden trap must be escaped before a detour (spec §10).
        if (
            ground_truth.is_sudden_trap
            and primitive == "Replan_Waypoint"
            and not self._escaped_already(prior_outputs)
        ):
            return Verdict(
                keep=False,
                reason=DROP_SAFETY,
                detail=(
                    f"same-round Replan_Waypoint on a sudden trap ({ground_truth.category!r}) "
                    "with no prior escape primitive"
                ),
            )
        return Verdict(keep=True, reason=KEEP, detail="attribution + recovery consistent with θ")

    def _escaped_already(self, prior_outputs: list[str] | None) -> bool:
        """True if a prior-round output already issued an escape primitive (Backstep/Topology)."""
        if not prior_outputs:
            return False
        return any(escape in out for out in prior_outputs for escape in self._tax.escape_primitives)
