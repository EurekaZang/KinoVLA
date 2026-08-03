"""A4.3 — interventional consequence analysis (Paper-A §4 A4.3, serves C3).

Pure-logic reduction of the interventional label-swap matrix ``M(s, ℓ)``. Consumes per-episode
:class:`Outcome` dicts (the matrix runner's output) and per-snapshot agent label distributions
(A2/A3 eval output) and produces the three C3 readouts:

  1. **``M(s, ℓ)``** — the causal object: success rate (+ catastrophic-event rates + mean physical
     cost) for each (scenario, forced-label) cell over N seeds. The claim lands iff the DIAGONAL
     (canonical label) ≫ off-diagonal, with the specific antisymmetries:
       - ``high_step`` on matched-O4 (adhesion) → catapult/immobilization (catastrophic) while
         ``backstep_detour`` on matched-O2 (mud) → success-but-slow (benign);
       - ``continue`` on nominal rows wins while any intervention strands (E4's misfire →
         ``Set_Constraint`` → stranded-at-3 pathology, now the MEASURED cost of false intervention).
  2. **Cost asymmetry → safe default.** A scalar ordinal physical cost per outcome; under a belief
     ``p = P(adhesion)`` the expected cost of each action crosses over at a SMALL ``p*`` because
     catastrophic ≫ benign ⇒ the conservative recovery (Backstep) is the expected-cost-minimizing
     action whenever adhesion is non-negligible. This is the decision-theoretic content of "Safe"
     that replaces the cut CBF chapter (experiments_design.md §0.3).
  3. **Action / attribution composition.** ``ERS(agent) = E_s[Σ_ℓ
     P̂_agent(ℓ|s)·cost(s,ℓ)]`` and true empirical regret relative to the best measured matrix
     action link A2/A3 predictions to physical cost.  The emitted action and the action implied by
     the predicted attribution are reported separately; conflating them can manufacture a causal
     bridge when a model emits an attribution/action-inconsistent response.

No Isaac here — tested with synthetic outcomes (``tests/test_a4_consequence.py``). Wilson CIs follow
the §7 statistics protocol; the matrix runner stamps every cell with (config hash, seeds, commit) +
the A0.1 permutation-variance certificate.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

# Default ordinal physical-cost levels (a fall is worst; a clean reach is free). The qualitative
# safe-default result (conservative recovery wins under adhesion-uncertainty) is ROBUST to these
# weights as long as catastrophic (fall/catapult/immobilized) ≫ benign (slow_success). Overridable
# via the cost config so the exact magnitudes are not magic numbers in code (QA 5.1.5).
DEFAULT_COST_LEVELS: dict[str, float] = {
    "clean_success": 0.0,
    "slow_success": 1.0,  # reached but slow / via a detour — the benign cost of a safe detour
    "stranded": 3.0,  # no fall, no reach — a wasted run (E4's Set_Constraint strand)
    "immobilized": 4.0,  # stalled in-hazard (v < ε for > 5 s) — cannot proceed
    "catapult": 5.0,  # tether-tear energy-release event — uncontrolled
    "fall": 6.0,  # the robot fell — catastrophic
}
DEFAULT_SLOW_THR_S = 20.0  # a reach slower than this is a "slow_success" (benign cost), not clean

# The forced-label vocabulary mirrors configs/eval/a0_registry.yaml forced_labels. A chosen recovery
# PRIMITIVE (what A2/A3 agents emit) maps onto exactly one forced label so the agent distribution
# composes with M. ``continue`` is the non-intervention pseudo-primitive (T5 nominal rows).
LABELS: tuple[str, ...] = (
    "continue",
    "backstep_detour",
    "high_step",
    "slow_low",
    "crawl",
    "hold_request",
    "detour_replan",
)

# Primitive-name → forced-label map (the A2/A3 agents emit primitives; M's columns are labels).
_PRIM_TO_LABEL: dict[str, str] = {
    "Backstep": "backstep_detour",
    "Set_Constraint": "slow_low",
    "Hold_and_Request": "hold_request",
    "Update_Topology": "detour_replan",
    "Replan_Waypoint": "detour_replan",
    "continue": "continue",
}
# Switch_Gait splits on its mode parameter: high_step → high_step label, crawl → crawl, else trot
# (treated as a power-through gait closest to high_step on mud). Adjust_Posture rides with the
# slow/low-posture family (its canonical use is the ice low-posture traverse).


@dataclass(frozen=True)
class Outcome:
    """One interventional episode: scenario s under forced label ℓ, seed."""

    scenario: str
    label: str
    seed: int
    reached: bool
    fell: bool
    immobilized: bool  # v < ε for > 5 s while in-hazard (observable, R8-clean)
    catapult: bool  # tether-tear energy-release event (observable motion signature)
    final_dist_m: float
    sim_time_s: float
    peak_omega: float  # max |yaw_rate| over the episode (rad/s)
    n_semantic_interventions: int
    left_hazard: bool  # the trunk exited the hazard rect with net forward progress
    success: bool  # the pre-registered registry-criterion verdict (R5)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson 95% CI on a binomial proportion (§7 statistics protocol)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def physical_cost(
    o: Outcome,
    levels: dict[str, float] | None = None,
    slow_thr_s: float = DEFAULT_SLOW_THR_S,
) -> float:
    """Scalar ordinal physical cost of one outcome (lower is better). Catastrophic events dominate;
    any registry SUCCESS (reach / safe_halt / escape / continue) is benign (the verdict defines the
    safe outcome — a Hold under overload is NOT a "strand"); a non-success non-catastrophe
    is a strand. The ordering, not magnitudes, drives the safe-default result."""
    lv = levels or DEFAULT_COST_LEVELS
    if o.fell:
        return lv["fall"]
    if o.catapult:
        return lv["catapult"]
    if o.immobilized:
        return lv["immobilized"]
    if o.success:  # any registry-criterion success ⇒ benign (reach, safe stop, escape, abstain)
        return lv["slow_success"] if o.sim_time_s > slow_thr_s else lv["clean_success"]
    return lv["stranded"]


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return round(sum(xs) / max(1, len(xs)), 4)


def cell_stats(
    cell: list[Outcome],
    levels: dict[str, float] | None = None,
    slow_thr_s: float = DEFAULT_SLOW_THR_S,
) -> dict:
    """Reduce one (scenario, label) cell over its seeds → success rate + Wilson CI + catastrophic
    rates + mean cost + mean peak |ω| + mean final distance."""
    n = len(cell)
    k = sum(1 for o in cell if o.success)
    costs = [physical_cost(o, levels, slow_thr_s) for o in cell]
    return {
        "n": n,
        "success_rate": round(k / max(1, n), 4),
        "success_ci": wilson(k, n),
        "fell_rate": round(sum(1 for o in cell if o.fell) / max(1, n), 4),
        "catapult_rate": round(sum(1 for o in cell if o.catapult) / max(1, n), 4),
        "immobilized_rate": round(sum(1 for o in cell if o.immobilized) / max(1, n), 4),
        "mean_cost": _mean(costs),
        "mean_peak_omega": _mean([o.peak_omega for o in cell]),
        "mean_final_dist_m": _mean([o.final_dist_m for o in cell]),
        "mean_time_s": _mean([o.sim_time_s for o in cell]),
    }


def M_matrix(
    outcomes: list[Outcome],
    scenarios: list[str],
    labels: tuple[str, ...] = LABELS,
    levels: dict[str, float] | None = None,
    slow_thr_s: float = DEFAULT_SLOW_THR_S,
) -> dict[str, dict[str, dict]]:
    """``M[s][ℓ] = cell_stats`` over seeds — A4's causal object (success + cost per cell)."""
    m: dict[str, dict[str, dict]] = {}
    for s in scenarios:
        m[s] = {}
        for lab in labels:
            cell = [o for o in outcomes if o.scenario == s and o.label == lab]
            m[s][lab] = (
                cell_stats(cell, levels, slow_thr_s)
                if cell
                else {
                    "n": 0,
                    "success_rate": 0.0,
                    "success_ci": [0.0, 0.0],
                    "fell_rate": 0.0,
                    "catapult_rate": 0.0,
                    "immobilized_rate": 0.0,
                    "mean_cost": 0.0,
                    "mean_peak_omega": 0.0,
                    "mean_final_dist_m": 0.0,
                    "mean_time_s": 0.0,
                }
            )
    return m


def safe_default_crossover(
    diag_cost: dict[str, float], off_cost: dict[str, float], actions: tuple[str, str]
) -> dict:
    """The cost-asymmetry → safe-default derivation for a two-action ambiguity pair.

    ``actions = (A, B)`` are the two candidate recoveries; ``p = P(truth matched to A)`` (so for the
    adhesion/mud pair pass ``(backstep_detour, high_step)`` with ``p = P(adhesion)``). ``diag_cost``
    is the cost of doing X when it is CORRECT; ``off_cost[X]`` is the cost of doing X when it is
    WRONG (the mis-attribution cost — e.g. high-stepping into adhesion). Then:

        E[cost|do A] = p·diag[A] + (1-p)·off[A]     (A correct on A-truth, A-wrong on B-truth)
        E[cost|do B] = p·off[B]  + (1-p)·diag[B]     (B-wrong on A-truth, B correct on B-truth)

    The optimal action flips at ``p*``. If ``off[B]`` (B's catastrophic mis-attribution) ≫
    ``off[A]`` (A's benign mis-attribution), ``p*`` is SMALL ⇒ A (the conservative recovery) is the
    expected-cost-minimizing action for almost any non-negligible ``p``. Returns ``p*``, the
    safe-default action on each side of it, and the cost-asymmetry ratio ``off[B]/off[A]``.
    """
    a, b = actions
    da, db = diag_cost[a], diag_cost[b]
    oa, ob = off_cost[a], off_cost[b]
    # Crossover: p·da + (1-p)·oa = p·ob + (1-p)·db  ⇒  p(da - oa - ob + db) = db - oa
    denom = da - oa - ob + db
    p_star = (db - oa) / denom if abs(denom) > 1e-12 else 0.5
    p_star = min(1.0, max(0.0, p_star))
    # Sign of (E[A]-E[B]) slope vs p = denom. denom<0 ⇒ E[A]-E[B] decreases with p, so A is cheaper
    # ABOVE p* and B below it.
    a_above = denom < 0.0
    asym = ob / oa if oa > 0 else float("inf")  # how many times worse B's mis-attribution is
    return {
        "actions": list(actions),
        "p_is": "P(truth matched to " + a + ")",
        "diag_cost": {a: da, b: db},
        "off_cost": {a: oa, b: ob},  # cost of the WRONG action on each truth
        "p_star": round(p_star, 4),
        "cost_asymmetry_ratio": round(asym, 4) if math.isfinite(asym) else None,
        "safe_default_below_pstar": b if a_above else a,
        "safe_default_above_pstar": a if a_above else b,
    }


def primitive_to_label(
    primitive: str, params: dict | None = None, *, strict_params: bool = False
) -> str:
    """Map an A2/A3 agent's emitted primitive (+ its params) onto a forced-label column of M."""
    if primitive in _PRIM_TO_LABEL:
        return _PRIM_TO_LABEL[primitive]
    if primitive == "Switch_Gait":
        mode = (params or {}).get("mode", "")
        if strict_params and mode not in {"high_step", "crawl"}:
            raise ValueError("Switch_Gait requires an observed mode for A4 action composition")
        return "crawl" if mode == "crawl" else "high_step"
    if primitive == "Adjust_Posture":
        return "slow_low"
    return "continue"


_ATTR_TO_PRIMITIVE: dict[str, tuple[str, dict]] = {
    "nominal": ("continue", {}),
    "low_friction": ("Set_Constraint", {}),
    "compliant_terrain": ("Switch_Gait", {"mode": "high_step"}),
    "region_collapse": ("Update_Topology", {}),
    "adhesion": ("Backstep", {}),
    "overload": ("Hold_and_Request", {}),
    "effort_decay": ("Switch_Gait", {"mode": "crawl"}),
    "external_push": ("Set_Constraint", {}),
    "invisible_obstacle": ("Update_Topology", {}),
    "high_centering": ("Adjust_Posture", {}),
    "obs_bias": ("Set_Constraint", {}),
}


def attribution_to_label(attribution: str | None) -> str:
    """Map a predicted cause through the frozen recovery taxonomy to an A4 matrix label.

    This is a mechanism diagnostic, not the action the robot necessarily executed.  The actual
    action must be obtained from ``primitive`` + the observed ``primitive_params``.
    """
    prim, params = _ATTR_TO_PRIMITIVE.get(str(attribution), ("continue", {}))
    return primitive_to_label(prim, params, strict_params=True)


def a3_row_to_a4_scenario(row: dict) -> str | None:
    """Map one frozen A3 evaluation row onto the corresponding A4 intervention scenario.

    T5 rows are decision-labelled ``nominal`` even though the simulator retains a weak physical
    operator category.  Their mapping must therefore use frozen operator/sample identity.  Keeping
    this bridge in the eval core prevents A4 and A5 from silently using different mappings.
    """
    cell, truth, sub = row.get("cell"), row.get("truth"), row.get("t3_sub", "")
    operator = str(row.get("operator", ""))
    sid = str(row.get("sid", ""))
    if cell == "T2" and truth == "adhesion":
        return "matched_O4_twophase"
    if cell == "T1" and truth == "compliant_terrain":
        return "matched_O2"
    if cell == "T1" and truth == "low_friction":
        return "O1_ice"
    if cell == "T3" and sub == "looks_safe":
        return "O7_looks_safe"
    if cell == "T3" and sub == "reverse":
        return "O7_reverse"
    if cell == "T3" and truth == "invisible_obstacle":
        return "O8_invisible"
    if cell == "T4" and truth == "overload":
        return "O5_payload_B"
    if cell == "T4" and truth == "effort_decay":
        return "O10_decay_B"
    if cell == "T5" and (operator == "O2_compliance" or "_O2_A_nominal_" in sid):
        return "O2_A_nominal"
    if cell == "T5" and (operator == "O1_mu_field" or "_O1_A_nominal_" in sid):
        return "O1_A_nominal"
    if cell == "T5" and (operator == "O10_effort_decay" or "_O10_A_nominal_" in sid):
        return "O10_A_nominal"
    if cell == "T5" and (operator == "O6_push" or "_O6_push_A_" in sid):
        return "O6_push_A"
    return None


def agent_label_distribution(per_item: list[dict], scenario: str) -> dict[str, float]:
    """``P̂_agent(ℓ | scenario)`` from the A2/A3 per-snapshot agent predictions: each snapshot's
    emitted primitive (+params) maps to a label; average over the scenario's snapshots."""
    rows = [r for r in per_item if r.get("scenario") == scenario]
    if not rows:  # fall back to the A3 per_item key ``cell`` if scenario absent
        rows = [r for r in per_item if r.get("cell") == scenario]
    n = len(rows)
    if n == 0:
        return {lab: 0.0 for lab in LABELS}
    counts: dict[str, int] = {lab: 0 for lab in LABELS}
    for r in rows:
        prim = r.get("primitive") or r.get("first_primitive")
        params = r.get("primitive_params") or {}
        if prim is None and r.get("attribution") in (None, "nominal"):
            prim = "continue"
        lab = primitive_to_label(prim, params) if prim else "continue"
        counts[lab] = counts.get(lab, 0) + 1
    return {lab: round(counts.get(lab, 0) / n, 4) for lab in LABELS}


def expected_outcome(
    agent_dist: dict[str, dict[str, float]],
    m: dict[str, dict[str, dict]],
    scenarios: list[str],
    *,
    metric: str = "mean_cost",
) -> dict[str, float]:
    """ERS: ``E_s[Σ_ℓ P̂_agent(ℓ|s)·M(s,ℓ)[metric]]`` — the agent's expected per-scenario
    success/cost composed from its label distribution and the matrix. ``metric='success_rate'`` ⇒
    spec ERS (higher better); ``metric='mean_cost'`` ⇒ expected physical cost (lower better)."""
    out: dict[str, float] = {}
    for s in scenarios:
        if s not in agent_dist:
            raise ValueError(f"agent distribution is missing matrix scenario {s!r}")
        dist = agent_dist[s]
        mass = sum(float(dist.get(lab, 0.0)) for lab in LABELS)
        if not math.isclose(mass, 1.0, abs_tol=1e-3):
            raise ValueError(f"agent distribution for {s!r} has probability mass {mass:.6f}, not 1")
        cells = m.get(s, {})
        total = 0.0
        for lab in LABELS:
            p = dist.get(lab, 0.0)
            cell = cells.get(lab, {})
            if p > 0.0 and int(cell.get("n", 0)) == 0:
                raise ValueError(f"M has no measured cell for scenario={s!r}, label={lab!r}")
            v = cell.get(metric, 0.0)
            total += p * v
        out[s] = round(total, 4)
    return out


@dataclass
class CompositionResult:
    """Expected physical outcome for one empirical policy over the matrix scenarios."""

    ers_per_scenario: dict[str, float]  # expected metric per scenario
    ers_mean: float  # E_s[ERS]
    success_per_scenario: dict[str, float]
    success_mean: float
    regret_per_scenario: dict[str, float]  # E[cost] - min measured cost; 0 ⇒ empirically optimal
    regret_mean: float
    canonical_gap_per_scenario: dict[str, float]  # signed E[cost] - canonical cost
    canonical_gap_mean: float
    mode_label_per_scenario: dict[str, str]  # most frequent emitted label (descriptive only)

    @property
    def argmin_label_per_scenario(self) -> dict[str, str]:
        """Backward-compatible alias; historical name was incorrect (this is an argmax mode)."""
        return self.mode_label_per_scenario


def compose_agent(
    agent_dist: dict[str, dict[str, float]],
    m: dict[str, dict[str, dict]],
    scenarios: list[str],
    canonical: dict[str, str],
) -> CompositionResult:
    """Compose an empirical label distribution with the measured intervention matrix.

    ERS and regret both integrate the full empirical distribution.  The old implementation used
    the distribution for ERS but its argmax for regret, which made the two estimands incomparable.
    ``regret`` is now relative to the lowest measured cost in each scenario.  The signed gap to the
    pre-registered canonical action is retained separately because weak matrix rows can contain a
    non-canonical action that empirically ties or beats the canonical controller.
    """
    ers = expected_outcome(agent_dist, m, scenarios, metric="mean_cost")
    success = expected_outcome(agent_dist, m, scenarios, metric="success_rate")
    regret: dict[str, float] = {}
    canonical_gap: dict[str, float] = {}
    mode: dict[str, str] = {}
    for s in scenarios:
        dist = agent_dist[s]
        cells = m[s]
        mode[s] = max(LABELS, key=lambda lab: dist.get(lab, 0.0))
        measured = [float(cells[lab]["mean_cost"]) for lab in LABELS if cells[lab].get("n", 0)]
        if not measured:
            raise ValueError(f"M has no measured labels for scenario {s!r}")
        best_cost = min(measured)
        c_canon = float(cells[canonical[s]]["mean_cost"])
        regret[s] = round(max(0.0, ers[s] - best_cost), 4)
        canonical_gap[s] = round(ers[s] - c_canon, 4)
    return CompositionResult(
        ers_per_scenario=ers,
        ers_mean=round(sum(ers.values()) / max(1, len(ers)), 4),
        success_per_scenario=success,
        success_mean=round(sum(success.values()) / max(1, len(success)), 4),
        regret_per_scenario=regret,
        regret_mean=round(sum(regret.values()) / max(1, len(regret)), 4),
        canonical_gap_per_scenario=canonical_gap,
        canonical_gap_mean=round(sum(canonical_gap.values()) / max(1, len(canonical_gap)), 4),
        mode_label_per_scenario=mode,
    )
