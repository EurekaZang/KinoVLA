"""Calibration-only structured gate for consequence-aware recovery selection.

The gate is deliberately downstream of the frozen VLA.  It may execute the VLA action only when
the action is the registry-canonical primitive for the VLA posterior category and calibration
shows that this observable stratum is never worse than the conservative fallback in any
appearance cluster, with a strictly lower aggregate physical cost.  Otherwise it falls back.

No privileged truth, scenario id, physical theta, or test-split statistic enters the deployed
decision.  Scenario ids and A4 costs are used only to score calibration/evaluation artifacts.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

import numpy as np

COST_LABEL_TO_PRIMITIVE = {
    "backstep_detour": "Backstep",
    "high_step": "Switch_Gait",
    "crawl": "Switch_Gait",
    "slow_low": "Set_Constraint",
    "hold_request": "Hold_and_Request",
    "detour_replan": "Update_Topology",
    "continue": "continue",
}


def _pair(row: dict[str, Any], category_field: str) -> tuple[str, str]:
    return str(row["agent_label"]), str(row[category_field])


def fit_structured_gate(
    calibration_rows: list[dict[str, Any]],
    canonical_primitive: dict[str, str],
    *,
    min_clusters: int = 2,
    tolerance: float = 1.0e-12,
    category_field: str = "posterior_argmax",
) -> dict[str, Any]:
    """Fit a registry-constrained gate using calibration rows only.

    A stratum is accepted iff:
    1. its action maps to the registry-canonical primitive for its posterior category;
    2. it differs from the fallback action;
    3. it occurs in at least ``min_clusters`` appearance clusters;
    4. agent-minus-safe cost is non-positive in every cluster and negative in aggregate.
    """
    if not calibration_rows:
        raise ValueError("structured gate needs non-empty calibration rows")
    bad_splits = {
        str(row.get("appearance_split"))
        for row in calibration_rows
        if str(row.get("appearance_split")) != "train"
    }
    if bad_splits:
        raise ValueError(f"fit received non-calibration appearance splits: {sorted(bad_splits)}")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in calibration_rows:
        grouped[_pair(row, category_field)].append(row)

    accepted: list[dict[str, str]] = []
    audit: dict[str, dict[str, Any]] = {}
    for (label, category), rows in sorted(grouped.items()):
        by_cluster: dict[str, list[float]] = defaultdict(list)
        deltas: list[float] = []
        for row in rows:
            delta = float(row["cost_agent"]) - float(row["cost_safe"])
            deltas.append(delta)
            by_cluster[str(row["cluster_id"])].append(delta)
        cluster_delta = {
            cluster: float(np.mean(values)) for cluster, values in sorted(by_cluster.items())
        }
        primitive = COST_LABEL_TO_PRIMITIVE.get(label)
        expected = canonical_primitive.get(category)
        semantic_consistent = primitive is not None and primitive == expected
        changes_fallback = any(label != str(row["safe_label"]) for row in rows)
        enough_clusters = len(by_cluster) >= int(min_clusters)
        aggregate_delta = float(np.mean(deltas))
        no_cluster_harm = max(cluster_delta.values()) <= tolerance
        strict_aggregate_gain = aggregate_delta < -tolerance
        use = all(
            (
                semantic_consistent,
                changes_fallback,
                enough_clusters,
                no_cluster_harm,
                strict_aggregate_gain,
            )
        )
        key = f"{label}|{category}"
        audit[key] = {
            "agent_label": label,
            category_field: category,
            "n": len(rows),
            "n_clusters": len(by_cluster),
            "primitive": primitive,
            "registry_canonical_primitive": expected,
            "semantic_consistent": semantic_consistent,
            "changes_fallback": changes_fallback,
            "aggregate_agent_minus_safe": round(aggregate_delta, 6),
            "cluster_agent_minus_safe": {
                name: round(value, 6) for name, value in cluster_delta.items()
            },
            "no_calibration_cluster_harm": no_cluster_harm,
            "accepted": use,
        }
        if use:
            accepted.append({"agent_label": label, category_field: category})

    return {
        "schema_version": 1,
        "decision_inputs": ["agent_label", category_field],
        "category_field": category_field,
        "forbidden_inputs": ["truth", "scenario", "theta", "appearance_split", "test_cost"],
        "fallback": "safe_label",
        "min_calibration_clusters": int(min_clusters),
        "accepted_strata": accepted,
        "calibration_audit": audit,
    }


def fit_supported_structured_gate(
    fit_rows: list[dict[str, Any]],
    canonical_primitive: dict[str, str],
    *,
    category_field: str = "attribution",
    support_field: str = "material_distance",
    calibration_split: str = "train",
    min_clusters: int = 2,
    min_fit_coverage: float = 0.10,
    tolerance: float = 1.0e-12,
    allowed_strata: list[tuple[str, str]] | None = None,
    validity_field: str | None = None,
    evidence_field: str | None = None,
) -> dict[str, Any]:
    """Fit a registry-consistent gate and a support radius before final evaluation.

    Action/category strata are learned only from the frozen calibration split.  A maximum support
    distance is then selected on all explicitly supplied method-development rows.  The largest
    radius is retained only if the resulting selective policy has non-positive cost delta in every
    appearance cluster, a strict aggregate gain, and the requested non-zero coverage.  Supplying a
    final/evaluation row here is an error.
    """
    if not fit_rows:
        raise ValueError("supported structured gate needs non-empty fit rows")
    bad = sorted(
        {
            str(row.get("method_split"))
            for row in fit_rows
            if str(row.get("method_split")) not in {calibration_split, "development"}
        }
    )
    if bad:
        raise ValueError(f"fit received final/evaluation method splits: {bad}")
    calibration = [
        {**row, "appearance_split": "train"}
        for row in fit_rows
        if str(row.get("method_split")) == calibration_split
    ]
    base = fit_structured_gate(
        calibration,
        canonical_primitive,
        min_clusters=min_clusters,
        tolerance=tolerance,
        category_field=category_field,
    )
    accepted = {
        (str(item["agent_label"]), str(item[category_field])) for item in base["accepted_strata"]
    }
    if allowed_strata is not None:
        allowed = {(str(label), str(category)) for label, category in allowed_strata}
        accepted &= allowed
        base["accepted_strata"] = [
            item
            for item in base["accepted_strata"]
            if (str(item["agent_label"]), str(item[category_field])) in accepted
        ]
    if not accepted:
        raise ValueError("calibration produced no registry-consistent beneficial stratum")

    observed_support = sorted(
        {
            float(row[support_field])
            for row in fit_rows
            if _pair(row, category_field) in accepted and np.isfinite(float(row[support_field]))
        }
    )
    if not observed_support:
        raise ValueError("no finite support values for accepted calibration strata")

    def boundaries(values: list[float]) -> list[float]:
        """Observed values plus between-value margins with identical decisions."""
        return sorted(
            {
                *values,
                *(
                    0.5 * (left + right)
                    for left, right in zip(values[:-1], values[1:], strict=True)
                ),
            }
        )

    candidate_values = boundaries(observed_support)
    if evidence_field is None:
        candidate_evidence: list[float | None] = [None]
    else:
        observed_evidence = sorted(
            {
                float(row[evidence_field])
                for row in fit_rows
                if _pair(row, category_field) in accepted
                and np.isfinite(float(row[evidence_field]))
            }
        )
        if not observed_evidence:
            raise ValueError("no finite evidence values for accepted calibration strata")
        candidate_evidence = boundaries(observed_evidence)

    radius_audit: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for radius in candidate_values:
        for minimum_evidence in candidate_evidence:
            item = _audit_supported_operating_point(
                fit_rows,
                accepted=accepted,
                category_field=category_field,
                support_field=support_field,
                radius=radius,
                evidence_field=evidence_field,
                minimum_evidence=minimum_evidence,
                validity_field=validity_field,
                min_fit_coverage=min_fit_coverage,
                tolerance=tolerance,
            )
            radius_audit.append(item)
            if item["eligible"]:
                eligible.append(item)
    if not eligible:
        raise ValueError(
            "no support radius satisfies coverage, gain, validity, and cluster no-harm constraints"
        )

    # Maximise verified deployment coverage.  Ties prefer a wider visual support margin and the
    # smallest anomaly threshold, both of which are more robust without changing fit decisions.
    selected = max(
        eligible,
        key=lambda item: (
            float(item["coverage"]),
            float(item["max_distance"]),
            -float(item.get("minimum_evidence", float("-inf"))),
        ),
    )
    gate = deepcopy(base)
    gate.update(
        {
            "schema_version": 2,
            "decision_inputs": [
                "agent_label",
                category_field,
                support_field,
                *([evidence_field] if evidence_field is not None else []),
            ],
            "support": {
                "field": support_field,
                "metric": "nearest_calibration_material_prototype_l2",
                "max_value": float(selected["max_distance"]),
            },
            "fit_method_splits": sorted({str(row["method_split"]) for row in fit_rows}),
            "min_fit_coverage": float(min_fit_coverage),
            "support_radius_selection": "largest verified coverage under per-cluster no-harm",
            "fit_validity_field": validity_field,
            "selected_radius_audit": selected,
            "radius_audit": radius_audit,
        }
    )
    if evidence_field is not None:
        gate["evidence"] = {
            "field": evidence_field,
            "metric": "maximum_tracking_error_in_observation_window",
            "min_value": float(selected["minimum_evidence"]),
        }
    gate["forbidden_inputs"] = [
        "truth",
        "scenario",
        "theta",
        "appearance_id",
        "appearance_split",
        "method_split",
        "test_cost",
    ]
    return gate


def _audit_supported_operating_point(
    fit_rows: list[dict[str, Any]],
    *,
    accepted: set[tuple[str, str]],
    category_field: str,
    support_field: str,
    radius: float,
    evidence_field: str | None,
    minimum_evidence: float | None,
    validity_field: str | None,
    min_fit_coverage: float,
    tolerance: float,
) -> dict[str, Any]:
    """Audit one pre-final support/evidence operating point."""
    by_cluster: dict[str, list[float]] = defaultdict(list)
    covered = 0
    invalid_acts = 0
    deltas: list[float] = []
    for row in fit_rows:
        act = (
            _pair(row, category_field) in accepted
            and float(row[support_field]) <= radius + tolerance
            and (
                evidence_field is None
                or float(row[evidence_field]) + tolerance >= float(minimum_evidence)
            )
        )
        delta = float(row["cost_agent"]) - float(row["cost_safe"]) if act else 0.0
        covered += int(act)
        invalid_acts += int(
            act and validity_field is not None and not bool(row.get(validity_field, False))
        )
        deltas.append(delta)
        by_cluster[str(row["cluster_id"])].append(delta)
    cluster_delta = {
        cluster: float(np.mean(values)) for cluster, values in sorted(by_cluster.items())
    }
    coverage = covered / len(fit_rows)
    aggregate_delta = float(np.mean(deltas))
    no_cluster_harm = max(cluster_delta.values()) <= tolerance
    valid = (
        coverage + tolerance >= float(min_fit_coverage)
        and aggregate_delta < -tolerance
        and no_cluster_harm
        and invalid_acts == 0
    )
    return {
        "max_distance": round(radius, 8),
        **(
            {"minimum_evidence": round(float(minimum_evidence), 8)}
            if minimum_evidence is not None
            else {}
        ),
        "coverage": round(coverage, 6),
        "aggregate_agent_minus_safe": round(aggregate_delta, 6),
        "max_cluster_agent_minus_safe": round(max(cluster_delta.values()), 6),
        "no_fit_cluster_harm": no_cluster_harm,
        "invalid_acts": invalid_acts,
        "all_acted_rows_valid": invalid_acts == 0,
        "eligible": valid,
    }


def gate_acts(row: dict[str, Any], gate: dict[str, Any]) -> bool:
    category_field = str(gate.get("category_field", "posterior_argmax"))
    accepted = {
        (str(item["agent_label"]), str(item[category_field])) for item in gate["accepted_strata"]
    }
    if _pair(row, category_field) not in accepted:
        return False
    support = gate.get("support")
    if support is None:
        support_ok = True
    else:
        field = str(support["field"])
        value = row.get(field)
        support_ok = (
            value is not None
            and np.isfinite(float(value))
            and float(value) <= float(support["max_value"])
        )
    evidence = gate.get("evidence")
    if evidence is None:
        evidence_ok = True
    else:
        field = str(evidence["field"])
        value = row.get(field)
        evidence_ok = (
            value is not None
            and np.isfinite(float(value))
            and float(value) >= float(evidence["min_value"])
        )
    return support_ok and evidence_ok


def evaluate_gate(rows: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        raise ValueError("structured gate evaluation needs non-empty rows")
    selective = [
        float(row["cost_agent"] if gate_acts(row, gate) else row["cost_safe"]) for row in rows
    ]
    baselines = {
        "agent": float(np.mean([float(row["cost_agent"]) for row in rows])),
        "safe": float(np.mean([float(row["cost_safe"]) for row in rows])),
        "continue": float(np.mean([float(row["cost_continue"]) for row in rows])),
    }
    cost = float(np.mean(selective))
    coverage = float(np.mean([gate_acts(row, gate) for row in rows]))
    return {
        "n": len(rows),
        "coverage": round(coverage, 4),
        "expected_cost": round(cost, 4),
        "baseline_cost": {name: round(value, 4) for name, value in baselines.items()},
        "paired_delta": {name: round(cost - value, 4) for name, value in baselines.items()},
    }


def bootstrap_gate(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
    episode_costs: dict[tuple[str, str], list[float]],
    *,
    reps: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Two-stage bootstrap over appearance clusters and A4 physical episodes."""
    by_scenario: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_scenario[str(row["scenario"])][str(row["cluster_id"])].append(row)
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in ("structured", "agent", "safe", "continue")}
    coverages: list[float] = []
    for _ in range(int(reps)):
        sampled: list[dict[str, Any]] = []
        for clusters in by_scenario.values():
            keys = sorted(clusters)
            for index in rng.integers(0, len(keys), size=len(keys)):
                sampled.extend(clusters[keys[int(index)]])
        costs = {
            key: float(np.mean(rng.choice(values, size=len(values), replace=True)))
            for key, values in episode_costs.items()
        }
        values = {name: [] for name in draws}
        covered = 0
        for row in sampled:
            act = gate_acts(row, gate)
            covered += int(act)
            labels = {
                "structured": row["agent_label"] if act else row["safe_label"],
                "agent": row["agent_label"],
                "safe": row["safe_label"],
                "continue": row["continue_label"],
            }
            for name, label in labels.items():
                values[name].append(costs[(str(row["scenario"]), str(label))])
        for name in draws:
            draws[name].append(float(np.mean(values[name])))
        coverages.append(covered / len(sampled))

    def ci(values: list[float]) -> list[float]:
        return [
            round(float(np.quantile(values, 0.025)), 4),
            round(float(np.quantile(values, 0.975)), 4),
        ]

    deltas = {
        f"structured_minus_{name}": [
            structured - baseline
            for structured, baseline in zip(draws["structured"], draws[name], strict=True)
        ]
        for name in ("agent", "safe", "continue")
    }
    delta_ci = {name: ci(values) for name, values in deltas.items()}
    return {
        "reps": int(reps),
        "unit": "appearance cluster within scenario + A4 episode within (scenario,label)",
        "cost_ci": {name: ci(values) for name, values in draws.items()},
        "coverage_ci": ci(coverages),
        "paired_delta_ci": delta_ci,
        "strictly_better_with_95ci": {
            name.removeprefix("structured_minus_"): bounds[1] < 0.0
            for name, bounds in delta_ci.items()
        },
    }
