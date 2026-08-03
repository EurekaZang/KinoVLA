"""Direct, leakage-resistant C4 selective-recovery evaluation.

C4 is a policy-level estimand.  The unit being compared is the same physical
case from the same pre-decision prefix under:

* the deployed selective policy; and
* the frozen always-safe policy.

Recovery-versus-continue matrices are useful side evidence, but are not
accepted here as a substitute for that direct comparison.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


ESTIMAND = "selective_policy_minus_always_safe_policy"
BRANCHES = {"selective", "always_safe"}


def _cluster_bootstrap(
    pairs: list[dict[str, Any]],
    field: str,
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    """Hierarchical paired bootstrap over scenes and cases within scenes."""
    by_scene: dict[str, list[float]] = defaultdict(list)
    for row in pairs:
        by_scene[str(row["scene_cluster"])].append(float(row[field]))
    scenes = sorted(by_scene)
    if not scenes:
        raise ValueError("C4 bootstrap needs at least one scene cluster")

    rng = np.random.default_rng(seed)
    samples = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        selected_scenes = rng.choice(scenes, size=len(scenes), replace=True)
        scene_values: list[float] = []
        for scene in selected_scenes:
            values = np.asarray(by_scene[str(scene)], dtype=np.float64)
            resampled = rng.choice(values, size=len(values), replace=True)
            scene_values.append(float(np.mean(resampled)))
        samples[draw] = float(np.mean(scene_values))

    scene_means = {
        scene: float(np.mean(values)) for scene, values in sorted(by_scene.items())
    }
    estimate = float(np.mean(list(scene_means.values())))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "estimate": estimate,
        "ci95": [float(lower), float(upper)],
        "scene_cluster_count": len(scenes),
        "case_count": len(pairs),
        "scene_cluster_means": scene_means,
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
        "two_sided_sign_p": float(
            min(
                1.0,
                2.0 * min(np.mean(samples <= 0.0), np.mean(samples >= 0.0)),
            )
        ),
    }


def _decision_audit(
    selective: dict[str, Any],
    always_safe: dict[str, Any],
    *,
    allowed_inputs: set[str],
    forbidden_inputs: set[str],
) -> dict[str, Any]:
    decision = selective.get("decision")
    if not isinstance(decision, dict):
        decision = {}
    safe_decision = always_safe.get("decision")
    if not isinstance(safe_decision, dict):
        safe_decision = {}

    used = {str(value) for value in decision.get("input_fields", [])}
    release = bool(decision.get("release", False))
    safe_action = str(decision.get("safe_action", ""))
    recovery_action = str(decision.get("recovery_action", ""))
    selective_action = str(selective.get("executed_action", ""))
    always_safe_action = str(always_safe.get("executed_action", ""))
    expected_selective_action = recovery_action if release else safe_action

    return {
        "same_frozen_decision_record": decision == safe_decision,
        "used_input_fields": sorted(used),
        "all_inputs_allowed": used <= allowed_inputs,
        "no_forbidden_input_used": not bool(used & forbidden_inputs),
        "release": release,
        "selective_action_matches_gate": (
            bool(expected_selective_action)
            and selective_action == expected_selective_action
        ),
        "always_safe_action_matches_gate": (
            bool(safe_action) and always_safe_action == safe_action
        ),
        "released_action_differs_from_safe": (
            not release or (
                bool(recovery_action)
                and recovery_action != safe_action
                and selective_action != always_safe_action
            )
        ),
    }


def _prefix_tolerance_passed(
    selective: dict[str, Any],
    always_safe: dict[str, Any],
    protocol: dict[str, Any],
) -> bool:
    specification = protocol.get("paired_prefix_audit", {})
    if not bool(specification.get("required", False)):
        return True
    tolerance = float(specification["tolerance"])
    audits = [
        row.get("predecision_tolerance_audit")
        for row in (selective, always_safe)
    ]
    if not all(isinstance(audit, dict) for audit in audits):
        return False
    return bool(
        audits[0] == audits[1]
        and all(bool(audit.get("passed")) for audit in audits)
        and all(
            float(audit.get("tolerance", float("nan"))) == tolerance
            for audit in audits
        )
        and all(
            float(audit.get("maximum_absolute_difference", float("inf")))
            <= tolerance
            for audit in audits
        )
        and selective.get("predecision_pair_certificate_sha256")
        == always_safe.get("predecision_pair_certificate_sha256")
        == selective.get("predecision_rows_sha256")
    )


def evaluate_direct_c4(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a frozen direct selective-versus-always-safe corpus."""
    if not rows:
        raise ValueError("direct C4 evaluation needs non-empty action outcomes")
    if protocol.get("estimand") != ESTIMAND:
        raise ValueError(f"protocol estimand must be {ESTIMAND!r}")

    allowed_inputs = {
        str(value) for value in protocol.get("allowed_decision_inputs", [])
    }
    forbidden_inputs = {
        str(value) for value in protocol.get("forbidden_decision_inputs", [])
    }
    if allowed_inputs & forbidden_inputs:
        raise ValueError("allowed and forbidden decision inputs overlap")

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)

    pair_rows: list[dict[str, Any]] = []
    pair_audits: dict[str, dict[str, Any]] = {}
    incomplete: list[str] = []
    for case_id, group in sorted(by_case.items()):
        by_branch = {str(row.get("branch")): row for row in group}
        if len(group) != 2 or set(by_branch) != BRANCHES:
            incomplete.append(case_id)
            continue
        selective = by_branch["selective"]
        always_safe = by_branch["always_safe"]
        same_step = (
            int(selective["decision_step"]) == int(always_safe["decision_step"])
        )
        same_prefix = (
            bool(selective.get("predecision_rows_sha256"))
            and selective.get("predecision_rows_sha256")
            == always_safe.get("predecision_rows_sha256")
        )
        branches_after_decision = (
            bool(selective.get("branch_started_strictly_after_decision"))
            and bool(always_safe.get("branch_started_strictly_after_decision"))
        )
        tolerance_passed = _prefix_tolerance_passed(
            selective, always_safe, protocol
        )
        decision_audit = _decision_audit(
            selective,
            always_safe,
            allowed_inputs=allowed_inputs,
            forbidden_inputs=forbidden_inputs,
        )
        prefix_passed = (
            same_step
            and same_prefix
            and branches_after_decision
            and tolerance_passed
        )
        execution_passed = all(
            bool(decision_audit[key])
            for key in (
                "same_frozen_decision_record",
                "all_inputs_allowed",
                "no_forbidden_input_used",
                "selective_action_matches_gate",
                "always_safe_action_matches_gate",
                "released_action_differs_from_safe",
            )
        )
        audit = {
            "same_decision_step": same_step,
            "same_predecision_rows_sha256": same_prefix,
            "branches_started_strictly_after_decision": branches_after_decision,
            "frozen_numerical_tolerance_audit_passed": tolerance_passed,
            "prefix_passed": prefix_passed,
            "execution_passed": execution_passed,
            "decision": decision_audit,
        }
        pair_audits[case_id] = audit

        release = bool(decision_audit["release"])
        predicted = str(selective["decision"].get("predicted_attribution", ""))
        truth = str(selective.get("truth_attribution", ""))
        pair_rows.append(
            {
                "case_id": case_id,
                "scene_cluster": str(selective["scene_cluster"]),
                "domain": str(selective["domain"]),
                "split": str(selective.get("split", "")),
                "operator": str(selective.get("operator", "")),
                "release": release,
                "released_attribution_correct": release and predicted == truth,
                "prefix_passed": prefix_passed,
                "execution_passed": execution_passed,
                "selective_minus_always_safe_cost": (
                    float(selective["terminal_cost"])
                    - float(always_safe["terminal_cost"])
                ),
                "selective_minus_always_safe_success": (
                    float(bool(selective["success"]))
                    - float(bool(always_safe["success"]))
                ),
                "selective": {
                    key: selective[key]
                    for key in ("executed_action", "terminal_cost", "success", "fell")
                },
                "always_safe": {
                    key: always_safe[key]
                    for key in ("executed_action", "terminal_cost", "success", "fell")
                },
            }
        )

    eligible = [
        row
        for row in pair_rows
        if row["prefix_passed"] and row["execution_passed"]
    ]
    released = [row for row in eligible if row["release"]]
    released_precision = (
        float(np.mean([row["released_attribution_correct"] for row in released]))
        if released
        else 0.0
    )
    coverage = len(released) / len(eligible) if eligible else 0.0

    bootstrap = protocol.get("bootstrap", {})
    draws = int(bootstrap.get("draws", 20_000))
    seed = int(bootstrap.get("seed", 20260724))
    if eligible:
        cost = _cluster_bootstrap(
            eligible,
            "selective_minus_always_safe_cost",
            seed=seed,
            draws=draws,
        )
        success = _cluster_bootstrap(
            eligible,
            "selective_minus_always_safe_success",
            seed=seed + 1,
            draws=draws,
        )
    else:
        cost = success = None

    requirements = protocol.get("acceptance", {})
    min_cases = int(requirements.get("minimum_paired_cases", 1))
    min_scenes = int(requirements.get("minimum_scene_clusters", 1))
    min_domains = int(requirements.get("minimum_domains", 1))
    min_coverage = float(requirements.get("minimum_release_coverage", 0.0))
    min_precision = float(
        requirements.get("minimum_released_attribution_precision", 0.95)
    )

    acceptance = {
        "all_pairs_complete": not incomplete and len(pair_rows) * 2 == len(rows),
        "all_rows_from_frozen_final_split": all(
            row["split"] == "test" for row in pair_rows
        ),
        "all_prefixes_exactly_matched": (
            bool(pair_rows) and all(row["prefix_passed"] for row in pair_rows)
        ),
        "all_policy_executions_match_frozen_gate": (
            bool(pair_rows) and all(row["execution_passed"] for row in pair_rows)
        ),
        "minimum_paired_cases_met": len(eligible) >= min_cases,
        "minimum_scene_clusters_met": (
            len({row["scene_cluster"] for row in eligible}) >= min_scenes
        ),
        "minimum_domains_met": (
            len({row["domain"] for row in eligible}) >= min_domains
        ),
        "minimum_release_coverage_met": coverage >= min_coverage,
        "released_attribution_precision_ge_0_95": (
            bool(released) and released_precision >= min_precision
        ),
        "selective_minus_always_safe_cluster_ci_upper_le_0": (
            cost is not None and float(cost["ci95"][1]) < 0.0
        ),
        "negative_null_and_harmful_cases_retained": len(pair_rows) == len(by_case),
    }
    complete = all(
        acceptance[key]
        for key in (
            "all_pairs_complete",
            "all_rows_from_frozen_final_split",
            "minimum_paired_cases_met",
            "minimum_scene_clusters_met",
            "minimum_domains_met",
        )
    )
    return {
        "schema_version": "kinofail.realistic-c4-direct-report.v1",
        "status": "confirmatory_complete" if complete else "incomplete",
        "estimand": ESTIMAND,
        "passed": all(acceptance.values()),
        "paired_case_count": len(pair_rows),
        "eligible_paired_case_count": len(eligible),
        "scene_cluster_count": len(
            {row["scene_cluster"] for row in eligible}
        ),
        "domains": sorted({row["domain"] for row in eligible}),
        "release_coverage": coverage,
        "released_case_count": len(released),
        "released_attribution_precision": released_precision,
        "primary_endpoint": cost,
        "secondary_success_endpoint": success,
        "acceptance": acceptance,
        "incomplete_cases": incomplete,
        "pair_audits": pair_audits,
        "paired_cases": pair_rows,
        "claim_boundary": (
            "This report supports C4 only if the frozen deployed selective policy is compared "
            "directly with the frozen always-safe policy from exact matched prefixes on the "
            "untouched test split. Recovery-minus-continue side evidence is not substituted."
        ),
    }
