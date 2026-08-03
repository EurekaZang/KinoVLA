#!/usr/bin/env python3
"""Build a machine-derived C1--C4 evidence ledger from frozen reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c1", type=Path, required=True)
    parser.add_argument("--c2", type=Path, required=True)
    parser.add_argument("--c3", type=Path, required=True)
    parser.add_argument("--c4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "C1": args.c1.resolve(),
        "C2": args.c2.resolve(),
        "C3": args.c3.resolve(),
        "C4": args.c4.resolve(),
    }
    reports = {claim: _json(path) for claim, path in paths.items()}
    c1, c2, c3, c4 = (reports[key] for key in paths)
    selected = c2.get("selected_method")
    if selected is None:
        candidates = [
            name
            for name in c2["results"]
            if str(name).startswith("structured_")
        ]
        if len(candidates) != 1:
            raise KeyError(
                "C2 report needs selected_method or one structured result"
            )
        selected = candidates[0]
    c2_method = str(selected)
    if c2_method not in c2["results"]:
        raise KeyError(
            f"C2 selected method absent from results: {c2_method}"
        )
    c2_ours = c2["results"][c2_method]
    c2_best = str(c2["best_baseline_on_formal_battery"])
    c2_core_checks = {
        key: bool(value)
        for key, value in c2["checks"].items()
        if key != "beats_every_baseline_point"
    }
    c2_core_supported = all(c2_core_checks.values())
    claims = {
        "C1": {
            "passed": c1.get("passed") is True,
            "core_claim_supported": c1.get("passed") is True,
            "evidence": {
                "cases": c1["counts"]["cases"],
                "samples": c1["counts"]["samples"],
                "scene_clusters": c1["counts"]["scene_clusters"],
                "domains": c1["counts"]["domains"],
                "visual_macro_within_scene_auc": c1["visual"][
                    "primary_macro_within_scene_roc_auc"
                ],
                "visual_scene_cluster_ci95": c1["visual"][
                    "scene_cluster_bootstrap_ci95"
                ],
                "visual_balanced_accuracy": c1["visual"][
                    "balanced_accuracy"
                ],
                "texture_swap_hard_consistency": c1["visual"][
                    "texture_swap"
                ]["hard_consistency"],
                "proprio_auc": c1["proprio"]["roc_auc"],
                "proprio_byte_identity_rate": c1["proprio"][
                    "byte_identity_rate"
                ],
            },
            "boundary": c1["claim_boundary"],
        },
        "C2": {
            "passed": c2.get("passed") is True,
            "core_claim_supported": c2_core_supported,
            "evidence": {
                "samples": c2["test_counts"]["samples"],
                "matched_cases": c2["test_counts"]["matched_cases"],
                "scene_clusters": c2["test_counts"]["scene_clusters"],
                "domains": c2["test_counts"]["domains"],
                "ours_balanced_accuracy": c2_ours["balanced_accuracy"],
                "ours_t2_accuracy": c2_ours["per_cell_accuracy"][
                    "T2_vision_decisive"
                ],
                "ours_t3_accuracy": c2_ours["per_cell_accuracy"][
                    "T3_proprio_decisive"
                ],
                "ours_worst_scene_accuracy": c2_ours[
                    "worst_scene_accuracy"
                ],
                "ours_texture_swap_hard_consistency": c2_ours[
                    "texture_swap_hard_consistency"
                ],
                "selected_method": c2_method,
                "best_baseline": c2_best,
                "best_baseline_balanced_accuracy": c2["results"][c2_best][
                    "balanced_accuracy"
                ],
                "matched_delta": c2["ours_minus_best_baseline"],
                "all_preregistered_checks_pass": all(
                    c2["checks"].values()
                ),
                "core_checks_excluding_strict_point_superiority": (
                    c2_core_checks
                ),
                "strict_point_superiority_passed": c2["checks"][
                    "beats_every_baseline_point"
                ],
            },
            "boundary": c2["claim_boundary"],
        },
        "C3": {
            "passed": c3.get("passed") is True,
            "core_claim_supported": c3.get("passed") is True,
            "evidence": {
                "controlled_episodes": c3[
                    "controlled_label_intervention"
                ]["episodes"],
                "cost_sensitive_scenarios": c3[
                    "controlled_label_intervention"
                ]["cost_sensitive_scenarios"],
                "success_sensitive_scenarios": c3[
                    "controlled_label_intervention"
                ]["success_sensitive_scenarios"],
                "unique_winner_seed_pairs": c3[
                    "controlled_label_intervention"
                ]["unique_winner_paired_seed_pairs"],
                "unique_winner_exact_sign_p_two_sided": c3[
                    "controlled_label_intervention"
                ]["unique_winner_exact_sign_p_two_sided"],
                "t2_cost_asymmetry_ratio": c3[
                    "asymmetric_t2_decision"
                ]["cost_asymmetry_ratio"],
                "posterior_crossing_p_star": c3[
                    "asymmetric_t2_decision"
                ]["posterior_crossing_p_star"],
                "realistic_bridge_paired_cases": c3[
                    "realistic_action_outcome_bridge"
                ]["paired_cases"],
            },
            "boundary": c3["claim_boundary"],
        },
        "C4": {
            "passed": c4.get("passed") is True,
            "core_claim_supported": c4.get("passed") is True,
            "evidence": {
                "paired_cases": c4["paired_case_count"],
                "scene_clusters": c4["scene_cluster_count"],
                "domains": len(c4["domains"]),
                "selective_minus_always_safe_cost": c4[
                    "primary_endpoint"
                ],
                "selective_minus_always_safe_success": c4[
                    "secondary_success_endpoint"
                ],
                "release_coverage": c4["release_coverage"],
                "released_attribution_precision": c4[
                    "released_attribution_precision"
                ],
            },
            "boundary": c4["claim_boundary"],
        },
    }
    all_frozen_acceptance_passed = all(
        item["passed"] for item in claims.values()
    )
    all_core_claims_supported = all(
        item["core_claim_supported"] for item in claims.values()
    )
    if all_frozen_acceptance_passed:
        status = "all_four_claims_pass_frozen_acceptance"
    elif all_core_claims_supported:
        status = (
            "all_four_core_claims_supported_"
            "c2_strict_superiority_not_confirmed"
        )
    else:
        status = "one_or_more_core_claims_not_supported"
    ledger = {
        "schema_version": "kinofail.realistic-c1-c4-evidence-ledger.v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "all_claims_passed": all_frozen_acceptance_passed,
        "all_frozen_acceptance_passed": all_frozen_acceptance_passed,
        "all_core_claims_supported": all_core_claims_supported,
        "claims": claims,
        "source_sha256": {
            claim: _sha(path) for claim, path in paths.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "all_claims_passed": ledger["all_claims_passed"],
                "all_core_claims_supported": (
                    ledger["all_core_claims_supported"]
                ),
                "output": str(args.output),
                "source_sha256": ledger["source_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if ledger["all_claims_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
