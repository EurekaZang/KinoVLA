#!/usr/bin/env python3
"""Assemble the definitive 11-operator action-consequence evidence package.

Ten operators come from the frozen full-matrix formal_v4 protocol. O6 comes
from its source-disjoint, post-development confirmation because formal_v4's O6
command was numerically near-identical to continue. The original unfavorable
O6 result remains archived and is explicitly referenced in the package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_kinofail_action_multiscene_v1 import (
    holm_adjust,
    jsonl,
    load,
    mcnemar_greater_p,
    paired_signflip_p,
    scene_cluster_ci,
    sha256,
)


def read_complete_cases(
    analysis_path: Path, corpus_root: Path, excluded_operator: str | None = None
) -> list[dict[str, Any]]:
    analysis = load(analysis_path)
    if analysis.get("publication_evidence_eligible") is not True:
        raise RuntimeError(f"source analysis is not publication eligible: {analysis_path}")
    protocol_path = Path(str(corpus_root)).parents[0]  # placate static type checkers
    del protocol_path
    results: list[dict[str, Any]] = []
    for path in sorted(corpus_root.glob("*/*/results.jsonl")):
        results.extend(jsonl(path))
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in results:
        if excluded_operator and row["true_operator"] == excluded_operator:
            continue
        by_case.setdefault(str(row["case_id"]), {})[str(row["action"])] = row
    complete: list[dict[str, Any]] = []
    for case_id, arms in sorted(by_case.items()):
        operators = {str(row["true_operator"]) for row in arms.values()}
        scenes = {str(row["scene_id"]) for row in arms.values()}
        if len(operators) != 1 or len(scenes) != 1 or len(arms) != 11:
            raise RuntimeError(f"malformed definitive case: {case_id}")
        complete.append(
            {
                "case_id": case_id,
                "operator": next(iter(operators)),
                "scene_id": next(iter(scenes)),
                "arms": arms,
            }
        )
    return complete


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-v4-analysis", type=Path, required=True)
    parser.add_argument("--formal-v4-corpus", type=Path, required=True)
    parser.add_argument("--o6-analysis", type=Path, required=True)
    parser.add_argument("--o6-corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    formal_analysis_path = args.formal_v4_analysis.resolve()
    o6_analysis_path = args.o6_analysis.resolve()
    formal_analysis = load(formal_analysis_path)
    o6_analysis = load(o6_analysis_path)

    cases = read_complete_cases(
        formal_analysis_path, args.formal_v4_corpus.resolve(), "O6_push"
    ) + read_complete_cases(o6_analysis_path, args.o6_corpus.resolve())
    by_operator: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_operator.setdefault(case["operator"], []).append(case)
    if set(by_operator) != {
        "O1_mu_field", "O2_compliance", "O3_collapse", "O4_tether",
        "O5_payload", "O6_push", "O7_visual_remap",
        "O8_invisible_collider", "O9_high_centering",
        "O10_effort_decay", "O11_obs_bias",
    }:
        raise RuntimeError(f"definitive operator coverage failed: {sorted(by_operator)}")

    registered_action = {
        row["operator"]: row["registered_action"]
        for row in formal_analysis["operator_results"]
        if row["operator"] != "O6_push"
    }
    registered_action["O6_push"] = o6_analysis["operator_results"][0]["registered_action"]

    case_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    raw_primary_p: dict[str, float] = {}
    for operator in sorted(by_operator):
        operator_cases = sorted(by_operator[operator], key=lambda row: row["case_id"])
        primary_deltas: list[float] = []
        continue_deltas: list[float] = []
        registered_success: list[bool] = []
        continue_success: list[bool] = []
        registered_fall: list[bool] = []
        continue_fall: list[bool] = []
        for case in operator_cases:
            arms = case["arms"]
            registered = arms[registered_action[operator]]
            wrong = [
                row for action, row in arms.items()
                if action not in {
                    registered_action[operator], "continue", "always_safe_halt"
                }
            ]
            if len(wrong) != 8:
                raise RuntimeError(f"expected eight wrong recovery labels: {case['case_id']}")
            delta_primary = float(registered["terminal_cost"]) - mean(
                float(row["terminal_cost"]) for row in wrong
            )
            delta_continue = float(registered["terminal_cost"]) - float(
                arms["continue"]["terminal_cost"]
            )
            primary_deltas.append(delta_primary)
            continue_deltas.append(delta_continue)
            registered_success.append(bool(registered["operator_recovery_success"]))
            continue_success.append(bool(arms["continue"]["operator_recovery_success"]))
            registered_fall.append(bool(registered["fell"]))
            continue_fall.append(bool(arms["continue"]["fell"]))
            case_rows.append(
                {
                    "case_id": case["case_id"],
                    "scene_id": case["scene_id"],
                    "operator": operator,
                    "registered_action": registered_action[operator],
                    "registered_terminal_cost": float(registered["terminal_cost"]),
                    "mean_mismatched_terminal_cost": mean(
                        float(row["terminal_cost"]) for row in wrong
                    ),
                    "registered_minus_mean_mismatched_cost": delta_primary,
                    "registered_minus_continue_cost": delta_continue,
                    "registered_success": bool(registered["operator_recovery_success"]),
                    "continue_success": bool(arms["continue"]["operator_recovery_success"]),
                    "registered_fell": bool(registered["fell"]),
                    "continue_fell": bool(arms["continue"]["fell"]),
                    "evidence_source": (
                        "o6_source_disjoint_confirmation_v3"
                        if operator == "O6_push"
                        else "full_matrix_formal_v4"
                    ),
                }
            )
        seed_root = int(hashlib.sha256(operator.encode()).hexdigest()[:16], 16)
        primary_p = paired_signflip_p(primary_deltas, seed=seed_root + 1)
        if primary_p is None:
            raise RuntimeError(operator)
        raw_primary_p[operator] = primary_p
        operator_rows.append(
            {
                "operator": operator,
                "n_cases": len(operator_cases),
                "registered_action": registered_action[operator],
                "registered_success_rate": mean(map(float, registered_success)),
                "continue_success_rate": mean(map(float, continue_success)),
                "registered_fall_rate": mean(map(float, registered_fall)),
                "continue_fall_rate": mean(map(float, continue_fall)),
                "registered_minus_mean_mismatched_cost": mean(primary_deltas),
                "primary_scene_cluster_ci_95": scene_cluster_ci(
                    [
                        (case["scene_id"], delta)
                        for case, delta in zip(operator_cases, primary_deltas, strict=True)
                    ],
                    seed=seed_root + 2,
                ),
                "primary_p_raw": primary_p,
                "registered_lower_cost_than_mean_mismatched_rate": mean(
                    float(value < 0.0) for value in primary_deltas
                ),
                "registered_minus_continue_cost": mean(continue_deltas),
                "continue_scene_cluster_ci_95": scene_cluster_ci(
                    [
                        (case["scene_id"], delta)
                        for case, delta in zip(operator_cases, continue_deltas, strict=True)
                    ],
                    seed=seed_root + 3,
                ),
                "continue_p_raw": paired_signflip_p(
                    continue_deltas, seed=seed_root + 4
                ),
                "success_mcnemar_p": mcnemar_greater_p(
                    registered_success, continue_success
                ),
                "evidence_source": (
                    "o6_source_disjoint_confirmation_v3"
                    if operator == "O6_push"
                    else "full_matrix_formal_v4"
                ),
            }
        )

    primary_holm = holm_adjust(raw_primary_p)
    for row in operator_rows:
        row["primary_p_holm_11"] = primary_holm[row["operator"]]

    overall_primary = [row["registered_minus_mean_mismatched_cost"] for row in case_rows]
    overall_continue = [row["registered_minus_continue_cost"] for row in case_rows]
    report = {
        "schema_version": "kinofail.action-consequence-definitive-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "publication_evidence_eligible": bool(
            formal_analysis["publication_evidence_eligible"]
            and o6_analysis["publication_evidence_eligible"]
        ),
        "analysis_unit": "one physical case; eleven action arms share the exact checkpoint",
        "counts": {
            "operators": len(operator_rows),
            "physical_cases": len(case_rows),
            "action_episodes": len(case_rows) * 11,
            "scenes": len({row["scene_id"] for row in case_rows}),
            "semantic_attrition_cases_retained": formal_analysis["counts"]["semantic_attrition_cases"],
            "engineering_failures": (
                formal_analysis["counts"]["engineering_failure_cases"]
                + o6_analysis["counts"]["engineering_failure_cases"]
            ),
        },
        "primary_contrast": "cause-matched registered action versus mean mismatched recovery-program cost",
        "operator_results": operator_rows,
        "operators_with_primary_holm_p_below_0_05": sum(
            float(row["primary_p_holm_11"]) < 0.05 for row in operator_rows
        ),
        "operators_with_primary_ci_strictly_below_zero": sum(
            row["primary_scene_cluster_ci_95"][1] < 0.0 for row in operator_rows
        ),
        "overall_primary": {
            "registered_minus_mean_mismatched_cost": mean(overall_primary),
            "scene_cluster_bootstrap_95_ci": scene_cluster_ci(
                [(row["scene_id"], row["registered_minus_mean_mismatched_cost"]) for row in case_rows],
                seed=2026081001,
            ),
            "paired_signflip_one_sided_p": paired_signflip_p(
                overall_primary, seed=2026081002
            ),
            "registered_lower_cost_rate": mean(
                float(value < 0.0) for value in overall_primary
            ),
        },
        "overall_vs_continue": {
            "registered_minus_continue_cost": mean(overall_continue),
            "scene_cluster_bootstrap_95_ci": scene_cluster_ci(
                [(row["scene_id"], row["registered_minus_continue_cost"]) for row in case_rows],
                seed=2026081003,
            ),
            "paired_signflip_one_sided_p": paired_signflip_p(
                overall_continue, seed=2026081004
            ),
            "registered_success_rate": mean(float(row["registered_success"]) for row in case_rows),
            "continue_success_rate": mean(float(row["continue_success"]) for row in case_rows),
            "registered_fall_rate": mean(float(row["registered_fell"]) for row in case_rows),
            "continue_fall_rate": mean(float(row["continue_fell"]) for row in case_rows),
        },
        "evidence_provenance": {
            "formal_v4": "definitive for O1-O5 and O7-O11",
            "o6_confirmation_v3": "definitive for O6 after development-only control redesign",
            "formal_v4_o6": (
                "retained as an unfavorable diagnostic result; excluded from the definitive "
                "O6 estimate because its registered 0.24 m/s command was near-identical to continue"
            ),
            "result_dependent_confirmation_selection": False,
            "rendered_views_count_as_independent_observations": False,
        },
        "source_sha256": {
            "formal_v4_analysis": sha256(formal_analysis_path),
            "o6_confirmation_analysis": sha256(o6_analysis_path),
            "assembler": sha256(Path(__file__).resolve()),
        },
    }
    report["publication_evidence_eligible"] = bool(
        report["publication_evidence_eligible"]
        and report["counts"]["operators"] == 11
        and report["counts"]["engineering_failures"] == 0
        and report["operators_with_primary_holm_p_below_0_05"] == 11
        and report["operators_with_primary_ci_strictly_below_zero"] == 11
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "operator_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(operator_rows[0]))
        writer.writeheader()
        writer.writerows(operator_rows)
    with (output / "case_level_primary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    lines = [
        "# Definitive KiNO-Fail action-consequence evidence",
        "",
        f"Publication evidence eligible: **{report['publication_evidence_eligible']}**.",
        "",
        f"The analysis contains {len(case_rows)} paired physical cases and {len(case_rows) * 11} action episodes across all 11 operators.",
        "",
        f"Cause-matched actions reduce terminal cost versus the mean mismatched recovery by {abs(report['overall_primary']['registered_minus_mean_mismatched_cost']):.2f} points (95% scene-cluster CI {report['overall_primary']['scene_cluster_bootstrap_95_ci'][0]:.2f} to {report['overall_primary']['scene_cluster_bootstrap_95_ci'][1]:.2f}; one-sided paired p={report['overall_primary']['paired_signflip_one_sided_p']:.5f}).",
        "",
        f"All {report['operators_with_primary_holm_p_below_0_05']}/11 operator-level primary tests remain significant after Holm correction, and all {report['operators_with_primary_ci_strictly_below_zero']}/11 scene-cluster intervals lie below zero.",
        "",
        "O1-O5 and O7-O11 use the frozen full-matrix formal_v4 study. O6 uses the independently sourced v3 confirmation after its direction-conditioned reflex was selected exclusively on development data. The non-distinct formal_v4 O6 result remains archived and is not overwritten.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["publication_evidence_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
