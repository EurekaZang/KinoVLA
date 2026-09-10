#!/usr/bin/env python3
"""Audit and summarize paired full-operator action-consequence studies.

The physical case, rather than an action arm or rendered view, is the unit of
analysis.  All contrasts are evaluated within the checkpoint shared by the
registered action, continue, and halt arms.  The script is valid for both the
three-arm development validation and the eleven-arm formal protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def rate(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return mean(float(bool(row[key])) for row in rows)


def average(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return mean(float(row[key]) for row in rows)


def number(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty quantile")
    position = (len(ordered) - 1) * probability
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def scene_cluster_ci(
    cells: list[tuple[str, float]], *, seed: int, draws: int = 20_000
) -> list[float] | None:
    by_scene: dict[str, list[float]] = defaultdict(list)
    for scene, value in cells:
        by_scene[scene].append(value)
    scenes = sorted(by_scene)
    if len(scenes) < 2:
        return None
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(draws):
        sample: list[float] = []
        for _cluster in scenes:
            sample.extend(by_scene[generator.choice(scenes)])
        estimates.append(mean(sample))
    return [quantile(estimates, 0.025), quantile(estimates, 0.975)]


def paired_signflip_p(values: list[float], *, seed: int) -> float | None:
    """One-sided randomization p-value for mean(values) < 0."""
    if not values:
        return None
    observed = mean(values)
    n = len(values)
    if n <= 20:
        extreme = 0
        total = 1 << n
        for mask in range(total):
            randomized = mean(
                value if mask & (1 << index) else -value
                for index, value in enumerate(values)
            )
            extreme += randomized <= observed + 1.0e-12
        return extreme / total
    generator = random.Random(seed)
    draws = 100_000
    extreme = 0
    for _ in range(draws):
        randomized = mean(
            value if generator.random() < 0.5 else -value for value in values
        )
        extreme += randomized <= observed + 1.0e-12
    return (extreme + 1.0) / (draws + 1.0)


def mcnemar_greater_p(
    registered: list[bool], comparator: list[bool]
) -> float | None:
    if not registered:
        return None
    favorable = sum(a and not b for a, b in zip(registered, comparator, strict=True))
    unfavorable = sum(not a and b for a, b in zip(registered, comparator, strict=True))
    discordant = favorable + unfavorable
    if discordant == 0:
        return 1.0
    return sum(
        math.comb(discordant, value) for value in range(favorable, discordant + 1)
    ) / (2.0**discordant)


def holm_adjust(raw: dict[str, float | None]) -> dict[str, float | None]:
    available = sorted(
        ((key, value) for key, value in raw.items() if value is not None),
        key=lambda item: item[1],
    )
    adjusted: dict[str, float | None] = {key: None for key in raw}
    running = 0.0
    total = len(available)
    for rank, (key, value) in enumerate(available):
        running = max(running, min(1.0, float(value) * (total - rank)))
        adjusted[key] = running
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-running", action="store_true")
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = load(protocol_path)
    schedule_path = Path(str(protocol["schedule"])).resolve()
    schedule = jsonl(schedule_path)
    schedule_by_id = {str(row["case_id"]): row for row in schedule}
    if len(schedule_by_id) != len(schedule):
        raise RuntimeError("duplicate case ids in schedule")
    if protocol.get("schedule_sha256") != sha256(schedule_path):
        raise RuntimeError("schedule hash mismatch")

    run_root = args.run_root.resolve()
    final_audit_path = run_root / "final_audit.json"
    if not final_audit_path.exists() and not args.allow_running:
        raise RuntimeError("collection is not terminal; pass --allow-running for a partial audit")
    attempt_paths = sorted((run_root / "attempts").glob("*.json"))
    attempts = [load(path) for path in attempt_paths]
    attempt_by_id = {str(row["case_id"]): row for row in attempts}

    corpus = args.corpus_root.resolve()
    results: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for path in sorted(corpus.glob("*/*/results.jsonl")):
        results.extend(jsonl(path))
    for path in sorted(corpus.glob("*/*/case_audits.jsonl")):
        audits.extend(jsonl(path))
    for path in sorted(corpus.glob("*/*/summary.json")):
        summaries.append(load(path))

    result_by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in results:
        case_id = str(row["case_id"])
        action = str(row["action"])
        if action in result_by_case[case_id]:
            raise RuntimeError(f"duplicate action arm: {case_id}/{action}")
        result_by_case[case_id][action] = row

    audit_by_case = {str(row["case_id"]): row for row in audits}
    correct_action = {
        str(key): str(value)
        for key, value in protocol["operator_to_registered_action"].items()
    }
    complete_cases: list[str] = []
    malformed: list[str] = []
    for case_id, row in schedule_by_id.items():
        expected = {str(action) for action in row["actions"]}
        actual = set(result_by_case.get(case_id, {}))
        audit = audit_by_case.get(case_id)
        if actual == expected and audit is not None and audit.get("passed") is True:
            complete_cases.append(case_id)
        elif case_id in attempt_by_id and attempt_by_id[case_id].get("state") == "terminal_accepted":
            malformed.append(case_id)

    by_operator: dict[str, list[str]] = defaultdict(list)
    for case_id in complete_cases:
        by_operator[str(schedule_by_id[case_id]["operator"])].append(case_id)

    table: list[dict[str, Any]] = []
    inference: dict[str, dict[str, Any]] = {}
    for operator in sorted(correct_action):
        case_ids = sorted(by_operator.get(operator, []))
        registered = [result_by_case[c][correct_action[operator]] for c in case_ids]
        continued = [result_by_case[c]["continue"] for c in case_ids]
        halted = [result_by_case[c]["always_safe_halt"] for c in case_ids]
        paired_cost_vs_continue = [
            float(result_by_case[c][correct_action[operator]]["terminal_cost"])
            - float(result_by_case[c]["continue"]["terminal_cost"])
            for c in case_ids
        ]
        paired_cost_vs_halt = [
            float(result_by_case[c][correct_action[operator]]["terminal_cost"])
            - float(result_by_case[c]["always_safe_halt"]["terminal_cost"])
            for c in case_ids
        ]
        strict_o9 = [
            bool(result_by_case[c]["continue"].get("strict_o9_semantics_seen"))
            for c in case_ids
        ] if operator == "O9_high_centering" else []
        planned = sum(
            str(row["operator"]) == operator for row in schedule_by_id.values()
        )
        table.append(
            {
                "operator": operator,
                "registered_action": correct_action[operator],
                "planned_cases": planned,
                "complete_paired_cases": len(case_ids),
                "registered_success_rate": rate(registered, "operator_recovery_success"),
                "continue_success_rate": rate(continued, "operator_recovery_success"),
                "halt_success_rate": rate(halted, "operator_recovery_success"),
                "registered_fall_rate": rate(registered, "fell"),
                "continue_fall_rate": rate(continued, "fell"),
                "halt_fall_rate": rate(halted, "fell"),
                "registered_mean_cost": average(registered, "terminal_cost"),
                "continue_mean_cost": average(continued, "terminal_cost"),
                "halt_mean_cost": average(halted, "terminal_cost"),
                "registered_minus_continue_mean_cost": (
                    mean(paired_cost_vs_continue) if paired_cost_vs_continue else None
                ),
                "registered_minus_halt_mean_cost": (
                    mean(paired_cost_vs_halt) if paired_cost_vs_halt else None
                ),
                "registered_lower_cost_than_continue_rate": (
                    mean(float(value < 0.0) for value in paired_cost_vs_continue)
                    if paired_cost_vs_continue else None
                ),
                "registered_lower_cost_than_halt_rate": (
                    mean(float(value < 0.0) for value in paired_cost_vs_halt)
                    if paired_cost_vs_halt else None
                ),
                "o9_continue_strict_semantics_rate": (
                    mean(float(value) for value in strict_o9) if strict_o9 else None
                ),
            }
        )
        scene_deltas = [
            (str(schedule_by_id[c]["scene_id"]), value)
            for c, value in zip(case_ids, paired_cost_vs_continue, strict=True)
        ]
        registered_success = [
            bool(row["operator_recovery_success"]) for row in registered
        ]
        continue_success = [
            bool(row["operator_recovery_success"]) for row in continued
        ]
        inference[operator] = {
            "primary_registered_minus_continue_cost": {
                "n_paired_cases": len(case_ids),
                "mean_difference": (
                    mean(paired_cost_vs_continue) if paired_cost_vs_continue else None
                ),
                "scene_cluster_bootstrap_95_ci": scene_cluster_ci(
                    scene_deltas,
                    seed=int(hashlib.sha256((operator + "/cost").encode()).hexdigest()[:16], 16),
                ),
                "paired_signflip_one_sided_p": paired_signflip_p(
                    paired_cost_vs_continue,
                    seed=int(hashlib.sha256((operator + "/sign").encode()).hexdigest()[:16], 16),
                ),
                "registered_lower_cost_rate": (
                    mean(float(value < 0.0) for value in paired_cost_vs_continue)
                    if paired_cost_vs_continue else None
                ),
            },
            "registered_minus_continue_success": {
                "difference": (
                    mean(float(value) for value in registered_success)
                    - mean(float(value) for value in continue_success)
                    if registered_success else None
                ),
                "mcnemar_one_sided_p": mcnemar_greater_p(
                    registered_success, continue_success
                ),
            },
        }

        mismatched_cost_deltas: list[float] = []
        best_wrong_cost_deltas: list[float] = []
        best_wrong_success: list[bool] = []
        for case_id in case_ids:
            arms = result_by_case[case_id]
            wrong = [
                row
                for action, row in arms.items()
                if action
                not in {
                    correct_action[operator],
                    "continue",
                    "always_safe_halt",
                }
            ]
            if not wrong:
                continue
            registered_row = arms[correct_action[operator]]
            mismatched_cost_deltas.append(
                float(registered_row["terminal_cost"])
                - mean(float(row["terminal_cost"]) for row in wrong)
            )
            best_wrong = min(wrong, key=lambda row: float(row["terminal_cost"]))
            best_wrong_cost_deltas.append(
                float(registered_row["terminal_cost"])
                - float(best_wrong["terminal_cost"])
            )
            best_wrong_success.append(bool(best_wrong["operator_recovery_success"]))
        if mismatched_cost_deltas:
            inference[operator]["registered_minus_mean_mismatched_recovery_cost"] = {
                "n_paired_cases": len(mismatched_cost_deltas),
                "mean_difference": mean(mismatched_cost_deltas),
                "scene_cluster_bootstrap_95_ci": scene_cluster_ci(
                    [
                        (str(schedule_by_id[c]["scene_id"]), value)
                        for c, value in zip(
                            case_ids, mismatched_cost_deltas, strict=True
                        )
                    ],
                    seed=int(
                        hashlib.sha256((operator + "/wrong-ci").encode()).hexdigest()[:16],
                        16,
                    ),
                ),
                "paired_signflip_one_sided_p": paired_signflip_p(
                    mismatched_cost_deltas,
                    seed=int(hashlib.sha256((operator + "/wrong").encode()).hexdigest()[:16], 16),
                ),
            }
            inference[operator]["registered_minus_oracle_best_wrong_cost"] = {
                "description": (
                    "the comparator is selected by minimum realized cost within each "
                    "case and is therefore deliberately favorable to the wrong actions"
                ),
                "n_paired_cases": len(best_wrong_cost_deltas),
                "mean_difference": mean(best_wrong_cost_deltas),
                "registered_lower_cost_rate": mean(
                    float(value < 0.0) for value in best_wrong_cost_deltas
                ),
                "paired_signflip_one_sided_p": paired_signflip_p(
                    best_wrong_cost_deltas,
                    seed=int(hashlib.sha256((operator + "/best-wrong").encode()).hexdigest()[:16], 16),
                ),
                "registered_minus_best_wrong_success": (
                    mean(float(value) for value in registered_success)
                    - mean(float(value) for value in best_wrong_success)
                ),
            }

    cost_raw = {
        operator: row["primary_registered_minus_continue_cost"][
            "paired_signflip_one_sided_p"
        ]
        for operator, row in inference.items()
    }
    success_raw = {
        operator: row["registered_minus_continue_success"]["mcnemar_one_sided_p"]
        for operator, row in inference.items()
    }
    cost_holm = holm_adjust(cost_raw)
    success_holm = holm_adjust(success_raw)
    mismatch_raw = {
        operator: row.get(
            "registered_minus_mean_mismatched_recovery_cost", {}
        ).get("paired_signflip_one_sided_p")
        for operator, row in inference.items()
    }
    mismatch_holm = holm_adjust(mismatch_raw)
    for operator in inference:
        inference[operator]["primary_registered_minus_continue_cost"][
            "holm_adjusted_p_across_operators"
        ] = cost_holm[operator]
        inference[operator]["registered_minus_continue_success"][
            "holm_adjusted_p_across_operators"
        ] = success_holm[operator]
        if "registered_minus_mean_mismatched_recovery_cost" in inference[operator]:
            inference[operator][
                "registered_minus_mean_mismatched_recovery_cost"
            ]["holm_adjusted_p_across_operators"] = mismatch_holm[operator]

    overall_cost_deltas: list[float] = []
    overall_scene_deltas: list[tuple[str, float]] = []
    overall_registered_success: list[bool] = []
    overall_continue_success: list[bool] = []
    overall_registered_fall: list[bool] = []
    overall_continue_fall: list[bool] = []
    for case_id in complete_cases:
        operator = str(schedule_by_id[case_id]["operator"])
        registered_row = result_by_case[case_id][correct_action[operator]]
        continue_row = result_by_case[case_id]["continue"]
        delta = float(registered_row["terminal_cost"]) - float(
            continue_row["terminal_cost"]
        )
        overall_cost_deltas.append(delta)
        overall_scene_deltas.append(
            (str(schedule_by_id[case_id]["scene_id"]), delta)
        )
        overall_registered_success.append(
            bool(registered_row["operator_recovery_success"])
        )
        overall_continue_success.append(
            bool(continue_row["operator_recovery_success"])
        )
        overall_registered_fall.append(bool(registered_row["fell"]))
        overall_continue_fall.append(bool(continue_row["fell"]))
    overall = {
        "n_paired_cases": len(overall_cost_deltas),
        "registered_success_rate": (
            mean(float(value) for value in overall_registered_success)
            if overall_registered_success else None
        ),
        "continue_success_rate": (
            mean(float(value) for value in overall_continue_success)
            if overall_continue_success else None
        ),
        "registered_fall_rate": (
            mean(float(value) for value in overall_registered_fall)
            if overall_registered_fall else None
        ),
        "continue_fall_rate": (
            mean(float(value) for value in overall_continue_fall)
            if overall_continue_fall else None
        ),
        "registered_minus_continue_mean_cost": (
            mean(overall_cost_deltas) if overall_cost_deltas else None
        ),
        "scene_cluster_bootstrap_95_ci": scene_cluster_ci(
            overall_scene_deltas, seed=20260809
        ),
        "paired_signflip_one_sided_p": paired_signflip_p(
            overall_cost_deltas, seed=20260810
        ),
        "registered_lower_cost_rate": (
            mean(float(value < 0.0) for value in overall_cost_deltas)
            if overall_cost_deltas else None
        ),
        "operators_with_negative_mean_cost_difference": sum(
            row["registered_minus_continue_mean_cost"] is not None
            and row["registered_minus_continue_mean_cost"] < 0.0
            for row in table
        ),
        "operators_with_complete_cases": sum(
            row["complete_paired_cases"] > 0 for row in table
        ),
    }
    overall_mismatched_deltas: list[float] = []
    overall_mismatched_scene_deltas: list[tuple[str, float]] = []
    for case_id in complete_cases:
        operator = str(schedule_by_id[case_id]["operator"])
        arms = result_by_case[case_id]
        wrong = [
            row
            for action, row in arms.items()
            if action
            not in {
                correct_action[operator],
                "continue",
                "always_safe_halt",
            }
        ]
        if not wrong:
            continue
        delta = float(arms[correct_action[operator]]["terminal_cost"]) - mean(
            float(row["terminal_cost"]) for row in wrong
        )
        overall_mismatched_deltas.append(delta)
        overall_mismatched_scene_deltas.append(
            (str(schedule_by_id[case_id]["scene_id"]), delta)
        )
    overall_mismatched = {
        "n_paired_cases": len(overall_mismatched_deltas),
        "registered_minus_mean_mismatched_recovery_cost": (
            mean(overall_mismatched_deltas) if overall_mismatched_deltas else None
        ),
        "scene_cluster_bootstrap_95_ci": scene_cluster_ci(
            overall_mismatched_scene_deltas, seed=20260811
        ),
        "paired_signflip_one_sided_p": paired_signflip_p(
            overall_mismatched_deltas, seed=20260812
        ),
        "registered_lower_cost_rate": (
            mean(float(value < 0.0) for value in overall_mismatched_deltas)
            if overall_mismatched_deltas else None
        ),
    }

    planned_cases = len(schedule)
    terminal_cases = len(attempts)
    accepted_cases = sum(row.get("state") == "terminal_accepted" for row in attempts)
    failed_cases = sum(row.get("state") == "terminal_failure" for row in attempts)
    semantic_attrition_case_ids: list[str] = []
    engineering_failure_case_ids: list[str] = []
    for row in attempts:
        if row.get("state") != "terminal_failure":
            continue
        case_id = str(row["case_id"])
        log_path = run_root / "logs" / f"{case_id}.log"
        log_text = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.exists()
            else ""
        )
        if (
            str(row.get("operator")) in {"O4_tether", "O9_high_centering"}
            and "operator onset missing" in log_text
        ):
            semantic_attrition_case_ids.append(case_id)
        else:
            engineering_failure_case_ids.append(case_id)
    o9_complete = by_operator.get("O9_high_centering", [])
    o9_strict_pass = all(
        bool(result_by_case[case_id]["continue"].get("strict_o9_semantics_seen"))
        for case_id in o9_complete
    )
    maximum_attrition = max(
        float(
            protocol.get(contract, {}).get(
                "maximum_overall_semantic_attrition_rate", 0.0
            )
        )
        for contract in ("o4_contract", "o9_contract")
    )
    publication_eligible = (
        final_audit_path.exists()
        and protocol.get("development_only") is False
        and len(complete_cases) + len(semantic_attrition_case_ids) == planned_cases
        and not engineering_failure_case_ids
        and len(semantic_attrition_case_ids) / planned_cases <= maximum_attrition
        and o9_strict_pass
        and not malformed
        and all(row.get("passed") is True for row in audits)
    )
    report = {
        "schema_version": "kinofail.action-multiscene-v1-analysis.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "development_only": bool(protocol.get("development_only")),
        "publication_evidence_eligible": publication_eligible,
        "analysis_unit": "one physical case with paired same-checkpoint action arms",
        "rendered_views_count_as_independent_observations": False,
        "counts": {
            "planned_cases": planned_cases,
            "terminal_attempts": terminal_cases,
            "accepted_cases": accepted_cases,
            "failed_cases": failed_cases,
            "semantic_attrition_cases": len(semantic_attrition_case_ids),
            "engineering_failure_cases": len(engineering_failure_case_ids),
            "complete_paired_cases": len(complete_cases),
            "result_arms": len(results),
            "malformed_accepted_cases": len(malformed),
        },
        "failed_case_ids": sorted(
            str(row["case_id"])
            for row in attempts
            if row.get("state") == "terminal_failure"
        ),
        "semantic_attrition_case_ids": sorted(semantic_attrition_case_ids),
        "engineering_failure_case_ids": sorted(engineering_failure_case_ids),
        "semantic_attrition_rate": (
            len(semantic_attrition_case_ids) / planned_cases
        ),
        "maximum_allowed_semantic_attrition_rate": maximum_attrition,
        "o9_all_complete_continue_arms_pass_strict_semantics": o9_strict_pass,
        "malformed_accepted_case_ids": sorted(malformed),
        "operator_results": table,
        "paired_inference": inference,
        "overall_registered_vs_continue": overall,
        "overall_registered_vs_mean_mismatched_recovery": overall_mismatched,
        "source_sha256": {
            "protocol": sha256(protocol_path),
            "schedule": sha256(schedule_path),
            "final_collection_audit": (
                sha256(final_audit_path) if final_audit_path.exists() else None
            ),
        },
        "unfavorable_outcomes_retained": True,
        "result_dependent_retry_or_selection": False,
    }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "analysis.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output / "operator_results.csv"
    if table:
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)

    lines = [
        "# KiNO-Fail full-operator action study audit",
        "",
        f"Protocol: `{protocol['protocol_id']}`",
        "",
        f"Complete paired cases: {len(complete_cases)}/{planned_cases}; "
        f"terminal failures: {failed_cases}.",
        "",
        "| Operator | n | Registered success | Continue | Halt | Reg. cost | Continue cost | Halt cost | Reg.-continue |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            "| {operator} | {complete_paired_cases}/{planned_cases} | {rs} | {cs} | {hs} | {rc} | {cc} | {hc} | {delta} |".format(
                **row,
                rs=number(row["registered_success_rate"]),
                cs=number(row["continue_success_rate"]),
                hs=number(row["halt_success_rate"]),
                rc=number(row["registered_mean_cost"]),
                cc=number(row["continue_mean_cost"]),
                hc=number(row["halt_mean_cost"]),
                delta=number(row["registered_minus_continue_mean_cost"]),
            )
        )
    lines.extend(
        [
            "",
            "All contrasts are paired within the exact restored checkpoint. "
            "Development protocols are never labeled as publication evidence.",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    return 0 if (args.allow_running or final_audit_path.exists()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
